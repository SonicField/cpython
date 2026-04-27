# Per-Instance Dynamic Watchpoint via tp_alloc Hook — Design

Per shepard 2026-04-26T14:28:49Z directive + pythia #165 #2: heavy-tier
discriminator design for W-PYTORCH-CM-(ii). Design-only (not
implementation) per supervisor stand-down + heavy-tier authorization
gate.

## Problem Recap

W-PYTORCH-CM-(ii): runtime SEGV at PyDict_SetItem (NULL+0xAB deref).
Mechanism narrowed to LSB-clear at obj-0x18 (PEP 697 IsValues encoding
0x97→0x96 → IsDict misinterpretation → NULL ob_type → SEGV). Writer
NOT in JIT'd cache slots (TYPE 0xd33020 + VALUE 0xd42018 + cache_
0xd5b2a0 all stable per generalist 13:06Z/13:39Z/14:16Z watchpoint
falsifications). Writer NOT identifiable by header-byte (refcnt +
type_ptr + first8 stable across recycled fresh _NoGrad instances per
testkeeper 14:20:44Z native D2+id falsifier).

Remaining classes: (a) narrow 1-byte writer / (b) wider write clipping
LSB / (d) two-instance conflation. All require per-instance discrimination
that cheap-tier substrates cannot provide.

## Design

Hook `_NoGrad` type's `tp_alloc` (or PyType_GenericAlloc replacement
for that class). On each allocation: set a hardware watchpoint on the
newly-allocated obj + 0x18, byte 0 (the LSB of managed-dict slot).
Watchpoint fires on any write to that byte. Handler captures backtrace
+ writer site. Clear watchpoint at tp_dealloc for safe recycling.

### Hook Site

```c
// In Objects/typeobject.c or in a small phx_pytorch_cm_debug.c
// Hook installed at module init (PHOENIX_W_PYTORCH_CM_DEBUG):

#ifdef PHOENIX_W_PYTORCH_CM_DEBUG
#include <sys/ptrace.h>
#include <sys/user.h>
#include <signal.h>

static PyTypeObject *phx_target_type = NULL;  // _NoGrad after lookup
static int phx_active_watchpoint_dr = -1;     // which DR0-DR3 used

static PyObject *phx_hooked_alloc(PyTypeObject *tp, Py_ssize_t nitems) {
    PyObject *obj = PyType_GenericAlloc(tp, nitems);
    if (obj && tp == phx_target_type && phx_active_watchpoint_dr < 0) {
        // Set hardware watchpoint on obj + 0x18, byte 0, write-only
        phx_set_hw_watchpoint((char *)obj + 0x18, /*size=*/1, /*write=*/1);
        phx_active_watchpoint_dr = 0;
    }
    return obj;
}

static void phx_hooked_dealloc(PyObject *obj) {
    if (phx_active_watchpoint_dr >= 0 && /*obj matches watched*/) {
        phx_clear_hw_watchpoint(phx_active_watchpoint_dr);
        phx_active_watchpoint_dr = -1;
    }
    // delegate to original tp_dealloc
}

void phx_install_pytorch_cm_hooks(void) {
    PyObject *m = PyImport_ImportModule("torch._C");
    PyObject *cls = PyObject_GetAttrString(m, "_NoGrad");
    phx_target_type = (PyTypeObject *)cls;
    // Install via tp_alloc/tp_dealloc swap or via type slot mutation
}
#endif
```

### Hardware Watchpoint Mechanism

x86_64: 4 hardware debug registers (DR0-DR3) usable for watchpoints.
Use `ptrace(PTRACE_POKEUSER, tid, offsetof(user, u_debugreg[N]), addr)`
to set, `+ DR7 control bits` to enable byte-granularity write-watch.
Or via prctl/perf_event_open for ptrace-free path (preferred — keeps
gdb available).

Constraints:
- x86_64: 4 watchpoints max
- Watchpoint fires on ANY write to byte (regardless of size; 1-byte
  watchpoint catches both narrow 1-byte writes AND wider writes whose
  store touches that byte)
- ARM64: differs (FAR_EL1 register), but Phoenix's primary x86 first

### Handler

When watchpoint fires (SIGTRAP), handler captures:
- Faulting instruction address (RIP / PC)
- Backtrace (libunwind or backtrace_symbols)
- Register dump (R8-R15, RAX-RDX, RSP, RBP)
- Watched address contents pre/post (compare to determine LSB-clear)
- Suppress secondary fires (prevent watchpoint-loop)

Handler logs to file (not stdout — stdout perturbs JIT timing per
D-1777190733 wrapper-perturbation finding):

```c
static void phx_wp_handler(int sig, siginfo_t *si, void *ucontext) {
    ucontext_t *uc = (ucontext_t *)ucontext;
    void *rip = (void *)uc->uc_mcontext.gregs[REG_RIP];
    char buf[1024];
    int n = snprintf(buf, sizeof buf,
        "WP HIT rip=%p slot=%p value=0x%02x\n",
        rip, watched_addr, *(unsigned char *)watched_addr);
    write(phx_log_fd, buf, n);
    // backtrace
    void *frames[32];
    int nframes = backtrace(frames, 32);
    backtrace_symbols_fd(frames, nframes, phx_log_fd);
    // re-arm watchpoint OR clear if first-fire-only
}
```

## Discrimination Outcomes

After this instrumentation runs:

- **(a) narrow 1-byte writer:** WP fires; backtrace shows a 1-byte
  store instruction (`mov %al, (mem)` / `andb $...,(mem)`). Writer
  site identified.

- **(b) wider write clipping LSB:** WP fires; backtrace shows wider
  store (`mov %rax, (mem)` / `mov %eax, (mem)`). Writer site identified
  + diagnose upstream (where did %rax/%eax get a value with low byte
  0x96?).

- **(d) two-instance conflation:** WP DOES NOT fire on watched
  instance, BUT D2 still captures LSB transition on a DIFFERENT
  instance. The watchpoint is on instance #N; the LSB transition is
  on instance #N+1 at recycled address (which would need its own
  watchpoint at tp_alloc).

  To strengthen (d) test: instrument tp_alloc to set watchpoint on
  EVERY new _NoGrad instance (cycling through DR0-DR3). 4 simultaneous
  watchpoints + reasonable workload coverage. If LSB transition occurs
  but NO watchpoint fires on the corrupted address, (d) confirmed
  AND the corruption isn't actually a write — it's address recycling.

## Perturbation Analysis

- tp_alloc hook: ~10ns overhead (1 ptr compare + maybe ptrace syscall
  on hit) — only fires on _NoGrad allocations
- Hardware watchpoint setup: ~1µs ptrace syscall (or ~100ns prctl path)
- Watchpoint fires: ~1µs context switch + handler ~10µs (backtrace
  dominates)
- Total per-instance overhead: ~10-15µs at allocation only

vs Python wrapper (D-1777190733 perturbed): wrapper added ~100µs per
__enter__ Python interpreter overhead. C-side hook is 1-2 orders of
magnitude lighter.

Caveat per pythia #157 #2: "perturbation-free vs Python wrapper" is
ASSERTION not proven. Hardware watchpoints add small SIGTRAP context-
switch overhead on every store-to-watched-byte; if writer fires
frequently (e.g., every iteration), cumulative overhead may shift JIT
timing. Mitigation: clear watchpoint after first fire to capture
single writer event, then re-arm for subsequent instances if needed.

## Substrate-Match Limit

This design discriminates writer-IDENTITY (which code path writes the
LSB byte) but does NOT determine WHO LOGICALLY OWNS the slot writeup.
If the writer is in JIT-emitted code, backtrace identifies the JIT
function but not the source-level Python construct that compiled to
it. JIT-emit writer tracing requires augmenting backtrace with
JIT-symbol-table (similar to existing CompiledFunction::disassemble).

This caveat replaces pythia #156 #1 substrate-mismatch concern at the
writer-identification layer.

## Resumption Procedure

1. Apply this design as small diff (~200 LOC) to Objects/typeobject.c
   or new Modules/_phoenix_pytorch_cm_debug.c module
2. Hook installed at module init (importable Python module)
3. Rebuild with PHOENIX_W_PYTORCH_CM_DEBUG defined
4. Run repro_s3.py
5. Inspect WP-handler log for fire events + backtraces
6. Discriminate per outcomes above

## Cost Estimate

- Implementation: ~200 LOC new file + ~10 LOC hook install + tp_alloc
  swap mechanism. ~30-60min implementation.
- Build: ~5min Phoenix rebuild with new flag.
- Run: ~1min bench_pytorch_cm with WP active.
- Analysis: ~5-15min trace interpretation.
- Total: ~45-90min from auth to evidence.

vs allocate-counter design (~200 LOC, also heavy-tier per governance
D-1777190699): roughly equivalent cost. Difference is what they
discriminate:
- allocate-counter: per-instance ID for D2 correlation (R1 vs R2)
- per-instance watchpoint: writer-identity localization (a vs b)

Both are heavy-tier and address different open classes. Both gated on
Alex direction OR explicit heavy-tier authorization.

## Out of Scope

- Implementation: design-only per stand-down. Implementation requires
  governance authorization.
- ARM64: x86_64-first per Phoenix priority; ARM64 watchpoint
  mechanism (FAR_EL1) differs and would need parallel design.
- Production deployment: PHOENIX_W_PYTORCH_CM_DEBUG gated; off by
  default; debug-only.
- Multi-thread safety: GIL-protected workload assumed; if
  workload spawns threads, watchpoint setup needs per-thread ptrace.

## Comparison with Allocate-Counter

| Concern               | Allocate-counter             | tp_alloc watchpoint       |
|-----------------------|------------------------------|---------------------------|
| Discriminates         | R1 vs R2 (instance identity) | writer site (a vs b)      |
| Cost                  | ~200 LOC + rebuild           | ~200 LOC + rebuild        |
| Perturbation risk     | side-table hash + mutex      | ptrace + SIGTRAP per fire |
| JIT-emit writer       | doesn't address              | catches via backtrace     |
| (d) discrimination    | direct (alloc_id mismatch)   | indirect (no fire on      |
|                       |                              | watched instance)         |
| Open issue post-run   | which writer (use this next) | which instance (use AC)   |

Recommendation: implement BOTH if heavy-tier authorized. Allocate-counter
discriminates (d); tp_alloc-watchpoint discriminates (a)/(b). Together
they close the remaining open classes.

If only ONE authorized: tp_alloc-watchpoint first — directly identifies
writer if (a) or (b); falsifies both if no fire (then (d) is the
remaining hypothesis, allocate-counter becomes load-bearing).
