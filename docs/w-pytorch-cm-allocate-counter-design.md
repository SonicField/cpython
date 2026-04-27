# Perturbation-Free C-Side Allocate-Counter Design

Per shepard 2026-04-26T08:13:27Z + supervisor 08:14:10Z directive.
Design-only (not implementation) per stand-down. Bound: ≤200 LOC,
≤1 file change. Ready-tool for resumption.

## Problem Recap

- W-PYTORCH-CM-PHOENIX-CRASH: D2 captures slot LSB transition 1→0 at
  same obj address but cannot disambiguate "same instance with mutated
  slot" (Hyp A) vs "different instances at recycled address with
  snapshot-conflation" (Hyp B refined). Both empirically anchored at
  premise level (T2.5 confirmed PyMem recycle; D2[1] slot value invalid
  per any PEP 697 state).
- Cheap printf class exhausted: Python __enter__ wrapper perturbs JIT
  timing enough to avoid trigger (testkeeper 08:03:56Z).
- D3d showed id() == &obj per CPython, so address-based identity
  conflates recycled instances. Need a per-instance identifier
  independent of address.

## Design

Add a 64-bit monotonic allocation counter to PyHeapTypeObject's
allocated instances via a side-table (not in-instance — modifying
PyObject layout breaks ABI + perturbs heap layout).

### Side-Table Approach (recommended)

Use a thread-local or global hash table keyed by obj-pointer with a
counter value, populated at tp_alloc completion + cleared at tp_dealloc.

```c
// In Objects/dictobject.c (single-file change per supervisor bound)
// near the existing T2.5 instrumentation:

#ifdef PHOENIX_W_PYTORCH_CM_DEBUG
#include <stdatomic.h>
static _Atomic uint64_t phx_alloc_counter = 0;
typedef struct {
    PyObject *obj;
    uint64_t alloc_id;
    PyTypeObject *tp;  // captured at alloc-time for filter
} PhxAllocRec;

#define PHX_TABLE_SZ 4096  // power-of-2; size to typical recycle-cycle
static PhxAllocRec phx_alloc_table[PHX_TABLE_SZ];
static pthread_mutex_t phx_table_mu = PTHREAD_MUTEX_INITIALIZER;

static inline size_t phx_hash(PyObject *obj) {
    return ((uintptr_t)obj >> 4) & (PHX_TABLE_SZ - 1);
}

void phx_record_alloc(PyObject *obj, PyTypeObject *tp) {
    uint64_t id = atomic_fetch_add(&phx_alloc_counter, 1);
    pthread_mutex_lock(&phx_table_mu);
    size_t h = phx_hash(obj);
    // Linear probe (open addressing) for collision; small N expected.
    for (size_t i = 0; i < PHX_TABLE_SZ; i++) {
        size_t idx = (h + i) & (PHX_TABLE_SZ - 1);
        if (phx_alloc_table[idx].obj == NULL || phx_alloc_table[idx].obj == obj) {
            phx_alloc_table[idx] = (PhxAllocRec){obj, id, tp};
            break;
        }
    }
    pthread_mutex_unlock(&phx_table_mu);
}

uint64_t phx_lookup_alloc_id(PyObject *obj) {
    pthread_mutex_lock(&phx_table_mu);
    size_t h = phx_hash(obj);
    uint64_t id = 0;
    for (size_t i = 0; i < PHX_TABLE_SZ; i++) {
        size_t idx = (h + i) & (PHX_TABLE_SZ - 1);
        if (phx_alloc_table[idx].obj == obj) {
            id = phx_alloc_table[idx].alloc_id;
            break;
        }
        if (phx_alloc_table[idx].obj == NULL) break;  // not found
    }
    pthread_mutex_unlock(&phx_table_mu);
    return id;
}
#endif
```

### Hook Points

1. **Record at tp_alloc completion**: in init_inline_values (where T2.5
   already instruments), call `phx_record_alloc(obj, tp)` for filtered
   types (_NoGrad, _Autocast). Captures alloc_id at instance birth.

2. **Lookup at D2 print**: in StoreAttrCache::invoke (or the existing
   D2 print site), call `phx_lookup_alloc_id(obj)` and add to D2
   output. D2 format becomes:
   ```
   D2[n] obj=0x... alloc_id=N refcnt=R ... slot=0x... LSB=X
   ```
   alloc_id distinguishes Instance N from Instance N+1 at same recycled
   address.

3. **(Optional) Clear at tp_dealloc**: call clear function to reuse
   slot. NOT strictly needed for n=200 D2 events; table can fill +
   wrap-around if collisions tolerable for diagnostic purposes.

## Perturbation Analysis

- Atomic counter increment: ~1ns, no JIT-timing impact
- Hash-table linear probe: O(load_factor) average, small for 4096-entry
  table with hundreds of live instances
- Mutex acquire/release: ~10-100ns; only on alloc + D2 print
- Total per-StoreAttr overhead: <1µs at D2 print site (200 events =
  <200µs total)
- vs Python wrapper: ~10-100µs per __enter__ call (Python interpreter
  overhead) — wrapper perturbed because Python-level overhead changes
  JIT-call-counter timing relative to compile threshold; C-side hooks
  at allocator + StoreAttrCache slow-path are pre-compile-threshold,
  shouldn't shift JIT activation window.

## Discrimination Outcomes

After this instrumentation runs, D2 output reveals one of:

- **Hyp A (same instance, mutated slot):** D2[0].alloc_id == D2[1].alloc_id
  AND D2[0].obj == D2[1].obj. Same instance, slot was mutated between
  the two StoreAttrCache invocations.

- **Hyp B (different instances at recycled address):** D2[0].alloc_id !=
  D2[1].alloc_id AND D2[0].obj == D2[1].obj. Two distinct instances,
  D2 conflated their snapshots.

- **Anomaly:** alloc_id == 0 (lookup miss) — instance allocated via path
  not hooked. Investigate: which allocator path missed phx_record_alloc?

## Resumption Procedure

1. Apply this design as 1-file diff to Objects/dictobject.c (≤200 LOC
   per supervisor bound)
2. Rebuild with PHOENIX_W_PYTORCH_CM_DEBUG defined
3. Run repro_s3.py
4. Inspect D2 output for alloc_id field
5. Discriminate Hyp A vs Hyp B per outcomes above
6. If Hyp A: writer-search continues (next: rr/ptrace per T1/T2 in
   tooling-note)
7. If Hyp B: D2 was misleading; original "1-byte writer" framing
   downgraded; mechanism reframes to instance-conflation, search
   shifts to "why does Instance N+1 see slot in unexpected state"

## Substrate-Match Limit (per pythia #157 #1)

This design discriminates Hyp A (single-instance mutation) vs Hyp B
(instance-conflation) by adding alloc_id correlation. It does NOT
discriminate WHO wrote the bad LSB byte if Hyp A confirms — and
crucially, JIT-EMITTED machine-code writes do NOT appear in
dictobject.c source. If the writer lives in JIT-emitted code,
dictobject.c instrumentation is silent.

This is the same substrate-match concern pythia #156 #1 raised against
the source-grep audit. Allocate-counter design inherits the limit:
narrows recycle-vs-source-side-write, NOT JIT-emit-write.

For JIT-emit writer hunt, rr/ptrace (T1/T2 in tooling-note) is the
substrate-matching tool. This design is upstream of that — answers
"is there a writer to hunt" first; rr/ptrace finds the writer if yes.

## Perturbation-Tolerance Caveat (per pythia #157 #2)

"Perturbation-free vs Python wrapper" is the design ASSERTION, not
proven. Bug already evaded printf-class instrumentation under D2+id
wrapper (testkeeper 08:03Z). C-side hooks are timing-different from
Python wrapper but still printf-class. If perturbation tolerance is
narrower than wrapper-specific (i.e., bug also evades C-side printf),
this design ALSO won't reproduce trigger.

Risk-mitigation: implement + test BEFORE assuming success. If
implementation reproduces trigger and discriminates A/B, design
validated. If trigger evades again, escalate to rr/ptrace immediately
(don't author another printf-class design layer per pythia #157 #4
six-month-regret pattern).

## Out of Scope

- Implementation: design-only per stand-down. Implementation requires
  governance authorization (currently stand-down for >5min rebuild).
- Multi-thread safety: pthread_mutex assumed; CPython GIL likely
  obviates per-instance lock contention concerns for typical workloads.
- Wrap-around: 64-bit counter doesn't wrap at any realistic test scale.
- Production deployment: PHOENIX_W_PYTORCH_CM_DEBUG gated; off by
  default; debug-only.