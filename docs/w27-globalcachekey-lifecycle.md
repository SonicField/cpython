# W27 — GlobalCacheKey Raw-Pointer Lifecycle Re-Architecture

**Status:** Filed (generalist, 2026-04-22, supervisor-authorized [chat ~16:04Z, 16:45Z W27-ownership update])
**Owner:** theologian (design), generalist (implementation) — per supervisor [chat 16:45Z #3]
**Schedule:** Entry-trigger = Tier 5 ZERO C++ complete AND no active push-class blocker. Exit by start of cinderx Tier 6 cleanup.
**Falsification test:** see §8 below — must land BEFORE re-architecture is considered design-complete.
**Origin:** push 59 ARM64 pydebug SEGV root-cause (lldb @ global_cache.cpp:325)

---

## 1. Problem statement

`jit::GlobalCacheKey` (Python/jit/global_cache.h) holds **raw `PyDictObject*` pointers** to a function's `globals` and `builtins` dicts:

```cpp
struct GlobalCacheKey {
  PyDictObject* builtins;        // raw pointer
  PyDictObject* globals;         // raw pointer
  Ref<PyUnicodeObject> name;     // refcounted
  ...
};
```

These pointers can outlive the dicts they reference. `GlobalCacheManager` watches each dict via the dict-watcher API, but the **PyDict_EVENT_DEALLOCATED** handler is intentionally a no-op (commit 314d0f2310, March 2026) due to a 2026-03-30 GC re-entrancy crash in `notifyDictUnwatch`.

Result: when `globals` is freed while `builtins` is still alive, the cache entry remains in `map_` with a stale `globals` pointer. Subsequent events on `builtins` (e.g. `notifyDictClear`) call `updateCache(cache, builtins, nullptr)`, which dereferences `cache.key().globals` → use-after-free.

## 2. Observed manifestation

- **Push 59** (b44f0752eb, 2026-04-22): falsifier `test_phoenix_jit_inline_except_closure` triggered ARM64 pydebug SEGV.
- lldb backtrace: `phoenix_dict_watcher → notifyDictClear → updateCache → hasOnlyUnicodeKeys → PyType_HasFeature(0xdddd…)`.
- Pattern: pydebug poison-fill on freed PyTypeObject reached via stale `globals` pointer.
- x86_64-release silent because freed memory still contains stale-but-readable type pointer.

## 3. Finalize-phase mitigation (LANDED, this commit)

Two defensive guards address the **finalize-specific** manifestation:
- `phoenix_dict_watcher`: early-return on `Py_IsFinalizing()` — skip processing all dict events during shutdown.
- `GlobalCacheManager::clear()`: short-circuit on `Py_IsFinalizing()` — drop in-memory state without calling `Ci_Watchers_UnwatchDict` (watcher infra also torn down by then).

These DO NOT address the deeper class.

## 4. Unaddressed class

**Mid-execution module unload**. If a Python module is unloaded (`del sys.modules[name]`, refcount on its `__dict__` reaches 0, dict deallocates) while the JIT cache holds a `GlobalCacheKey` keyed on that dict, the same UAF triggers — but `Py_IsFinalizing()` is `false`, so neither guard fires.

Reproduction sketch (not yet tested):
```python
import sys, weakref, gc
import some_jit_compiled_module as m
m.compiled_function()  # registers global cache entry
ref = weakref.ref(m.__dict__)
del sys.modules['some_jit_compiled_module']
del m
gc.collect()
assert ref() is None  # globals freed
# Any subsequent event on builtins that touches the stale cache → UAF
```

## 5. Re-architecture options

### Option A: Refcount the cached dicts
Hold `Ref<PyDictObject>` instead of raw `PyDictObject*` in `GlobalCacheKey`. Pro: simplest. Con: extends dict lifetime, may interfere with module unload semantics (modules expected to be GC'd after `del sys.modules[name]`).

### Option B: Weak references via dict watcher
Implement true weak-reference semantics by handling `PyDict_EVENT_DEALLOCATED` correctly. Requires resolving the 2026-03-30 GC re-entrancy crash. Suggested approach: defer cache cleanup to a per-thread queue drained on the next safe point, rather than synchronous `notifyDictUnwatch` from inside the watcher callback.

### Option C: Cache validation on read
Mark cache entries invalid via a separate shadow map indexed by dict pointer; check validity at every cache read. Pro: lifecycle-independent. Con: per-read overhead.

### Option D: Restructure cache key
Drop `globals`/`builtins` raw pointers from the key entirely; key on the JIT'd function's identity, look up dicts dynamically. Pro: no aliasing issue. Con: changes hash semantics, cache invalidation gets complex.

### Option E: Sync inline mark-invalidate, no companion-touch (W-A4-mid landing)
Per theologian 2026-04-24T19:47:12Z reconciliation. Refines Option B
without the deferred-queue infrastructure: handle `PyDict_EVENT_DEALLOCATED`
inline, but ONLY on Phoenix-local state — never call
`Ci_Watchers_UnwatchDict` (the 2026-03-30 crash trigger was specifically
companion-dict touch). Operations:
1. iterate `GlobalCacheManager::map_` for entries with key `globals==dict`
   or `builtins==dict`; mark each cache entry value-cleared / drop
2. erase `watch_map_[dict]` entry
All operations on Phoenix's own GIL-protected state; no PyDict internals
touched, no companion-dict touched. Compliant with CPython "watcher
callback should not mutate watched state" guidance (mutates only
Phoenix-local maps, not the dict).

Estimate: ~1 session vs Option B's 3-4hr (no queue infra needed).

Likely best path: **Option E** for the W-A4-mid mid-execution-unload
class. Option B (deferred-queue) remains the cleaner long-term shape if
future events require additional callback work that COULD touch the dict
(in which case the queue's safe-point drain is needed); for the
mid-execution-unload class alone, Option E's "no companion-touch"
constraint is sufficient.

## 6. Sequencing

- Pre-req: Tier 5 close (zero C++ in builder.cpp emit methods)
- Falsifier needed before fix lands: mid-execution module-unload reproducer
- Estimated scope: 3-4 hours for Option B, 1-2 hours for Option A (with caveats)
- Risk: medium — touches global cache invariants, requires re-verification of all global-cache-related tests

## 7. Cross-references

- Push 59 root-cause analysis: chat 2026-04-22 ~15:55-16:04Z
- Original DEALLOCATED-skip: 314d0f2310 (Alex, 2026-03-30)
- Finalize-phase mitigation: this commit (paired with push 59)
- W27 ownership + falsification gate: pythia #76 [chat 16:45Z], supervisor revision [chat 16:45Z #3]
- Push 58 ARM64-pydebug coredump finding (independent confirmation of pre-existing class):
  testkeeper [chat 2026-04-22 16:45Z] — 8b7b9351a1 binary on the SAME falsifier
  workload also produced shutdown coredump, just non-deterministic vs push 59's
  deterministic SEGV. Pythia #75 (b1) outcome confirmed.
- Related lifecycle pattern: feedback_no_workarounds.md

## 8. Falsification test (REQUIRED before re-architecture is design-complete)

**Purpose:** ensure W27 has a revisit-trigger and is not silently fallow.

**Workload sketch** (`Lib/test/test_phoenix_w27_module_unload_uaf.py`, NOT YET WRITTEN):

```python
import sys, weakref, gc, importlib, unittest

class TestModuleUnloadUAF(unittest.TestCase):
    def test_globals_unloaded_while_cache_alive(self):
        """Verify cache entry keyed on a freed globals dict doesn't UAF
        when builtins receives a subsequent dict event.

        EXPECTED FAIL pre-W27: SEGV under pydebug (poison-fill on stale
        globals pointer) when GlobalCacheManager processes builtins
        event and derefs cache.key().globals.

        EXPECTED PASS post-W27: cache entry dropped on globals
        deallocate (Option B deferred-cleanup or Option A refcount).
        """
        # 1. Import a module whose JIT-compiled function uses LOAD_GLOBAL
        # 2. Run the function above auto-compile threshold (1000 calls)
        #    to register a GlobalCacheKey for (module_globals, builtins, name)
        # 3. weakref the module's __dict__
        # 4. del sys.modules[module_name] + del local module ref + gc.collect()
        # 5. Assert weakref is dead (globals freed)
        # 6. Trigger a builtins event (e.g., assign a new builtin via
        #    __builtins__.foo = bar) to fire phoenix_dict_watcher
        # 7. Pre-W27 expect: SEGV under pydebug ARM64
        # 8. Post-W27 expect: clean (cache entry dropped on dealloc)
```

**Status:** SKIPPED until W27 work begins — `@unittest.skip("W27 falsification, written before re-architecture lands; reproducer for mid-execution module-unload UAF")`. Skip reason includes the W27 doc reference so test discovery surfaces this as the W27 trigger.

**Acceptance criteria:**
- Test reproduces SEGV at HEAD (pre-W27) under ARM64 pydebug — proves the latent class is testable, not theoretical.
- Test passes after W27 fix lands — proves the re-architecture closed it.
- If the test cannot reproduce the UAF (i.e., the latent class is harder to trigger than predicted), W27 scope re-opens for design re-evaluation; do not silently close.

**Implementation owner:** generalist (writes test before re-architecture work begins, per W27 falsification-first discipline).

---

## 9. W-A4-mid attempt 2026-04-24: CLOSED-by-non-reproduction

Per W27 §8 acceptance criteria 'If the test cannot reproduce the UAF
(i.e., the latent class is harder to trigger than predicted), W27
scope re-opens for design re-evaluation; do not silently close.'

Attempt summary (theologian + supervisor + testkeeper + generalist
2026-04-24T19:44Z–19:59Z):

1. Synthetic falsifier `Lib/test/test_phoenix_w27_module_unload_uaf.py`
   written per §8 sketch. Does NOT reproduce: function's `__globals__`
   strong-ref keeps mod.__dict__ alive past `del mod`; the freed-dict
   scenario the test tries to construct never happens (testkeeper
   19:54:21Z + valgrind-clean confirmation 19:56:02Z).

2. Substitute falsifier: re-use existing `test_phoenix_jit_inline_except_closure`
   (W27 doc §2 push-59 origin). Result: SIGABRT under x86_64 pydebug
   on `test_load_deref_in_except_basic`, but NOT a W27 dict-watcher
   UAF — fires the Phase B I1 invariant from
   push 40 4145fe3fb0 SECOND-PILOT (`bc_block_array.count !=
   block_map_phx.count`). DIFFERENT bug class surfaced as a side
   effect (testkeeper 19:58:25Z); separately tracked as
   W-PHASE-B-PYDEBUG.

3. ARM64 pydebug at devgpu004 commit 66921f69b1 (older than push 40):
   `test_phoenix_jit_inline_except_closure` PASSES. Does not surface
   the W27 push-59 ARM64 SEGV any more either — the 2026-04-22 origin
   condition appears to have been implicitly affected by intervening
   pushes in the W22 / W-C2 / DSecondary chain (pushes 38–49) but
   the precise causation is not traced.

Outcome per §8 + supervisor 19:59:18Z:

- W-A4-mid status: **CLOSED-by-non-reproduction** as of
  2026-04-24T19:59Z. The empirical falsifier the §8 discipline
  required does not reproduce the latent UAF on the canonical
  workload OR a synthetic substitute.

- Finalize-phase mitigation (§3) remains as standing partial
  protection (`Py_IsFinalizing` early-return in
  `phoenix_dict_watcher` + `GlobalCacheManager::clear`).

- Mid-execution-unload class is NOT proven absent — only proven
  not-currently-reproducible by the available falsifiers. Re-open
  on new evidence (e.g. a future workload SEGV in
  `phoenix_dict_watcher → notifyDictClear → updateCache`, or a
  per-W-PreExistingAudit cinderx_dev oracle finding that surfaces
  it).

- The synthetic test
  `Lib/test/test_phoenix_w27_module_unload_uaf.py` is preserved
  as a regression sentinel: it currently passes; if Phoenix changes
  ever cause it to FAIL, that re-opens W27 with concrete evidence.
