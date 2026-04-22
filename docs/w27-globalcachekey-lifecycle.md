# W27 — GlobalCacheKey Raw-Pointer Lifecycle Re-Architecture

**Status:** Filed (generalist, 2026-04-22, supervisor-authorized [chat ~16:04Z])
**Owner:** unassigned (Tier 6+ scope)
**Schedule:** Post-Tier-5-close; not blocking any current work
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

Likely best path: **Option B** with deferred-cleanup queue. Matches CPython's "watcher callback should not mutate watched state" guidance and resolves the 2026-03-30 re-entrancy.

## 6. Sequencing

- Pre-req: Tier 5 close (zero C++ in builder.cpp emit methods)
- Falsifier needed before fix lands: mid-execution module-unload reproducer
- Estimated scope: 3-4 hours for Option B, 1-2 hours for Option A (with caveats)
- Risk: medium — touches global cache invariants, requires re-verification of all global-cache-related tests

## 7. Cross-references

- Push 59 root-cause analysis: chat 2026-04-22 ~15:55-16:04Z
- Original DEALLOCATED-skip: 314d0f2310 (Alex, 2026-03-30)
- Finalize-phase mitigation: this commit (paired with push 59)
- Related lifecycle pattern: feedback_no_workarounds.md
