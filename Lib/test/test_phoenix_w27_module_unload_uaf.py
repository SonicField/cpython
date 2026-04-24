"""W27 falsifier test: GlobalCacheKey raw-pointer lifecycle UAF on
mid-execution module unload.

Per docs/w27-globalcachekey-lifecycle.md §8 (theologian + supervisor
2026-04-24T19:47:12Z): test reproduces the SEGV pre-W-A4-mid (Option E
inline mark-invalidate fix) and passes post-fix. Ensures W27 has a
revisit-trigger and is not silently fallow.

Pre-fix expected outcome (cc081f1de0..76f2c863d5):
  ARM64 pydebug → SEGV in phoenix_dict_watcher → notifyDictClear →
  updateCache → hasOnlyUnicodeKeys → PyType_HasFeature(0xdddd...) when
  cache entry's stale globals pointer is dereferenced after the module's
  __dict__ has been freed.
  x86_64 release → silent UAF (freed memory still readable).

Post-fix expected outcome (W-A4-mid Option E lands):
  PyDict_EVENT_DEALLOCATED handler invokes
  GlobalCacheManager::invalidateForDeallocatedDict; cache entry dropped
  before the stale pointer can be dereferenced. Test passes clean.
"""

import gc
import sys
import unittest


class TestModuleUnloadUAF(unittest.TestCase):
    """Reproducer for W27 GlobalCacheKey raw-pointer UAF.

    Pattern:
      1. Define a JIT-eligible function in a synthetic module that uses
         LOAD_GLOBAL (registers GlobalCacheKey for its globals dict)
      2. Force_compile or auto-compile the function (cache entry written)
      3. Drop all references to the module + GC.collect (globals freed)
      4. Trigger any builtins event (e.g. assignment to __builtins__) so
         phoenix_dict_watcher fires and walks the cache. With stale
         globals pointer, derefs poisoned/freed memory.

    Pre-W-A4-mid: SEGV/UAF on step 4.
    Post-W-A4-mid: clean (cache invalidated on globals dealloc in step 3).
    """

    def setUp(self):
        try:
            import _cinderx  # noqa: F401
            import cinderjit  # noqa: F401
        except ImportError:
            self.skipTest("cinderjit not available")
        self.cinderjit = sys.modules['cinderjit']

    def test_globals_unloaded_while_cache_alive(self):
        """Verify cache entry keyed on a freed globals dict doesn't UAF."""
        import types

        # 1. Create a synthetic module with a function that uses LOAD_GLOBAL.
        #    'len' is a builtin lookup → globals dict accessed via fallback.
        mod = types.ModuleType('w27_uaf_test_mod')
        exec(
            "def use_global(x):\n"
            "    return len(x) + 1\n",
            mod.__dict__,
        )

        # Register module so JIT compile sees a real globals dict identity.
        sys.modules['w27_uaf_test_mod'] = mod

        # 2. force_compile to register a GlobalCacheKey entry for
        #    (mod.__dict__, builtins.__dict__, 'len').
        self.cinderjit.force_compile(mod.use_global)
        self.assertTrue(self.cinderjit.is_jit_compiled(mod.use_global))

        # Sanity-call to ensure the cache path executed at least once.
        self.assertEqual(mod.use_global([1, 2, 3]), 4)

        # 3. Drop module references + GC; the dict will be freed if no
        #    refs survive. PyDict has no tp_weaklistoffset so we can't use
        #    weakref to verify; the SEGV on step 4 IS the proof. (testkeeper
        #    19:49:45Z: weakref.ref(mod.__dict__) raises TypeError;
        #    dropped per (d) 'just trigger UAF without explicit liveness
        #    check'.)
        del sys.modules['w27_uaf_test_mod']
        del mod
        gc.collect()

        # 4. Trigger a builtins event by assigning a new builtin attribute.
        #    This fires phoenix_dict_watcher → notifyDictUpdate or
        #    notifyDictClear depending on event class. The handler walks
        #    GlobalCacheManager.map_ which still contains the entry keyed
        #    on the (potentially stale) mod.__dict__ pointer.
        import builtins
        builtins._w27_test_marker = 42
        try:
            # If we got here without SEGV, the cache invalidation worked
            # (or the dict happened not to be freed yet, in which case
            # the test is non-falsifying for this run — rare).
            pass
        finally:
            del builtins._w27_test_marker


if __name__ == '__main__':
    unittest.main()
