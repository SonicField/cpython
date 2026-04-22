"""Phoenix JIT — D-1774910012 falsifier: closure LOAD_DEREF in except blocks.

Tests the post-threshold path of emitInlineExceptionMatch under JIT compilation
when the except-clause body uses LOAD_DEREF on a closure variable. This is the
exact failure mode documented at D-1774910012 (28b4ee14b3): closure LOAD_DEREF
in except blocks caused deopt stack mismatch (Py_None placeholder + POP_EXCEPT
semantics). Bug only triggered above auto-compile threshold (1000 calls).

The C++ fix shipped without a regression test. The C port (push 59
emitInlineExceptionMatch) preserves the same invariant. This test is the
regression falsifier per pythia #73 + theologian + supervisor 14:53:28Z.

Test design (4 steps):
  1. Define function with closure capturing outer variable
  2. except-clause body uses LOAD_DEREF on the closure variable
  3. Call 1001+ times to trigger threshold=1000 → JIT compile
  4. Verify post-threshold call exercises exception path under JIT-compiled
     code with no SegFault, no exc_info corruption, expected result preserved

Run with: ./python -m test test_phoenix_jit_inline_except_closure
"""

import sys
import unittest

try:
    import _cinderx
    import cinderjit
    HAS_JIT = True
except ImportError:
    HAS_JIT = False

WARMUP = 1100  # threshold=1000, add margin


@unittest.skipUnless(HAS_JIT, "requires JIT")
class TestJitInlineExceptClosure(unittest.TestCase):
    """Falsifier for D-1774910012 (inline exception handler deopt fix).

    Each test defines an outer function that creates an inner closure where
    the except-clause body LOAD_DEREFs the captured variable. Warming the
    inner past WARMUP triggers the JIT path that previously corrupted
    exc_info chain.
    """

    def test_load_deref_in_except_basic(self):
        """LOAD_DEREF on closure var inside except: assert no corruption."""
        def make_inner():
            captured = "captured-value"
            def inner(should_raise):
                try:
                    if should_raise:
                        raise ValueError("boom")
                    return "ok"
                except ValueError:
                    # LOAD_DEREF on captured (closure cell) inside except body
                    return f"caught-with-{captured}"
            return inner

        inner = make_inner()
        # Warmup non-raising path 1100 times to trigger JIT
        for _ in range(WARMUP):
            self.assertEqual(inner(False), "ok")
        self.assertTrue(cinderjit.is_jit_compiled(inner),
                        "inner was NOT JIT-compiled after warmup")
        # Now exercise the except path under JIT
        self.assertEqual(inner(True), "caught-with-captured-value")
        # Run exception path many more times to surface any latent corruption
        for _ in range(100):
            self.assertEqual(inner(True), "caught-with-captured-value")
            self.assertEqual(inner(False), "ok")

    def test_load_deref_with_pop_except_chain(self):
        """exc_info chain integrity: nested try/except using LOAD_DEREF in both."""
        def make_inner():
            outer_msg = "outer-closure"
            def inner(level):
                try:
                    if level == 0:
                        raise ValueError(f"v-{outer_msg}")
                    elif level == 1:
                        raise TypeError(f"t-{outer_msg}")
                    return ("clean", outer_msg)
                except ValueError as e:
                    return ("ve", str(e), outer_msg)
                except TypeError as e:
                    return ("te", str(e), outer_msg)
            return inner

        inner = make_inner()
        # Warmup with all 3 paths
        for i in range(WARMUP):
            inner(2)  # clean path dominates warmup
        self.assertTrue(cinderjit.is_jit_compiled(inner),
                        "inner was NOT JIT-compiled after warmup")

        # Exercise each path many times, verify result + exc_info clean
        for _ in range(100):
            r0 = inner(0)
            r1 = inner(1)
            r2 = inner(2)
            self.assertEqual(r0, ("ve", "v-outer-closure", "outer-closure"))
            self.assertEqual(r1, ("te", "t-outer-closure", "outer-closure"))
            self.assertEqual(r2, ("clean", "outer-closure"))
            # exc_info should be clean between calls (no chain corruption)
            self.assertIsNone(sys.exc_info()[1],
                              "sys.exc_info corrupted post-call")

    def test_load_deref_after_pop_except(self):
        """LOAD_DEREF after POP_EXCEPT: prev_exc placeholder pop semantics."""
        def make_inner():
            shared = [0]
            def inner(should_raise):
                try:
                    if should_raise:
                        raise RuntimeError("trigger")
                    return ("noraise", shared[0])
                except RuntimeError:
                    pass
                # After POP_EXCEPT, LOAD_DEREF on shared (mutates closure)
                shared[0] += 1
                return ("post-except", shared[0])
            return inner

        inner = make_inner()
        # Warmup non-raising path
        for _ in range(WARMUP):
            inner(False)
        self.assertTrue(cinderjit.is_jit_compiled(inner),
                        "inner was NOT JIT-compiled after warmup")

        # Run exception path; verify shared[0] mutates correctly post-POP_EXCEPT
        before = inner(False)[1]
        # Exception path must increment shared[0] each call
        r1 = inner(True)
        self.assertEqual(r1, ("post-except", before + 1))
        r2 = inner(True)
        self.assertEqual(r2, ("post-except", before + 2))


@unittest.skipUnless(HAS_JIT, "requires JIT")
class TestJitCallExceptionHandler(unittest.TestCase):
    """Falsifier for emitCallExceptionHandler D1-D3 unique invariants.

    Per theologian [chat 17:23:36Z] spec — push 60 emitCallExceptionHandler C
    port (9630005502) introduced D1 (setSuppressExceptionDeopt + pop result
    pre-call), D2 (DeoptTC zero re-push), D3 (RefineType + result push) on top
    of the shared P2+P3+P4 helper. The shared helper is exercised by
    TestJitInlineExceptClosure above; D1-D3 unique invariants are exercised
    here via function-call-in-try shape (NOT BINARY_SUBSCR_DICT).

    Pythia #77 #3 + supervisor [chat 17:23:12Z] authorized this as pre-push-61
    gate (regression-test-pre-exists discipline matching D-1774910012).
    """

    def test_call_in_try_simple_except(self):
        """C1 — function call in try, ValueError matched, return from except.

        Triggers D1 (setSuppressExceptionDeopt + pop) and D3 (RefineType +
        result push). D3 omission would NameError-on-result on OK path; D1
        omission would deopt instead of inline-handle.
        """
        def _helper(should_raise):
            if should_raise:
                raise ValueError("test")
            return 42

        def caller(should_raise):
            try:
                return _helper(should_raise)
            except ValueError:
                return -1

        for _ in range(WARMUP):
            self.assertEqual(caller(False), 42)
        self.assertTrue(cinderjit.is_jit_compiled(caller),
                        "caller was NOT JIT-compiled after warmup")

        # Exercise both OK (D3) and except (D1) paths under JIT
        self.assertEqual(caller(False), 42)
        self.assertEqual(caller(True), -1)
        for _ in range(100):
            self.assertEqual(caller(False), 42)
            self.assertEqual(caller(True), -1)

    def test_call_in_try_with_closure_load_deref(self):
        """C2 — function call in try, except body LOAD_DEREFs closure variable.

        Exercises BOTH D1-D3 (call-handler entry path) AND PA (D-1774910012
        prev_exc Py_None placeholder push + POP_EXCEPT pop semantics). PA
        failure would corrupt sys.exc_info post-loop.
        """
        def _helper(should_raise):
            if should_raise:
                raise ValueError("test")
            return 42

        def make_handler(default):
            def caller(should_raise):
                try:
                    return _helper(should_raise)
                except ValueError:
                    return default  # LOAD_DEREF on closure
            return caller

        handler = make_handler(-99)
        for _ in range(WARMUP):
            self.assertEqual(handler(False), 42)
        self.assertTrue(cinderjit.is_jit_compiled(handler),
                        "handler was NOT JIT-compiled after warmup")

        self.assertEqual(handler(False), 42)
        self.assertEqual(handler(True), -99)
        # exc_info chain must remain clean post-loop (PA invariant)
        for _ in range(100):
            self.assertEqual(handler(True), -99)
            self.assertEqual(handler(False), 42)
            self.assertIsNone(sys.exc_info()[1],
                              "sys.exc_info corrupted by call-handler path")

    def test_call_in_try_deopt_path(self):
        """C3 — function call in try, exception NOT matched (TypeError vs ValueError).

        Triggers D2 (DeoptTC zero re-push, no left/right context). D2 omission
        would corrupt deopt frame state when unmatched exception propagates
        through the call-handler path.
        """
        def _maybe_raise(should_raise):
            if should_raise:
                raise TypeError("wrong type")
            return 42

        def caller(should_raise):
            try:
                return _maybe_raise(should_raise)
            except ValueError:  # WRONG type — TypeError will not match
                return -1

        for _ in range(WARMUP):
            self.assertEqual(caller(False), 42)
        self.assertTrue(cinderjit.is_jit_compiled(caller),
                        "caller was NOT JIT-compiled after warmup")

        # Exercise deopt path: TypeError propagates out of caller (unmatched)
        with self.assertRaises(TypeError):
            caller(True)
        # OK path still correct post-deopt
        self.assertEqual(caller(False), 42)
        # Run several deopts in a row
        for _ in range(50):
            with self.assertRaises(TypeError):
                caller(True)
            self.assertEqual(caller(False), 42)


if __name__ == "__main__":
    unittest.main()
