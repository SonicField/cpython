"""Test that UseType instructions with float types survive the Simplify pass.

This test targets the funcTypeChecks assertion (compiler.cpp:68) which verifies
that UseType instructions store the correct type after the Simplify pass.
The bug: C-factory-created UseType<TFloatExact> stored type as 0x8000Top
(corrupted) on ARM64 with --with-pydebug, causing funcTypeChecks to fail.

If this test passes, the UseType type round-trip through C factories is correct.
If it crashes with 'Assertion failed: (funcTypeChecks)', the bug is present.
"""

import unittest

try:
    import _cinderx
    import cinderjit
    HAS_JIT = True
except ImportError:
    HAS_JIT = False


@unittest.skipUnless(HAS_JIT, "requires JIT")
class TestUseTypeFloat(unittest.TestCase):
    """Test float operations that trigger UseType<TFloatExact> in Simplify."""

    def _jit_test(self, func, *args):
        interp_result = func(*args)
        cinderjit.force_compile(func)
        self.assertTrue(cinderjit.is_jit_compiled(func),
                        f"{func.__name__} not compiled")
        jit_result = func(*args)
        self.assertEqual(interp_result, jit_result,
                         f"{func.__name__}: interp={interp_result} jit={jit_result}")

    def test_float_add(self):
        """Float add triggers simplifyBinaryOp → emitUseType(TFloatExact)."""
        def f(x, y):
            return x + y
        self._jit_test(f, 1.5, 2.5)

    def test_float_multiply(self):
        """Float multiply — same path."""
        def f(x, y):
            return x * y
        self._jit_test(f, 3.0, 4.0)

    def test_float_mixed_ops(self):
        """Multiple float ops in one function — multiple UseType emissions."""
        def f(x, y):
            a = x + y
            b = x * y
            c = a / b
            return c
        self._jit_test(f, 2.0, 3.0)

    def test_float_accumulate(self):
        """Float accumulation in loop — the pattern that crashes."""
        def f(n):
            total = 0.0
            for i in range(n):
                total += float(i)
            return total
        self._jit_test(f, 100)

    def test_float_denormal(self):
        """Denormal float operations — the specific test_phoenix_float failure."""
        def f():
            tiny = 5e-324
            return (tiny + tiny, tiny * 2, tiny / 2)
        self._jit_test(f)


if __name__ == "__main__":
    unittest.main()
