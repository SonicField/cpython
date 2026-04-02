"""Tests for Phoenix JIT float arithmetic correctness on ARM64 and x86_64.

These tests verify that JIT-compiled float operations produce identical
results to the interpreter. Each test:
1. Runs a function under the interpreter
2. Force-compiles the function via cinderjit.force_compile()
3. Runs the same function under JIT
4. Asserts the results are identical (within floating-point tolerance)

This catches silent float codegen bugs where the JIT produces wrong answers
without crashing — e.g., returning -0.0 instead of 2469.7.
"""

import math
import sys
import unittest

try:
    import _cinderx
    import cinderjit
    HAS_JIT = True
except ImportError:
    HAS_JIT = False


def _force_compile(func):
    """Force-compile a function and return whether it was compiled."""
    if not HAS_JIT:
        return False
    cinderjit.force_compile(func)
    return cinderjit.is_jit_compiled(func)


def _compare_float(a, b, tol=1e-9):
    """Compare two float values with tolerance."""
    if math.isnan(a) and math.isnan(b):
        return True
    if math.isinf(a) and math.isinf(b):
        return (a > 0) == (b > 0)
    return abs(a - b) < tol * max(1.0, abs(a))


@unittest.skipUnless(HAS_JIT, "Phoenix JIT not available")
class TestFloatArithmetic(unittest.TestCase):
    """Test basic float arithmetic operations under JIT."""

    def _check_jit_correctness(self, func, *args, tol=1e-9):
        """Run func under interpreter, force-compile, run under JIT, compare."""
        interp_result = func(*args)
        compiled = _force_compile(func)
        self.assertTrue(compiled, f"{func.__name__} did not compile")
        jit_result = func(*args)

        if isinstance(interp_result, float):
            self.assertTrue(
                _compare_float(interp_result, jit_result, tol),
                f"{func.__name__}: interpreter={interp_result}, JIT={jit_result}"
            )
        elif isinstance(interp_result, list):
            self.assertEqual(len(interp_result), len(jit_result))
            for i, (a, b) in enumerate(zip(interp_result, jit_result)):
                self.assertTrue(
                    _compare_float(a, b, tol),
                    f"{func.__name__}[{i}]: interpreter={a}, JIT={b}"
                )
        else:
            self.assertEqual(interp_result, jit_result)

    def test_float_add_accumulate(self):
        def f(n):
            t = 0.0
            for i in range(n):
                t += float(i) * 1.1
            return t
        self._check_jit_correctness(f, 1000)

    def test_float_sub_accumulate(self):
        def f(n):
            t = 1000000.0
            for i in range(n):
                t -= float(i) * 0.001
            return t
        self._check_jit_correctness(f, 1000)

    def test_float_mul_accumulate(self):
        def f(n):
            t = 1.0
            for i in range(1, n + 1):
                t *= (1.0 + 1.0 / float(i))
            return t
        self._check_jit_correctness(f, 100)

    def test_float_div_accumulate(self):
        def f(n):
            t = 1000000.0
            for i in range(1, n + 1):
                t /= (1.0 + 0.0001 * float(i))
            return t
        self._check_jit_correctness(f, 100)

    def test_float_mixed_ops(self):
        """Matches the float_arith benchmark pattern that returned -0.0."""
        def f(n):
            t = 0.0
            for i in range(n):
                for j in range(1, 101):
                    t = (t + float(j) * 1.1 - float(j) / 3.3) % 10000.0
            return t
        self._check_jit_correctness(f, 100)

    def test_float_fma_pattern(self):
        def f(n):
            a, b, c = 1.5, 2.5, 3.5
            for i in range(n):
                a = a * b + c
                b = b * c - a
                c = (c + a) * 0.5
                a = abs(a) % 1000
                b = abs(b) % 1000
                c = abs(c) % 1000
            return a + b + c
        self._check_jit_correctness(f, 1000)


@unittest.skipUnless(HAS_JIT, "Phoenix JIT not available")
class TestFloatEdgeCases(unittest.TestCase):
    """Test float edge cases under JIT."""

    def _check_jit_correctness(self, func, *args, tol=1e-9):
        interp_result = func(*args)
        _force_compile(func)
        jit_result = func(*args)
        if isinstance(interp_result, float):
            self.assertTrue(
                _compare_float(interp_result, jit_result, tol),
                f"{func.__name__}: interpreter={interp_result}, JIT={jit_result}"
            )
        else:
            self.assertEqual(interp_result, jit_result)

    def test_negative_zero(self):
        def f(n):
            x = 0.0
            for i in range(n):
                x = x - 0.0
            return x
        self._check_jit_correctness(f, 10)

    def test_infinity(self):
        def f(n):
            return 1e308 * 2.0
        self._check_jit_correctness(f, 1)

    def test_denormal(self):
        def f(n):
            x = 5e-324
            for i in range(n):
                x = x * 0.5
            return x
        self._check_jit_correctness(f, 10)


@unittest.skipUnless(HAS_JIT, "Phoenix JIT not available")
class TestFloatDataStructures(unittest.TestCase):
    """Test float operations with data structure mutation (nbody-like patterns)."""

    def _check_jit_correctness(self, func, *args, tol=1e-9):
        interp_result = func(*args)
        _force_compile(func)
        jit_result = func(*args)
        if isinstance(interp_result, (list, tuple)):
            self.assertEqual(len(interp_result), len(jit_result))
            for i, (a, b) in enumerate(zip(interp_result, jit_result)):
                self.assertTrue(
                    _compare_float(a, b, tol),
                    f"{func.__name__}[{i}]: interpreter={a}, JIT={b}"
                )
        elif isinstance(interp_result, float):
            self.assertTrue(
                _compare_float(interp_result, jit_result, tol),
                f"{func.__name__}: interpreter={interp_result}, JIT={jit_result}"
            )

    def test_list_store_mutation(self):
        """Pattern matching nbody — in-place list element mutation with float ops."""
        def f(n):
            data = [float(i) for i in range(10)]
            for _ in range(n):
                for i in range(len(data)):
                    data[i] = data[i] * 1.01 + 0.5
                    data[i] = data[i] % 1000.0
            return sum(data)
        self._check_jit_correctness(f, 100)

    def test_nested_loop_accumulate(self):
        def f(n):
            total = 0.0
            for i in range(n):
                inner = 0.0
                for j in range(100):
                    inner += float(j) * 0.01
                total += inner
            return total
        self._check_jit_correctness(f, 100)

    def test_multi_body_interaction(self):
        """Simplified nbody interaction pattern."""
        def f(n):
            bodies = [[1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 1.0],
                      [4.0, 5.0, 6.0, 0.4, 0.5, 0.6, 2.0]]
            dt = 0.01
            for _ in range(n):
                for i in range(len(bodies)):
                    bi = bodies[i]
                    for j in range(i + 1, len(bodies)):
                        bj = bodies[j]
                        dx = bi[0] - bj[0]
                        dy = bi[1] - bj[1]
                        dz = bi[2] - bj[2]
                        dist = (dx * dx + dy * dy + dz * dz) ** 0.5
                        mag = dt / (dist * dist * dist)
                        bi[3] -= dx * bj[6] * mag
                        bi[4] -= dy * bj[6] * mag
                        bi[5] -= dz * bj[6] * mag
                        bj[3] += dx * bi[6] * mag
                        bj[4] += dy * bi[6] * mag
                        bj[5] += dz * bi[6] * mag
                for b in bodies:
                    b[0] += dt * b[3]
                    b[1] += dt * b[4]
                    b[2] += dt * b[5]
            return [round(x, 10) for b in bodies for x in b]
        self._check_jit_correctness(f, 50)


if __name__ == "__main__":
    unittest.main()
