"""Phoenix JIT Arithmetic and Numeric Operations Tests.

Verifies that JIT-compiled arithmetic functions produce identical results
to the interpreter across all numeric opcode patterns. For each test:
1. Define a Python function exercising specific opcode(s)
2. Run under interpreter to get the expected result
3. Force-compile via cinderjit.force_compile()
4. Run under JIT and assert the result matches

Run with: ./python -m test test_phoenix_jit_arithmetic
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


@unittest.skipUnless(HAS_JIT, "requires JIT")
class TestJitArithmetic(unittest.TestCase):
    """Test JIT compilation of arithmetic and numeric operations."""

    def _jit_test(self, func, *args):
        """Run func interpreted, then JIT-compiled, assert same result."""
        interp_result = func(*args)
        cinderjit.force_compile(func)
        self.assertTrue(
            cinderjit.is_jit_compiled(func),
            f"{func.__name__} not compiled",
        )
        jit_result = func(*args)
        self.assertEqual(
            interp_result,
            jit_result,
            f"{func.__name__}: interp={interp_result!r}, jit={jit_result!r}",
        )
        return jit_result

    def _jit_test_float(self, func, *args, rel_tol=1e-12):
        """Like _jit_test but uses approximate float comparison."""
        interp_result = func(*args)
        cinderjit.force_compile(func)
        self.assertTrue(
            cinderjit.is_jit_compiled(func),
            f"{func.__name__} not compiled",
        )
        jit_result = func(*args)
        if isinstance(interp_result, float) and isinstance(jit_result, float):
            if math.isnan(interp_result):
                self.assertTrue(
                    math.isnan(jit_result),
                    f"{func.__name__}: expected nan, got {jit_result!r}",
                )
            elif math.isinf(interp_result):
                self.assertEqual(
                    interp_result,
                    jit_result,
                    f"{func.__name__}: interp={interp_result!r}, jit={jit_result!r}",
                )
            else:
                self.assertAlmostEqual(
                    interp_result,
                    jit_result,
                    msg=f"{func.__name__}: interp={interp_result!r}, jit={jit_result!r}",
                )
        else:
            self.assertEqual(
                interp_result,
                jit_result,
                f"{func.__name__}: interp={interp_result!r}, jit={jit_result!r}",
            )
        return jit_result

    # ─── BINARY_OP: basic int arithmetic ────────────────────────────

    def test_int_add(self):
        def f(a, b):
            return a + b
        self._jit_test(f, 3, 7)

    def test_int_subtract(self):
        def f(a, b):
            return a - b
        self._jit_test(f, 10, 4)

    def test_int_multiply(self):
        def f(a, b):
            return a * b
        self._jit_test(f, 6, 7)

    def test_int_truediv(self):
        def f(a, b):
            return a / b
        self._jit_test(f, 10, 3)

    def test_int_floordiv(self):
        def f(a, b):
            return a // b
        self._jit_test(f, 17, 5)

    def test_int_modulo(self):
        def f(a, b):
            return a % b
        self._jit_test(f, 17, 5)

    def test_int_power(self):
        def f(a, b):
            return a ** b
        self._jit_test(f, 2, 10)

    def test_int_negative_operands(self):
        def f(a, b):
            return a + b, a - b, a * b, a // b, a % b
        self._jit_test(f, -13, 5)

    def test_int_negative_both(self):
        def f(a, b):
            return a + b, a - b, a * b, a // b, a % b
        self._jit_test(f, -13, -5)

    # ─── BINARY_OP: small and large integers ────────────────────────

    def test_int_small_values(self):
        def f():
            return 0 + 0, 1 + 0, 0 - 1, 1 * 1, -1 * -1
        self._jit_test(f)

    def test_int_large_add(self):
        def f(a, b):
            return a + b
        self._jit_test(f, 10**50, 10**50)

    def test_int_large_multiply(self):
        def f(a, b):
            return a * b
        self._jit_test(f, 10**30, 10**30)

    def test_int_large_power(self):
        def f(a, b):
            return a ** b
        self._jit_test(f, 2, 256)

    def test_int_large_floordiv(self):
        def f(a, b):
            return a // b
        self._jit_test(f, 10**100, 7)

    def test_int_large_modulo(self):
        def f(a, b):
            return a % b
        self._jit_test(f, 10**100 + 3, 10**50)

    def test_int_overflow_boundary(self):
        """Test around sys.maxsize (word-size boundary)."""
        def f(a, b):
            return a + b, a * 2, a - b
        self._jit_test(f, sys.maxsize, 1)

    def test_int_negative_large(self):
        def f(a):
            return -a, a * -1, a + (-a)
        self._jit_test(f, 10**80)

    # ─── BINARY_OP: float arithmetic ────────────────────────────────

    def test_float_add(self):
        def f(a, b):
            return a + b
        self._jit_test_float(f, 1.5, 2.5)

    def test_float_subtract(self):
        def f(a, b):
            return a - b
        self._jit_test_float(f, 10.7, 3.2)

    def test_float_multiply(self):
        def f(a, b):
            return a * b
        self._jit_test_float(f, 3.14, 2.0)

    def test_float_truediv(self):
        def f(a, b):
            return a / b
        self._jit_test_float(f, 22.0, 7.0)

    def test_float_floordiv(self):
        def f(a, b):
            return a // b
        self._jit_test_float(f, 7.5, 2.0)

    def test_float_modulo(self):
        def f(a, b):
            return a % b
        self._jit_test_float(f, 7.5, 2.0)

    def test_float_power(self):
        def f(a, b):
            return a ** b
        self._jit_test_float(f, 2.5, 3.0)

    def test_float_negative_operands(self):
        def f(a, b):
            return a + b, a - b, a * b, a / b
        self._jit_test(f, -3.5, 1.2)

    # ─── Float edge cases ──────────────────────────────────────────

    def test_float_inf_add(self):
        def f():
            return float('inf') + 1.0
        result = self._jit_test_float(f)
        self.assertTrue(math.isinf(result))

    def test_float_inf_sub(self):
        def f():
            return float('inf') - float('inf')
        interp_result = f()
        cinderjit.force_compile(f)
        self.assertTrue(cinderjit.is_jit_compiled(f))
        jit_result = f()
        self.assertTrue(math.isnan(interp_result))
        self.assertTrue(math.isnan(jit_result))

    def test_float_inf_mul(self):
        def f():
            return float('inf') * -1.0
        result = self._jit_test_float(f)
        self.assertEqual(result, float('-inf'))

    def test_float_inf_div(self):
        def f():
            return 1.0 / float('inf')
        result = self._jit_test_float(f)
        self.assertEqual(result, 0.0)

    def test_float_neg_inf(self):
        def f(a):
            return a + 1.0, a * 2.0, a - 1.0
        self._jit_test(f, float('-inf'))

    def test_float_nan_arithmetic(self):
        """NaN propagates through arithmetic."""
        def f():
            n = float('nan')
            return n + 1.0, n * 2.0, n - 1.0, n / 2.0
        interp_result = f()
        cinderjit.force_compile(f)
        self.assertTrue(cinderjit.is_jit_compiled(f))
        jit_result = f()
        for i_val, j_val in zip(interp_result, jit_result):
            self.assertTrue(math.isnan(i_val), f"interp not nan: {i_val!r}")
            self.assertTrue(math.isnan(j_val), f"jit not nan: {j_val!r}")

    def test_float_nan_comparison(self):
        """NaN is not equal to itself."""
        def f():
            n = float('nan')
            return n == n, n != n, n < n, n > n
        self._jit_test(f)

    def test_float_negative_zero(self):
        def f():
            return -0.0
        result = self._jit_test(f)
        self.assertEqual(result, 0.0)
        self.assertTrue(math.copysign(1.0, result) == -1.0)

    
    def test_float_negative_zero_arithmetic(self):
        def f():
            nz = -0.0
            # Note: 1.0 / -0.0 raises ZeroDivisionError in CPython,
            # so we test only operations that preserve -0.0 sign.
            return nz + 0.0, nz * 1.0, nz - 0.0
        interp_result = f()
        cinderjit.force_compile(f)
        self.assertTrue(cinderjit.is_jit_compiled(f))
        jit_result = f()
        for i_val, j_val in zip(interp_result, jit_result):
            self.assertEqual(i_val, j_val)
            self.assertEqual(
                math.copysign(1.0, i_val),
                math.copysign(1.0, j_val),
                f"sign mismatch: interp={i_val!r}, jit={j_val!r}",
            )

    def test_float_denormal(self):
        def f():
            d = 5e-324  # smallest positive denormal
            return d, d + d, d * 2.0, d / 2.0
        self._jit_test(f)

    def test_float_very_large(self):
        def f():
            return 1.7976931348623157e+308 * 0.5
        self._jit_test_float(f)

    # ─── Mixed int/float arithmetic ─────────────────────────────────

    def test_mixed_int_float_add(self):
        def f(a, b):
            return a + b
        self._jit_test(f, 3, 2.5)

    def test_mixed_int_float_multiply(self):
        def f(a, b):
            return a * b
        self._jit_test(f, 7, 0.5)

    def test_mixed_int_float_truediv(self):
        def f(a, b):
            return a / b
        self._jit_test(f, 10, 3.0)

    def test_mixed_int_float_power(self):
        def f(a, b):
            return a ** b
        self._jit_test(f, 2, 0.5)

    def test_mixed_int_float_floordiv(self):
        def f(a, b):
            return a // b
        self._jit_test(f, 7, 2.5)

    def test_mixed_int_float_modulo(self):
        def f(a, b):
            return a % b
        self._jit_test(f, 7, 2.5)

    # ─── UNARY operations ──────────────────────────────────────────

    def test_unary_negative_int(self):
        def f(a):
            return -a
        self._jit_test(f, 42)

    def test_unary_negative_float(self):
        def f(a):
            return -a
        self._jit_test(f, 3.14)

    def test_unary_negative_zero(self):
        def f(a):
            return -a
        result = self._jit_test(f, 0)
        self.assertEqual(result, 0)

    def test_unary_positive_int(self):
        def f(a):
            return +a
        self._jit_test(f, -42)

    def test_unary_positive_float(self):
        def f(a):
            return +a
        self._jit_test(f, -3.14)

    def test_unary_not_truthy(self):
        def f(a):
            return not a
        self._jit_test(f, 42)

    def test_unary_not_falsy(self):
        def f(a):
            return not a
        self._jit_test(f, 0)

    def test_unary_not_bool(self):
        def f(a):
            return not a
        self._jit_test(f, True)
        # Also test False
        def g(a):
            return not a
        self._jit_test(g, False)

    def test_unary_invert_int(self):
        def f(a):
            return ~a
        self._jit_test(f, 42)

    def test_unary_invert_negative(self):
        def f(a):
            return ~a
        self._jit_test(f, -1)

    def test_unary_invert_zero(self):
        def f(a):
            return ~a
        self._jit_test(f, 0)

    def test_unary_invert_large(self):
        def f(a):
            return ~a
        self._jit_test(f, 10**50)


    def test_unary_invert_bool(self):
        def f(a):
            return ~a
        self._jit_test(f, True)

    # ─── Augmented assignment (INPLACE ops) ─────────────────────────

    def test_iadd(self):
        def f(a, b):
            a += b
            return a
        self._jit_test(f, 10, 5)

    def test_isub(self):
        def f(a, b):
            a -= b
            return a
        self._jit_test(f, 10, 5)

    def test_imul(self):
        def f(a, b):
            a *= b
            return a
        self._jit_test(f, 6, 7)

    def test_itruediv(self):
        def f(a, b):
            a /= b
            return a
        self._jit_test(f, 10, 3)

    def test_ifloordiv(self):
        def f(a, b):
            a //= b
            return a
        self._jit_test(f, 17, 5)

    def test_imod(self):
        def f(a, b):
            a %= b
            return a
        self._jit_test(f, 17, 5)

    def test_ipow(self):
        def f(a, b):
            a **= b
            return a
        self._jit_test(f, 2, 10)

    def test_iand(self):
        def f(a, b):
            a &= b
            return a
        self._jit_test(f, 0xFF, 0x0F)

    def test_ior(self):
        def f(a, b):
            a |= b
            return a
        self._jit_test(f, 0xF0, 0x0F)

    def test_ixor(self):
        def f(a, b):
            a ^= b
            return a
        self._jit_test(f, 0xFF, 0x0F)

    def test_ilshift(self):
        def f(a, b):
            a <<= b
            return a
        self._jit_test(f, 1, 10)

    def test_irshift(self):
        def f(a, b):
            a >>= b
            return a
        self._jit_test(f, 1024, 3)

    def test_iadd_float(self):
        def f(a, b):
            a += b
            return a
        self._jit_test(f, 1.5, 2.5)

    def test_imul_float(self):
        def f(a, b):
            a *= b
            return a
        self._jit_test(f, 3.0, 4.5)

    # ─── Bitwise operations ─────────────────────────────────────────

    def test_bitwise_and(self):
        def f(a, b):
            return a & b
        self._jit_test(f, 0b11110000, 0b10101010)

    def test_bitwise_or(self):
        def f(a, b):
            return a | b
        self._jit_test(f, 0b11110000, 0b00001111)

    def test_bitwise_xor(self):
        def f(a, b):
            return a ^ b
        self._jit_test(f, 0b11110000, 0b10101010)

    def test_bitwise_lshift(self):
        def f(a, b):
            return a << b
        self._jit_test(f, 1, 20)

    def test_bitwise_rshift(self):
        def f(a, b):
            return a >> b
        self._jit_test(f, 0xFFFF, 8)

    def test_bitwise_large_shift(self):
        def f(a, b):
            return a << b
        self._jit_test(f, 1, 100)

    def test_bitwise_negative_and(self):
        def f(a, b):
            return a & b
        self._jit_test(f, -1, 0xFF)

    def test_bitwise_negative_or(self):
        def f(a, b):
            return a | b
        self._jit_test(f, -256, 0xFF)

    def test_bitwise_negative_xor(self):
        def f(a, b):
            return a ^ b
        self._jit_test(f, -1, 0xFF)

    def test_bitwise_negative_rshift(self):
        """Arithmetic right shift preserves sign."""
        def f(a, b):
            return a >> b
        self._jit_test(f, -1024, 3)

    def test_bitwise_chained(self):
        def f(a, b, c):
            return (a & b) | c
        self._jit_test(f, 0xFF00, 0x0FF0, 0x000F)

    # ─── Complex number arithmetic ──────────────────────────────────

    def test_complex_add(self):
        def f(a, b):
            return a + b
        self._jit_test(f, (1+2j), (3+4j))

    def test_complex_subtract(self):
        def f(a, b):
            return a - b
        self._jit_test(f, (5+3j), (2+1j))

    def test_complex_multiply(self):
        def f(a, b):
            return a * b
        self._jit_test(f, (1+2j), (3+4j))

    def test_complex_truediv(self):
        def f(a, b):
            return a / b
        self._jit_test(f, (1+2j), (3+4j))

    def test_complex_power(self):
        def f(a, b):
            return a ** b
        self._jit_test(f, (1+1j), 2)

    def test_complex_negation(self):
        def f(a):
            return -a
        self._jit_test(f, (3+4j))

    def test_complex_abs(self):
        def f(a):
            return abs(a)
        self._jit_test_float(f, (3+4j))

    def test_complex_mixed_int(self):
        def f(a, b):
            return a + b, a * b
        self._jit_test(f, (1+2j), 3)

    def test_complex_mixed_float(self):
        def f(a, b):
            return a + b, a * b
        self._jit_test(f, (1+2j), 1.5)

    # ─── Boolean arithmetic ─────────────────────────────────────────

    def test_bool_add(self):
        def f(a, b):
            return a + b
        self._jit_test(f, True, True)

    def test_bool_multiply(self):
        def f(a, b):
            return a * b
        self._jit_test(f, False, 3)

    def test_bool_subtract(self):
        def f(a, b):
            return a - b
        self._jit_test(f, True, False)

    def test_bool_add_int(self):
        def f(a, b):
            return a + b
        self._jit_test(f, True, 41)

    def test_bool_mul_float(self):
        def f(a, b):
            return a * b
        self._jit_test(f, True, 3.14)

    def test_bool_power(self):
        def f(a, b):
            return a ** b
        self._jit_test(f, True, 100)

    def test_bool_false_mul_int(self):
        def f(a, b):
            return a * b
        self._jit_test(f, False, 999)

    def test_bool_bitwise(self):
        def f(a, b):
            return a & b, a | b, a ^ b
        self._jit_test(f, True, False)

    # ─── divmod ─────────────────────────────────────────────────────

    def test_divmod_int(self):
        def f(a, b):
            return divmod(a, b)
        self._jit_test(f, 17, 5)

    def test_divmod_negative(self):
        def f(a, b):
            return divmod(a, b)
        self._jit_test(f, -17, 5)

    def test_divmod_float(self):
        def f(a, b):
            return divmod(a, b)
        self._jit_test(f, 7.5, 2.0)

    def test_divmod_large(self):
        def f(a, b):
            return divmod(a, b)
        self._jit_test(f, 10**50 + 7, 13)

    def test_divmod_negative_divisor(self):
        def f(a, b):
            return divmod(a, b)
        self._jit_test(f, 17, -5)

    # ─── abs() builtin ──────────────────────────────────────────────

    def test_abs_positive_int(self):
        def f(a):
            return abs(a)
        self._jit_test(f, 42)

    def test_abs_negative_int(self):
        def f(a):
            return abs(a)
        self._jit_test(f, -42)

    def test_abs_zero_int(self):
        def f(a):
            return abs(a)
        self._jit_test(f, 0)

    def test_abs_positive_float(self):
        def f(a):
            return abs(a)
        self._jit_test(f, 3.14)

    def test_abs_negative_float(self):
        def f(a):
            return abs(a)
        self._jit_test(f, -3.14)

    def test_abs_neg_zero_float(self):
        def f(a):
            return abs(a)
        result = self._jit_test(f, -0.0)
        self.assertEqual(result, 0.0)
        self.assertEqual(math.copysign(1.0, result), 1.0)

    def test_abs_inf(self):
        def f(a):
            return abs(a)
        self._jit_test(f, float('-inf'))

    def test_abs_large_int(self):
        def f(a):
            return abs(a)
        self._jit_test(f, -(10**80))

    # ─── matmul operator (@) ────────────────────────────────────────

    def test_matmul_custom(self):
        """Test @ operator with a custom class implementing __matmul__."""
        class Vec:
            def __init__(self, items):
                self.items = list(items)
            def __matmul__(self, other):
                # Dot product
                return sum(a * b for a, b in zip(self.items, other.items))
            def __eq__(self, other):
                if isinstance(other, Vec):
                    return self.items == other.items
                return self.items == other if isinstance(other, list) else NotImplemented

        def f(a, b):
            return a @ b

        v1 = Vec([1, 2, 3])
        v2 = Vec([4, 5, 6])
        self._jit_test(f, v1, v2)

    # ─── Compound / multi-step arithmetic ───────────────────────────

    def test_compound_expression(self):
        def f(a, b, c):
            return (a + b) * c - a // b
        self._jit_test(f, 10, 3, 7)

    def test_chained_augmented(self):
        def f(a):
            a += 10
            a *= 2
            a -= 5
            a //= 3
            return a
        self._jit_test(f, 7)

    def test_nested_power(self):
        def f(a, b, c):
            return a ** (b ** c)
        self._jit_test(f, 2, 2, 3)

    def test_mixed_operations_chain(self):
        """Exercise many different opcodes in one function."""
        def f(x):
            a = x + 10
            b = a * 3
            c = b - 7
            d = c // 4
            e = d % 5
            g = e ** 2
            h = ~g
            i = -h
            j = abs(i)
            return j
        self._jit_test(f, 5)

    def test_fibonacci_arithmetic(self):
        """Fibonacci uses add + swap — good compound test."""
        def fib(n):
            a, b = 0, 1
            for _ in range(n):
                a, b = b, a + b
            return a
        self._jit_test(fib, 50)

    def test_euclidean_gcd(self):
        """GCD uses modulo and comparison."""
        def gcd(a, b):
            while b:
                a, b = b, a % b
            return a
        self._jit_test(gcd, 48, 18)

    def test_sum_of_squares(self):
        """Loop with multiply and add."""
        def f(n):
            total = 0
            for i in range(1, n + 1):
                total += i * i
            return total
        self._jit_test(f, 100)

    def test_factorial(self):
        """Iterative factorial — multiply and augmented assignment."""
        def factorial(n):
            result = 1
            for i in range(2, n + 1):
                result *= i
            return result
        self._jit_test(factorial, 20)

    def test_alternating_sum(self):
        """Subtraction, negation, addition in a loop."""
        def f(n):
            total = 0
            for i in range(1, n + 1):
                if i % 2 == 0:
                    total -= i
                else:
                    total += i
            return total
        self._jit_test(f, 100)

    # ─── Division edge cases ────────────────────────────────────────

    def test_int_truediv_exact(self):
        """10 / 2 should yield 5.0 (float)."""
        def f(a, b):
            return a / b
        result = self._jit_test(f, 10, 2)
        self.assertIsInstance(result, float)

    def test_int_truediv_negative(self):
        def f(a, b):
            return a / b
        self._jit_test(f, -7, 2)

    def test_floordiv_negative(self):
        """Python floor division rounds toward negative infinity."""
        def f(a, b):
            return a // b
        self._jit_test(f, -7, 2)

    def test_modulo_negative_divisor(self):
        """Python modulo sign follows the divisor."""
        def f(a, b):
            return a % b
        self._jit_test(f, 7, -3)

    def test_zero_division_int(self):
        def f(a, b):
            try:
                return a // b
            except ZeroDivisionError:
                return "caught"
        self._jit_test(f, 10, 0)

    def test_zero_division_float(self):
        def f(a, b):
            try:
                return a / b
            except ZeroDivisionError:
                return "caught"
        self._jit_test(f, 10.0, 0.0)

    # ─── Power edge cases ──────────────────────────────────────────

    def test_power_zero_exponent(self):
        def f(a):
            return a ** 0
        self._jit_test(f, 999)

    def test_power_one_exponent(self):
        def f(a):
            return a ** 1
        self._jit_test(f, 42)

    def test_power_negative_exponent(self):
        def f(a, b):
            return a ** b
        self._jit_test(f, 2, -3)

    def test_power_float_exponent(self):
        def f(a, b):
            return a ** b
        self._jit_test_float(f, 27.0, 1.0 / 3.0)

    def test_power_zero_base(self):
        def f(a, b):
            return a ** b
        self._jit_test(f, 0, 10)

    # ─── Three-arg pow ──────────────────────────────────────────────

    def test_pow_three_arg(self):
        """pow(base, exp, mod) — modular exponentiation."""
        def f(a, b, c):
            return pow(a, b, c)
        self._jit_test(f, 2, 10, 1000)

    def test_pow_three_arg_large(self):
        def f(a, b, c):
            return pow(a, b, c)
        self._jit_test(f, 7, 256, 13)

    # ─── round() and int/float conversion ───────────────────────────

    def test_int_to_float(self):
        def f(a):
            return float(a)
        self._jit_test(f, 42)

    def test_float_to_int(self):
        def f(a):
            return int(a)
        self._jit_test(f, 3.7)

    def test_float_to_int_negative(self):
        def f(a):
            return int(a)
        self._jit_test(f, -3.7)

    def test_round_float(self):
        def f(a):
            return round(a)
        self._jit_test(f, 3.5)

    def test_round_float_ndigits(self):
        def f(a, n):
            return round(a, n)
        self._jit_test(f, 3.14159, 2)


if __name__ == '__main__':
    unittest.main()
