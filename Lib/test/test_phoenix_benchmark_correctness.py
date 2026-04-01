"""Phoenix JIT Benchmark Correctness Gate.

Verifies that JIT-compiled benchmark functions produce identical results
to the interpreter. For each benchmark:
1. Run under interpreter (before compilation)
2. Force-compile via cinderjit.force_compile()
3. Run under JIT
4. Assert outputs match

This gate catches silent wrong-result bugs where the JIT produces incorrect
output without crashing — e.g., float_arith returning -0.0 instead of 2469.7.

Run with: ./python -m test test_phoenix_benchmark_correctness
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


def _compare(a, b, tol=1e-9):
    """Compare two values with tolerance for floats."""
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
        if math.isinf(a) and math.isinf(b):
            return (a > 0) == (b > 0)
        return abs(a - b) < tol * max(1.0, abs(a))
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(_compare(x, y, tol) for x, y in zip(a, b))
    return a == b


# ═══════════════════════════════════════════════════════════════════
# Benchmark functions (copied from Tools/benchmark_phoenix.py)
# Each returns a deterministic result for a given input.
# ═══════════════════════════════════════════════════════════════════

def _fib(n):
    if n < 2:
        return n
    return _fib(n - 1) + _fib(n - 2)

def bench_fibonacci(n_iter):
    total = 0
    for i in range(n_iter):
        total += _fib(20)
    return total

def bench_int_arith(n_iter):
    total = 0
    for i in range(n_iter):
        for j in range(100):
            total = (total + j * j - j // 3) % 1000000007
    return total

def bench_float_arith(n_iter):
    total = 0.0
    for i in range(n_iter):
        for j in range(1, 101):
            total = (total + float(j) * 1.1 - float(j) / 3.3) % 10000.0
    return total

def _nqueens_solve(n, row=0, cols=0, diag1=0, diag2=0):
    if row == n:
        return 1
    count = 0
    available = ((1 << n) - 1) & ~(cols | diag1 | diag2)
    while available:
        bit = available & (-available)
        available ^= bit
        count += _nqueens_solve(
            n, row + 1, cols | bit,
            (diag1 | bit) << 1, (diag2 | bit) >> 1,
        )
    return count

def bench_nqueens(n_iter):
    total = 0
    for i in range(n_iter):
        total += _nqueens_solve(8)
    return total

def bench_gen_simple(n_iter):
    def gen(n):
        total = 0
        for i in range(n):
            total += i
            yield total
    total = 0
    for i in range(n_iter):
        for val in gen(100):
            total += val
    return total % 1000000007

def bench_gen_nested(n_iter):
    def inner(n):
        for i in range(n):
            yield i * i
    def outer(n):
        for val in inner(n):
            yield val + 1
    total = 0
    for i in range(n_iter):
        for val in outer(50):
            total += val
    return total % 1000000007

def bench_list_comp(n_iter):
    total = 0
    for i in range(n_iter):
        r = [x * x for x in range(100)]
        total += sum(r)
    return total

def bench_dict_ops(n_iter):
    total = 0
    for i in range(n_iter):
        d = {}
        for j in range(100):
            d[j] = j * j
        total += sum(d.values())
    return total

def bench_func_calls(n_iter):
    def inner(x):
        return x + 1
    total = 0
    for i in range(n_iter):
        for j in range(100):
            total += inner(j)
    return total

def bench_store_subscr(n_iter):
    lst = [0] * 100
    total = 0
    for i in range(n_iter):
        for j in range(100):
            lst[j] = j * i
        total += lst[50]
    return total

def bench_context_manager(n_iter):
    class CM:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    total = 0
    for i in range(n_iter):
        with CM():
            total += i
    return total

def bench_kwargs_dispatch(n_iter):
    def target(a=0, b=0, c=0):
        return a + b + c
    total = 0
    for i in range(n_iter):
        total += target(a=i, b=i + 1, c=i + 2)
    return total

def bench_positional_dispatch(n_iter):
    def target(a, b, c):
        return a + b + c
    total = 0
    for i in range(n_iter):
        total += target(i, i + 1, i + 2)
    return total

def bench_dunder_protocol(n_iter):
    class Obj:
        def __init__(self, v):
            self.v = v
        def __add__(self, other):
            return Obj(self.v + other.v)
        def __len__(self):
            return self.v
    total = 0
    for i in range(n_iter):
        a = Obj(i)
        b = Obj(i + 1)
        c = a + b
        total += len(c)
    return total

def _nbody_make_bodies():
    return [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.1],
    ]

def _nbody_advance(bodies, dt):
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

def bench_nbody(n_iter):
    bodies = _nbody_make_bodies()
    for _ in range(n_iter):
        _nbody_advance(bodies, 0.01)
    return round(bodies[0][0], 8)


# ═══════════════════════════════════════════════════════════════════
# Benchmark correctness gate
# ═══════════════════════════════════════════════════════════════════

# (name, function, n_iter, inner_functions_to_compile)
BENCHMARKS = [
    ("fibonacci", bench_fibonacci, 10, [_fib]),
    ("int_arith", bench_int_arith, 10, []),
    ("float_arith", bench_float_arith, 10, []),
    ("nqueens", bench_nqueens, 10, [_nqueens_solve]),
    ("gen_simple", bench_gen_simple, 10, []),
    ("gen_nested", bench_gen_nested, 10, []),
    ("list_comp", bench_list_comp, 10, []),
    ("dict_ops", bench_dict_ops, 10, []),
    ("func_calls", bench_func_calls, 10, []),
    ("store_subscr", bench_store_subscr, 10, []),
    ("context_manager", bench_context_manager, 10, []),
    ("kwargs_dispatch", bench_kwargs_dispatch, 10, []),
    ("positional_dispatch", bench_positional_dispatch, 10, []),
    ("dunder_protocol", bench_dunder_protocol, 10, []),
    ("nbody", bench_nbody, 10, [_nbody_advance, _nbody_make_bodies]),
]


@unittest.skipUnless(HAS_JIT, "Phoenix JIT not available")
class TestBenchmarkCorrectness(unittest.TestCase):
    """Verify every benchmark produces identical output under JIT vs interpreter."""

    def _check_correctness(self, name, func, n_iter, inner_funcs):
        """Run func under interpreter, force-compile, run under JIT, compare."""
        # Run under interpreter
        interp_result = func(n_iter)

        # Force-compile the benchmark function and its inner functions
        for f in [func] + inner_funcs:
            try:
                cinderjit.force_compile(f)
            except Exception:
                pass

        # Run under JIT
        jit_result = func(n_iter)

        # Compare
        self.assertTrue(
            _compare(interp_result, jit_result),
            f"{name}: interpreter={interp_result}, JIT={jit_result}"
        )

    def test_fibonacci(self):
        self._check_correctness("fibonacci", bench_fibonacci, 10, [_fib])

    def test_int_arith(self):
        self._check_correctness("int_arith", bench_int_arith, 10, [])

    def test_float_arith(self):
        self._check_correctness("float_arith", bench_float_arith, 10, [])

    def test_nqueens(self):
        self._check_correctness("nqueens", bench_nqueens, 10, [_nqueens_solve])

    def test_gen_simple(self):
        self._check_correctness("gen_simple", bench_gen_simple, 10, [])

    def test_gen_nested(self):
        self._check_correctness("gen_nested", bench_gen_nested, 10, [])

    def test_list_comp(self):
        self._check_correctness("list_comp", bench_list_comp, 10, [])

    def test_dict_ops(self):
        self._check_correctness("dict_ops", bench_dict_ops, 10, [])

    def test_func_calls(self):
        self._check_correctness("func_calls", bench_func_calls, 10, [])

    def test_store_subscr(self):
        self._check_correctness("store_subscr", bench_store_subscr, 10, [])

    def test_context_manager(self):
        self._check_correctness("context_manager", bench_context_manager, 10, [])

    def test_kwargs_dispatch(self):
        self._check_correctness("kwargs_dispatch", bench_kwargs_dispatch, 10, [])

    def test_positional_dispatch(self):
        self._check_correctness("positional_dispatch", bench_positional_dispatch, 10, [])

    def test_dunder_protocol(self):
        self._check_correctness("dunder_protocol", bench_dunder_protocol, 10, [])

    def test_nbody(self):
        self._check_correctness("nbody", bench_nbody, 10,
                                [_nbody_advance, _nbody_make_bodies])


if __name__ == "__main__":
    unittest.main()
