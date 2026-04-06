#!/usr/bin/env python3
"""Phoenix JIT benchmark suite.

Adapted from CinderX benchmark_cinderx.py (commit a2920ee6).

METHODOLOGY: All comparisons use ABBA interleaving (A, B, B, A) to
control for thermal drift and co-located workload noise.

SUBCOMMANDS:
  abba    — Builtin micro-benchmarks: JIT function vs interpreter function
            (in-process, same Python, independent function objects)
  jit     — Overall JIT benchmarks across many workloads
            (in-process, single Python with auto-compilation)
  all     — Run all of the above

Phoenix uses auto-compilation only (threshold=1000 calls). There is no
force_compile Python API. Functions are JIT-compiled after reaching the
call threshold via the func watcher + counting trampoline.

FALSIFICATION:
  - Control: run without JIT → delta should be ~0 for in-process tests
  - IQR must not span zero for a result to be marked significant
  - Raw block deltas printed for manual drift inspection

USAGE:
  ./python Tools/benchmark_phoenix.py abba
  ./python Tools/benchmark_phoenix.py jit
  ./python Tools/benchmark_phoenix.py all
  ./python Tools/benchmark_phoenix.py abba --blocks=20 --iters=100000
"""

import argparse
import contextlib
import functools
import math
import os
import platform
import statistics
import sys
import time

# ═══════════════════════════════════════════════════════════════════════════
# Early JIT init — must happen BEFORE function definitions so the func
# watcher can install the counting trampoline on module-level functions.
# Without this, functions defined before JIT init never auto-compile.
# ═══════════════════════════════════════════════════════════════════════════
try:
    import _cinderx
    # If --no-jit requested, disable JIT immediately before any function
    # reaches threshold. Must happen before function definitions below.
    if '--no-jit' in sys.argv:
        import cinderjit
        cinderjit.disable()
except ImportError:
    pass

# ═══════════════════════════════════════════════════════════════════════════
# Configuration defaults
# ═══════════════════════════════════════════════════════════════════════════

ABBA_BLOCKS = 15
BENCH_ITERS = 50_000
INNER_ITERS = 100
WARMUP_ITERS = 5_000
AUTO_COMPILE_THRESHOLD = 1000  # Phoenix auto-compilation threshold

# ═══════════════════════════════════════════════════════════════════════════
# Phoenix JIT helpers
# ═══════════════════════════════════════════════════════════════════════════

_jit_disabled = False  # Set by --no-jit flag

def check_jit_available():
    """Check if Phoenix JIT is loaded. Returns True if _cinderx is importable."""
    if _jit_disabled:
        return False
    try:
        import _cinderx
        return True
    except ImportError:
        return False

def warmup_function(func, iters=None, arg=1):
    """Warmup a function to trigger auto-compilation at threshold."""
    iters = iters or WARMUP_ITERS
    for _ in range(iters):
        func(arg)

# ═══════════════════════════════════════════════════════════════════════════
# ABBA engine
# ═══════════════════════════════════════════════════════════════════════════

def time_one(func, n):
    """Time a single benchmark invocation. Returns seconds."""
    t0 = time.perf_counter()
    func(n)
    t1 = time.perf_counter()
    return t1 - t0

def run_abba(func_a, func_b, n_blocks, bench_iters):
    """Run ABBA interleaved comparison.

    Each block: A, B, B, A. Monotonic drift within a block cancels.

    Returns dict with raw times, deltas, median, IQR, significance.
    """
    a_times = []
    b_times = []
    deltas = []

    for _ in range(n_blocks):
        ta1 = time_one(func_a, bench_iters)
        tb1 = time_one(func_b, bench_iters)
        tb2 = time_one(func_b, bench_iters)
        ta2 = time_one(func_a, bench_iters)

        a_times.extend([ta1, ta2])
        b_times.extend([tb1, tb2])

        block_a_mean = (ta1 + ta2) / 2
        block_b_mean = (tb1 + tb2) / 2
        deltas.append(block_a_mean - block_b_mean)

    a_times.sort()
    b_times.sort()
    deltas.sort()

    median_a = statistics.median(a_times)
    median_b = statistics.median(b_times)
    median_delta = statistics.median(deltas)

    q1_idx = len(deltas) // 4
    q3_idx = 3 * len(deltas) // 4
    iqr_lo = deltas[q1_idx]
    iqr_hi = deltas[q3_idx]

    # Significant if IQR does not span zero
    significant = (iqr_lo > 0 and iqr_hi > 0) or (iqr_lo < 0 and iqr_hi < 0)

    total_calls = bench_iters * INNER_ITERS
    ns_a = median_a / total_calls * 1e9 if total_calls > 0 else 0
    ns_b = median_b / total_calls * 1e9 if total_calls > 0 else 0
    pct = (median_b - median_a) / median_b * 100 if median_b > 0 else 0

    return {
        "a_times": a_times,
        "b_times": b_times,
        "deltas": deltas,
        "median_a": median_a,
        "median_b": median_b,
        "ns_a": ns_a,
        "ns_b": ns_b,
        "median_delta": median_delta,
        "iqr_lo": iqr_lo,
        "iqr_hi": iqr_hi,
        "significant": significant,
        "pct_improvement": pct,
    }

def print_abba_results(results, labels=("A", "B")):
    """Print a table of ABBA results."""
    label_a, label_b = labels
    print(
        f"{'Benchmark':20s} {label_a + ' ns/call':>12s} {label_b + ' ns/call':>12s} "
        f"{'Improv%':>8s} {'Signif':>7s} {'IQR':>22s}"
    )
    print("-" * 85)

    for r in results:
        sig = "YES" if r["significant"] else "no"
        iqr_str = f"[{r['iqr_lo']*1e3:+.3f}, {r['iqr_hi']*1e3:+.3f}] ms"
        label = r.get("label", "?")[:20]
        print(
            f"  {label:20s} {r['ns_a']:10.1f}   {r['ns_b']:10.1f}   "
            f"{r['pct_improvement']:+6.1f}%  {sig:>7s}  {iqr_str}"
        )

    print()
    sig_count = sum(1 for r in results if r["significant"])
    print(f"Significant results: {sig_count}/{len(results)}")
    print()
    print("Interpretation:")
    print("  Signif=YES: IQR of per-block deltas does not span zero.")
    print("  Signif=no:  IQR spans zero. Cannot distinguish from noise.")
    print("  Improv%:    Positive = A faster than B. Negative = B faster.")
    print()
    print("Raw per-block deltas (ms) -- inspect for drift patterns:")
    for r in results:
        deltas_ms = [f"{d*1e3:+.3f}" for d in r["deltas"]]
        label = r.get("label", "?")[:20]
        print(f"  {label:20s} [{', '.join(deltas_ms)}]")
    print()

# ═══════════════════════════════════════════════════════════════════════════
# Benchmark definitions
# ═══════════════════════════════════════════════════════════════════════════

# --- ABBA micro-benchmark targets ---

class _Obj:
    __slots__ = ("x", "y", "z")
    def __init__(self):
        self.x = 1
        self.y = 2
        self.z = 3

class _Animal:
    pass

class _Dog(_Animal):
    pass

def _gen():
    while True:
        yield 1

def make_isinstance():
    def bench(n):
        obj = _Dog()
        total = 0
        for _ in range(n):
            for _ in range(INNER_ITERS):
                total += isinstance(obj, _Animal)
        return total
    return bench, INNER_ITERS

def make_issubclass():
    def bench(n):
        total = 0
        for _ in range(n):
            for _ in range(INNER_ITERS):
                total += issubclass(_Dog, _Animal)
        return total
    return bench, INNER_ITERS

def make_hasattr():
    def bench(n):
        obj = _Obj()
        total = 0
        for _ in range(n):
            for _ in range(INNER_ITERS):
                total += hasattr(obj, "x")
        return total
    return bench, INNER_ITERS

def make_getattr_bench():
    def bench(n):
        obj = _Obj()
        total = 0
        for _ in range(n):
            for _ in range(INNER_ITERS):
                total += getattr(obj, "x")
        return total
    return bench, INNER_ITERS

def make_next_bench():
    def bench(n):
        g = _gen()
        total = 0
        for _ in range(n):
            for _ in range(INNER_ITERS):
                total += next(g)
        return total
    return bench, INNER_ITERS

def make_next_default():
    def bench(n):
        g = _gen()
        total = 0
        for _ in range(n):
            for _ in range(INNER_ITERS):
                total += next(g, 0)
        return total
    return bench, INNER_ITERS

def make_divmod_bench():
    def bench(n):
        total = 0
        for _ in range(n):
            for _ in range(INNER_ITERS):
                q, r = divmod(1000007, 37)
                total += q
        return total
    return bench, None

ABBA_BENCHMARKS = [
    ("isinstance",   make_isinstance),
    ("issubclass",   make_issubclass),
    ("hasattr",      make_hasattr),
    ("getattr",      make_getattr_bench),
    ("next",         make_next_bench),
    ("next_default", make_next_default),
    ("divmod",       make_divmod_bench),
]

# --- JIT benchmark functions ---

def _fib(n):
    if n < 2:
        return n
    return _fib(n - 1) + _fib(n - 2)

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

_SPECTRAL_N = 100

def _spectral_A(i, j):
    return 1.0 / ((i + j) * (i + j + 1) // 2 + i + 1)

def _spectral_mul_Av(v):
    n = _SPECTRAL_N
    return [sum(_spectral_A(i, j) * v[j] for j in range(n)) for i in range(n)]

def _spectral_mul_Atv(v):
    n = _SPECTRAL_N
    return [sum(_spectral_A(j, i) * v[j] for j in range(n)) for i in range(n)]

def _spectral_mul_AtAv(v):
    return _spectral_mul_Atv(_spectral_mul_Av(v))

def bench_fibonacci(n_iter):
    """Recursive fibonacci -- tests function call overhead."""
    total = 0
    for _ in range(n_iter // 10):
        total += _fib(20)
    return total

class _Packet:
    __slots__ = ('link', 'ident', 'kind', 'datum', 'data')
    def __init__(self, link, ident, kind):
        self.link = link
        self.ident = ident
        self.kind = kind
        self.datum = 0
        self.data = [0] * 4

_TASK_IDLE = 1
_TASK_WORK = 2
_TASK_HANDLER_A = 3
_TASK_HANDLER_B = 4
_DEVICE_A = 5
_DEVICE_B = 6

_K_DEV = 1000
_K_WORK = 1001

class _Task:
    __slots__ = ('link', 'ident', 'pri', 'input', 'state',
                 'handle', 'v1', 'v2', 'layout')
    def __init__(self, link, ident, pri, input_queue, state, handle):
        self.link = link
        self.ident = ident
        self.pri = pri
        self.input = input_queue
        self.state = state
        self.handle = handle
        self.v1 = 0
        self.v2 = 0
        self.layout = 0

_STATE_RUN = 0
_STATE_RUNPKT = 1
_STATE_WAIT = 2
_STATE_WAITPKT = 3
_STATE_HOLD = 4

class _RichardsRunner:
    __slots__ = ('task_list', 'task_table', 'queue_count', 'hold_count',
                 'current_task', 'current_id', 'layout')
    def __init__(self):
        self.task_list = None
        self.task_table = [None] * 10
        self.queue_count = 0
        self.hold_count = 0
        self.current_task = None
        self.current_id = 0
        self.layout = 0

    def add_task(self, ident, pri, input_queue, state, handle):
        t = _Task(self.task_list, ident, pri, input_queue, state, handle)
        self.task_list = t
        self.task_table[ident] = t

    def find_task(self, ident):
        return self.task_table[ident]

    def hold_current(self):
        self.hold_count += 1
        self.current_task.state = _STATE_HOLD
        return self.current_task.link

    def release(self, ident):
        t = self.find_task(ident)
        if t is None:
            return None
        if t.state == _STATE_HOLD:
            t.state = _STATE_RUN
        if t.pri > self.current_task.pri:
            return t
        return self.current_task

    def queue_packet(self, pkt):
        t = self.find_task(pkt.ident)
        if t is None:
            return None
        self.queue_count += 1
        pkt.link = None
        pkt.ident = self.current_id
        if t.input is None:
            t.input = pkt
            if t.state == _STATE_WAIT:
                t.state = _STATE_RUNPKT
            if t.pri > self.current_task.pri:
                return t
        else:
            p = t.input
            while p.link is not None:
                p = p.link
            p.link = pkt
        return self.current_task

    def schedule(self):
        t = self.task_list
        while t is not None:
            if t.state == _STATE_RUN or t.state == _STATE_RUNPKT:
                self.current_task = t
                self.current_id = t.ident
                pkt = None
                if t.state == _STATE_RUNPKT:
                    pkt = t.input
                    t.input = pkt.link
                    if t.input is None:
                        t.state = _STATE_RUN
                    else:
                        t.state = _STATE_RUNPKT
                next_task = t.handle(self, t, pkt)
                if next_task is not None:
                    t = next_task
                    continue
            t = t.link

def _idle_fn(runner, task, pkt):
    task.v1 -= 1
    if task.v1 == 0:
        return runner.hold_current()
    if (task.v2 & 1) == 0:
        task.v2 >>= 1
        return runner.release(_DEVICE_A)
    else:
        task.v2 = (task.v2 >> 1) ^ 0xD008
        return runner.release(_DEVICE_B)

def _work_fn(runner, task, pkt):
    if pkt is None:
        return runner.hold_current()
    dest = _TASK_HANDLER_A if task.v1 == _TASK_HANDLER_B else _TASK_HANDLER_B
    task.v1 = dest
    pkt.ident = dest
    pkt.datum = 0
    for i in range(4):
        task.v2 += 1
        if task.v2 > 26:
            task.v2 = 1
        pkt.data[i] = 65 + task.v2 - 1
    return runner.queue_packet(pkt)

def _handler_fn(runner, task, pkt):
    if pkt is not None:
        if pkt.kind == _K_WORK:
            task.v1 = pkt
        else:
            task.v2 = pkt
    work = task.v1
    if work is not None and isinstance(work, _Packet):
        count = work.datum
        if count >= 4:
            task.v1 = work.link
            return runner.queue_packet(work)
        dev = task.v2
        if dev is not None and isinstance(dev, _Packet):
            task.v2 = dev.link
            dev.datum = work.data[count]
            work.datum = count + 1
            return runner.queue_packet(dev)
    return runner.hold_current()

def _device_fn(runner, task, pkt):
    if pkt is None:
        v = task.v1
        if v is not None and isinstance(v, _Packet):
            task.v1 = None
            return runner.queue_packet(v)
        return runner.hold_current()
    task.v1 = pkt
    return runner.hold_current()

def _run_richards_once():
    r = _RichardsRunner()
    wkq = _Packet(None, _TASK_WORK, _K_WORK)
    wkq = _Packet(wkq, _TASK_WORK, _K_WORK)
    r.add_task(_TASK_IDLE, 0, wkq, _STATE_RUNPKT, _idle_fn)
    r.find_task(_TASK_IDLE).v1 = 100
    r.find_task(_TASK_IDLE).v2 = 0x3039

    wkq = _Packet(None, _TASK_HANDLER_A, _K_WORK)
    wkq = _Packet(wkq, _TASK_HANDLER_A, _K_WORK)
    r.add_task(_TASK_WORK, 1000, wkq, _STATE_RUNPKT, _work_fn)
    r.find_task(_TASK_WORK).v1 = _TASK_HANDLER_A
    r.find_task(_TASK_WORK).v2 = 0

    wkq = _Packet(None, _DEVICE_A, _K_DEV)
    wkq = _Packet(wkq, _DEVICE_A, _K_DEV)
    wkq = _Packet(wkq, _DEVICE_A, _K_DEV)
    r.add_task(_TASK_HANDLER_A, 2000, wkq, _STATE_RUNPKT, _handler_fn)

    wkq = _Packet(None, _DEVICE_B, _K_DEV)
    wkq = _Packet(wkq, _DEVICE_B, _K_DEV)
    wkq = _Packet(wkq, _DEVICE_B, _K_DEV)
    r.add_task(_TASK_HANDLER_B, 3000, wkq, _STATE_RUNPKT, _handler_fn)

    r.add_task(_DEVICE_A, 4000, None, _STATE_WAIT, _device_fn)
    r.add_task(_DEVICE_B, 5000, None, _STATE_WAIT, _device_fn)

    r.schedule()
    return (r.queue_count, r.hold_count)

class _RichardsTask:
    __slots__ = ("id", "pri", "nxt", "state")
    def __init__(self, tid, pri):
        self.id = tid
        self.pri = pri
        self.nxt = None
        self.state = 0

def bench_richards_full(n_iter):
    """Full Richards scheduler -- method dispatch, attribute access, control flow."""
    total_q = 0
    total_h = 0
    for _ in range(n_iter // 100):
        q, h = _run_richards_once()
        total_q += q
        total_h += h
    return total_q + total_h

def bench_richards_slots(n_iter):
    """Richards scheduler with __slots__ -- attribute access."""
    total = 0
    for _ in range(n_iter):
        tasks = [_RichardsTask(i, i * 10) for i in range(10)]
        for i in range(len(tasks) - 1):
            tasks[i].nxt = tasks[i + 1]
        t = tasks[0]
        while t is not None:
            total += t.pri
            t = t.nxt
    return total

def bench_nqueens(n_iter):
    """N-queens solver -- recursive backtracking, bit operations."""
    total = 0
    for _ in range(n_iter // 100):
        total += _nqueens_solve(8)
    return total

def bench_spectral_norm(n_iter):
    """Spectral norm -- list comprehensions, floating point."""
    total = 0.0
    for _ in range(n_iter // 1000):
        u = [1.0] * _SPECTRAL_N
        for _ in range(5):
            v = _spectral_mul_AtAv(u)
            u = _spectral_mul_AtAv(v)
        vBv = sum(ui * vi for ui, vi in zip(u, v))
        vv = sum(vi * vi for vi in v)
        total += math.sqrt(vBv / vv)
    return total

def bench_float_arith(n_iter):
    """Float arithmetic -- math module, basic ops."""
    total = 0.0
    for i in range(n_iter):
        x = float(i) * 0.001
        total += math.sin(x) * math.cos(x) + math.sqrt(abs(x) + 1.0)
    return total

def bench_gen_simple(n_iter):
    """Simple generator iteration."""
    def gen(n):
        for i in range(n):
            yield i
    total = 0
    for _ in range(n_iter // 100):
        for v in gen(100):
            total += v
    return total

def bench_gen_nested(n_iter):
    """Nested generator with function calls."""
    def compute(a, b):
        return a * b + a - b
    def gen(n):
        for i in range(n):
            yield compute(i, i + 1)
    total = 0
    for _ in range(n_iter // 100):
        for v in gen(100):
            total += v
    return total

def bench_list_comp(n_iter):
    """List comprehension -- creation and iteration."""
    total = 0
    for _ in range(n_iter // 100):
        xs = [i * i for i in range(100)]
        total += sum(xs)
    return total

def bench_dict_ops(n_iter):
    """Dictionary operations -- creation, lookup, iteration."""
    total = 0
    for _ in range(n_iter // 100):
        d = {i: i * i for i in range(100)}
        for k, v in d.items():
            total += v
    return total

def bench_func_calls(n_iter):
    """Function call overhead -- simple argument passing."""
    def add3(a, b, c):
        return a + b + c
    total = 0
    for i in range(n_iter):
        total += add3(i, i + 1, i + 2)
    return total

def bench_int_arith(n_iter):
    """Pure integer arithmetic."""
    total = 0
    a, b = 3, 7
    for i in range(n_iter):
        total += a * i + b
        a = (a + 1) % 127
        b = (b + 3) % 131
    return total

def bench_store_subscr(n_iter):
    """List and dict subscript store."""
    xs = [0] * 100
    d = {}
    total = 0
    for i in range(n_iter):
        idx = i % 100
        xs[idx] = i
        d[idx] = i
        total += xs[idx] + d[idx]
    return total

class _CMgr:
    __slots__ = ('val',)
    def __init__(self):
        self.val = 0
    def __enter__(self):
        self.val += 1
        return self
    def __exit__(self, *args):
        self.val -= 1
        return False

def bench_context_manager(n_iter):
    """Context manager dispatch -- __enter__/__exit__ protocol."""
    mgr = _CMgr()
    total = 0
    for _ in range(n_iter):
        with mgr as m:
            total += m.val
    return total

def _kwargs_callee(**kwargs):
    return kwargs.get('a', 0) + kwargs.get('b', 0)

def bench_kwargs_dispatch(n_iter):
    """Keyword argument unpacking -- **kwargs overhead."""
    total = 0
    for i in range(n_iter):
        total += _kwargs_callee(a=i, b=i+1)
    return total

def _positional_callee(a=0, b=0):
    return a + b

def bench_positional_dispatch(n_iter):
    """Keyword call-site to positional callee."""
    total = 0
    for i in range(n_iter):
        total += _positional_callee(a=i, b=i+1)
    return total

class _DunderObj:
    __slots__ = ('_data',)
    def __init__(self):
        self._data = list(range(10))
    def __getitem__(self, idx):
        return self._data[idx % 10]
    def __len__(self):
        return len(self._data)
    def __iter__(self):
        return iter(self._data)
    def __contains__(self, item):
        return item in self._data

def bench_dunder_protocol(n_iter):
    """Dunder protocol dispatch -- __getitem__, __len__, __contains__."""
    obj = _DunderObj()
    total = 0
    for i in range(n_iter):
        total += obj[i] + len(obj)
        if (i % 10) in obj:
            total += 1
    return total

# --- N-body simulation ---

_NBODY_PI = 3.14159265358979323846
_NBODY_SOLAR_MASS = 4 * _NBODY_PI * _NBODY_PI
_NBODY_DAYS_PER_YEAR = 365.24

def _nbody_make_bodies():
    return [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, _NBODY_SOLAR_MASS],
        [4.84143144246472090, -1.16032004402742839, -1.03622044471123109e-01,
         1.66007664274403694e-03 * _NBODY_DAYS_PER_YEAR,
         7.69901118419740425e-03 * _NBODY_DAYS_PER_YEAR,
         -6.90460016972063023e-05 * _NBODY_DAYS_PER_YEAR,
         9.54791938424326609e-04 * _NBODY_SOLAR_MASS],
        [8.34336671824457987, 4.12479856412430479, -4.03523417114321381e-01,
         -2.76742510726862411e-03 * _NBODY_DAYS_PER_YEAR,
         4.99852801234917238e-03 * _NBODY_DAYS_PER_YEAR,
         2.30417297573763929e-05 * _NBODY_DAYS_PER_YEAR,
         2.85885980666130812e-04 * _NBODY_SOLAR_MASS],
        [1.28943695621391310e+01, -1.51111514016986312e+01, -2.23307578892655734e-01,
         2.96460137564761618e-03 * _NBODY_DAYS_PER_YEAR,
         2.37847173959480950e-03 * _NBODY_DAYS_PER_YEAR,
         -2.96589568540237556e-05 * _NBODY_DAYS_PER_YEAR,
         4.36624404335156298e-05 * _NBODY_SOLAR_MASS],
        [1.53796971148509165e+01, -2.59193146099879641e+01, 1.79258772950371181e-01,
         2.68067772490389322e-03 * _NBODY_DAYS_PER_YEAR,
         1.62824170038242295e-03 * _NBODY_DAYS_PER_YEAR,
         -9.51592254519715870e-05 * _NBODY_DAYS_PER_YEAR,
         5.15138902046611451e-05 * _NBODY_SOLAR_MASS],
    ]

def _nbody_advance(bodies, dt):
    n = len(bodies)
    for i in range(n):
        bi = bodies[i]
        bix, biy, biz = bi[0], bi[1], bi[2]
        bivx, bivy, bivz = bi[3], bi[4], bi[5]
        bim = bi[6]
        for j in range(i + 1, n):
            bj = bodies[j]
            dx = bix - bj[0]
            dy = biy - bj[1]
            dz = biz - bj[2]
            dist2 = dx * dx + dy * dy + dz * dz
            mag = dt / (dist2 * dist2 ** 0.5)
            bjm = bj[6]
            bivx -= dx * bjm * mag
            bivy -= dy * bjm * mag
            bivz -= dz * bjm * mag
            bj[3] += dx * bim * mag
            bj[4] += dy * bim * mag
            bj[5] += dz * bim * mag
        bi[3] = bivx
        bi[4] = bivy
        bi[5] = bivz
    for b in bodies:
        b[0] += dt * b[3]
        b[1] += dt * b[4]
        b[2] += dt * b[5]

def _nbody_energy(bodies):
    e = 0.0
    n = len(bodies)
    for i in range(n):
        bi = bodies[i]
        e += 0.5 * bi[6] * (bi[3]*bi[3] + bi[4]*bi[4] + bi[5]*bi[5])
        for j in range(i + 1, n):
            bj = bodies[j]
            dx = bi[0] - bj[0]
            dy = bi[1] - bj[1]
            dz = bi[2] - bj[2]
            dist = (dx*dx + dy*dy + dz*dz) ** 0.5
            e -= bi[6] * bj[6] / dist
    return e

def bench_nbody(n_iter):
    bodies = _nbody_make_bodies()
    px = py = pz = 0.0
    for b in bodies:
        px -= b[3] * b[6]
        py -= b[4] * b[6]
        pz -= b[5] * b[6]
    bodies[0][3] = px / _NBODY_SOLAR_MASS
    bodies[0][4] = py / _NBODY_SOLAR_MASS
    bodies[0][5] = pz / _NBODY_SOLAR_MASS
    for _ in range(n_iter // 10):
        _nbody_advance(bodies, 0.01)
    return _nbody_energy(bodies)

# --- Decorator chain benchmark ---

def _timer_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

def _validator_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result
    return wrapper

def _logger_decorator(func):
    count = [0]
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        count[0] += 1
        return func(*args, **kwargs)
    wrapper.call_count = count
    return wrapper

def _cacher_decorator(func):
    cache = {}
    @functools.wraps(func)
    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    wrapper.cache = cache
    return wrapper

class _DecoratedCompute:
    @_timer_decorator
    @_validator_decorator
    @_logger_decorator
    def add(self, a, b):
        return a + b

    @_timer_decorator
    @_validator_decorator
    def multiply(self, a, b):
        return a * b

    @_cacher_decorator
    def fibonacci(self, n):
        if n < 2:
            return n
        return self.fibonacci(n - 1) + self.fibonacci(n - 2)

    @staticmethod
    def static_op(x, y):
        return x * y + y

    @classmethod
    def class_op(cls, x):
        return x * 2

def _make_adder(offset):
    def adder(x):
        return x + offset
    return adder

def bench_decorator_chain(n_iter):
    """Decorator chain -- functools.wraps, stacked decorators, closures."""
    comp = _DecoratedCompute()
    adders = [_make_adder(i * 0.1) for i in range(10)]
    total = 0.0
    for i in range(n_iter // 10):
        total += comp.add(total % 100, float(i % 50))
        total += comp.multiply(total % 100, 0.99)
        fib_val = comp.fibonacci(i % 20)
        total += fib_val * 0.001
        total += _DecoratedCompute.static_op(total % 100, 0.5)
        total += _DecoratedCompute.class_op(total % 100)
        for adder in adders:
            total = adder(total % 100)
        total = total % 10000.0
    return total

# --- Deep class hierarchy ---

class _DCBase:
    def __init__(self, name):
        self.name = name
        self.training = True
        self._forward_hooks = []
    def parameters(self):
        return [v for k, v in self.__dict__.items() if isinstance(v, float)]
    def train(self, mode=True):
        self.training = mode
        return self

class _DCLayer(_DCBase):
    def __init__(self, name, in_features, out_features):
        super().__init__(name)
        self.in_features = in_features
        self.out_features = out_features
        self.weight = 0.01 * in_features * out_features
        self.bias = 0.01 * out_features
    def forward(self, x):
        return x * self.weight + self.bias

class _DCBlock(_DCLayer):
    def __init__(self, name, features, num_layers=3):
        super().__init__(name, features, features)
        self.num_layers = num_layers
        self.scale = 1.0 / num_layers
        self.layers = [_DCLayer(f"{name}_sub_{i}", features, features)
                       for i in range(num_layers)]
    def forward(self, x):
        residual = x
        for layer in self.layers:
            x = layer.forward(x) * self.scale
        return x + residual

class _DCNetwork(_DCBlock):
    def __init__(self, name, features, num_blocks=2):
        super().__init__(name, features, num_layers=3)
        self.num_blocks = num_blocks
        self.blocks = [_DCBlock(f"{name}_block_{i}", features)
                       for i in range(num_blocks)]
    def forward(self, x):
        for block in self.blocks:
            x = block.forward(x)
        return x

class _DCModel(_DCNetwork):
    def __init__(self, name, features=64, num_blocks=2):
        super().__init__(name, features, num_blocks)
        self.classifier_weight = 0.01 * features
        self.classifier_bias = 0.001
    def forward(self, x):
        x = super().forward(x)
        return x * self.classifier_weight + self.classifier_bias
    def __repr__(self):
        return (f"Model({self.name}, features={self.in_features}, "
                f"blocks={self.num_blocks})")

def bench_deep_class_super(n_iter):
    """5-level class hierarchy with super() -- MRO, isinstance, repr."""
    total = 0.0
    for _ in range(n_iter // 10):
        model = _DCModel("bench", features=32, num_blocks=2)
        result = model.forward(1.0)
        total += result % 100.0
        if isinstance(model, _DCModel):
            total += 0.001
        if isinstance(model, _DCNetwork):
            total += 0.001
        if isinstance(model, _DCBlock):
            total += 0.001
        if isinstance(model, _DCLayer):
            total += 0.001
        if isinstance(model, _DCBase):
            total += 0.001
        _ = model.training
        _ = model.in_features
        _ = model.num_layers
        _ = model.num_blocks
        model.train(False)
        params = model.parameters()
        total += len(params) * 0.001
        _ = repr(model)
    return total

# --- PyTorch-style context managers ---

class _NoGrad:
    _enabled = True
    def __enter__(self):
        self._prev = _NoGrad._enabled
        _NoGrad._enabled = False
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        _NoGrad._enabled = self._prev
        return False

class _Autocast:
    _mode = 'float32'
    def __init__(self, mode='float16'):
        self._target = mode
    def __enter__(self):
        self._prev = _Autocast._mode
        _Autocast._mode = self._target
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        _Autocast._mode = self._prev
        return False

class _ProfileScope:
    _depth = 0
    _total = 0
    def __init__(self, name):
        self._name = name
    def __enter__(self):
        _ProfileScope._depth += 1
        _ProfileScope._total += 1
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        _ProfileScope._depth -= 1
        return False

@contextlib.contextmanager
def _training_mode(model_dict, mode=True):
    prev = model_dict.get('training', True)
    model_dict['training'] = mode
    try:
        yield model_dict
    finally:
        model_dict['training'] = prev

def bench_pytorch_cm(n_iter):
    """PyTorch-style context managers -- nested, contextlib, state toggle."""
    model = {'training': True, 'weight': 1.0, 'bias': 0.0}
    total = 0.0
    for i in range(n_iter // 10):
        with _NoGrad():
            total += model['weight'] * float(i % 100) + model['bias']
        with _NoGrad():
            with _Autocast('float16'):
                total += total % 1000 * 0.99
        with _ProfileScope('forward'):
            with _NoGrad():
                with _Autocast('bfloat16'):
                    total = (total % 1000) + 0.001
        with _training_mode(model, mode=False) as m:
            total += m['weight'] * 0.5
        for j in range(5):
            with _ProfileScope(f'layer_{j}'):
                total = (total + float(j)) % 10000
    return total

# --- Benchmark registries ---

JIT_BENCHMARKS = [
    ("fibonacci",           bench_fibonacci),
    ("richards_full",       bench_richards_full),
    ("richards_slots",      bench_richards_slots),
    ("nqueens",             bench_nqueens),
    ("spectral_norm",       bench_spectral_norm),
    ("float_arith",         bench_float_arith),
    ("gen_simple",          bench_gen_simple),
    ("gen_nested",          bench_gen_nested),
    ("list_comp",           bench_list_comp),
    ("dict_ops",            bench_dict_ops),
    ("func_calls",          bench_func_calls),
    ("store_subscr",        bench_store_subscr),
    ("int_arith",           bench_int_arith),
    ("context_manager",     bench_context_manager),
    ("kwargs_dispatch",     bench_kwargs_dispatch),
    ("positional_dispatch", bench_positional_dispatch),
    ("dunder_protocol",     bench_dunder_protocol),
    ("nbody",               bench_nbody),
    ("decorator_chain",     bench_decorator_chain),
    ("deep_class_super",    bench_deep_class_super),
    ("pytorch_cm",          bench_pytorch_cm),
]

# ═══════════════════════════════════════════════════════════════════════════
# Subcommand: abba
# ═══════════════════════════════════════════════════════════════════════════

def cmd_abba(args):
    """Run ABBA micro-benchmarks.

    Creates two independent copies of each benchmark function. Warms up
    copy A past the auto-compilation threshold (so it gets JIT-compiled),
    keeps copy B below threshold (interpreter-only). Measures A vs B
    using ABBA interleaving.
    """
    print("=" * 72)
    print("ABBA Interleaved Benchmark -- JIT vs Interpreter (in-process)")
    print("=" * 72)
    print(f"Python:       {sys.version}")
    print(f"Platform:     {platform.machine()}")
    print(f"JIT active:   {check_jit_available()}")
    print(f"ABBA_BLOCKS:  {args.blocks}")
    print(f"BENCH_ITERS:  {args.iters}")
    print(f"INNER_ITERS:  {INNER_ITERS}")
    print(f"WARMUP_ITERS: {args.warmup}")
    print()

    jit_available = check_jit_available()
    if not jit_available:
        print("MODE: CONTROL (no Phoenix JIT available)")
        print("A and B use identical code paths. Delta should be ~0.")
    else:
        print("MODE: JIT vs Interpreter")
        print("A = warmed past auto-compile threshold, B = cold (interpreter).")
    print()

    all_results = []

    for bench_name, factory in ABBA_BENCHMARKS:
        func_a, expected = factory()
        func_b, _ = factory()

        # Correctness check
        if expected is not None:
            result_a = func_a(1)
            assert result_a == expected, (
                f"{bench_name} correctness: got {result_a}, expected {expected}"
            )

        # Warmup A past auto-compilation threshold
        warmup_function(func_a, args.warmup)
        # B stays cold (below threshold)

        label = f"{bench_name}"
        result = run_abba(func_a, func_b, args.blocks, args.iters)
        result["label"] = label
        all_results.append(result)

    print_abba_results(all_results, labels=("JIT", "Interp"))
    print("=" * 72)

# ═══════════════════════════════════════════════════════════════════════════
# Subcommand: jit
# ═══════════════════════════════════════════════════════════════════════════

def cmd_jit(args):
    """Run JIT benchmarks -- absolute timing with auto-compilation."""
    print("=" * 72)
    print("Phoenix JIT Benchmarks -- Auto-compilation (threshold=1000)")
    print("=" * 72)
    print(f"Python:       {sys.version}")
    print(f"Platform:     {platform.machine()}")
    print(f"JIT active:   {check_jit_available()}")
    print(f"N iterations: 100,000")
    print(f"Warmup:       3 rounds")
    print(f"Measure:      5 rounds")
    print()

    n_iter = 100_000
    n_warmup = 3
    n_measure = 5

    benchmarks = JIT_BENCHMARKS
    if args.only:
        only_set = set(args.only.split(","))
        benchmarks = [(n, f) for n, f in JIT_BENCHMARKS if n in only_set]
        if not benchmarks:
            print(f"ERROR: no benchmarks match --only={args.only}")
            print(f"Available: {', '.join(n for n, _ in JIT_BENCHMARKS)}")
            sys.exit(1)

    print(f"{'Benchmark':24s} {'Mean ms':>10s} {'Min ms':>10s} {'Stdev ms':>10s}")
    print("-" * 60)

    for name, func in benchmarks:
        # Warmup (triggers auto-compilation for hot functions)
        for _ in range(n_warmup):
            func(n_iter)

        # Measure
        times = []
        for _ in range(n_measure):
            t0 = time.perf_counter()
            func(n_iter)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)  # ms

        mean_ms = statistics.mean(times)
        min_ms = min(times)
        stdev_ms = statistics.stdev(times) if len(times) > 1 else 0.0

        print(f"  {name:24s} {mean_ms:8.1f}   {min_ms:8.1f}   {stdev_ms:8.1f}")

    print()
    print("=" * 72)

# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Phoenix JIT benchmark suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--no-jit", action="store_true",
                        help="Disable JIT: skip _cinderx import, measure interpreter only")
    sub = parser.add_subparsers(dest="command")

    # abba
    p_abba = sub.add_parser("abba", help="ABBA micro-benchmarks")
    p_abba.add_argument("--blocks", type=int, default=ABBA_BLOCKS)
    p_abba.add_argument("--iters", type=int, default=BENCH_ITERS)
    p_abba.add_argument("--warmup", type=int, default=WARMUP_ITERS)

    # jit
    p_jit = sub.add_parser("jit", help="JIT benchmarks")
    p_jit.add_argument("--only", type=str, default=None,
                       help="Comma-separated benchmark names to run")

    # all
    sub.add_parser("all", help="Run all benchmarks")

    args = parser.parse_args()

    if args.no_jit:
        global _jit_disabled
        _jit_disabled = True
        try:
            import cinderjit
            cinderjit.compile_after_n_calls(999999999)
            # Disable the func watcher so no new trampolines are installed
            cinderjit.disable()
        except (ImportError, AttributeError):
            pass

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "abba":
        cmd_abba(args)
    elif args.command == "jit":
        cmd_jit(args)
    elif args.command == "all":
        # Run abba with defaults
        abba_args = argparse.Namespace(
            blocks=ABBA_BLOCKS, iters=BENCH_ITERS, warmup=WARMUP_ITERS
        )
        cmd_abba(abba_args)
        print()
        # Run jit with defaults
        jit_args = argparse.Namespace(only=None)
        cmd_jit(jit_args)

if __name__ == "__main__":
    main()
