#!/usr/bin/env python3
"""Consolidated CinderX benchmark suite for aarch64.

Replaces: benchmark_abba.py, benchmark_g1_next_abba.py,
          cinderx_jit_benchmark.sh, benchmark_specialisation.sh

METHODOLOGY: All comparisons use ABBA interleaving (A, B, B, A) to
control for thermal drift and co-located workload noise.

SUBCOMMANDS:
  abba    — Builtin micro-benchmarks: JIT function vs interpreter function
            (in-process, same Python, independent function objects)
  g1      — G1 fast path: JIT caller+JIT gen vs JIT caller+interp gen
            (in-process, isolates JITRT_InvokeIterNext contribution)
  jit     — Overall JIT vs vanilla Python across many workloads
            (subprocess isolation: venv CinderX vs system Python -I)
  spec    — Specialisation ON vs OFF (enable_specialized_opcodes effect)
            (subprocess isolation: same Python, different config)
  all     — Run all of the above

COMPILE MODES:
  --compile=auto   Use cinderjit.auto() and warmup to trigger compilation (default)
                   Matches production: adaptive interpreter specialises bytecodes
                   before JIT compiles, producing better code.
  --compile=force  Force-compile via cinderjit.force_compile()
                   Dev/debug: compiles before full adaptation. Results are NOT
                   comparable with auto mode.

FALSIFICATION:
  - Control: run without CinderX → delta should be ~0 for in-process tests
  - IQR must not span zero for a result to be marked significant
  - Raw block deltas printed for manual drift inspection

USAGE:
  # On devgpu (aarch64) with CinderX venv:
  PYTHONJIT=1 /path/to/venv/bin/python3 benchmark_cinderx.py abba
  PYTHONJIT=1 /path/to/venv/bin/python3 benchmark_cinderx.py all
  PYTHONJIT=1 /path/to/venv/bin/python3 benchmark_cinderx.py jit --reps=3
  PYTHONJIT=1 /path/to/venv/bin/python3 benchmark_cinderx.py spec --compile=auto

  # Worker mode (used internally for subprocess-isolated benchmarks):
  python3 benchmark_cinderx.py --worker=jit --condition=on
"""

# GUARD: cinderjit.auto() must precede all stdlib imports to prevent SIGSEGV.
#
# Without this, site.py loads _cinderx.so which activates the JIT with
# compile_after_n_calls=0 (compile everything immediately). Import-time
# functions get JIT-compiled, accumulate guard failures, and trigger deopt
# backoff → SIGSEGV in importlib._get_spec.
#
# cinderjit.auto() sets compile_after_n_calls=1000, preventing import-time
# JIT compilation. Workers use -S flag and call init_cinderjit() explicitly.
#
# GATED behind __name__ == "__main__": importing this module from external
# code must NOT activate JIT. Previous unconditional activation caused six
# measurement reversals (D-1773813690). JIT is always active for direct
# execution — no PYTHONJIT env var required.
if __name__ == "__main__":
    try:
        import _cinderx  # Phoenix: load JIT module
        import cinderjit; cinderjit.auto()
    except ImportError:
        pass

import argparse
import contextlib
import functools
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
# ═══════════════════════════════════════════════════════════════════════════
# Configuration defaults
# ═══════════════════════════════════════════════════════════════════════════

ABBA_BLOCKS = 15
BENCH_ITERS = 50_000
INNER_ITERS = 100
WARMUP_ITERS = 5_000
COMPILE_THRESHOLD = 999_999_999  # Prevent auto-compilation when not wanted
# ═══════════════════════════════════════════════════════════════════════════
# CinderX helpers
# ═══════════════════════════════════════════════════════════════════════════

def init_cinderjit(compile_mode="force"):
    """Initialise CinderX/Phoenix JIT. Returns cinderjit module or None."""
    try:
        import _cinderx  # Phoenix: built-in module, no cinderx.init() needed
        import cinderjit

        if compile_mode == "auto":
            cinderjit.auto()
        else:
            # Prevent auto-compilation; we will force-compile selectively
            try:
                cinderjit.compile_after_n_calls(COMPILE_THRESHOLD)
            except (AttributeError, TypeError):
                pass

        return cinderjit
    except (ImportError, AttributeError):
        return None

def _check_preconditions():
    """Verify benchmark environment is correctly configured.

    Fails loudly if preconditions are not met, preventing silently
    invalid results. Each check addresses a specific failure mode
    observed in practice.
    """
    # Phoenix: no -S flag needed. JIT auto-activates with threshold=1000
    # via phoenix_init.cpp, not site.py. The SIGSEGV issue from CinderX's
    # compile_after_n_calls=0 during site.py loading does not apply.

    # 2. Initialise CinderX/Phoenix JIT.
    #    Phoenix: _cinderx is a built-in module, no cinderx.init() needed.
    try:
        import _cinderx
    except ImportError as e:
        print(
            f"ERROR: cannot load _cinderx: {e}\n"
            "Ensure Phoenix JIT is built into the Python binary.",
            file=sys.stderr,
        )
        sys.exit(1)

    # 3. cinderjit must be importable (registered by cinderx.init()).
    try:
        import cinderjit
    except ImportError:
        print(
            "ERROR: cannot import cinderjit after cinderx.init().\n"
            "CinderX may not have been built with JIT support.",
            file=sys.stderr,
        )
        sys.exit(1)

    # 4. JIT must be functional after cinderjit.auto().
    if not hasattr(cinderjit, "auto"):
        print(
            "ERROR: cinderjit.auto() not available — build may be "
            "incomplete or corrupt.",
            file=sys.stderr,
        )
        sys.exit(1)

    # 5. Verify JIT is actually active (catches false-positive
    #    configurations where cinderjit imports but JIT never engages).
    try:
        if hasattr(cinderjit, "is_enabled") and not cinderjit.is_enabled():
            print(
                "WARNING: cinderjit loaded but JIT is not enabled.\n"
                "Results may not reflect JIT performance.",
                file=sys.stderr,
            )
    except Exception:
        pass  # is_enabled may not exist in all builds

    print("Precondition checks: PASS")
def verify_jit_preconditions(condition):
    """Fail loudly if JIT preconditions are not met.

    Prevents false positives: silently running without JIT produces
    interpreter results labelled as JIT results. These assertions
    convert institutional knowledge into executable invariants.
    """
    if condition != "on":
        return  # No JIT needed for baseline/off conditions

    # -S flag: site.py must not have loaded _cinderx.so early
    # (compile_after_n_calls=0 causes SIGSEGV in spec_from_loader)
    if "site" in sys.modules:
        print("FATAL: Running without -S flag.", file=sys.stderr)
        print("  site.py loads _cinderx.so with compile_after_n_calls=0,", file=sys.stderr)
        print("  causing import-time JIT compilation and crashes.", file=sys.stderr)
        print("  Run with: python3 -S benchmark_cinderx.py ...", file=sys.stderr)
        sys.exit(1)

    # Phoenix: _cinderx is built-in, no cinderx.init() needed.
    try:
        import _cinderx
    except ImportError:
        pass

    # cinderjit must be importable
    try:
        import cinderjit
    except ImportError:
        print("FATAL: cinderjit not importable.", file=sys.stderr)
        print("  Is _cinderx.so on sys.path? Check PYTHONPATH.", file=sys.stderr)
        print("  Expected: PYTHONPATH includes cinderx/PythonLib/", file=sys.stderr)
        sys.exit(1)

    # cinderjit.auto must exist (catches partial/broken builds)
    if not hasattr(cinderjit, "auto"):
        print("FATAL: cinderjit.auto() not available.", file=sys.stderr)
        print("  Build may be incomplete or incompatible.", file=sys.stderr)
        sys.exit(1)

def warmup_function(func, iters=None):
    """Warmup a function with small inputs."""
    iters = iters or WARMUP_ITERS
    for _ in range(iters):
        func(1)
def force_compile(func, cinderjit_mod):
    """Force JIT-compile a function. Returns True if compiled."""
    if not cinderjit_mod:
        return False
    try:
        cinderjit_mod.force_compile(func)
        return is_compiled(func, cinderjit_mod)
    except Exception:
        return False
def is_compiled(func, cinderjit_mod):
    """Check if function is JIT-compiled."""
    if not cinderjit_mod:
        return False
    try:
        return func in cinderjit_mod.get_compiled_functions()
    except Exception:
        try:
            return cinderjit_mod.is_jit_compiled(func)
        except Exception:
            return False
def enable_specialised_opcodes(cinderjit_mod):
    """Enable specialised opcodes if available."""
    if not cinderjit_mod:
        return False
    try:
        cinderjit_mod.enable_specialized_opcodes()
        return True
    except (AttributeError, Exception):
        return False
def print_config_header(args):
    """Print configuration and comparability warning."""
    print(f"Compile mode: {args.compile}")
    print()
    print("*** COMPARE LIKE WITH LIKE ***")
    print("Results from --compile=auto and --compile=force are NOT comparable.")
    print("  auto:  JIT compiles after adaptive interpreter specialises (production)")
    print("  force: JIT compiles immediately via force_compile (dev/debug)")
    print(f"Current: --compile={args.compile}  --reps={args.reps}  "
          f"--iters={args.iters}  --blocks={args.blocks}")
    print()
# ═══════════════════════════════════════════════════════════════════════════
# ABBA engine (shared by all subcommands)
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
    print("Raw per-block deltas (ms) — inspect for drift patterns:")
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
# --- G1 fast path targets ---

def _gen_jit():
    """Generator function — will be JIT-compiled."""
    while True:
        yield 1

def _gen_interp():
    """Generator function — stays interpreter-only."""
    while True:
        yield 1

def make_g1_caller(gen_func):
    """Create a G1 benchmark caller. Closure variable determines gen state."""
    def bench(n):
        g = gen_func()
        total = 0
        for _ in range(n):
            for _ in range(INNER_ITERS):
                total += next(g)
        return total
    return bench
# --- JIT vs vanilla benchmark functions ---
# These are used by the subprocess worker mode.

def _fib(n):
    if n < 2:
        return n
    return _fib(n - 1) + _fib(n - 2)

class _RichardsTask:
    __slots__ = ("id", "pri", "nxt", "state")
    def __init__(self, tid, pri):
        self.id = tid
        self.pri = pri
        self.nxt = None
        self.state = 0

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
    """Recursive fibonacci — tests function call overhead."""
    total = 0
    for _ in range(n_iter // 10):
        total += _fib(20)
    return total
# --- Richards OS task scheduler benchmark ---

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
            task.v1 = pkt  # work_in
        else:
            task.v2 = pkt  # dev_in
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

def bench_richards_full(n_iter):
    """Full Richards scheduler — method dispatch, attribute access, control flow."""
    total_q = 0
    total_h = 0
    for _ in range(n_iter // 100):
        q, h = _run_richards_once()
        total_q += q
        total_h += h
    return total_q + total_h

def bench_richards_slots(n_iter):
    """Richards scheduler with __slots__ — attribute access."""
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
    """N-queens solver — recursive backtracking, bit operations."""
    total = 0
    for _ in range(n_iter // 100):
        total += _nqueens_solve(8)
    return total

def bench_spectral_norm(n_iter):
    """Spectral norm — list comprehensions, floating point."""
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
    """Float arithmetic — math module, basic ops."""
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
    """List comprehension — creation and iteration."""
    total = 0
    for _ in range(n_iter // 100):
        xs = [i * i for i in range(100)]
        total += sum(xs)
    return total

def bench_dict_ops(n_iter):
    """Dictionary operations — creation, lookup, iteration."""
    total = 0
    for _ in range(n_iter // 100):
        d = {i: i * i for i in range(100)}
        for k, v in d.items():
            total += v
    return total

def bench_func_calls(n_iter):
    """Function call overhead — simple argument passing."""
    def add3(a, b, c):
        return a + b + c
    total = 0
    for i in range(n_iter):
        total += add3(i, i + 1, i + 2)
    return total

# --- Inlining frontier benchmark helpers ---
# Module-level callees so they can be force-compiled independently.

def _callee_with_import():
    """Callee containing EAGER_IMPORT_NAME (import os)."""
    import os
    return os.sep

def _callee_with_try():
    """Callee containing exception handler (try/except)."""
    try:
        return 42
    except Exception:
        return -1
def bench_import_callee(n_iter):
    """Hot loop calling callee with import -- EAGER_IMPORT_NAME inlining."""
    total = 0
    for _ in range(n_iter):
        total += len(_callee_with_import())
    return total

def bench_try_except_callee(n_iter):
    """Hot loop calling callee with try/except -- exception handler inlining."""
    total = 0
    for _ in range(n_iter):
        total += _callee_with_try()
    return total

def bench_store_subscr(n_iter):
    """List and dict subscript store -- STORE_SUBSCR specialisation."""
    xs = [0] * 100
    d = {}
    total = 0
    for i in range(n_iter):
        idx = i % 100
        xs[idx] = i
        d[idx] = i
        total += xs[idx] + d[idx]
    return total

def bench_int_arith(n_iter):
    """Pure integer arithmetic -- BINARY_OP_ADD_INT / BINARY_OP_MULTIPLY_INT."""
    total = 0
    a, b = 3, 7
    for i in range(n_iter):
        total += a * i + b
        a = (a + 1) % 127
        b = (b + 3) % 131
    return total

# --- Adversarial benchmarks (CinderX known weak spots) ---

class _CMgr:
    """Minimal context manager for benchmarking."""
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
    """Context manager dispatch — __enter__/__exit__ protocol."""
    mgr = _CMgr()
    total = 0
    for _ in range(n_iter):
        with mgr as m:
            total += m.val
    return total

def _kwargs_callee(**kwargs):
    return kwargs.get('a', 0) + kwargs.get('b', 0)

def bench_kwargs_dispatch(n_iter):
    """Keyword argument unpacking — **kwargs overhead."""
    total = 0
    for i in range(n_iter):
        total += _kwargs_callee(a=i, b=i+1)
    return total

def _positional_callee(a=0, b=0):
    return a + b

def bench_positional_dispatch(n_iter):
    """Keyword call-site to positional callee — ResolveKwargs target."""
    total = 0
    for i in range(n_iter):
        total += _positional_callee(a=i, b=i+1)
    return total

class _DunderObj:
    """Object implementing common dunder protocols."""
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
    """Dunder protocol dispatch — __getitem__, __len__, __contains__."""
    obj = _DunderObj()
    total = 0
    for i in range(n_iter):
        total += obj[i] + len(obj)
        if (i % 10) in obj:
            total += 1
    return total
# --- N-body simulation (pyperformance-derived) ---
# Float-heavy, tuple-heavy. Good test for EA/SR: intermediate
# coordinate differences are temporary tuples that never escape.

_NBODY_PI = 3.14159265358979323846
_NBODY_SOLAR_MASS = 4 * _NBODY_PI * _NBODY_PI
_NBODY_DAYS_PER_YEAR = 365.24

def _nbody_make_bodies():
    return [
        # Sun
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, _NBODY_SOLAR_MASS],
        # Jupiter
        [4.84143144246472090, -1.16032004402742839, -1.03622044471123109e-01,
         1.66007664274403694e-03 * _NBODY_DAYS_PER_YEAR,
         7.69901118419740425e-03 * _NBODY_DAYS_PER_YEAR,
         -6.90460016972063023e-05 * _NBODY_DAYS_PER_YEAR,
         9.54791938424326609e-04 * _NBODY_SOLAR_MASS],
        # Saturn
        [8.34336671824457987, 4.12479856412430479, -4.03523417114321381e-01,
         -2.76742510726862411e-03 * _NBODY_DAYS_PER_YEAR,
         4.99852801234917238e-03 * _NBODY_DAYS_PER_YEAR,
         2.30417297573763929e-05 * _NBODY_DAYS_PER_YEAR,
         2.85885980666130812e-04 * _NBODY_SOLAR_MASS],
        # Uranus
        [1.28943695621391310e+01, -1.51111514016986312e+01, -2.23307578892655734e-01,
         2.96460137564761618e-03 * _NBODY_DAYS_PER_YEAR,
         2.37847173959480950e-03 * _NBODY_DAYS_PER_YEAR,
         -2.96589568540237556e-05 * _NBODY_DAYS_PER_YEAR,
         4.36624404335156298e-05 * _NBODY_SOLAR_MASS],
        # Neptune
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
    # Offset momentum (standard nbody setup)
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

def bench_deep_class(n_iter):
    """Deep class hierarchy — heavy instance attribute access."""
    class Base:
        def __init__(self, name):
            self.name = name
            self.training = True
        def parameters(self):
            return [v for k, v in self.__dict__.items() if isinstance(v, float)]

    class Layer(Base):
        def __init__(self, name, feat):
            Base.__init__(self, name)
            self.in_features = feat
            self.weight = 0.01 * feat
            self.bias = 0.01
        def forward(self, x):
            return x * self.weight + self.bias

    class Network(Layer):
        def __init__(self, name, feat, n=3):
            Layer.__init__(self, name, feat)
            self.layers = [Layer(f"{name}_{i}", feat) for i in range(n)]
        def forward(self, x):
            for layer in self.layers:
                x = layer.forward(x)
            return x

    total = 0.0
    for _ in range(n_iter // 100):
        net = Network("bench", 32)
        result = net.forward(1.0)
        total += result
        _ = net.training
        _ = net.in_features
    return total
# --- Decorator chain benchmark (functools.wraps, closure capture) ---
# Exercises: wrapper function dispatch through stacked decorators,
# closure variable capture, @staticmethod/@classmethod resolution,
# dict-based memoisation cache. These patterns are common in PyTorch
# (@torch.no_grad, @torch.jit.export) and general Python frameworks.
# Uses n_iter // 10 so warmup is sufficient in both compile modes:
#   auto:  10K iterations per warmup call — JIT compilation triggers
#          early and compiled code stabilises before measurement.
#   force: 10 iterations per func(100) warmup call (100 total) —
#          enough for bytecode specialisation before force_compile.

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
    """Closure factory — mimics torch.no_grad() context."""
    def adder(x):
        return x + offset
    return adder

def bench_decorator_chain(n_iter):
    """Decorator chain — functools.wraps, stacked decorators, closures."""
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
# --- Deep class with super() chains (5-level MRO) ---
# Exercises: super().__init__() through a 5-level hierarchy (PyTorch
# nn.Module pattern), MRO method resolution, isinstance checks across
# the hierarchy, __repr__, attribute lookup through inheritance chain.
# Complements the existing bench_deep_class (3-level, no super()).
# Uses n_iter // 10 so warmup is sufficient in both compile modes:
#   auto:  10K iterations per warmup call — JIT compilation triggers
#          early and compiled code stabilises before measurement.
#   force: 10 iterations per func(100) warmup call (100 total) —
#          enough for bytecode specialisation before force_compile.
#          (At // 100 only 10 total iterations occur, which is
#          borderline for CPython's adaptive specialisation threshold.)

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
    """5-level class hierarchy with super() — MRO, isinstance, repr."""
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
# --- PyTorch-style context managers (nested, contextlib, state toggle) ---
# Exercises: __enter__/__exit__ dispatch through nested with-statements,
# @contextlib.contextmanager (generator-based CM), class-variable state
# toggle (mimics torch.no_grad/autocast global flag pattern), rapid
# per-layer enter/exit cycling. Complements the existing bench_context_manager
# (single simple CM). These patterns are the primary target of the Stage 1/2
# callee resolution work (simplifyVectorCallBoundMethod).
# Uses n_iter // 10 for adequate warmup in both auto and force modes.

class _NoGrad:
    """Mimics torch.no_grad() — sets/restores a global flag."""
    _enabled = True
    def __enter__(self):
        self._prev = _NoGrad._enabled
        _NoGrad._enabled = False
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        _NoGrad._enabled = self._prev
        return False

class _Autocast:
    """Mimics torch.autocast() — sets/restores precision mode."""
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
    """Mimics profiler scope — tracks entry/exit counts."""
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
    """Mimics model.train()/model.eval() as context manager."""
    prev = model_dict.get('training', True)
    model_dict['training'] = mode
    try:
        yield model_dict
    finally:
        model_dict['training'] = prev

def bench_pytorch_cm(n_iter):
    """PyTorch-style context managers — nested, contextlib, state toggle."""
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
JIT_BENCHMARKS = [
    ("fibonacci",       bench_fibonacci),
    ("richards_full",   bench_richards_full),
    ("richards_slots",  bench_richards_slots),
    ("nqueens",         bench_nqueens),
    ("spectral_norm",   bench_spectral_norm),
    ("float_arith",     bench_float_arith),
    ("gen_simple",      bench_gen_simple),
    ("gen_nested",      bench_gen_nested),
    ("list_comp",       bench_list_comp),
    ("dict_ops",        bench_dict_ops),
    ("func_calls",      bench_func_calls),
    ("import_callee",      bench_import_callee),
    ("try_except_callee",  bench_try_except_callee),
    ("store_subscr",       bench_store_subscr),
    ("int_arith",          bench_int_arith),
    ("context_manager",    bench_context_manager),
    ("kwargs_dispatch",    bench_kwargs_dispatch),
    ("positional_dispatch", bench_positional_dispatch),
    ("dunder_protocol",   bench_dunder_protocol),
    ("nn_module_forward", bench_deep_class),  # TEMP: skip known crash
    ("nbody",             bench_nbody),
    ("decorator_chain",   bench_decorator_chain),
    ("deep_class_super",  bench_deep_class_super),  # TEMP: skip known crash
    ("pytorch_cm",        bench_pytorch_cm),
]

# Functions to force-compile for JIT benchmarks
_JIT_COMPILABLE = [
    _fib, _nqueens_solve, _spectral_A, _spectral_mul_Av,
    _spectral_mul_Atv, _spectral_mul_AtAv,
    _callee_with_import, _callee_with_try,
    _kwargs_callee,
    _positional_callee,
    _run_richards_once, _idle_fn, _work_fn, _handler_fn, _device_fn,
    _nbody_advance, _nbody_energy, _nbody_make_bodies,
    _make_adder, _training_mode,
]
# --- Specialisation benchmark targets ---
# Selected to exercise LOAD_ATTR_INSTANCE_VALUE, STORE_ATTR, LOAD_ATTR_MODULE

class _Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def bench_attr_access(n_iter):
    """Pure attribute access — LOAD_ATTR_INSTANCE_VALUE / STORE_ATTR."""
    p = _Point(42, 99)
    total = 0
    for _ in range(n_iter):
        total += p.x + p.y
        p.x = total % 1000
    return total

def bench_module_attr(n_iter):
    """Module attribute access — LOAD_ATTR_MODULE."""
    total = 0.0
    for _ in range(n_iter):
        total += math.pi + math.e
    return total
SPEC_BENCHMARKS = [
    ("deep_class",     bench_deep_class),  # TEMP: skip known crash
    ("attr_access",    bench_attr_access),
    ("module_attr",    bench_module_attr),
    ("richards_slots", bench_richards_slots),
    ("func_calls",     bench_func_calls),
    ("list_comp",      bench_list_comp),
]
# ═══════════════════════════════════════════════════════════════════════════
# Subcommand: abba
# ═══════════════════════════════════════════════════════════════════════════

def cmd_abba(args):
    """Run ABBA micro-benchmarks: JIT function vs interpreter function."""
    print("=" * 72)
    print("ABBA Interleaved Benchmark — JIT vs Interpreter (in-process)")
    print("=" * 72)
    print(f"Python:       {sys.version}")
    print(f"ABBA_BLOCKS:  {args.blocks}")
    print(f"BENCH_ITERS:  {args.iters}")
    print(f"INNER_ITERS:  {INNER_ITERS}")
    print(f"WARMUP_ITERS: {args.warmup}")
    print_config_header(args)

    cinderjit = init_cinderjit(args.compile)

    if not cinderjit:
        print("MODE: CONTROL (no CinderX JIT available)")
        print("A and B use identical code paths. Delta should be ~0.")
        print()
        mode = "control"
    else:
        print("MODE: JIT vs Interpreter")
        print("A = JIT-compiled function, B = interpreter-only duplicate.")
        print()
        mode = "jit_vs_interp"

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

        if mode == "jit_vs_interp":
            warmup_function(func_a, args.warmup)
            if args.compile == "force":
                force_compile(func_a, cinderjit)
            warmup_function(func_b, args.warmup)
            # B: warmed up but NOT force-compiled
            a_jit = is_compiled(func_a, cinderjit)
            b_jit = is_compiled(func_b, cinderjit)
            label = f"{bench_name} (A:JIT={a_jit})"
        else:
            warmup_function(func_a, args.warmup)
            warmup_function(func_b, args.warmup)
            label = f"{bench_name} (control)"

        result = run_abba(func_a, func_b, args.blocks, args.iters)
        result["label"] = label
        all_results.append(result)

    print_abba_results(all_results, labels=("JIT", "Interp"))
    print("=" * 72)
# ═══════════════════════════════════════════════════════════════════════════
# Subcommand: g1
# ═══════════════════════════════════════════════════════════════════════════

def cmd_g1(args):
    """Run G1 fast path benchmark: JIT gen vs interp gen."""
    print("=" * 72)
    print("G1 Fast Path ABBA Benchmark — JITRT_InvokeIterNext")
    print("=" * 72)
    print(f"Python:       {sys.version}")
    print(f"ABBA_BLOCKS:  {args.blocks}")
    print_config_header(args)

    cinderjit = init_cinderjit(args.compile)

    caller_a = make_g1_caller(_gen_jit)
    caller_b = make_g1_caller(_gen_interp)

    # Verify shared code object
    same_code = caller_a.__code__ is caller_b.__code__
    print(f"Caller code objects identical: {same_code}")
    if not same_code:
        print("WARNING: callers have different code objects — bias possible.")
    print()

    # Correctness
    assert caller_a(1) == INNER_ITERS, "caller_a correctness failed"
    assert caller_b(1) == INNER_ITERS, "caller_b correctness failed"
    print("Correctness: PASS")
    print()

    if cinderjit:
        print("Compilation setup:")
        warmup_function(caller_a, args.warmup)
        warmup_function(caller_b, args.warmup)

        if args.compile == "force":
            a_ok = force_compile(caller_a, cinderjit)
            b_ok = force_compile(caller_b, cinderjit)
            g_jit_ok = force_compile(_gen_jit, cinderjit)
        else:
            a_ok = is_compiled(caller_a, cinderjit)
            b_ok = is_compiled(caller_b, cinderjit)
            g_jit_ok = is_compiled(_gen_jit, cinderjit)

        g_interp_compiled = is_compiled(_gen_interp, cinderjit)

        print(f"  caller_a (JIT gen):     {'JIT' if a_ok else 'INTERP'}")
        print(f"  caller_b (interp gen):  {'JIT' if b_ok else 'INTERP'}")
        print(f"  gen_jit:                {'JIT' if g_jit_ok else 'INTERP'}")
        print(f"  gen_interp:             {'JIT' if g_interp_compiled else 'INTERP'}")
        print()

        if a_ok and b_ok and g_jit_ok and not g_interp_compiled:
            print("Preconditions: ALL MET")
        else:
            print("Preconditions: PARTIAL — results may be unreliable")
        print()
    else:
        print("MODE: CONTROL (no CinderX)")
        warmup_function(caller_a, args.warmup)
        warmup_function(caller_b, args.warmup)
        print()

    print(f"Running {args.blocks} ABBA blocks...")
    print()

    result = run_abba(caller_a, caller_b, args.blocks, args.iters)

    print("=" * 72)
    print("RESULTS")
    print("=" * 72)
    print()
    print(f"  A (JIT gen):     {result['ns_a']:.1f} ns/call")
    print(f"  B (interp gen):  {result['ns_b']:.1f} ns/call")
    print(f"  Improvement:     {result['pct_improvement']:+.1f}%")
    print(f"  Significant:     {'YES' if result['significant'] else 'NO'}")
    print(
        f"  IQR:             [{result['iqr_lo']*1e3:+.3f}, "
        f"{result['iqr_hi']*1e3:+.3f}] ms"
    )
    print()

    if result["significant"] and result["pct_improvement"] > 0:
        print("VERDICT: G1 fast path provides a REAL speedup.")
    elif result["significant"] and result["pct_improvement"] < 0:
        print("VERDICT: G1 fast path is SLOWER (unexpected).")
    else:
        print("VERDICT: G1 fast path shows NO significant difference.")
    print()

    # Raw deltas
    deltas_ms = [f"{d*1e3:+.3f}" for d in result["deltas"]]
    print(f"Raw per-block deltas (ms): [{', '.join(deltas_ms)}]")
    print()
    print("=" * 72)
# ═══════════════════════════════════════════════════════════════════════════
# Subcommand: jit (subprocess isolation)
# ═══════════════════════════════════════════════════════════════════════════
def _resolve_cinderx_python():
    """Resolve the CinderX Python executable.

    Fallback chain:
      1. CINDERX_PYTHON envvar (explicit path)
      2. CINDERX_VENV/bin/python3 (venv directory)
      3. sys.executable (current interpreter)
    """
    import os
    explicit = os.environ.get("CINDERX_PYTHON")
    if explicit:
        return explicit
    venv = os.environ.get("CINDERX_VENV")
    if venv:
        return os.path.join(venv, "bin/python3")
    return sys.executable
def _check_cinderx_available(python_path):
    """Check that the given Python can import _cinderx.

    Returns True if _cinderx is importable, False otherwise.
    Prints a diagnostic message on failure.
    """
    import subprocess
    try:
        result = subprocess.run(
            [python_path, "-c", "import _cinderx"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            print(f"WARNING: {python_path} cannot import _cinderx")
            print(f"  stderr: {result.stderr.strip()}")
            return False
        return True
    except Exception as e:
        print(f"WARNING: Failed to check _cinderx availability: {e}")
        return False
def _run_worker(python_cmd, condition, compile_mode, only=None):
    """Run this script as a subprocess worker, return JSON results.

    Returns dict with benchmark results on success, or None on failure.
    Prints clear diagnostic on failure including signal number for crashes.
    """
    env = os.environ.copy()
    # Enable HIR inliner for JIT-ON workers (improves benchmark accuracy).
    # Strip from OFF/baseline workers for experimental hygiene.
    if condition == "on":
        env.setdefault("PYTHONJITENABLEHIRINLINER", "1")
    else:
        env.pop("PYTHONJITENABLEHIRINLINER", None)
    # -S flag: skip site.py to prevent _cinderx.so from loading at Python
    # startup with compile_after_n_calls=0. Without -S, JIT activates before
    # the worker script runs, causing deopt backoff crashes during import.
    # The worker script calls cinderjit.auto() explicitly after startup.
    cmd = python_cmd + ["-S",
        os.path.abspath(__file__),
        f"--worker=jit",
        f"--condition={condition}",
        f"--compile={compile_mode}",
    ] + ([f"--only={only}"] if only else [])
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300, env=env,
        )
        if result.returncode != 0:
            if result.returncode < 0:
                import signal
                sig = -result.returncode
                sig_name = signal.Signals(sig).name if sig in signal._value2member_map_ else f"signal {sig}"
                print(f"CRASHED ({sig_name})")
            else:
                print(f"ERROR (exit {result.returncode}): {result.stderr[:200]}")
            return None
        return json.loads(result.stdout)
    except subprocess.TimeoutExpired:
        print("TIMEOUT (300s)")
        return None
    except json.JSONDecodeError:
        print(f"BAD OUTPUT: {result.stdout[:200]}")
        return None
def _worker_jit(args):
    """Worker mode: run JIT benchmarks, output JSON."""
    condition = args.condition
    compile_mode = args.compile

    cinderjit_mod = None
    if condition == "on":
        verify_jit_preconditions(condition)
        cinderjit_mod = init_cinderjit(compile_mode)
        if cinderjit_mod is None:
            print("FATAL: JIT requested (condition=on) but cinderjit failed to initialise.", file=sys.stderr)
            print("  init_cinderjit() returned None. Check build and PYTHONPATH.", file=sys.stderr)
            sys.exit(1)
        if cinderjit_mod and compile_mode == "force":
            # Warmup for bytecode specialisation
            for _, func in JIT_BENCHMARKS:
                for _ in range(10):
                    try:
                        func(100)
                    except Exception:
                        pass
            # Force-compile
            for func in _JIT_COMPILABLE:
                try:
                    cinderjit_mod.force_compile(func)
                except Exception:
                    pass
            for _, func in JIT_BENCHMARKS:
                try:
                    cinderjit_mod.force_compile(func)
                except Exception:
                    pass
        if cinderjit_mod:
            enable_specialised_opcodes(cinderjit_mod)

    n_iter = 100_000
    n_warmup = 3
    n_measure = 5

    benchmarks = JIT_BENCHMARKS
    if args.only:
        only_set = set(args.only.split(","))
        benchmarks = [(n, f) for n, f in JIT_BENCHMARKS if n in only_set]
        if not benchmarks:
            print(f"FATAL: no benchmarks match --only={args.only}", file=sys.stderr)
            print(f"Available: {', '.join(n for n, _ in JIT_BENCHMARKS)}", file=sys.stderr)
            sys.exit(1)

    results = {
        "condition": condition,
        "benchmarks": {},
    }

    for name, func in benchmarks:
        # Warmup
        for _ in range(n_warmup):
            func(n_iter)

        # Measure
        times = []
        for _ in range(n_measure):
            t0 = time.perf_counter_ns()
            func(n_iter)
            t1 = time.perf_counter_ns()
            times.append((t1 - t0) / 1e6)  # ms

        results["benchmarks"][name] = {
            "times_ms": times,
            "mean_ms": sum(times) / len(times),
            "min_ms": min(times),
        }

    print(json.dumps(results))
def _worker_spec(args):
    """Worker mode: run spec benchmarks, output JSON."""
    condition = args.condition
    compile_mode = args.compile

    cinderjit_mod = init_cinderjit(compile_mode)
    if cinderjit_mod and condition == "on":
        enable_specialised_opcodes(cinderjit_mod)

    if cinderjit_mod and compile_mode == "force":
        for _, func in SPEC_BENCHMARKS:
            warmup_function(func, WARMUP_ITERS)
            try:
                cinderjit_mod.force_compile(func)
            except Exception:
                pass
    else:
        for _, func in SPEC_BENCHMARKS:
            warmup_function(func, WARMUP_ITERS)

    n_iter = 100_000
    n_measure = 5

    results = {
        "condition": condition,
        "benchmarks": {},
    }

    for name, func in SPEC_BENCHMARKS:
        times = []
        for _ in range(n_measure):
            t0 = time.perf_counter_ns()
            func(n_iter)
            t1 = time.perf_counter_ns()
            times.append((t1 - t0) / 1e6)

        results["benchmarks"][name] = {
            "times_ms": times,
            "mean_ms": sum(times) / len(times),
            "min_ms": min(times),
        }

    print(json.dumps(results))
def cmd_jit(args):
    """Run JIT vs vanilla Python benchmarks (per-benchmark subprocess isolation).

    Each benchmark runs in its own subprocess to ensure clean JIT state.
    This prevents cross-benchmark contamination: icache pressure, type
    watcher accumulation, deopt backoff state, and CodeExtra allocations
    from one benchmark affecting another. It also means a crash in one
    benchmark (e.g. IMPORT_NAME SIGSEGV) does not kill the others.
    """
    print("=" * 72)
    print("CinderX JIT vs Vanilla Python — Per-Benchmark Subprocess ABBA")
    print("=" * 72)
    print(f"Platform:     {platform.machine()}")
    print(f"Reps:         {args.reps} ABBA cycles per benchmark")
    print_config_header(args)

    # Determine Python commands
    venv_python = _resolve_cinderx_python()
    if not _check_cinderx_available(venv_python):
        print(f"FATAL: CinderX not available at {venv_python}")
        print("Set CINDERX_PYTHON or CINDERX_VENV to point to a CinderX venv.")
        return
    vanilla_python = os.environ.get(
        "VANILLA_PYTHON",
        "/usr/local/fbcode/platform010-aarch64/bin/python3.12",
    )

    venv_cmd = [venv_python]
    vanilla_cmd = [vanilla_python, "-I"]

    print(f"JIT ON:  {venv_python}")
    print(f"JIT OFF: {vanilla_python} -I")
    print()

    # Determine which benchmarks to run
    bench_names = [name for name, _ in JIT_BENCHMARKS]
    if args.only:
        only_set = set(args.only.split(","))
        bench_names = [n for n in bench_names if n in only_set]
        if not bench_names:
            print(f"FATAL: no benchmarks match --only={args.only}")
            print(f"Available: {', '.join(n for n, _ in JIT_BENCHMARKS)}")
            return

    print(f"Benchmarks:   {len(bench_names)} (each in isolated subprocess)")
    print()

    # Per-benchmark ABBA: each benchmark gets its own subprocess workers
    # so JIT state from one benchmark cannot affect another.
    all_on = {}   # bench_name -> [mean_ms, ...]
    all_off = {}  # bench_name -> [mean_ms, ...]

    for bi, bench_name in enumerate(bench_names, 1):
        print(f"[{bi}/{len(bench_names)}] {bench_name}")

        on_times = []
        off_times = []
        run_num = 0

        for rep in range(1, args.reps + 1):
            for condition in ["on", "off", "off", "on"]:
                run_num += 1
                cmd = venv_cmd if condition == "on" else vanilla_cmd
                label = "ON " if condition == "on" else "OFF"
                print(
                    f"  Run {run_num}/{args.reps * 4}: JIT_{label} "
                    f"(rep {rep}) ... ",
                    end="", flush=True,
                )

                result = _run_worker(cmd, condition, args.compile, only=bench_name)
                if result and bench_name in result.get("benchmarks", {}):
                    ms = result["benchmarks"][bench_name]["mean_ms"]
                    print(f"{ms:.2f}ms")
                    if condition == "on":
                        on_times.append(ms)
                    else:
                        off_times.append(ms)
                else:
                    print("FAILED")

                time.sleep(1)  # Let CPU settle

        all_on[bench_name] = on_times
        all_off[bench_name] = off_times
        print()

    # Comparison table
    print("=" * 75)
    print(f"CinderX JIT Performance Comparison ({platform.machine()})")
    print("=" * 75)
    print()

    print(
        f"{'Benchmark':<22} {'Vanilla':>10} {'CinderX':>10} "
        f"{'Speedup':>9} {'Δ%':>7}"
    )
    print("-" * 65)

    total_on = 0
    total_off = 0

    for bench_name in bench_names:
        on_times = all_on.get(bench_name, [])
        off_times = all_off.get(bench_name, [])

        if not on_times and not off_times:
            print(f"  {bench_name:<20} {'*** CRASHED BOTH CONDITIONS ***':>49}")
            continue
        if not on_times:
            print(f"  {bench_name:<20} {'*** JIT_ON CRASHED ***':>49}")
            continue
        if not off_times:
            print(f"  {bench_name:<20} {'*** JIT_OFF CRASHED ***':>49}")
            continue

        on_mean = sum(on_times) / len(on_times)
        off_mean = sum(off_times) / len(off_times)

        total_on += on_mean
        total_off += off_mean

        speedup = off_mean / on_mean if on_mean > 0 else 0
        delta_pct = ((off_mean - on_mean) / off_mean * 100) if off_mean > 0 else 0

        marker = "**" if speedup > 1.05 else ("!!" if speedup < 0.95 else "  ")
        print(
            f"  {bench_name:<20} {off_mean:>8.2f}ms {on_mean:>8.2f}ms "
            f"{speedup:>8.2f}x {delta_pct:>6.1f}% {marker}"
        )

    print("-" * 65)
    if total_on > 0:
        overall = total_off / total_on
        overall_pct = ((total_off - total_on) / total_off) * 100
        print(
            f"  {'TOTAL':<20} {total_off:>8.2f}ms {total_on:>8.2f}ms "
            f"{overall:>8.2f}x {overall_pct:>6.1f}%"
        )
    print("=" * 75)
    print()
    print("** = JIT >5% faster   !! = JIT >5% slower")
    print()
# ═══════════════════════════════════════════════════════════════════════════
# Subcommand: spec (subprocess isolation)
# ═══════════════════════════════════════════════════════════════════════════

def cmd_spec(args):
    """Run specialisation ON vs OFF benchmarks (subprocess isolated)."""
    print("=" * 72)
    print("CinderX Specialisation ON vs OFF — Subprocess ABBA")
    print("=" * 72)
    print(f"Platform:     {platform.machine()}")
    print(f"Reps:         {args.reps}")
    print_config_header(args)

    venv_python = _resolve_cinderx_python()
    if not _check_cinderx_available(venv_python):
        print(f"FATAL: CinderX not available at {venv_python}")
        print("Set CINDERX_PYTHON or CINDERX_VENV to point to a CinderX venv.")
        return
    python_cmd = [venv_python]
    print(f"Python: {venv_python}")
    print()

    # Falsification: verify enable_specialized_opcodes works
    print("--- Falsification check ---")
    check_result = _run_worker(python_cmd, "on", args.compile)
    if check_result:
        print("Spec ON worker: OK")
    else:
        print("FATAL: Spec ON worker failed. Cannot run spec benchmark.")
        return

    check_result = _run_worker(python_cmd, "off", args.compile)
    if check_result:
        print("Spec OFF worker: OK")
    else:
        print("FATAL: Spec OFF worker failed.")
        return
    print()

    # ABBA runs
    on_results = []
    off_results = []

    run_num = 0
    for rep in range(1, args.reps + 1):
        for condition in ["on", "off", "off", "on"]:
            run_num += 1
            print(
                f"  Run {run_num}/{args.reps * 4}: "
                f"SPEC_{'ON' if condition == 'on' else 'OFF'} "
                f"(rep {rep}) ... ",
                end="", flush=True,
            )

            result = _run_worker(python_cmd, condition, args.compile)
            if result:
                total_ms = sum(
                    b["mean_ms"] for b in result["benchmarks"].values()
                )
                print(f"{total_ms:.1f}ms total")
                if condition == "on":
                    on_results.append(result)
                else:
                    off_results.append(result)
            else:
                print("FAILED")

            time.sleep(2)

    if not on_results or not off_results:
        print("\nERROR: Not enough results.")
        return

    # Comparison
    print()
    print("=" * 75)
    print(f"Specialisation Effect ({platform.machine()})")
    print("=" * 75)
    print()

    all_benchmarks = sorted(
        set().union(*(r["benchmarks"].keys() for r in on_results + off_results))
    )

    print(
        f"{'Benchmark':<22} {'Spec OFF':>10} {'Spec ON':>10} "
        f"{'Ratio':>9} {'Δ%':>7}"
    )
    print("-" * 65)

    for b in all_benchmarks:
        on_means = [
            r["benchmarks"][b]["mean_ms"]
            for r in on_results if b in r["benchmarks"]
        ]
        off_means = [
            r["benchmarks"][b]["mean_ms"]
            for r in off_results if b in r["benchmarks"]
        ]

        on_mean = sum(on_means) / len(on_means) if on_means else 0
        off_mean = sum(off_means) / len(off_means) if off_means else 0

        if on_mean > 0:
            ratio = off_mean / on_mean
            delta_pct = ((off_mean - on_mean) / off_mean) * 100
        else:
            ratio = 0
            delta_pct = 0

        print(
            f"  {b:<20} {off_mean:>8.2f}ms {on_mean:>8.2f}ms "
            f"{ratio:>8.4f}x {delta_pct:>6.1f}%"
        )

    print("=" * 75)
    print()
    print("Ratio > 1.0 = spec ON is faster. Ratio ≈ 1.0 = neutral.")
    print()
# ═══════════════════════════════════════════════════════════════════════════
# Subcommand: all
# ═══════════════════════════════════════════════════════════════════════════

def cmd_all(args):
    """Run all benchmark suites."""
    print("Running all benchmark suites...")
    print()

    cmd_abba(args)
    print()
    cmd_g1(args)
    print()
    cmd_jit(args)
    print()
    cmd_spec(args)
# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description=f"Consolidated CinderX benchmark suite for {platform.machine()}.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  benchmark_cinderx.py abba              # Builtin micro-benchmarks
  benchmark_cinderx.py g1                # G1 fast path
  benchmark_cinderx.py jit --reps=3      # JIT vs vanilla (3 ABBA cycles)
  benchmark_cinderx.py jit --only=richards_full          # Single benchmark
  benchmark_cinderx.py jit --only=fibonacci,nqueens      # Multiple benchmarks
  benchmark_cinderx.py spec --compile=auto  # Spec ON vs OFF, auto-compile
  benchmark_cinderx.py all               # Run everything

Environment variables:
  CINDERX_PYTHON   Path to CinderX venv Python (default: $CINDERX_VENV/bin/python3,
                   then sys.executable)
  CINDERX_VENV     Path to CinderX venv directory
  VANILLA_PYTHON   Path to vanilla Python (default: system python3.12)
  PYTHONJITENABLEHIRINLINER  Set to 1 to enable HIR inliner (opt-in, passed
                   to JIT-ON workers only; stripped from baseline/OFF workers)
""",
    )

    # Worker mode (internal use for subprocess isolation)
    parser.add_argument(
        "--worker", choices=["jit", "spec"],
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--condition", choices=["on", "off"],
        help=argparse.SUPPRESS,
    )

    # Common options
    parser.add_argument(
        "--only", type=str, default=None,
        help="Run only the named benchmark (e.g. --only=richards_full). "
             "Comma-separated for multiple (e.g. --only=fibonacci,nqueens).",
    )
    parser.add_argument(
        "--compile", choices=["force", "auto"], default="auto",
        help="Compile mode: auto (warmup-driven, default) or force (force_compile)",
    )
    parser.add_argument(
        "--blocks", type=int, default=ABBA_BLOCKS,
        help=f"Number of ABBA blocks for in-process tests (default: {ABBA_BLOCKS})",
    )
    parser.add_argument(
        "--iters", type=int, default=BENCH_ITERS,
        help=f"Iterations per measurement (default: {BENCH_ITERS})",
    )
    parser.add_argument(
        "--warmup", type=int, default=WARMUP_ITERS,
        help=f"Warmup iterations (default: {WARMUP_ITERS})",
    )
    parser.add_argument(
        "--reps", type=int, default=2,
        help="ABBA repetitions for subprocess tests (default: 2)",
    )

    # Subcommand (positional, optional — worker mode has no subcommand)
    parser.add_argument(
        "subcommand", nargs="?",
        choices=["abba", "g1", "jit", "spec", "all"],
        help="Benchmark suite to run",
    )

    args = parser.parse_args()

    # Worker mode — output JSON, no banner
    if args.worker == "jit":
        _worker_jit(args)
        return
    if args.worker == "spec":
        _worker_spec(args)
        return

    # Precondition checks — fail loudly if environment is wrong
    _check_preconditions()

    # Normal mode — require subcommand
    if not args.subcommand:
        parser.print_help()
        sys.exit(1)

    # Architecture check
    if platform.machine() != "aarch64":
        print(
            f"WARNING: Running on {platform.machine()}, "
            f"designed for aarch64. Results may differ."
        )
        print()

    dispatch = {
        "abba": cmd_abba,
        "g1": cmd_g1,
        "jit": cmd_jit,
        "spec": cmd_spec,
        "all": cmd_all,
    }

    dispatch[args.subcommand](args)
if __name__ == "__main__":
    main()
