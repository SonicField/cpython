# BRC/DECREF Optimisation Benchmark Plan

## Objective

Measure the performance impact of two optimisations:
1. **BRC Queue Sharding** (`Py_BRC_SHARDED`)
2. **Fast Decref via Atomic ADD** (`Py_BRC_FAST_DECREF`)

## Constraints

- **Serial GC only**: Use `--gc-serial` to isolate BRC/DECREF changes from parallel GC
- **5 runs per configuration**: For statistical significance
- **Full documentation**: Every step, every result, every rebuild

---

## Phase 1: Code Changes

### Step 1.1: Add `Py_BRC_FAST_DECREF` flag

**File:** `Include/internal/pycore_brc.h`

Add after `Py_BRC_SHARDED` definition:
```c
// Enable fast decref path using atomic ADD when object is already queued/merged.
// When disabled, always uses CAS loop.
#define Py_BRC_FAST_DECREF 1
```

**File:** `Objects/object.c`

Wrap fast path in `_Py_DecRefSharedIsDead()` with conditional:
```c
#if Py_BRC_FAST_DECREF
    // Fast path: check if already queued/merged...
    Py_ssize_t shared = o->ob_ref_shared;
    if ((shared & _Py_REF_SHARED_FLAG_MASK) >= _Py_REF_QUEUED) {
        // ... existing fast path code ...
    }
#endif
    // Slow path: CAS loop
```

### Step 1.2: Verify changes compile

```bash
make clean && make -j
./python -c "print('OK')"
```

**Document:** Commit hash, any compilation warnings

---

## Build Configuration (CRITICAL)

All builds MUST use identical optimisation flags:

```bash
./configure --disable-gil --with-lto CFLAGS=-O3
```

- `--disable-gil`: Free-threading build
- `--with-lto`: Link Time Optimisation
- `CFLAGS=-O3`: Aggressive compiler optimisation

**Before starting:** Verify current configuration:
```bash
grep -E "configure.*--" config.status
```

Expected output should include `--disable-gil --with-lto CFLAGS=-O3`

---

## Phase 2: Configurations

Four configurations to test:

| Config | `Py_BRC_SHARDED` | `Py_BRC_FAST_DECREF` | Description |
|--------|------------------|----------------------|-------------|
| A | 0 | 0 | Baseline (no optimisations) |
| B | 1 | 0 | Sharding only |
| C | 0 | 1 | Fast decref only |
| D | 1 | 1 | Both (current) |

For each configuration:
1. Edit `Include/internal/pycore_brc.h` to set both flags
2. Run `make clean && make -j` (LTO rebuild required for each config change)
3. Verify: `./python -c "import sys; print(sys.version)"`
4. **Document:** Copy of the flag values before building

---

## Phase 3: Realistic Benchmark (gc_realistic_benchmark.py)

### Parameters

- Duration: 30 seconds
- Threads: 4 and 8
- Runs: 5 per configuration per thread count
- GC mode: Serial (omit `--parallel` flag)

### Procedure

For each configuration (A, B, C, D):
```bash
# Rebuild with LTO (takes several minutes)
make clean && make -j

# Verify build
./python -c "import sys; print(sys.version)"

# 4 threads, 5 runs (serial GC = omit --parallel flag)
for i in 1 2 3 4 5; do
    echo "=== Run $i ===" >> results_config_X_4threads.txt
    ./python Lib/test/gc_realistic_benchmark.py \
        --threads 4 --duration 30 \
        >> results_config_X_4threads.txt 2>&1
done

# 8 threads, 5 runs
for i in 1 2 3 4 5; do
    echo "=== Run $i ===" >> results_config_X_8threads.txt
    ./python Lib/test/gc_realistic_benchmark.py \
        --threads 8 --duration 30 \
        >> results_config_X_8threads.txt 2>&1
done
```

**Note:** Omitting `--parallel` uses serial GC, which isolates the BRC/DECREF changes.

### Metrics to Extract

From each run:
- Throughput (workloads/sec)
- GC overhead (%)
- Max pause (ms)

### Statistical Analysis

For each metric:
- Mean
- Standard deviation
- 95% confidence interval
- Comparison: % change from baseline (Config A)

---

## Phase 4: Pyperformance Suite

### Setup

Pyperformance and pyperf are cloned to `/data/users/alexturner/parallel_gc/`:
- `pyperformance/` - benchmark suite
- `pyperf/` - benchmark runner

Run individual benchmarks with:
```bash
cd /data/users/alexturner/parallel_gc
PYTHONPATH=./pyperf ./cpython/python pyperformance/benchmarks/bm_XXX/run_benchmark.py \
    -o results_config_X_bm_XXX.json
```

### Benchmarks to Run

Core benchmarks (representative of real workloads):

| Benchmark | Category | Notes |
|-----------|----------|-------|
| deltablue | Constraint solving | HIGH_CYCLES (416 cycles/iter) |
| richards | Task scheduling | MINIMAL_CYCLES |
| nbody | Numerical | MINIMAL_CYCLES |
| float | Numerical | NO_CYCLES |
| regex_compile | Text processing | NO_CYCLES |
| json_loads | Serialisation | NO_CYCLES |
| deepcopy | Object copying | HIGH_CYCLES (242 cycles/iter) |
| async_tree | Async | HIGH_CYCLES (242 cycles/iter) |
| comprehensions | Core Python | NO_CYCLES |
| generators | Core Python | NO_CYCLES |
| gc_collect | GC specific | Direct GC measurement |
| gc_traversal | GC specific | Direct GC measurement |

### Procedure

For each configuration (A, B, C, D):
```bash
# Rebuild with LTO
make clean && make -j

cd /data/users/alexturner/parallel_gc

# Run each benchmark (5 iterations is default)
for bm in deltablue richards nbody float regex_compile json_loads deepcopy \
          async_tree comprehensions generators gc_collect gc_traversal; do
    echo "Running $bm..."
    PYTHONPATH=./pyperf ./cpython/python \
        pyperformance/benchmarks/bm_$bm/run_benchmark.py \
        -o results_config_X_$bm.json
done
```

### Comparison

Use pyperf to compare results:
```bash
PYTHONPATH=./pyperf ./cpython/python -m pyperf compare_to \
    results_config_A_deltablue.json \
    results_config_D_deltablue.json
```

---

## Phase 5: Results Documentation

### Required Outputs

1. **Code diff:** Show exact changes made for conditional compilation
2. **Build logs:** For each configuration (at least first build)
3. **Raw results:** All output files from benchmarks
4. **Summary table:**

```
| Config | Benchmark | Threads | Mean Throughput | Std Dev | vs Baseline |
|--------|-----------|---------|-----------------|---------|-------------|
| A      | realistic | 4       | X               | Y       | -           |
| B      | realistic | 4       | X               | Y       | +Z%         |
...
```

5. **Pyperformance summary:** Geometric mean across all benchmarks
6. **Conclusion:** Which optimisations help, by how much, recommendation

---

## Phase 6: Cleanup

After benchmarking:
1. Restore `Py_BRC_SHARDED = 1` and `Py_BRC_FAST_DECREF = 1`
2. Rebuild
3. Verify tests pass

---

## Progress Log

| Step | Status | Notes | Timestamp |
|------|--------|-------|-----------|
| 1.1 Add flag | | | |
| 1.2 Verify compile | | | |
| 2.A Build config A | | | |
| 3.A.4 Realistic 4 threads | | | |
| 3.A.8 Realistic 8 threads | | | |
| 4.A Pyperformance | | | |
| 2.B Build config B | | | |
| ... | | | |

---

## Estimated Effort

- 4 configurations × (rebuild + 2 thread counts × 5 runs × 30s) = ~40 minutes realistic benchmark
- 4 configurations × pyperformance (~30 min each) = ~2 hours pyperformance
- Analysis and documentation: ~1 hour

Total: ~3-4 hours of execution time
