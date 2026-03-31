# Phoenix JIT Benchmark How-To

How to build, run, interpret, and save ABBA benchmark results.

## Prerequisites

1. **CPython + JIT built** via `scripts/build_phoenix.sh` (x86_64 with LTO) or manual build (ARM64 without LTO to match cinderx_dev).
2. **Vanilla baseline binary** for comparison:
   - x86_64: build vanilla CPython with `scripts/build_vanilla.sh`
   - ARM64 (devgpu004): use cinderx_dev's CPython at `~/local/cinderx_dev/python-install/bin/python3.12`
3. **benchmark_phoenix_full.py** in `Tools/` (committed in repo).

## Building

### x86_64

```bash
cd cpython
scripts/build_phoenix.sh    # Builds JIT library + CPython with LTO
scripts/build_vanilla.sh    # Builds vanilla CPython with LTO (no JIT)
```

### ARM64 (devgpu004)

```bash
cd ~/local/phoenix-cpython-git

# Build JIT library (no LTO, matches cinderx_dev)
mkdir -p Python/jit_build/build && cd Python/jit_build/build
cmake .. -DPHOENIX_ASM=ON \
    -DCMAKE_CXX_FLAGS="-DPHOENIX_ASM" \
    -DCMAKE_C_FLAGS="-DPHOENIX_ASM" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++
cmake --build . -- -j$(nproc)
mkdir -p _deps/asmjit-build && llvm-ar rcs _deps/asmjit-build/libasmjit.a

# Build CPython WITHOUT LTO (matches cinderx_dev flags)
cd ~/local/phoenix-cpython-git
./configure --without-pydebug
rm -f python && make -j$(nproc)
```

## Running ABBA Benchmarks

### x86_64

```bash
cd cpython
VANILLA_PYTHON=cpython-vanilla/python \
    ./python Tools/benchmark_phoenix_full.py jit --reps=3
```

### ARM64 (devgpu004)

```bash
cd ~/local/phoenix-cpython-git

# Create a wrapper for vanilla Python (needs LD_LIBRARY_PATH)
cat > /tmp/run_vanilla.sh << 'EOF'
#!/bin/bash
LD_LIBRARY_PATH=~/local/cinderx_dev/python-install/lib \
    ~/local/cinderx_dev/python-install/bin/python3.12 "$@"
EOF
chmod +x /tmp/run_vanilla.sh

VANILLA_PYTHON=/tmp/run_vanilla.sh \
    ./python Tools/benchmark_phoenix_full.py jit --reps=3
```

### Options

- `--reps=N`: number of ABBA repetitions per benchmark (default 3, more = less noise)
- `--only=bench1,bench2`: run specific benchmarks only
- `--timeout=N`: per-benchmark timeout in seconds

## Interpreting Results

The benchmark harness uses **ABBA methodology**: alternates JIT-ON and JIT-OFF runs to cancel out thermal/load drift. Each benchmark runs in a **subprocess** for isolation.

Key metrics:
- **Total speedup**: sum(vanilla_times) / sum(jit_times) — weighted by benchmark duration
- **Geometric mean**: product(speedups)^(1/N) — equal weight per benchmark
- **Per-benchmark speedup**: > 1.0 means JIT is faster, < 1.0 means JIT is slower

### Acceptance criteria (ARM64 vs cinderx_dev)

- Geometric mean >= 1.0x
- No individual benchmark > 30% slower (speedup >= 0.7x)
- Build flags must match comparison target

### Known regressions

These benchmarks are consistently slower with the JIT (same pattern as CinderX):
- `gen_simple`: generator overhead from JIT compilation
- `try_except_callee`: exception handler deopt overhead
- `gen_nested`: nested generator overhead

## Saving Results

Save results to `docs/benchmarks/` with this naming convention:

```
<arch>_abba_YYYY-MM-DD.md
```

Each file must include:
- Commit hash and branch
- Build flags (LTO, PGO, optimization level)
- Platform (architecture, machine name)
- Methodology (ABBA reps, subprocess isolation, auto-compile threshold)
- Full per-benchmark table with speedup ratios
- Tolerance check result (PASS/FAIL with criteria)
- Known issues (crashes, regressions)
- Raw output file path

See `x86_64_abba_2026-03-31.md` and `arm64_abba_2026-03-31.md` for examples.
