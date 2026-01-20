#!/bin/bash
# BRC/DECREF Optimisation Benchmark Script
# Usage: ./run_brc_benchmarks.sh [verify|realistic|pyperformance|all] [config_name]
#
# Examples:
#   ./run_brc_benchmarks.sh verify           # Check benchmark setup
#   ./run_brc_benchmarks.sh realistic A      # Run realistic benchmark for config A
#   ./run_brc_benchmarks.sh pyperformance A  # Run pyperformance for config A
#   ./run_brc_benchmarks.sh all A            # Run all benchmarks for config A

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CPYTHON_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
PARALLEL_GC_DIR="$(dirname "$CPYTHON_DIR")"
PYPERFORMANCE_DIR="$PARALLEL_GC_DIR/pyperformance"
PYPERF_DIR="$PARALLEL_GC_DIR/pyperf"
PYTHON="$CPYTHON_DIR/python"
RESULTS_DIR="$SCRIPT_DIR/brc_results"

# Benchmarks to run
PYPERFORMANCE_BENCHMARKS=(
    deltablue
    richards
    nbody
    float
    regex_compile
    json_loads
    deepcopy
    async_tree
    comprehensions
    generators
    gc_collect
    gc_traversal
)

# Number of runs for realistic benchmark
REALISTIC_RUNS=5
REALISTIC_DURATION=30

usage() {
    echo "Usage: $0 [verify|realistic|pyperformance|all] [config_name]"
    echo ""
    echo "Commands:"
    echo "  verify           Check that all benchmarks are available"
    echo "  realistic X      Run realistic benchmark (5 runs at 4 and 8 threads)"
    echo "  pyperformance X  Run pyperformance benchmarks"
    echo "  all X            Run all benchmarks"
    echo ""
    echo "Config names: A, B, C, D (see plan for flag settings)"
    exit 1
}

verify_setup() {
    echo "=== Verifying Benchmark Setup ==="
    echo ""

    # Check Python exists
    if [ -x "$PYTHON" ]; then
        echo "OK: Python executable found at $PYTHON"
        echo "    Version: $($PYTHON --version 2>&1)"
    else
        echo "ERROR: Python not found at $PYTHON"
        exit 1
    fi

    # Check pyperf
    if [ -d "$PYPERF_DIR" ]; then
        echo "OK: pyperf found at $PYPERF_DIR"
    else
        echo "ERROR: pyperf not found at $PYPERF_DIR"
        exit 1
    fi

    # Check pyperformance
    if [ -d "$PYPERFORMANCE_DIR" ]; then
        echo "OK: pyperformance found at $PYPERFORMANCE_DIR"
    else
        echo "ERROR: pyperformance not found at $PYPERFORMANCE_DIR"
        exit 1
    fi

    # Check each benchmark
    echo ""
    echo "Checking pyperformance benchmarks:"
    local all_ok=true
    for bm in "${PYPERFORMANCE_BENCHMARKS[@]}"; do
        bm_dir="$PYPERFORMANCE_DIR/benchmarks/bm_$bm"
        if [ -d "$bm_dir" ]; then
            echo "  OK: bm_$bm"
        else
            echo "  MISSING: bm_$bm"
            all_ok=false
        fi
    done

    # Check realistic benchmark
    echo ""
    if [ -f "$SCRIPT_DIR/gc_realistic_benchmark.py" ]; then
        echo "OK: gc_realistic_benchmark.py found"
    else
        echo "ERROR: gc_realistic_benchmark.py not found"
        exit 1
    fi

    # Create results directory
    mkdir -p "$RESULTS_DIR"
    echo ""
    echo "OK: Results directory: $RESULTS_DIR"

    if $all_ok; then
        echo ""
        echo "=== All checks passed ==="
    else
        echo ""
        echo "=== Some benchmarks missing - check pyperformance clone ==="
        exit 1
    fi
}

run_realistic() {
    local config="$1"
    if [ -z "$config" ]; then
        echo "ERROR: Config name required (A, B, C, or D)"
        exit 1
    fi

    echo "=== Running Realistic Benchmark for Config $config ==="
    echo "Duration: ${REALISTIC_DURATION}s per run"
    echo "Runs: $REALISTIC_RUNS"
    echo ""

    mkdir -p "$RESULTS_DIR"

    for threads in 4 8; do
        outfile="$RESULTS_DIR/realistic_config_${config}_${threads}threads.txt"
        echo "Running $threads threads..."
        echo "Output: $outfile"

        echo "=== Config $config, $threads threads ===" > "$outfile"
        echo "Started: $(date)" >> "$outfile"
        echo "" >> "$outfile"

        for run in $(seq 1 $REALISTIC_RUNS); do
            echo "  Run $run of $REALISTIC_RUNS..."
            echo "=== Run $run ===" >> "$outfile"
            "$PYTHON" "$SCRIPT_DIR/gc_realistic_benchmark.py" \
                --threads "$threads" \
                --duration "$REALISTIC_DURATION" \
                >> "$outfile" 2>&1
            echo "" >> "$outfile"
        done

        echo "Completed: $(date)" >> "$outfile"
        echo "  Done."
    done

    echo ""
    echo "=== Realistic benchmark complete ==="
    echo "Results in: $RESULTS_DIR/realistic_config_${config}_*.txt"
}

run_pyperformance() {
    local config="$1"
    if [ -z "$config" ]; then
        echo "ERROR: Config name required (A, B, C, or D)"
        exit 1
    fi

    echo "=== Running Pyperformance for Config $config ==="
    echo ""

    mkdir -p "$RESULTS_DIR"

    cd "$PARALLEL_GC_DIR"

    for bm in "${PYPERFORMANCE_BENCHMARKS[@]}"; do
        bm_dir="$PYPERFORMANCE_DIR/benchmarks/bm_$bm"
        if [ ! -d "$bm_dir" ]; then
            echo "SKIP: bm_$bm (not found)"
            continue
        fi

        outfile="$RESULTS_DIR/pyperformance_config_${config}_${bm}.json"
        echo "Running bm_$bm..."

        PYTHONPATH="$PYPERF_DIR" "$PYTHON" \
            "$bm_dir/run_benchmark.py" \
            -o "$outfile" \
            2>&1 | grep -E "^(bm_|$bm:|Mean|WARNING)" || true

        echo "  -> $outfile"
    done

    echo ""
    echo "=== Pyperformance complete ==="
    echo "Results in: $RESULTS_DIR/pyperformance_config_${config}_*.json"
}

# Main
case "${1:-}" in
    verify)
        verify_setup
        ;;
    realistic)
        run_realistic "$2"
        ;;
    pyperformance)
        run_pyperformance "$2"
        ;;
    all)
        run_realistic "$2"
        run_pyperformance "$2"
        ;;
    *)
        usage
        ;;
esac
