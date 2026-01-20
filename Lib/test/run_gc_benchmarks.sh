#!/bin/bash
#
# Parallel GC Benchmark Runner
#
# Runs consistent benchmark configurations for measuring parallel GC performance.
# Results are saved to results/ with timestamped filenames in markdown format.
#
# Usage:
#   ./run_gc_benchmarks.sh               # Standard benchmarks (default)
#   ./run_gc_benchmarks.sh quick         # Quick sanity check
#   ./run_gc_benchmarks.sh abandoned     # Test abandoned thread handling
#   ./run_gc_benchmarks.sh throughput    # Throughput-only benchmarks
#
# Standard benchmark parameters (chosen to be convincing to observers):
#   - Workers: 4, 8, 16
#   - Cleanup workers: 0 (serial), 4 (parallel)
#   - Heap types: all 8 (chain, tree, wide_tree, graph, layered, independent, ai_workload, web_server)
#   - Heap size: 500k objects
#   - Survivor ratio: 80%
#   - Creation threads: 4 (with persistent pool)
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CPYTHON_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${CPYTHON_ROOT}/python"
RESULTS_DIR="${SCRIPT_DIR}/results"
TIMESTAMP=$(date +%Y-%m-%d-%H-%M)

# Create results directory if it doesn't exist
mkdir -p "$RESULTS_DIR"

# Verify python exists
if [ ! -x "$PYTHON" ]; then
    echo "ERROR: Python executable not found at: $PYTHON"
    echo "Please build Python first with: ./configure --disable-gil --with-lto && make -j"
    exit 1
fi

# Detect build type
BUILD_TYPE=$($PYTHON -c "import sys; print('ftp' if hasattr(sys, '_is_gil_enabled') and not sys._is_gil_enabled() else 'gil')" 2>/dev/null || echo "unknown")

echo "========================================"
echo "Parallel GC Benchmark Runner"
echo "========================================"
echo "Build type: $BUILD_TYPE"
echo "Python: $PYTHON"
echo "Results directory: $RESULTS_DIR"
echo "Timestamp: $TIMESTAMP"
echo ""

run_gc_benchmark() {
    local name="$1"
    local args="$2"
    local output_file="${RESULTS_DIR}/${TIMESTAMP}-${name}.md"

    echo "Running: $name"
    echo "  Args: $args"
    echo "  Output: $output_file"
    echo ""

    $PYTHON "${SCRIPT_DIR}/gc_benchmark.py" $args -o "$output_file" 2>&1 | tee "${RESULTS_DIR}/${TIMESTAMP}-${name}-console.txt"

    echo ""
    echo "  Saved to: $output_file"
    echo ""
}

run_throughput_benchmark() {
    local name="$1"
    local args="$2"
    local output_file="${RESULTS_DIR}/${TIMESTAMP}-${name}.txt"

    echo "Running: $name"
    echo "  Args: $args"
    echo "  Output: $output_file"
    echo ""

    $PYTHON "${SCRIPT_DIR}/gc_throughput_benchmark.py" $args 2>&1 | tee "$output_file"

    echo ""
    echo "  Saved to: $output_file"
    echo ""
}

case "${1:-standard}" in
    quick)
        echo "Mode: QUICK (sanity check)"
        echo ""
        echo "Purpose: Fast verification that benchmarks run correctly."
        echo ""

        # Minimal GC benchmark
        run_gc_benchmark "gc-quick" \
            "--workers 4,8 --cleanup-workers 0,4 --heap-type chain,wide_tree --heap-size 50k --survivor-ratio 0.8 --creation-threads 4 --keep-threads --warmup 1 --iterations 2"

        # Minimal throughput
        run_throughput_benchmark "throughput-quick" \
            "--heap-size 100k --duration 10 --parallel 8 --cleanup-workers 4 --threads 4 --keep-threads"
        ;;

    abandoned)
        echo "Mode: ABANDONED (regression test for thread abandonment)"
        echo ""
        echo "Purpose: Verify parallel GC correctly handles abandoned thread heaps."
        echo "         This is a 'have I broken anything' test, not a performance benchmark."
        echo ""

        # Test abandoned thread handling - don't use --keep-threads
        run_gc_benchmark "gc-abandoned" \
            "--workers 4,8 --cleanup-workers 0,4 --heap-type chain,wide_tree --heap-size 100k --survivor-ratio 0.5 --creation-threads 4 --warmup 2 --iterations 3"

        # Throughput with abandoned threads
        run_throughput_benchmark "throughput-abandoned" \
            "--heap-size 200k --duration 20 --parallel 8 --cleanup-workers 4 --threads 4"
        ;;

    throughput)
        echo "Mode: THROUGHPUT (throughput benchmarks only)"
        echo ""
        echo "Purpose: Measure real-world throughput impact of parallel cleanup."
        echo ""

        # Compare serial vs parallel with different cleanup_workers
        echo "--- Serial baseline ---"
        run_throughput_benchmark "throughput-serial" \
            "--heap-size 500k --duration 60 --threads 4 --keep-threads"

        echo "--- Parallel 8 workers, cleanup_workers=0 (serial cleanup) ---"
        run_throughput_benchmark "throughput-p8-cw0" \
            "--heap-size 500k --duration 60 --parallel 8 --cleanup-workers 0 --threads 4 --keep-threads"

        echo "--- Parallel 8 workers, cleanup_workers=4 (parallel cleanup) ---"
        run_throughput_benchmark "throughput-p8-cw4" \
            "--heap-size 500k --duration 60 --parallel 8 --cleanup-workers 4 --threads 4 --keep-threads"

        echo "--- Parallel 16 workers, cleanup_workers=4 ---"
        run_throughput_benchmark "throughput-p16-cw4" \
            "--heap-size 500k --duration 60 --parallel 16 --cleanup-workers 4 --threads 4 --keep-threads"

        # AI workload (more realistic memory patterns)
        echo "--- AI workload: Parallel 8 workers, cleanup_workers=4 ---"
        run_throughput_benchmark "throughput-ai-p8-cw4" \
            "--heap-size 500k --duration 60 --parallel 8 --cleanup-workers 4 --threads 4 --keep-threads --heap-type ai"
        ;;

    standard|*)
        echo "Mode: STANDARD (default benchmarks)"
        echo ""
        echo "Purpose: Comprehensive benchmarks for measuring parallel GC performance."
        echo "         Parameters chosen to be convincing to observers."
        echo ""
        echo "Parameters:"
        echo "  - Workers: 4, 8, 16"
        echo "  - Cleanup workers: 0 (serial), 4 (parallel)"
        echo "  - Heap types: all 7"
        echo "  - Heap size: 500k objects"
        echo "  - Survivor ratio: 80%"
        echo "  - Creation threads: 4 (persistent pool)"
        echo ""

        # Main GC benchmark - all heap types, key worker counts
        run_gc_benchmark "gc-standard" \
            "--workers 4,8,16 --cleanup-workers 0,4 --heap-type chain,tree,wide_tree,graph,layered,independent,ai_workload,web_server --heap-size 500k --survivor-ratio 0.8 --creation-threads 4 --keep-threads --warmup 3 --iterations 5"

        # Throughput benchmarks
        echo ""
        echo "=== Throughput Benchmarks ==="
        echo ""

        # Serial baseline
        run_throughput_benchmark "throughput-serial" \
            "--heap-size 500k --duration 60 --threads 4 --keep-threads"

        # Parallel with serial cleanup (baseline for cleanup_workers comparison)
        run_throughput_benchmark "throughput-p8-cw0" \
            "--heap-size 500k --duration 60 --parallel 8 --cleanup-workers 0 --threads 4 --keep-threads"

        # Parallel with parallel cleanup
        run_throughput_benchmark "throughput-p8-cw4" \
            "--heap-size 500k --duration 60 --parallel 8 --cleanup-workers 4 --threads 4 --keep-threads"

        # Higher worker count
        run_throughput_benchmark "throughput-p16-cw4" \
            "--heap-size 500k --duration 60 --parallel 16 --cleanup-workers 4 --threads 4 --keep-threads"

        # AI workload
        run_throughput_benchmark "throughput-ai-p8-cw4" \
            "--heap-size 500k --duration 60 --parallel 8 --cleanup-workers 4 --threads 4 --keep-threads --heap-type ai"

        # Web server workload (isolated requests, ideal for tid-based cleanup)
        run_throughput_benchmark "throughput-web-p8-cw4" \
            "--heap-size 500k --duration 60 --parallel 8 --cleanup-workers 4 --threads 4 --keep-threads --heap-type web_server"
        ;;
esac

echo "========================================"
echo "Benchmarks complete"
echo "========================================"
echo ""
echo "Results saved to: $RESULTS_DIR"
echo ""
echo "Generated files:"
ls -1 "${RESULTS_DIR}"/${TIMESTAMP}-* 2>/dev/null | while read f; do
    echo "  $(basename "$f")"
done
