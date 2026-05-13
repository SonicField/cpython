# lib_preflight.sh — shared preflight invariant checks (I1+I4) for
# gate_phoenix.sh + test_preflight_i*_negative.sh per theologian
# 20:30:40Z + gatekeeper 20:31:18Z pre-spec under supervisor 20:18:04Z
# RESEQUENCE (D-1778705500). Sourced (not executed).
#
# Invariants:
#   I1 BINARY IDENTITY: $PYTHON md5+mtime captured once post-build, asserted
#     pre-stage. Closes B3 PYTHON→python_gate alias silent-pass class
#     (catalog: D-1778698980).
#   I4 BUILD-PRODUCT FRESHNESS: $PYTHON binary mtime > newest tracked source
#     mtime (excluding generated headers + build dirs). Run once at gate
#     start.
#
# I3-COMPLIANT: explicit if-then under set -euo pipefail; exit 1 on fail;
# no silent || true.

EXPECTED_BINARY_MD5=""
EXPECTED_BINARY_MTIME=""

# _preflight_log: route output through tee -a $RESULTS_FILE if set;
# otherwise plain echo. Lib-friendly (no $RESULTS_FILE dependency).
_preflight_log() {
    if [ -n "${RESULTS_FILE:-}" ]; then
        echo "$@" | tee -a "$RESULTS_FILE"
    else
        echo "$@"
    fi
}

# preflight_capture_binary: capture md5+mtime baseline for $PYTHON.
# Called once after Step 1 build completes (gate_phoenix.sh line ~86).
preflight_capture_binary() {
    if [ ! -f "$PYTHON" ]; then
        _preflight_log "I1 BINARY IDENTITY capture FAIL — \$PYTHON ($PYTHON) does not exist"
        exit 1
    fi
    EXPECTED_BINARY_MD5=$(md5sum "$PYTHON" | awk '{print $1}')
    EXPECTED_BINARY_MTIME=$(stat -c %Y "$PYTHON")
    _preflight_log "I1 BINARY IDENTITY captured: md5=$EXPECTED_BINARY_MD5 mtime=$EXPECTED_BINARY_MTIME path=$PYTHON"
}

# preflight_recapture_binary: documented re-capture API for legitimate
# mid-run rebuild stages (e.g., (3-gate) bad-path substep — gatekeeper
# 20:31:18Z (b)). Caller MUST cite a reason so re-capture is auditable in
# the gate transcript.
preflight_recapture_binary() {
    local reason="${1:-unspecified}"
    if [ ! -f "$PYTHON" ]; then
        _preflight_log "I1 BINARY IDENTITY re-capture FAIL (reason=$reason) — \$PYTHON missing"
        exit 1
    fi
    local prev_md5="$EXPECTED_BINARY_MD5"
    local prev_mtime="$EXPECTED_BINARY_MTIME"
    EXPECTED_BINARY_MD5=$(md5sum "$PYTHON" | awk '{print $1}')
    EXPECTED_BINARY_MTIME=$(stat -c %Y "$PYTHON")
    _preflight_log "I1 BINARY IDENTITY re-captured (reason=$reason): prev_md5=$prev_md5 new_md5=$EXPECTED_BINARY_MD5 prev_mtime=$prev_mtime new_mtime=$EXPECTED_BINARY_MTIME"
}

# preflight_check_binary <stage-label>: assert $PYTHON md5+mtime match
# baseline captured by preflight_capture_binary. exit 1 on drift.
preflight_check_binary() {
    local stage="${1:-unspecified}"
    if [ -z "$EXPECTED_BINARY_MD5" ]; then
        _preflight_log "I1 BINARY IDENTITY [stage=$stage] FAIL — preflight_capture_binary never called (gate-script bug)"
        exit 1
    fi
    if [ ! -f "$PYTHON" ]; then
        _preflight_log "I1 BINARY IDENTITY [stage=$stage] FAIL — gate binary $PYTHON disappeared mid-run"
        exit 1
    fi
    local actual_md5
    local actual_mtime
    actual_md5=$(md5sum "$PYTHON" | awk '{print $1}')
    actual_mtime=$(stat -c %Y "$PYTHON")
    if [ "$actual_md5" != "$EXPECTED_BINARY_MD5" ] || [ "$actual_mtime" != "$EXPECTED_BINARY_MTIME" ]; then
        _preflight_log "I1 BINARY IDENTITY drift [stage=$stage] FAIL — gate binary changed mid-run"
        _preflight_log "  expected_md5=$EXPECTED_BINARY_MD5 actual_md5=$actual_md5"
        _preflight_log "  expected_mtime=$EXPECTED_BINARY_MTIME actual_mtime=$actual_mtime"
        _preflight_log "  path=$PYTHON"
        exit 1
    fi
}

# preflight_check_freshness: assert $PYTHON binary mtime > newest tracked
# source mtime under Python/, Modules/, Include/, Objects/ and Parser/.
# Excludes generated headers per gatekeeper 20:31:18Z (a):
#   - Python/jit_build/      cmake JIT build artifacts
#   - Python/cinderx/build/  cmake cinderx build artifacts
#   - */build/generated/     build-generated source
#   - pyconfig.h             autoconf-generated
#   - Modules/clinic/        Argument Clinic-generated
#   - Modules/_decimal/libmpdec  vendored 3rd-party (rarely edited)
preflight_check_freshness() {
    if [ ! -f "$PYTHON" ]; then
        _preflight_log "I4 BUILD-PRODUCT FRESHNESS FAIL — \$PYTHON ($PYTHON) does not exist"
        exit 1
    fi
    local roots
    roots="$CPYTHON_ROOT/Python $CPYTHON_ROOT/Modules $CPYTHON_ROOT/Include $CPYTHON_ROOT/Objects $CPYTHON_ROOT/Parser"
    # find -prune skips matching dirs; the post-prune branch matches .c/.h/.cpp
    # files newer than $PYTHON. head -5 caps diagnostic noise.
    local newer_files
    newer_files=$(find $roots \
        \( -path '*/jit_build/*' -o -path '*/cinderx/build/*' -o -path '*/build/generated/*' \
           -o -path '*/Modules/clinic/*' -o -path '*/_decimal/libmpdec/*' \
           -o -name 'pyconfig.h' \) -prune \
        -o -type f \( -name '*.c' -o -name '*.h' -o -name '*.cpp' \) -newer "$PYTHON" -print 2>/dev/null | head -5)
    if [ -n "$newer_files" ]; then
        local newest
        newest=$(echo "$newer_files" | head -1)
        local source_mtime
        local binary_mtime
        source_mtime=$(stat -c %Y "$newest")
        binary_mtime=$(stat -c %Y "$PYTHON")
        local delta=$((source_mtime - binary_mtime))
        _preflight_log "I4 BUILD-PRODUCT FRESHNESS FAIL — source(s) newer than gate binary (stale build)"
        _preflight_log "  newest_source=$newest"
        _preflight_log "  source_mtime=$source_mtime binary_mtime=$binary_mtime delta_seconds=$delta"
        _preflight_log "  path=$PYTHON"
        _preflight_log "  additional newer files (up to 4 more):"
        echo "$newer_files" | tail -n +2 | while read -r f; do
            _preflight_log "    $f"
        done
        exit 1
    fi
    _preflight_log "I4 BUILD-PRODUCT FRESHNESS OK — gate binary $PYTHON is newer than all tracked sources"
}

# preflight_check_env_flag <expected_flag_string> <cmake_log_path>: assert
# every -D token in expected_flag_string appears in cmake-invocation log.
# Closes B1 EXTRA_CMAKE_FLAGS:189 hard-reset class — proves caller-supplied
# flags actually reached cmake (not silently dropped by env-passthrough).
#
# expected_flag_string: typically the script-received EXTRA_CMAKE_FLAGS
# (snapshot taken BEFORE any internal mutation like ASAN augmentation).
# Per-flag iteration (gatekeeper 21:22:00Z (a)): split on -D, grep each
# -DKEY=VAL token in log. Tolerates whitespace/ordering variation that
# canonical-string match would miss.
#
# Caller responsibility: only invoke when expected_flag_string is non-empty
# (gatekeeper 21:22:00Z (c) edge case — empty vs forgotten-set distinction
# is at caller layer; helper is no-op-on-empty defensive).
preflight_check_env_flag() {
    local expected="${1:-}"
    local log_path="${2:-}"
    if [ -z "$expected" ]; then
        return 0
    fi
    if [ -z "$log_path" ] || [ ! -f "$log_path" ]; then
        _preflight_log "I2 ENV-FLAG READ-BACK FAIL — cmake log path missing or unreadable (log_path=$log_path)"
        exit 1
    fi
    # Per-flag iteration: split expected into -D tokens. Each token is one
    # -DKEY=VAL or -DKEY (without =). Strip leading whitespace then split
    # on ' -' boundary (preserving leading -D on each token via re-prepend).
    local missing=()
    local token
    # Use awk to split on ' -' boundary; produces one token per line
    while IFS= read -r token; do
        token="${token# }"  # trim leading space
        [ -z "$token" ] && continue
        # Re-prepend '-' since awk split consumed the leading dash
        local probe="-$token"
        # Skip if not a -D flag (e.g., empty after trim)
        [[ "$probe" =~ ^-D ]] || continue
        if ! grep -q -- "$probe" "$log_path"; then
            missing+=("$probe")
        fi
    done < <(echo "$expected" | awk '{n=split($0, a, " -"); for (i=1;i<=n;i++) print a[i]}')
    if [ "${#missing[@]}" -ne 0 ]; then
        _preflight_log "I2 ENV-FLAG READ-BACK FAIL — expected flag(s) absent from cmake invocation log"
        _preflight_log "  expected_flag_string='$expected'"
        _preflight_log "  log_path=$log_path"
        _preflight_log "  missing_tokens: ${missing[*]}"
        _preflight_log "  cmake-invocation log tail (last 20 lines):"
        tail -20 "$log_path" | while IFS= read -r line; do
            _preflight_log "    $line"
        done
        exit 1
    fi
    _preflight_log "I2 ENV-FLAG READ-BACK OK — all -D tokens from EXTRA_CMAKE_FLAGS=' $expected' verified in $log_path"
}

# i3_pipe_grep_count <var-name> <input-string> <grep-args...>: pipe-grep -c
# with rc-safe handling per I3 SUBSHELL ERROR-VISIBILITY (theologian 21:44:42Z
# + gatekeeper 21:45:16Z + supervisor 21:48:29Z). Sets named variable to count
# (0 on no-match; grep -c rc=1 is OK = no-match); WARNs to stderr on grep
# internal error (rc>1) — surfaces silent grep-broken-by-corrupt-input class
# that bare 'grep -c X || true' would mask as 'no matches'.
#
# WARN-class severity (per gatekeeper 21:45:16Z (b)): grep failure is a
# best-effort parse, downstream may use default; CALLER decides if the count
# matters and applies FAIL action if needed.
i3_pipe_grep_count() {
    local var="$1"
    local input="$2"
    shift 2
    local count rc
    # PIPESTATUS doesn't survive command-substitution boundary, so use a
    # subshell with pipefail to make the substitution's rc reflect the
    # pipeline's overall rc (last non-zero command). 0 = matches found,
    # 1 = no match (legitimate count=0), >1 = grep internal error (WARN).
    set +e
    count=$(set -o pipefail; printf '%s' "$input" | grep -c "$@")
    rc=$?
    set -e
    if [ "$rc" -gt 1 ]; then
        echo "WARNING: i3_pipe_grep_count: grep -c rc=$rc (internal error, NOT no-match); args=$*; pattern may be malformed or input encoding broken" >&2
    fi
    eval "$var=${count:-0}"
}

# preflight_check_env_readiness: I5γ γ-2 ENV-READINESS PREFLIGHT
# (theologian 22:14:12Z + gatekeeper 22:15:07Z + supervisor 22:17:44Z).
#
# Verifies critical build tools are available in $PATH at script-start;
# fails loudly with diagnostic + install hint when missing. Closes the
# wlei-cmake brittleness operational risk surfaced at I1+I4 ARM64 ratify
# (generalist 21:12:10Z + supervisor 21:03:41Z) — script bails fast with
# actionable message rather than running half a configure step before
# cmake-not-found error at line 226.
#
# Tool list per gatekeeper 22:15:07Z (c) audit recommendation: minimal
# toolchain explicitly checked (cmake/clang/clang++/python3); ar/ld/m4/
# autoconf rely on default PATH-resolution from /usr/bin (system-default,
# rarely missing). Augment list at impl-time if those classes go missing.
#
# I3-COMPLIANT: explicit if-then under set -euo pipefail; exit 1 on fail.
preflight_check_env_readiness() {
    local missing=()
    local tool
    for tool in cmake clang clang++ python3; do
        if ! command -v "$tool" >/dev/null 2>&1; then
            missing+=("$tool")
        fi
    done
    if [ "${#missing[@]}" -ne 0 ]; then
        echo "FAIL: γ-2 ENV-READINESS — required build tool(s) not in PATH"
        echo "  missing: ${missing[*]}"
        echo "  current PATH: $PATH"
        echo "  install hint: dnf install cmake clang OR augment PATH (e.g.,"
        echo "    export PATH=/data/users/wlei/miniconda3-arm/bin:\$PATH"
        echo "    on devgpu004 ARM64 where wlei has miniconda3-arm cmake)"
        exit 1
    fi
}
