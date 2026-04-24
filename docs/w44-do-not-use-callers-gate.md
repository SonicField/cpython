# W44 — `_DO_NOT_USE_*` Helper Production-Path Caller Gate

**Status:** FILED authorized 2026-04-23T19:26:05Z per supervisor +
theologian 2026-04-23T19:25:28Z + librarian follow-through
2026-04-23T19:36:45Z.

**Owner:** generalist (impl post-W22-push) + theologian (spec).

**Estimated cost:** ~30min infra (template available:
`scripts/caller_grep.sh` from push 92 D-1776930053).

**Schedules:** AFTER W22 push 678e9905a8 lands; don't pull focus from
current gate cycle.

---

## 1. Problem statement

The C-port codebase contains explicit `_DO_NOT_USE_*` helper
functions (e.g., `hir_c_create_cond_branch`, `hir_c_create_branch`)
which bypass `Edge::set_to()` and leave target blocks' `in_edges_`
unset. The header at `Include/internal/cinderx/Jit/hir/hir_instr_c.h:1170-1172`
+ `:1184-1185` carries verbatim "DO NOT USE for production"
warnings.

**Header warnings have failed twice as a guardrail:**

1. **First incident** (06e2ecb652/7d0aff1e42 → fix 61c319ca49):
   pure-C terminator factories used directly → DominatorAnalysis
   null in_edge SEGFAULT.

2. **Second incident** (678e9905a8 W22 cluster, 2026-04-23): 3
   production-path callers in `Python/jit/hir/builder_emit_c.c`
   (lines 4207 emitSend, 4458 emitMatchClass, 4522
   emitMatchMappingSequence) used pure-C `hir_c_create_cond_branch`
   instead of `_cpp` variant. CFG corruption cascaded to:
   - W22 yield-from + await SIGSEGV (multi-iteration cluster)
   - test_phoenix_jit_controlflow regression
   - test_phoenix_jit_comparisons regression
   - test_phoenix_jit_autocompile (test_gen_chain) regression
   ~3 hours of cluster iteration.

Per pythia #102 2026-04-23: "the third occurrence is not 'if' but
'when'."

## 2. Resolution

Mechanical CI gate: `scripts/check_do_not_use_callers.sh` greps for
production-path callers of any `_DO_NOT_USE_*` symbol. Fails gate
if any caller is found in non-test/non-helper code paths.

### 2.1 Implementation pattern

Adapt from `scripts/caller_grep.sh` (push 92 D-1776930053, ~118
lines). Key differences:

```bash
# Symbols to check (extensible list)
DO_NOT_USE_SYMBOLS=(
    "hir_c_create_branch"          # _DO_NOT_USE_ pattern
    "hir_c_create_cond_branch"     # _DO_NOT_USE_ pattern
    # ... extensible
)

# Production paths (callers must use _cpp variants here)
PRODUCTION_PATHS=(
    "Python/jit/hir/builder_emit_c.c"
    "Python/jit/hir/refcount_pass_c.c"
    "Python/jit/hir/refcount_env_c.c"
    "Python/jit/hir/simplify_c.c"
    # ... extensible
)

# For each symbol, grep production paths for non-test calls
# Fail with clear message + offending file:line + recommended _cpp
# alternative
```

### 2.2 Whitelist mechanism

Some callers may legitimately bypass `Edge::set_to()` (test code,
diagnostic probes). Maintain at top of script:

```bash
WHITELIST=(
    "Python/jit/hir/hir_instr_c_verify.cpp"  # static_assert verifier
    "Lib/test/test_phoenix_*.py"             # test fixtures
)
```

### 2.3 Gate integration

Add to `scripts/gate_phoenix.sh` standard gate sequence (~5min cost
per push). Failure mode: clear error message identifying:
- Which symbol called
- Which file:line
- Recommended `_cpp` alternative
- Cross-link to `feedback_edge_management.md` for context

## 3. Acceptance criteria

1. Script identifies the W22 cluster's 3 sites pre-fix (run against
   pre-678e9905a8 tree)
2. Script PASSES on current HEAD (post-W22-fix)
3. Integration into `gate_phoenix.sh` adds <5min per push
4. False-positive rate: zero (whitelist mechanism handles legitimate
   cases)
5. Future regression: a new caller of `_DO_NOT_USE_*` in production
   path FAILS the gate before reaching commit

## 4. Cross-link

- Pythia identification: pythia #102 2026-04-23 (3) actionable
  recurrence-prevention
- Theologian filing: 2026-04-23T19:25:28Z
- Supervisor authorization: 2026-04-23T19:26:05Z
- Librarian follow-through: 2026-04-23T19:36:45Z
- Template: `scripts/caller_grep.sh` (push 92, D-1776930053)
- Memory: `feedback_edge_management.md` (Edge::set_to manages
  in_edges_)
- Empirical surface: 06e2ecb652/7d0aff1e42 → 61c319ca49 (first
  incident) + 678e9905a8 W22 cluster (second incident)
- Sibling workstreams: W33 (zero-bridge verifier), W42 (refcount-
  correctness verifier), W43 (pydebug+match pre-existing class)
