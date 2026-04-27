# W33 — Mechanical ZERO-Bridge Verifier Script

**Status:** FILED (deferred per supervisor [chat L2935]). ~30min infra
work; symmetric to `scripts/caller_grep.sh` (push 92) +
auto-numstat (push 85). File post-Batch-2-G when current burndown
stabilizes.

**Owner:** TBD (testkeeper most likely; theologian draft)

---

## 1. Problem statement

The 'ZERO new bridges' claim in commit messages and chat reports is
asserted but never mechanically verified. D-1776903189 carved out
post-Step-B falsification (W26 §4 mutation) for ZERO-bridge batches,
trusting the claim at face value.

Per pythia #87 (2026-04-23): "(a) 'ZERO new bridges' is asserted in
commit messages but never verified by mechanical check (e.g. diff of
[bridge symbols] across base..HEAD) — unlike the auto-numstat script
(b02b18e4cb) that mechanized count-discipline."

The risk-mass differs by batch type (per theologian L2934):
- **Header-inline** (Batch 2-E/2-F/2-G): safe-by-construction, C++
  preserved byte-identical.
- **Relocation** (Batch 2-B/2-C): body moves .cpp → .c (ABI shift,
  #include surface change). Highest risk for silent bridge addition.
- **Pure deletion** (Batch 2-D): no regression class.
- **Sole-path C-port** (W27a/b/c/d emit-methods): bridge inventory
  closed by W27 spec; mechanical check would have caught accidentals.

A mechanical verifier closes the gap for relocation + sole-path
classes, and provides cheap insurance for header-inline.

## 2. Resolution — `scripts/zero_bridge_check.sh`

### Inputs
- `--base=<sha>` base commit (default: `git merge-base HEAD origin/main`)
- `--head=<sha>` head commit (default: `HEAD`)

### Outputs
- Exit 0 + `ZERO BRIDGES VERIFIED` if no new bridge symbols added
- Exit 1 + diff of new bridge symbols if any added (lists file:line)
- stdout: enumeration of bridge symbol classes inspected

### Bridge symbol classes inspected
Grep `git diff <base>..<head> --` for added (`^+`) lines matching:
- `^\+(extern\s+\"C\"\s+)?[a-zA-Z_:&* <>0-9]+\s+phx_[a-z_]+\(` —
  new C-side bridge functions (PhxArray, PhxLabel, etc.)
- `^\+(extern\s+\"C\"\s+)?[a-zA-Z_:&* <>0-9]+\s+hir_[a-z_]+_c\(` —
  new HIR C-API bridge functions
- `^\+\s*HirCFG\s+|^\+\s*HirInstr\s+|^\+\s*HirBasicBlock\s+` —
  new typed-bridge struct usages (W25b boundary)
- Filter to `Python/jit/hir/`, `Python/jit/lir/`,
  `Include/internal/cinderx/Jit/` paths (configurable)

### False-positive handling
- Whitelist: existing wrapper bridges that move between files but don't
  change signature. Maintained at top of script.
- Comments-only changes are excluded by `git diff -G` regex.

## 3. Integration

Add to gate cycle as advisory check (NOT blocking initially):
- `scripts/build_phoenix.sh` runs after build, prints `ZERO BRIDGES
  VERIFIED` or warns.
- Generalist Step A scope draft cites the script output as part of
  ZERO-bridge claim, alongside diff stats and Cat-A/B classification.

After 5 batches of clean output, promote to mandatory in
gate_phoenix.sh per the W26 §4b pattern (necessary-not-sufficient
gates layered with W21 golden + manual smoke).

## 4. When to schedule

Per supervisor [chat L2935]: file post-Batch-2-G; ~30min infra work.
Useful for any Batch 2-H+ relocation work (e.g., builder.cpp 6719L
write-path eventually) and for the W34 __static__ retroactive test
pass corroborating the claim.

Cross-link:
- Pythia hypothesis: #87 (2026-04-23) — re-issue from L2933
- Theologian batch-class breakdown: [chat L2934]
- Supervisor concur: [chat L2935]
- Sibling infra: scripts/caller_grep.sh (push 92 4cfd894a27),
  auto-numstat (push 85 b02b18e4cb)
