# Synthetic-Falsifier-at-Gate (Defensive-Infra Discipline)

**Authored:** theologian, 2026-05-12 (per supervisor 22:23:05Z disposition of pythia #366 + 23:38:14Z directive for post-c13 codification)

**Scope:** Every commit landing defensive-infrastructure code MUST be preceded by a synthetic-falsifier test that demonstrates the infrastructure actually catches the failure class it claims to defend against. This document codifies the discipline as a structural rule, supplementing the chat-resident decision log.

## Motivation

On 2026-05-12, the pre-commit-build-check.sh hook (commit a2e8808ade) shipped through normal review (gatekeeper APPROVE on bash -n + diff inspection at 18:07:08Z). Three substrate commits (c8 aa06bc70bf, c9 f938c22a59, c10 c6da14168c) then PASSED the hook in 92-104s each before pythia #364 prompted a synthetic-falsifier test (supervisor 21:08:23Z), which immediately surfaced a bash if-then-fi bug (the hook returned exit 0 regardless of build status). The hook never actually blocked any commit; c8/c9/c10 were carried by Tier 1 gate, not by the hook (commit 82592f64e6 fix landed 21:27:25Z).

The failure mode: defensive infrastructure shipped, looked correct under normal review, fired green on every gate, and appeared to be working — while contributing zero actual enforcement. The hook surface (output strings, exit code paths, log messages) looked right; the runtime behavior did not.

A synthetic-falsifier test (intentional regression → verify guard catches → revert) at the original a2e8808ade landing would have caught the bash if-then-fi bug immediately, at zero cost relative to the eventual rework + 3-commit hook-theatrical window.

## Rule

When a commit modifies any of the following defensive-infrastructure paths, the commit author (or the gatekeeper, before APPROVE) MUST run a synthetic-falsifier test BEFORE the commit lands. Failure to do so is a process violation (medic-flaggable, peer to `feedback_compile_before_commit` violations).

### Covered paths

- `scripts/pre-commit-*` (and any related pre-commit hook scripts)
- `.git/hooks/*` (any installed git hook)
- `scripts/gate_phoenix.sh` (the master Tier 1 gate script)
- `scripts/build_phoenix.sh` (build-system enforcement)
- Any new or modified script in `scripts/` that BLOCKs commits, BLOCKs pushes, or rejects gate runs
- Any change to `nbs-sidecar`, `nbs-launch-agent`, fixup, or restart-class infrastructure that asserts "agent is healthy" / "session is alive"
- Any new or modified static_assert verifier file (`*_verify.cpp`) — synthetic-falsifier here = intentional struct field offset mismatch in source, verify the assert FAILS, revert

### Test pattern

The synthetic-falsifier test is a 3-step pre-commit ritual:

1. **Inject a regression** that should fire the guard. The injected regression must be precisely scoped to exercise the SPECIFIC failure class the infrastructure claims to defend (not a generic "make build fail" — match the class).
2. **Run the guard**, verify it fires (BLOCKs commit / fails build / surfaces a warning / etc.).
3. **Revert the regression** before the actual infrastructure commit lands. The actual commit message MUST cite the falsifier test outcome ("Synthetic falsifier verified: [class] regression caught at [stage]").

If the guard does NOT fire, the infrastructure is broken (or mis-targeted) and the commit cannot land until either the guard is fixed or the scope is corrected.

### Examples

**Pre-commit build hook:** inject an `extern int undefined_function();` call into a header included by a static-inline body. Commit. Verify hook BLOCKS (linker undefined-reference). Revert. Commit the actual hook change.

**Tier 1 gate script:** inject an intentional Phoenix test failure (`assert False` in a force_compile fixture). Run gate. Verify gate reports FAIL. Revert. Commit the gate change.

**static_assert verifier file:** inject a wrong bit-pattern in one of the C-side constants that the verifier checks. Build. Verify the static_assert FAILS at compile time. Revert. Commit the verifier change.

**Sidecar/restart enforcement:** kill an agent process; verify the fixup mechanism reports it as DEAD and respawns; verify the restart actually produces an alive sidecar. Then commit the enforcement change.

## Self-falsifying disclaimer

This rule is itself defensive infrastructure (process-level rather than code-level). Per its own discipline, it should be falsifier-tested. The natural test:

1. Author a commit that violates this rule (lands defensive-infra without synthetic-falsifier evidence).
2. Verify it gets caught at gatekeeper APPROVE OR medic-flag.
3. Revert.

The 2026-05-12 incident itself constitutes the unintentional first falsifier test of this rule's NEED — the bash bug shipped, the substrate-cadence absorbed it via Tier 1 redundancy, the rule was authored after the fact. Future violations should be caught via gatekeeper checklist + medic agent-side enforcement.

## Enforcement

- **Author-side:** author runs synthetic-falsifier test BEFORE the commit lands; cites outcome in commit body.
- **Gatekeeper:** APPROVE checklist for defensive-infra-class commits MUST include "synthetic falsifier verified" line citing the author's evidence (commit body or chat post).
- **Medic:** [MEDIC-WARNING] on any defensive-infra commit landing without synthetic-falsifier evidence in commit body or chat.
- **Supervisor:** disposes flags from medic; may dispatch synthetic-falsifier as a follow-up commit if the infrastructure has already landed without one.

## Out-of-scope

- Routine code commits (substrate work, refactors, ports). The rule applies ONLY to defensive infrastructure.
- Documentation-only commits (no enforcement behavior to falsify).
- Test-only commits in `Lib/test/` (those ARE the falsifier; they don't need their own synthetic falsifier).
- Bug fixes to existing defensive infrastructure where the bug itself surfaced via natural failure (no synthetic regression needed; the natural failure is the falsifier).

## Non-coverage caveat

Synthetic-falsifier tests verify the guard catches the SPECIFIC failure class injected. They do NOT prove the guard catches ALL failure classes in its scope. A guard that passes one synthetic test may still have other coverage gaps. This rule is necessary but not sufficient for defensive-infra correctness; it raises the floor without raising the ceiling.
