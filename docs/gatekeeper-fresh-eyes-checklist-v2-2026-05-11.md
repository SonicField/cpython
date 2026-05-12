# Gatekeeper FRESH-EYES Checklist v2 — Phase 5.A3 + 5.B and forward
**Version:** v2 (supersedes D-1778183896 v1 gatekeeper-fresh-eyes-checklist-2026-05-07.md)
**Source:** post-Stage-B batch-codification per D-1778492262 (supervisor 09:37:10Z 2026-05-11) + D-1778500218 v2 draft (generalist 11:23Z 2026-05-11) + 9 amendments + A9 empirical update (testkeeper 16:16:37Z + theologian 16:17:14Z + supervisor 16:25:27Z + 17:02:10Z 2026-05-11)
**Trigger:** at Duo-unblock OR pre-push for any 5.B/5.C ratify cycle, BEFORE executor authorization

---

## Procedural-Discipline Preamble (v2 A6 — POST-RATIFY-ITEM-5)

When supervisor reviews gatekeeper-authored procedure (or vice-versa) and considers an amendment:
  (a) Default to QUERY: "am I missing X?" / "what's the rationale for Y?"
  (b) PIVOT (substitute alternate procedure) only after QUERY surfaces explicit gap or rationale-disconnect
  (c) Cross-post window risk: when authoring takes >2min, re-read chat between authoring start and post-time per feedback_chat_recheck_before_claim

Per D-1778490709 (supervisor 10:10:56Z 2026-05-11 self-correction).

## Scope Tags (v2 A4 — POST-RATIFY-ITEM-3)

Whenever a step references a binary or symbol, tag the scope:
  - `[BARE-SUBSTRATE]` refers to current ARM64 HEAD without dry-run cherry-picks
  - `[DRY-RUN-X-Y]` refers to BARE-SUBSTRATE + cherry-pick(X,Y) for validation
  - `[POST-PUSH-TIP]` refers to remote tip after Step 6 push lands

Symbol-presence checks must specify scope: e.g. "falsifier_assert_pivot_matches_parsed PRESENT in [DRY-RUN-d89aebb4ef] release binary" not "PRESENT in release binary". Per D-1778500067 (supervisor 09:01:25Z scope-correction during medic 09:01:48Z fabrication-class false-positive).

---

## Step 1 — LOCAL pre-execution sanity (unchanged from v1)
```
cd cpython
git rev-parse HEAD                          # MUST = canonical FREEZE HEAD
git status --porcelain | grep -v "^?? "     # MUST be empty (zero tracked changes)
git rev-parse <expected dangling SHAs>      # MUST resolve, dangling commits intact
md5sum ../cpython-vanilla/python            # MUST = canonical x86_64 vanilla md5 (per current FREEZE)
```
HALT on any divergence; investigate before proceeding.

## Step 2 — Dual dry-run replay (unchanged from v1, with off-by-one fix per gatekeeper 07:38:53Z 2026-05-11 finding)
PASS-path replay (throwaway phoenix-pass-dryrun-replay) per existing v1 procedure.
FAIL-path replay (throwaway phoenix-fail-dryrun-replay) per existing v1 procedure.

NOTE on FAIL-path verification: chain length is N commits (where N = number of cherry-picks + push.sh-prime + 5.B c1, currently 6 for 5.A3+5.B class). Use `HEAD~N` for anchor preservation check, not literal `HEAD~7` — count from subjects-list length (`git log --pretty=%s -N | wc -l == N`).

Structural verification per drift-defense regime D-1778198653; SHA equality NOT canonical for committer-timestamp-drift classes.

HALT on structural mismatch.

## Step 3 — ARM64 substrate probe (extended with A2 build-env preflight)
**v1 base probe (unchanged):**
```
ssh devgpu004 "cd ~/local/phoenix/cpython && git log --oneline -1"  # MUST = canonical FREEZE HEAD or post-tip bundle
ssh devgpu004 "ls -la ~/local/phoenix/phoenix-arm64.bundle 2>/dev/null; df -h ~/local/phoenix/ | tail -1"
ssh devgpu004 "md5sum <ARM64-canonical vanilla python path> 2>/dev/null"  # MUST = ARM64-canonical vanilla md5 (post-ratify codification per A3)
ssh devgpu004 "ls ~/local/phoenix/cpython/scripts/build_phoenix.sh && ls ~/local/phoenix/cpython/scripts/gate_phoenix.sh"
```

**v2 A2 EXTENSION — REQUIRED-BINARIES sub-probe (POST-RATIFY-ITEM-2):**
```
Step 3 (extended): ARM64 substrate probe — REQUIRED-BINARIES sub-probe
  (a) Existing: HEAD + bundle + disk + python_vanilla md5 + scripts present
  (b) NEW: probe required build environment (preferred: scripts/preflight_arm64.sh per Q2)
      - command -v cmake (else /home/alexturner/.local/bin/cmake)
      - command -v clang (else /usr/local/bin/clang.par + cert WARNING acceptance)
      - test -f /usr/include/zlib.h (else sudo dnf install zlib-devel.aarch64)
      - test -f /usr/lib64/libz.so (zlib runtime)
      - bash scripts/build_phoenix.sh --dry-run (smoke without full build)
  (c) On gap: surface required workaround inventory (CPATH=/tmp/jit-extra-includes/,
      LIBRARY_PATH=/tmp/jit-extra-libs/, EXTRA_CMAKE_FLAGS-clobber awareness)
      BEFORE Step 3b dispatch — not as discovery during build.
```
Origin: generalist 07:53:31Z (clang env failure) + testkeeper 08:14:30Z (cmake env) + 08:25:50Z (zlib-devel) + 08:41:07Z (EXTRA_CMAKE_FLAGS clobber) + 08:50:12Z (libz LIBRARY_PATH) 2026-05-11.

**Sub-amendment A2.1 (script defect tracking):**
build_phoenix.sh:189 silently resets `EXTRA_CMAKE_FLAGS=""` unless `--asan` flag set. Fix path: replace with `EXTRA_CMAKE_FLAGS="${EXTRA_CMAKE_FLAGS:-}"` (preserve env var). Land outside ratify-FREEZE window.

HALT on FAIL → INFRA-debug, NOT ARM64-FAIL recipe (avoid dropping valid Phase chain commits to fix substrate).

## Step 4 — Silent-supersession check (unchanged from v1)
DO NOT use `nbs-scribe-query --last` default — reads stale .nbs/scribe/phoenix-log.md (mtime 2026-04-30, D-1778066328 parked tool routing bug).
USE: `nbs-scribe-query --chat=.nbs/chat/phoenix.chat --by=supervisor --after=<recent-cutoff>`
OR direct read `.nbs/scribe/live-log.md` for current live log.
Verify named playbook decisions NOT silently superseded.

## Step 5 — RATIFY GATE three-bar conjunction (rewritten per A5 + A9)

Authorize executor only when all THREE bars PASS on ARM64 v/vi:
- **(i) Geomean ≥ baseline-anchor x vs anchor commit** (within ±2% noise window per theologian 10:54:46Z 2026-05-11 framing; default baseline = anchor's measured geomean, not historic x86_64 1.07x absolute — see A3 + post-ratify codification queue for ARM64-anchor commit holding 1.07x line).
- **(ii) Zero crash** across full bench + test run + functional-clean DIFF=0 vs baseline (per (Y) controlled experiment per Alex feedback_assume_phoenix_regression).
- **(iii) Commit-match: ARM64 HEAD^{tree} = local HEAD^{tree}** (canonical tree-hash equivalence per drift-defense D-1778198653; literal SHA equality is metadata-only proxy and is NOT canonical for committer-timestamp-drift classes including rebase / cherry-pick / git am). Per D-1778500096 + gatekeeper 09:01:39Z 2026-05-11 (POST-RATIFY-ITEM-4). Pair with A4 SCOPE TAGS for full disambiguation.

**KNOWN LIMITATION DISCHARGE PRECONDITION (v2 A9 — POST-RATIFY-ITEM A9):**
Any specified runtime-observation mechanism MUST have prior empirical-feasibility validation on the target substrate (pydebug ARM64 in this class). Paper-only specification of a mechanism is NOT a discharge gate; it is a discharge-mechanism-candidate. First-empirical-validation on a ratify-cycle becomes a discharge-mechanism test, not a substrate test — and outcome lands in this checklist as either 'mechanism validated' or 'mechanism infeasible, alternate proposed'.

  - **A9 EMPIRICAL UPDATE (testkeeper 16:16:37Z 2026-05-11):** gdb-breakpoint-on-JIT-compiled-function in ARM64 pydebug = MECHANISM VALIDATED. Refutes paper-only-mechanism-infeasibility worry from pythia #323/#326/#327/#329.
  - **A9 sub-class disambiguation (theologian 16:17:14Z 2026-05-11):** Two PARTIAL-DISCHARGE sub-classes:
      (a) mechanism-infeasibility — gdb couldn't break (paper-only gate). REFUTED for ARM64-pydebug-JIT-compiled per testkeeper 16:16:37Z.
      (b) trigger-inadequacy — gdb works, chosen trigger doesn't execute the falsifier path. ACTIVE for 5.B c1 reach-witness on isinstance trigger.
  - **A9(c) maturity-date precondition (supervisor 16:25:27Z + 17:02:10Z 2026-05-11):** Class-(b) PARTIAL-DISCHARGE has a discharge maturity ceiling: CAST-trigger runtime-reach witness REQUIRED before 5.B-cycle-N (where N = c4 OR Phase 5.C close, whichever comes first); if CAST-trigger substrate doesn't exist by then, AUTHORED requirement (not deferral). Without a maturity ceiling the (b) carve-out becomes de-facto pattern for every Cat-A/B reach-witness gate (pythia #331 catch — exactly the precedent this section is intended to prevent).

**Per-bench INVESTIGATE-FLAG (per D-1778127738(1)):** any per-bench >5pct regression vs anchor = INVESTIGATE-FLAG (testkeeper analysis required), NOT auto-HALT. x86-calibrated threshold; ARM64 noise-band uncalibrated.

**Cumulative drift gate (per feedback_no_perf_drift + supervisor 17:02:10Z disposition on pythia #332):** gatekeeper checks 5+ batch trend on per-bench drops, BLOCK on monotone or >7% cumulative drift even if per-batch passes. ARM64 anchor-commit codification: TBD post-ratify per A3 resolution path.

Auto-HALT solely on (i)/(ii)/(iii) failure.

**ABBA report template (per A3 — POST-RATIFY-ITEM-1):**
```
ABBA report MUST include:
  - Phoenix binary build config (LTO yes/no, compiler version, debug flags)
  - Vanilla binary build config (LTO yes/no, compiler version, debug flags)
  - Symmetry assessment: same config (apples-to-apples) or asymmetric (apples-to-oranges)
  - Tip-vs-anchor delta is canonical gate (substrate-class noise cancels per supervisor 09:37:10Z)
  - Absolute Phoenix-vs-vanilla ratio is ARM64-canonical-establishment number, NOT historic-baseline match
```

Resolution path (deferred): Either (a) solve clang env on devgpu004 + rebuild vanilla --with-lto matching Phoenix toolchain, or (b) rebuild Phoenix --without --with-lto for paired-config ARM64 ABBA only. Decision deferred to W-PERF queue per project_w_perf_owner.md (theologian owner).

## Step 5.5 — PRE-PUSH REMOTE FETCH + DIVERGENCE CHECK (NEW v2 A1 — POST-RATIFY-ITEM-7, REQUIRED gate-blocking per Q1)

```
Step 5.5: PRE-PUSH REMOTE FETCH + DIVERGENCE CHECK
  (a) git fetch origin <push-target-branch> (e.g. phoenix-asm-integration)
  (b) Capture FETCH_HEAD as REMOTE_TIP
  (c) git merge-base HEAD REMOTE_TIP → MERGE_BASE
  (d) Topology classification:
      - REMOTE_TIP == HEAD → already-pushed; nothing to do
      - HEAD ancestor of REMOTE_TIP → local stale; pull/refresh required
      - REMOTE_TIP ancestor of HEAD → fast-forward push (Step 6 standard path)
      - SIBLING (neither ancestor) → REBASE required onto REMOTE_TIP before push
        - Verify remote-only commits (git log MERGE_BASE..REMOTE_TIP) are
          file-disjoint from local-only commits (Step 5.5 auto-merge precondition)
        - Execute git rebase --onto REMOTE_TIP MERGE_BASE HEAD
        - Re-run structural verification (parent + diff name-only + content match)
        - Branch reattachment: git checkout -B <branch>
  (e) Surface topology classification + chosen action in chat before Step 6
```

Origin: theologian 10:25:15Z 2026-05-11 catch + generalist 10:24:37Z 2026-05-11 push REJECTED non-fast-forward (remote b1cf9119e5 had 5.B spec from 2026-05-05 unauthorized push D-1777993894 not in local 085a44b68f).

Per gatekeeper 11:28:21Z Q1 review: REQUIRED gate-blocking, not optional.

## Step 6 — Path execution (split per A7 — POST-RATIFY-ITEM-6)

Each Stage requires its own pre+post-ARM64 APPROVE + supervisor PUSH AUTHORIZED chain.

**Stage A — Phase chain head (e.g. 5.A3 push.sh-prime equivalent):**
  - Pre-condition: Step 5 ALL 3 BARS PASS on `[BARE-SUBSTRATE]`
  - Procedure: rebase + structural-verify + scripts/push.sh per drift-defense
  - Post-condition: gatekeeper post-push VERIFY (tree-hash + parent + content)

**Stage B — Cherry-picked dependent commit (e.g. 5.B c1 d89aebb4ef equivalent):**
  - Pre-condition: SEPARATE Stage A close + ARM64 build+test cycle on combination (refresh-substrate + pydebug build + test_phoenix.sh + reach-witness + ABBA)
  - Pre-condition: SEPARATE Step 5'-style verdict
  - Procedure: rebase onto Stage A tip + structural-verify + scripts/push.sh per drift-defense
  - Post-condition: gatekeeper post-push VERIFY (tree-hash + parent + content)

Origin: supervisor 10:18:22Z + theologian 10:17:36Z 2026-05-11. Closes ratify-lifecycle granularity gap where v1 bundled both stages.

**Per-arch fallback (ARM64-FAIL):** unchanged from v1 (cherry-pick survivors on anchor + bisect c5+c7).

## Post-ratify §5 paraphrase mechanical check (per D-1778125474(C), unchanged from v1)

At 5.C §5 commit review (post-ratify, post-§3a, gatekeeper-side):
Grep doc §5 against canonical playbook §5 HOLD-LIFT GATE paragraph (PASS condition through DOC-BINDING lines).
Verbatim string-match required (whitespace-tolerant).
BLOCK on diff with paraphrase-violation flag.

## A8 Recurrence Memos (institutional memory, not gate-modifying)

**A8.1 Medic detector dangling-commit scope** — feedback_fabrication_detector_verify
3rd instance (after 2026-04-22 + 2026-05-08 compound-Bash + 2026-05-11 falsifier-symbol). Search method `git log --all -S` excludes dangling commits in reflog. Scope-fix: include `--reflog` flag OR `git fsck --dangling` enumeration when claim references a dangling SHA. Per gatekeeper 09:04:52Z 2026-05-11 + Q3 review (codification stronger than memory): include in v2 normative checklist not memo-only.

**A8.2 One-witness-post-retract rule** (D-1778491287, shepard 09:20:57Z 2026-05-11)
Once an accusation has been formally RETRACTED, additional independent verification posts become pile-on (over-witnessing). One witness post-retract is sufficient for the record.

## A10 Fabrication-verdict probe precondition (v2.1 — supervisor 18:42:34Z 2026-05-11)

When accusing FABRICATION (HALLUCINATION-class verdict on another agent's grep / symbol-existence / file-content claim), search method MUST include before issuing CONFIRMED-FABRICATION verdict:
  (1) `git status --porcelain` on the relevant file path(s)
  (2) `git diff` on any `M` files
  (3) `git show HEAD:<path>` and/or `stat -c '%y'` for HEAD-vs-WT mtime check

**Rationale:** 5 instances of feedback_fabrication_detector_verify class-violation in 5-day window (medic dangling-commit miss 09:01:48Z, medic md5sum 2026-05-08, medic compound-Bash 2026-05-08, medic folded-PTY 2026-05-08, supervisor temporal-window 18:21:22Z propagating through librarian 18:21:45Z). Common pattern: grep on current state misses in-flight modifications, dangling-commit, or reflog scope. `git status / diff / stat` is the canonical pre-verdict probe; raw working-tree grep alone is insufficient.

### A10.1 Application-trace requirement (v2.1.1 — supervisor 22:39Z + 01:05:42Z 2026-05-12)

A10's verdict-side probe is self-attested; an additional application-trace requirement applies SOLELY to FABRICATION-VERDICT-class accusations against another agent (not routine verification posts). When issuing a CONFIRMED-FABRICATION verdict, the verdict post MUST inline:

  (a) **Verbatim git command output snippet** supporting the accusation — file paths + 1-3 line excerpt, sufficient for any reader to reproduce. This implements peer-checkable application-trace per pythia #336(3) + #340(1)+(3).
  (b) **Explicit HEAD-at-claim disambiguation** — state which HEAD/commit/working-tree-state the original (allegedly fabricated) claim was about, e.g. `claim-time HEAD = X` vs `current HEAD = Y` if they differ. Closes the #340(2) grep-scope-mismatch gap that produced both POST-A10 incidents (librarian 21:26Z + supervisor 22:36Z self-violation).

**Explicit non-coverage:** routine verification claims — gate posts, state reports, ABBA results, build-status, push-verify — remain governed by feedback_no_tool_execution_citation (Alex 2026-04-29 directive: plain assertions, gate-side verifies). A10.1 verbatim-snippet requirement is FABRICATION-VERDICT-CLASS-ONLY, not generic. The two rules coexist via claim-class distinction.

**Rationale:** 7 instances of feedback_fabrication_detector_verify class-violation in 5-day window (5 pre-A10 + 2 POST-A10-codification: librarian 21:26Z, supervisor self 22:36Z). Common pattern of POST-A10 incidents: A10's `git status/diff/stat` probe ran against current working tree but the original claim referenced a different reference frame (pre-c2 HEAD), and the probe-result confirmed "absent now" without surfacing the temporal mismatch. A10.1's HEAD-at-claim disambiguation makes the temporal frame inspectable; the verbatim-snippet requirement makes the probe-result peer-verifiable. Authored under structural escalation from pythia #335(c) (POST-A10 empirical class-repeat trigger fired at supervisor 22:39Z).

## Open caveats (post-ratify codification queue)

- §3a methodology arch-portability: EXECUTION-CLEAN ≠ VALIDATION-CLEAN (sub-case A may be vacuous-correct, not validating-correct)
- ARM64 §3a wall-clock unmodelled (~2-3hr realistic, factor at Alex bandwidth window)
- nbody +3.8pct ARM64 arch-asymmetry (theologian decision-tree at docs/w-perf-nbody-arm64-asymmetry-decision-tree-WIP-2026-05-07.md)
- INFRA chat_file.c:907 5+ instance ASSERT severs new ephemeral handle attribution
- 7+ FILED pythia post-ratify codification queue (#297-#332) needs discharge owner
- A3 resolution path (clang env / paired-config Phoenix) deferred to W-PERF queue (theologian owner)
- ARM64-anchor commit holding 1.07x line vs ratchet (per pythia #332 + supervisor 17:02:10Z)
- A9(c) class-(b) maturity-date enforcement: who/what fails-closed at 5.B-cycle-N if CAST-trigger substrate absent (per pythia #331 + supervisor 17:02:10Z amendment-pending)
- kwargs_dispatch +6.7% INVESTIGATE-FLAG from 5.B c1 cycle (theologian W-PERF queue per project_w_perf_owner.md)

## Adoption record

- v1 D-1778183896 (gatekeeper 19:56:25Z 2026-05-07) SUPERSEDED by this artefact
- v2 adoption authorized supervisor 17:39:46Z 2026-05-11 per D-1778492262 cadence + Stage A (Stage 6 PASS-path Step 2 fd0fff739e) + Stage B (Step 6 PASS-path Step 3 56fed762a5) both closed on SonicField/cpython
- 9 amendments (A1-A9) consolidated from POST-RATIFY-ITEMs 1-7 + 2 recurrence-class items per generalist 11:23Z draft + gatekeeper 11:28:21Z review + A9 EMPIRICAL UPDATE
- Prep-commit by generalist on top of 56fed762a5 per supervisor 17:39:46Z
- v2.1 A10 amendment authorized supervisor 18:42:34Z 2026-05-11 per shepard 18:41:46Z this-cycle dispatch (5th feedback_fabrication_detector_verify violation in 5-day window). Prep-commit by generalist on top of 5.B c2 commit 63718ca58b, then pivot-rebased onto v2 docs 2da1b31d2d per supervisor 22:36:12Z PIVOT (pythia #336(1) chain-decouple). Canonical-pushed at 352cc1f9 22:40Z.
- v2.1.1 A10.1 amendment authorized supervisor 01:05:42Z 2026-05-12 per pythia #341 independent-push-window observation + structural escalation from supervisor 22:39Z (7th feedback_fabrication_detector_verify instance, 1st POST-A10-codification). Prep-commit by generalist on top of v2.1 352cc1f9.
