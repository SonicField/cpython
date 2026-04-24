# Phoenix Project Ground Rules

## Gate-Before-Push (MANDATORY)

All commits on the phoenix-asm-integration branch require gatekeeper APPROVE in chat before git push to SonicField/cpython. The pushing agent MUST:

1. Commit locally
2. Post commit hash to chat requesting gatekeeper review
3. Wait for gatekeeper to post "APPROVE — <commit hash> clear to push"
4. Push only after APPROVE is posted
5. Cite the gatekeeper approval when reporting the push

Any push without prior gatekeeper APPROVE is a process violation. Medic MUST flag violations via [MEDIC-WARNING].

Gatekeeper MUST NOT issue APPROVE until testkeeper confirms BUILD PASS. Review-only approval (git show --stat without build confirmation) is invalid. This has been violated 3 times — it is now a hard gate.

## Push Authorization (MANDATORY — added 2026-04-17)

Generalist CANNOT push until supervisor posts "PUSH AUTHORIZED" in chat citing testkeeper's ARM64 PASS message timestamp. The sequence is:

1. Testkeeper posts ARM64 gate PASS with timestamp
2. Gatekeeper posts APPROVE with git show excerpts citing ARM64 PASS timestamp
3. Supervisor posts "PUSH AUTHORIZED citing testkeeper ARM64 PASS at <timestamp>"
4. ONLY THEN may generalist run git push

This rule exists because push-before-ARM64-gate occurred 3 times with social-only enforcement. Mechanical rule: no PUSH AUTHORIZED message = no push. Crashes block gate FOREVER (Alex directive 2026-04-17).

Retroactive approvals are NOT acceptable for Phase 3D commits (codegen conversion touches 57K lines — unwinding a bad push is costly).

## Build Verification (MANDATORY)

Every BUILD PASS report must include: (1) the build command run, (2) binary timestamp, (3) commit hash verified in the built binary AND working tree clean during the build (no uncommitted edits influencing the binary), (4) the verbatim build trailer block (from `=== Build complete ===` through `Binary timestamp: ...`). Reports without all four are UNVERIFIED.

Two build-verification failures in the 2026-04-16 session (stale binary + phantom build) proved that "it builds" claims without evidence are unreliable. Item (3)'s tree-clean addition is per testkeeper 2026-04-22 L2186 self-flag (cited HEAD didn't reflect actual binary content during uncommitted-edit window). Item (4) verbatim-trailer paste is per testkeeper L2233 + librarian L2255 (closes memory-decay-gap on PASS-on-claim convention).

**Refuting build-class claims (filesystem-first):** any agent disputing a BUILD PASS report (e.g. medic-class hallucination warnings) MUST cite filesystem evidence — `stat -c %Y` on the binary, `.o` file existence checks, or grep of build stdout — BEFORE posting the warning. Session-log scans alone are insufficient because builds may run via Bash tool not nbs-ts. Per medic 2026-04-22 L2226 false-positive (session-log-only check missed Bash-tool builds, escalated to HALT + Alex-threat before filesystem check; supervisor + medic both self-flagged).

**Counts in commit messages and chat reports (script-driven):** any quantitative claim (TU counts, extern counts, file counts, method counts) MUST come from script output (`scripts/count_w25_b1b_tus.sh`, `scripts/count_emit_methods.sh`, or equivalent) — not memory or recollection. Quote the script command and output. Per repeated /N-class lapses (4+ instances across 2026-04-16 + 2026-04-22 sessions despite feedback memory).

## Build Lock (MANDATORY — Phase 3D)

During Phase 3D, ONLY testkeeper may run make, cmake, configure, build_phoenix.sh, make distclean, or any build command in the cpython directory. All other agents are restricted to file editing and git operations.

Violation is a process violation equivalent to pushing without gatekeeper approval. Medic MUST flag violations via [MEDIC-WARNING].

If testkeeper is dead or unavailable, supervisor may temporarily designate ONE other agent as builder. The designation must be posted to chat before any build command runs.

## JIT Header Change Protocol (MANDATORY)

After ANY change to JIT headers that affect struct layout (hir.h, intrusive_list.h, hir_instr_c.h, lir_types_c.h), a full `make distclean` + rebuild is REQUIRED. Incremental `make` does NOT recompile cmake-built JIT .o files — stale objects with old struct layouts cause silent data corruption.

Use `build_phoenix.sh` (which does cmake clean) or manual `make distclean` + configure + make. This protocol is non-negotiable — stale builds caused 3 false debugging trails in the 2026-04-10 session.

## Debug-First for Unknown Crashes (MANDATORY)

When investigating an unknown crash or segfault, agents MUST use GDB/LLDB instrumentation BEFORE making code changes:

1. Reproduce the crash under GDB/LLDB (use nbs-local-session or nbs-remote-session for persistent sessions)
2. Set watchpoints on the corrupted memory to find the EXACT code path that creates the bad state
3. Only after the root cause is identified via debugger evidence: implement the fix
4. Verify the fix locally (testkeeper builds, agent runs test) BEFORE committing

Speculative fixes (guessing the code path and adding guards) are NOT acceptable. The dict_fromkeys investigation demonstrated that 7 speculative attempts failed before 1 watchpoint-guided fix succeeded. See docs/phase3d-debug-protocol.md for the full protocol.

No workarounds, deopt bail-outs, or interpreter fallbacks — Alex's standing directive.

## Per-Commit Benchmark Gate (MANDATORY — Phase 3D)

After EVERY Phase 3D commit, before push, run a 4-benchmark check:

```bash
cp ./python ./python_bench
VANILLA_PYTHON=../cpython-vanilla/python JIT_ENABLE=1 ./python_bench Tools/benchmark_phoenix.py jit --compile=auto --reps=3 --only=fibonacci,nqueens,gen_simple,func_calls
```

- Subset: fibonacci, nqueens, gen_simple, func_calls
- Mode: --compile=auto --reps=3 (3-rep ABBA, ~10 min)
- BLOCK threshold: geo-mean drop >5% vs prior commit = BLOCK push
- Alert threshold: any single benchmark drop >10% = manual review (NOT auto-block)
- Vanilla: ../cpython-vanilla/python (md5 fcb1dddcbf5d1edbf54c478e705deccc)
- Report: raw per-benchmark ratios + geo-mean posted to chat

Calibrated 2026-04-13: 5 identical runs showed per-benchmark noise of 2.6-10.2% spread.
Individual 2% thresholds produce false positives. Geo-mean with 3-rep ABBA has ~2.4% CV.

Session-boundary gate: full 24-benchmark 3-rep ABBA every 5 conversions. Geo-mean must
stay within 5% of pinned LTO baseline (1.10x at HEAD, re-baselined 2026-04-14 with LTO).
Prior non-LTO baseline was 1.06x at f6f2f7ad8e. Phase 3D confirmed performance-neutral
to positive (+4% from pre-Phase-3D LTO 1.06x to HEAD LTO 1.10x).

Hard performance floor (Alex directive 2026-04-14): geo-mean must NEVER drop below 1.0x
(vanilla CPython parity). Below 1.0x = BLOCK push until fixed. Performance optimization
comes after Phase 3D is complete.

This prevents silent performance regressions like c4616abb1a (jit_get_config sync overhead, 0.92x regression undetected for 4+ sessions).

## Benchmark Binary Copy (RECOMMENDED)

Before running benchmarks, copy the binary to avoid build contention:

```bash
cp ./python ./python_bench
```

Use `./python_bench` for benchmarks while development builds continue on `./python`. This prevents benchmark crashes from concurrent builds (3 crashes in 2026-04-12 session).

## Push & Remote Access Protocol (MANDATORY)

All network access (GitHub push, ARM64 SSH) MUST use `nbs-local-run` — direct proxy/SSH from agent sessions does NOT work.

```bash
# Push to SonicField/cpython
nbs-local-run '/data/users/alexturner/phoenix/push.sh'

# ARM64 SSH (requires Alex's Duo 2FA auth first)
nbs-local-run 'ssh devgpu004.kcm2.facebook.com "<command>"'

# ARM64 SCP
nbs-local-run 'scp <local-file> devgpu004.kcm2.facebook.com:<remote-path>'
```

Do NOT attempt `git push` or `ssh` directly — they will fail with proxy 403 or hang on 2FA.

## HirType/Type Conversion (MANDATORY)

Never use `reinterpret_cast` between `HirType` and `Type`. These types have different binary layouts (C++ bitfields vs manual shift packing). Use:
- `Type::toHirType(t)` — C++ Type → C HirType
- `Type::fromHirType(h)` — C HirType → C++ Type

Gatekeeper MUST grep for `reinterpret_cast.*HirType` and `reinterpret_cast.*const Type.*&` in every review. Any match is an automatic BLOCK.

## Pydebug Gate Protocol

ARM64 gate runs with `--with-pydebug` + `CMAKE_BUILD_TYPE=Debug` to catch JIT assertion bugs invisible in optimized builds. Use `scripts/build_phoenix.sh --pydebug --clean`.

Pydebug gate checks auto-compilation path assertions (funcTypeChecks, JIT_DCHECK). force_compile tests may not work under pydebug (cinderjit module may not load).

## ARM64 Commit-Match: Tree-Match Canonical

CLAUDE.md's ARM64-commit-match rule guards against silent-bundle-failure (x86 and ARM64 building different content). The substantive check is content equivalence, not literal SHA equality. Verify via `git rev-parse <SHA>^{tree}` cross-arch + parent-SHA match.

When devgpu004 receives a commit via `git am` (because the local bundle is multi-commits behind the canonical github remote), `git am` re-stamps the committer-date and the resulting SHA differs from the x86 SHA — even with `git am --committer-date-is-author-date`, because the chain of parent SHAs starts diverging the moment the bundle is stale. The trees stay byte-identical because the patch applies the same diff onto the same parent tree.

In that case, **tree-match is canonical**: x86_64 TREE == ARM64 TREE proves the content is identical. SHA-match is a metadata artifact, not a content check. Testkeeper reports both SHAs in this scenario and gatekeeper accepts tree-match as the gate.

For full SHA-match, refresh the bundle on devgpu004 between pushes (`git bundle create` on x86 + `scp` to devgpu004), or pursue W30 (point devgpu004's git origin at github directly).

## Pre-Edit Working-Tree Integrity (NEW)

Multi-file edit sequences (≥2 files) require working-tree integrity discipline:

1. **Pre-edit baseline snapshot.** Before multi-file edit, capture `git status --short` + `git diff --name-only HEAD`. Confirm tree state is clean OR known-modified.

2. **Mid-edit integrity check.** If any touched file's mtime/content changes UNEXPECTEDLY during edit (not as result of agent's own write), HALT immediately. Restore to baseline via `git checkout HEAD -- <file>` per Alex directive D-1776434533.

3. **Pre-commit verification.** At commit time, re-verify staged diff matches expected file-set + line-count forecast. If staged diff is missing expected files OR shows unexpected files, HALT (do NOT commit; investigate).

4. **HALT response.** When integrity violation detected, restore to clean HEAD per gate-#3 prescription (librarian 01:46:19Z) + post observed-but-not-attributed report per don't-speculate-on-cause (generalist 01:24:43Z precedent). Surface trigger for supervisor disposition; do NOT auto-resume.

Rationale: 5+ prior incidents (D-1775810621, D-1775669703, D-1776414469, D-1776434533, D-1776887480/D-1776890644) caused gate failures + false BUILD PASS + ARM64 build BLOCK. Per pythia #104 (3): gate #3 (HEAD==binary AND tree-clean during build) is REACTIVE; this discipline is PROACTIVE complement.

**Honest framing per pythia #105 2026-04-24:** This is an UNDIAGNOSED-RECURRING class. The 5+ prior incidents closed at the SYMPTOM layer (build break, gate fail, Alex directive) without confirmed root-cause attribution. This 4-step discipline codifies a reactive HALT — it does not provide automated detection. Detection still depends on the agent noticing mid-edit (which generalist did at 01:23Z without the discipline). Recurrence-prevention claim does not exceed what this text mechanizes; automated detection (file-watcher / git-hook) feasibility is fixup PIR territory and remains open.
