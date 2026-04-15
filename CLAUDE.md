# Phoenix Project Ground Rules

## Gate-Before-Push (MANDATORY)

All commits on the phoenix-asm-integration branch require gatekeeper APPROVE in chat before git push to SonicField/cpython. The pushing agent MUST:

1. Commit locally
2. Post commit hash to chat requesting gatekeeper review
3. Wait for gatekeeper to post "APPROVE — <commit hash> clear to push"
4. Push only after APPROVE is posted
5. Cite the gatekeeper approval when reporting the push

Any push without prior gatekeeper APPROVE is a process violation. Medic MUST flag violations via [MEDIC-WARNING].

Retroactive approvals are NOT acceptable for Phase 3D commits (codegen conversion touches 57K lines — unwinding a bad push is costly).

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
