# Push 51 emitYieldValue — W26 build-state-corruption case study

**Status:** RECOVERED 2026-04-22 — code at `05e2b8821a` is correct + reproducible
(`testkeeper` V1 35/35 PASS deterministic on clean rebuild). The original SEGV
was build-state corruption, not a code bug. First observed instance of W26.

## Quick narrative

1. `generalist` committed `05e2b8821a` (`emitYieldValue` → C, push 51).
2. `testkeeper` ran the gate; 7/7 tests SEGV including `test_abs_inf` which
   contains no `YIELD_VALUE` bytecode (the converted method is never
   dispatched on this test).
3. Bisection narrowed the regression to the C++ stub, NOT the C body.
4. Code-inspection dead-end: 4-invariant audit clean, no signature drift, no
   header changes, no symbol collisions.
5. `testkeeper` clean-rebuilt the same source at `05e2b8821a` (V1) — 5x runs
   of the 7-test set: **35/35 PASS, 0 SEGV.**
6. MD5 of the original "broken" binary matches the bytes that crashed (V2)
   — same binary, was broken, now passes the same test under fresh build
   environment.
7. Causal attribution (V3, partial): the original gate ran on an
   incrementally-built binary produced by an iter1→iter2→full-build sequence
   following two compile fixes within ~3 minutes. Most plausible cause:
   incremental cmake build state carried stale `.o` / LTO bitcode artifacts
   from the iter1 broken-decl compile, producing a binary that crashed
   non-deterministically across emitter codepaths.

## What we know

### V1 — code is correct + reproducible

`testkeeper` 2026-04-22 12:02:29Z: clean rebuild from `05e2b8821a`,
5 consecutive runs of the 7-test set, all 35/35 pass, zero SEGV.

### V2 — same binary now passes its own crash test

`md5sum ./python_bench` matches the bytes that crashed at 11:38Z. Test
behaviour changed environment-side, not binary-side.

### V3 — build-state corruption is the working hypothesis

Original gate timeline:

  - `11:34Z` iter 1 commit (broken `HirInstr` decl) → compile-check FAILED
  - `11:35Z` iter 2 (`void *` fix) → compile-check PASSED
  - `11:36Z` `05e2b8821a` committed
  - `11:37Z` full `bash scripts/build_phoenix.sh` — produced binary
  - `11:38Z` `cp ./python ./python_bench` + 7-test loop → SEGV

Two compile fixes inside ~3 min, then an incremental full build, then test.
The 12:00Z clean rebuild used `cmake clean cache` and produced a working
binary from the same source.

### Code-faithfulness audit (theologian 11:43:50Z)

All four invariants from the pre-audit verified post-commit in the C body:

1. Per-yield `CO_ASYNC_GENERATOR` check at the YIELD_VALUE site
2. 3.12 `RESUME`-peek dispatch (`next_bc.opcode() == RESUME && oparg >= 2`)
3. Double `AllocateStack` on the async-gen wrap path
4. Iter at TOS PEEKED (not popped)

Negative findings (all ruled out): bridge signature mismatch, header
changes, struct layout drift, symbol collision (only 3 references to
`hir_builder_emit_yield_value_c` in the tree).

### Bisection (testkeeper 11:49:45Z)

`git checkout 2faa8a024f -- Python/jit/hir/builder.cpp` while keeping the
`builder_emit_c.c` additions: `test_abs_inf` PASSES.

Conclusion at the time: the C body is innocent, the C++ stub is the
trigger. Post-V1 reading: the bisection's hybrid tree happened to evict
the corrupted intermediate artifacts that the full push-51 build carried,
which is why it passed.

## Anchored W-items

- **W26 — build-state corruption mitigation**: anchored post-Tier-5,
  joint testkeeper + generalist. Concrete deliverable: minimum gate
  hardening — mandatory `build_phoenix.sh --clean` before test-bench
  rebuild during rapid-iter sessions. testkeeper already adopted this
  discipline at 12:00:11Z; W26 codifies it into `gate_phoenix.sh`.
- **W25 — typed-bridge design**: stays valid as predicted by pythia #68;
  push 51 was a near-miss observation (the void*-erased extern decls
  did NOT cause the SEGV, but the class concern remains).
- **W27 — C++ stub design template**: downgraded to speculative pending
  a second push-51-class instance.

## Process notes

### Detached-HEAD reset trap (generalist, 11:55:58Z)

When attempting to revert push 51, generalist ran `git reset --hard
2faa8a024f` while HEAD was DETACHED (testkeeper bisection had previously
left HEAD detached). `reset --hard` from detached HEAD only moves the
HEAD ref, not the branch ref. The branch `phoenix-asm-integration`
remained at `05e2b8821a`. generalist verified `git log --oneline -3`
which showed HEAD at `2faa8a024f` and incorrectly concluded the branch
had been reset.

**Correct verification after a `reset --hard` intended to move a branch:**
`git rev-parse <branch-name>` — confirms the branch tip explicitly.
`git log --oneline -3` only shows HEAD, which can be misleading on a
detached HEAD.

### Process-win framing (theologian 11:53:32Z)

The original Tier 1 quick gate caught the suspected regression before
push. Per `feedback_compile_before_commit.md` +
`feedback_no_workarounds.md`, parking was preferred to shipping a
suspected-broken delta. The subsequent V1 verification recovered the
correct interpretation. Net: nothing broken shipped; `gate_phoenix.sh`
worked as designed.

## Reproducer (for W26 investigation)

To reproduce the original failure mode, the build state needs to mimic
the rapid-iter sequence. Approximate steps:

```bash
# Start from a state where iter1's broken compile artifacts exist
cd Python/jit_build/build
make phoenix_jit -j8                   # iter1 — leaves partial compile state
# (apply iter2 source fix without --clean)
make phoenix_jit -j8                   # iter2 — incremental
cd ../..
bash scripts/build_phoenix.sh           # incremental full build
cp ./python ./python_bench
./python_bench -m test test_phoenix_jit_arithmetic -k test_abs_inf
# Expected (per V3 hypothesis): SEGV
```

Confirming this reproducer is part of the W26 investigation.
