# Wiring & Gate Catches Tally

Running tally of regressions caught by the rotating wiring gate, the ARM64
stash bracket, and other gate-infra falsifiers. Per supervisor 2026-04-22
00:02:54Z (rotating-cadence acceptance criterion): if zero catches accrue
over the next 30 emit-method conversions AND a regression slips through,
raise the cadence from 1-per-10 to 1-per-5.

Each entry records: `<date> <commit hash where caught> <function/test that
caught it> <root cause class> <test-bug | emit-bug | gate-script bug>`.

## Catches

| # | Date       | Commit       | Caught by                              | Root cause                                                                                          | Class            |
|---|------------|--------------|----------------------------------------|-----------------------------------------------------------------------------------------------------|------------------|
| 1 | 2026-04-21 | 0e62d9108b   | wiring smoke `load_attr_generic`       | `import sys` in body → IMPORT_NAME → JIT PYJIT_RESULT_UNKNOWN_ERROR at force_compile                | test-bug         |
| 2 | 2026-04-21 | 15feea07c9   | ARM64 gate (push 36 stash bracket)     | `git stash push -u` captured SCP'd bundle inside cpython tree → fetch failed silently               | gate-script bug  |
| 3 | 2026-04-21 | 91d5b60f5d   | ARM64 commit-match check               | `grep -oP 'ARM64_COMMIT=\K\S+'` matched heredoc body literal `$(git rev-parse...)` not runtime echo | gate-script bug  |

## Meta-observation (supervisor 2026-04-22 00:17:13Z)

The push 36 ARM64 stash bracket (15feea07c9 + 3b7e566ad3) materially changed
gate observable behavior — adding the heredoc body echo to ARM64_OUTPUT,
adding stash markers around the remote run, exercising the dirty-tree path
on every gate. Two adjacent gate-script bugs (#2 and #3 above) became
visible in the next two gate runs. Structural fixes accelerate latent-bug
discovery by changing the observable surface; pre-push-36, both bugs were
present but silent.

Pattern: when adopting a structural fix, expect 1-2 secondary catches in
the immediate next gate runs. Treat them as evidence the fix is working as
intended, not as setbacks.

## Cadence accounting

- Conversions completed when last catch logged: 84 emit methods (push 37, 0e62d9108b)
- Conversions until cadence review (per supervisor 00:02:54Z): 30
- Next cadence review at: 114 emit methods (~push 47-50 depending on batching)
- Default cadence: 1 wiring function added per 10 converted emit methods
