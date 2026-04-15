# Perf Stat: Fibonacci fib(30) — x86_64

Date: 2026-04-13
Commit: 0cd00ad1d2 (post jit_get_config sync-once fix)
Binary: LTO, --compile=force

## JIT (force_compile)

```
task-clock:     145.08 ms
cpu-cycles:     360,869,705  (2.5 GHz)
instructions:   942,807,314  (2.6 IPC)
branches:       193,891,983
branch-misses:  1,031,257    (0.5%)
page-faults:    1,421
time elapsed:   0.146 sec
```

## Vanilla CPython 3.12.13

```
task-clock:     203.49 ms
cpu-cycles:     505,822,072  (2.5 GHz)
instructions:   1,452,294,691  (2.9 IPC)
branches:       229,982,570
branch-misses:  2,200,190    (1.0%)
page-faults:    827
time elapsed:   0.204 sec
```

## Comparison

| Metric        | JIT       | Vanilla    | Ratio         |
|---------------|-----------|------------|---------------|
| task-clock    | 145 ms    | 204 ms     | 1.41x faster  |
| cpu-cycles    | 361M      | 506M       | 1.40x fewer   |
| instructions  | 943M      | 1452M      | 1.54x fewer   |
| branches      | 194M      | 230M       | 1.19x fewer   |
| branch-misses | 1.0M (0.5%) | 2.2M (1.0%) | 2.1x fewer |
| IPC           | 2.6       | 2.9        | vanilla higher |

JIT generates 35% fewer instructions and 29% fewer cycles.
fibonacci target: 2.50x. Current: 1.81x (harness) / 1.41x (perf stat).
Gap is Python call dispatch overhead on recursive function calls.
