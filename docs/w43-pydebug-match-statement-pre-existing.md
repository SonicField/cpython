# W43 — Pydebug + Match-Statement Pre-Existing Crash Class

**Status:** FILED 2026-04-23T18:55:58Z per supervisor — pre-existing
crash class confirmed via controlled experiment (testkeeper
2026-04-23T18:55:30Z fresh `--pydebug --clean` rebuild at baseline
05e2b8821a also crashes; not f47bcc9a8a-introduced).

**Owner:** TBD.

**Estimated cost:** unknown — needs GDB investigation per Alex
2026-04-23T17:53:08Z directive.

---

## 1. Problem statement

Under pydebug build (`scripts/build_phoenix.sh --pydebug --clean`),
match-statement code crashes when JIT-compiled. Empirically
reproducible:

- python_baseline (05e2b8821a, fresh --pydebug --clean rebuild,
  md5 6c3af73b6d): match-statement EXIT=139 SIGSEGV in JIT-emitted
  code
- python (HEAD f47bcc9a8a, fresh --pydebug --clean): match-statement
  EXIT=139 SIGSEGV in JIT-emitted code
- python_debug (Apr 18 76dc1c2a13, pydebug): match-statement EXIT=134
  SIGABRT 'free(): invalid pointer' from glibc malloc detector

Pre-dates the W22 work entirely (also crashes at 05e2b8821a baseline
which is pre-W25→W27 velocity window).

## 2. Crash classes

Two distinct crash signatures observed depending on build vintage:

1. **SIGSEGV 139 in JIT-emitted code** (current HEAD + 05e2b8821a
   baseline pydebug)
   - Stack: crash inside JIT-emitted code, frame #0 at instruction
     pointer in JIT code arena, called from `PyObject_Vectorcall` /
     `_PyEval_EvalFrameDefault` bytecodes.c:2715
   - Likely cause: refcount-correctness or JIT-emitted-code
     correctness bug surfaced under pydebug strict-refcount

2. **SIGABRT 134 'free(): invalid pointer'** (Apr 18 76dc1c2a13
   pydebug)
   - glibc malloc-detector pointer-validity check fails
   - Likely cause: similar refcount class fixed/refactored between
     Apr 18 and current; signature changed

## 3. Investigation plan

Per Alex 2026-04-23T17:53:08Z + memory `feedback_gdb_first.md`: GDB
FIRST.

1. Build current HEAD pydebug
2. Run /tmp/test_match_simple.py (or representative match-statement
   reproducer) under GDB
3. Capture stack trace at SIGSEGV — identify JIT-emitted instruction
   pointer location + walk backward to source HIR/builder
4. Identify which HIR opcode emits the crashing JIT code
5. Implement root-cause fix

Likely shares root with W42 (refcount-correctness verifier) since
match-statement is a key W42 fixture per existing scope.

## 4. Relationship to W22 + W42

- W22 yield-from fix at f47bcc9a8a is correct and unrelated to W43
  (controlled experiment confirms)
- W43 is in the same pre-existing bug-class as W22 + W39 await +
  W40 controlflow + W41 comparisons — all surfaced after broken-gate
  repair
- W42 verifier is the systemic mitigation; W43 is one specific
  empirical instance

W43 fix may be a useful W42 fixture: 'JIT-emitted code passes
pydebug refcount check on match-statement'.

## 5. When to schedule

Per supervisor 2026-04-23T18:55:58Z: continue GDB sequentially with
W22-residual bugs (await + gen_chain). W43 takes priority as needed
per cycle.

Cross-link:
- W22 fix: 66850a4ba1 + f47bcc9a8a (yield-from + match runtime)
- W42: docs/w42-refcount-correctness-verifier.md (systemic mitigation)
- W39 candidate: await SIGSEGV (deopt v27-not-live, separate)
- W41 candidate: gen_chain SIGSEGV (hir_remove_unreachable_blocks_c
  NULL, separate)
- Memory: feedback_gdb_first.md (GDB-FIRST methodology) +
  feedback_no_workarounds.md (pydebug-refleak-gate-for-refcount-class)
- Empirical surface: testkeeper controlled experiment 2026-04-23T18:55:30Z
