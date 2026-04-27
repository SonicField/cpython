# W-PreExistingAudit — CPython 9-failure set

**Status:** Filed (generalist scaffold, 2026-04-24, queued P2 per supervisor 21:14:29Z)
**Owner:** theologian (oracle methodology), generalist (per-item investigation), testkeeper (oracle runs)
**Trigger:** pythia #130 #2 (D-1777060712) + theologian 19:58:22Z + supervisor 19:57:21Z+21:13:53Z. The 9-failure CPython set has been carried `PreExistingProven` across pushes 39–49 by set-equality discharge alone, never tested for Phoenix-causation under Alex policy 'all bugs presumed Phoenix-introduced'.
**Cost estimate:** ~30min/item × 9 = ~5hr; sub-batchable.
**Sequencing:** post-W-PHASE-B-PYDEBUG (push 50) per supervisor 21:14:29Z queue.
**Naming:** W-PreExistingAudit canonical (theologian first-mover 19:58:22Z); W-CPython9 alias retired per supervisor 21:14:29Z.

---

## 1. Failure set (per testkeeper push-49 gate, identical across pushes 39–49)

| # | Test | Module | Failure mode | cinderx_dev oracle | Phoenix-introduced? | Disposition |
|---|---|---|---|---|---|---|
| 1 | test_misc | test.test_gdb | (TBD: capture failure surface) | TBD | TBD | TBD |
| 2 | test_pretty_print | test.test_gdb | (TBD) | TBD | TBD | TBD |
| 3 | (whole module) | test_cmd_line | (TBD) | TBD | TBD | TBD |
| 4 | (whole module) | test_pdb | (TBD) | TBD | TBD | TBD |
| 5 | (whole module) | test_peg_generator | (TBD) | TBD | TBD | TBD |
| 6 | (whole module) | test_posixpath | (TBD) | TBD | TBD | TBD |
| 7 | (whole module) | test_urllib | (TBD) | TBD | TBD | TBD |
| 8 | (whole module) | test_urllib2 | (TBD) | TBD | TBD | TBD |
| 9 | (TBD; testkeeper reported "9 failures across 8 modules") | TBD | TBD | TBD | TBD | TBD |

testkeeper to clarify the 9th failure (one module produces 2 failures; needs gate-log inspection).

---

## 2. Oracle methodology

Per pythia #131 #3 + librarian 21:18:22Z prior-decision chain.

### 2.1 cinderx_dev runtime parameters

- **Path:** `devgpu004.kcm2:~/local/cinderx_dev/cinderx` (D-1774988351 + D-1774988639 reinstated)
- **Build flags:** NO PGO, NO LTO, RelWithDebInfo (-O2)
- **LTO symmetry decision:** ARM64 no-LTO to match cinderx_dev; x86_64 keeps LTO symmetric on both sides (D-1774989074)

### 2.2 Phoenix runtime parameters (per-test comparison environment)

- **Path:** `cpython/python` at HEAD post-push-50 (or whichever push is being audited)
- **Build flags:** **TBD per pythia #131 #3** — must match cinderx_dev environment as closely as possible. Options:
  - (A) Build a release Phoenix matching cinderx_dev's NO-PGO + NO-LTO + RelWithDebInfo profile for the audit; reduces diff to source revision only
  - (B) Use Phoenix's standard build (LTO on x86_64); accept LTO as a known asymmetry; document it in each per-item disposition
- **Recommendation: (A)** for audit cleanliness; the differential should be between two distinct revisions, NOT two distinct environments.

### 2.3 Test-harness equivalence

Both Phoenix and cinderx_dev run the same `Lib/test/test_*.py` files (from CPython source); no harness divergence at the test-file level. Confirm pre-audit:
- Phoenix CPython source = cpython tree HEAD
- cinderx_dev CPython source = whichever vendored CPython 3.12 they have

If versions differ (Phoenix 3.12.13 vs cinderx_dev's pinned 3.12.x), record the version delta per item — the failure may be CPython-test-version-specific, NOT Phoenix-introduced.

---

## 3. Per-item procedure

For each of the 9 failures:

1. **Capture failure surface from Phoenix gate log** (testkeeper has gate logs at `docs/gates/<commit>.log`): exact `FAIL`/`ERROR`/`CRASH` lines, traceback, exit code.
2. **Run same test on cinderx_dev** (`PYTHONPATH=cinderx/cinderx/PythonLib LD_LIBRARY_PATH=python-install/lib JIT_ENABLE=1 ./python-install/bin/python3.12 -m test <module>` per librarian 13:58:02Z recipe).
3. **Disposition:**
   - cinderx_dev FAILS same way → cinderx-shared bug; investigate as Phoenix-introduced default per Alex policy (cinderx_dev failure ≠ pre-existing-acceptable; that framing was retired)
   - cinderx_dev PASSES → Phoenix-introduced; open per-item fix workstream
   - cinderx_dev cannot run the test (missing module, env mismatch) → record blocker, defer item

4. Record findings in §1 table.

---

## 4. Acceptance criteria

W-PreExistingAudit closes when:
- All 9 items have `cinderx_dev oracle` + `Phoenix-introduced?` columns populated
- Phoenix-introduced items have a fix workstream opened (W-CPython-{module-name})
- Cinderx-shared items have an explicit disposition (defer-to-cinderx-upstream OR fix-in-Phoenix-anyway)
- Audit doc `parked-bug-audit-2026-04-24.md` updated with the W-PreExistingAudit closure summary

---

## 5. Cross-references

- Audit Group A+C closure: `docs/parked-bug-audit-2026-04-24.md`
- Phoenix-presumed-regression policy: `feedback_assume_phoenix_regression.md` (Alex 2026-04-24 13:47:22Z)
- cinderx_dev oracle recipe: librarian 13:58:02Z (chat); D-1774988351 + D-1774988639 + D-1774989074
- Pythia #130 #2 surfacing: 19:57:38Z
- Naming reconciliation: supervisor 21:14:29Z
