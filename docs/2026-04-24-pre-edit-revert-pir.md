# Pre-Edit External-Revert Post-Incident Review (PIR)

**Date:** 2026-04-24
**Incident:** Tier 8 pilot Phase A execution HALT (01:23Z, generalist
01:24:43Z observed-but-not-attributed)
**Owner:** theologian (per supervisor 2026-04-24T02:35:05Z redirect
from fixup; fixup-6bbfee5a session ended 02:14:01Z pre-assignment per
librarian 02:34:46Z)
**Triggered by:** pythia #105 (1)+(3) 2026-04-24T02:12:18Z + supervisor
02:14:16Z scope assignment + pythia #104 (3) 2026-04-24T01:36:43Z

---

## 1. Incident class: undiagnosed-recurring file-state-revert

Six incidents on file (this PIR's 2026-04-24 incident + 5 priors per
librarian 2026-04-24T01:46:19Z institutional memory):

| Incident | Decision-log ID | Resolution class | Root cause attribution |
|---|---|---|---|
| Dirty tree partial revert → 48 vs 40 byte Instr layout mismatch | D-1775810621 | SYMPTOM-only | NONE — no actor identified |
| Uncommitted builder.h decl → ARM64 build failure, gate BLOCKED | D-1775669703 | SYMPTOM-only | NONE — no actor identified |
| Uncommitted hir.h edit → false BUILD PASS | D-1776414469 | SYMPTOM-only | NONE — no actor identified |
| Alex directive: always full commit checkouts, no partial reverts | D-1776434533 | DIRECTIVE-only | NONE — no diagnosis of cause |
| Gate item #3 hardened: HEAD==binary AND tree-clean during build | D-1776887480 / D-1776890644 | GATE-hardening | NONE — no diagnosis of cause |
| 2026-04-24T01:23Z external file-state revert during Tier 8 Phase A | (this incident) | SYMPTOM-only | NONE — generalist 01:24:43Z observed-but-not-attributed |

**Conclusion:** 6 incidents, 0 root cause attributions on file. Class
labeled UNDIAGNOSED-RECURRING per pythia #105 (3) + CLAUDE.md
amendment 03a0dcb569 (theologian 2026-04-24T02:14:46Z).

---

## 2. Automated-detection feasibility analysis

### 2.1 Option (a) inotify file-watcher daemon

- **Feasibility:** YES, technically feasible (Linux inotify or
  fanotify; ~100-200L Python/C daemon process)
- **Coverage:** mid-edit divergence (writes invalidate snapshot)
- **Cost:** separate process management; agent integration (agents
  don't currently spawn watchdogs)
- **Blast radius:** NBS-suite extension required for agent
  integration; multi-session workstream
- **Disposition:** DEFERRED to W48 future workstream; not required
  for immediate Tier 8 Phase A resume

### 2.2 Option (b) pre-commit git hook

- **Feasibility:** YES (standard git infra)
- **Coverage:** commit-time only (no mid-edit detection)
- **Cost:** ~10-20L bash hook + integration into push.sh / commit
  workflow
- **Power vs gate #3:** SAME coverage class — gate #3 already
  enforces HEAD==binary AND tree-clean during build. Pre-commit
  hook would duplicate; no detection improvement
- **Disposition:** REJECTED — duplicates existing gate #3 without
  net new detection power

### 2.3 Option (c) agent-side mtime-checkpoint discipline

- **Feasibility:** YES, agent-side discipline (no infra)
- **Coverage:** mid-edit divergence (per-edit mtime baseline +
  verify-before-next-write)
- **Cost:** ~5L per agent edit-loop in CLAUDE.md discipline
- **Power:** detects revert between writes via mtime mismatch;
  HALT before further damage
- **Disposition:** RECOMMEND ADOPT as Pre-Edit WT Integrity rule 1
  extension. Lightweight, in-session feasible

### 2.4 Recommendation

ADOPT (c) agent-side mtime-checkpoint discipline as immediate
extension to CLAUDE.md Pre-Edit WT Integrity rule 1. File W48
NBS-suite tree-watcher extension as future workstream (deferred;
not required for Tier 8 Phase A resume).

---

## 3. Honest framing per pythia #105

This PIR confirms: 6 incidents, 0 root causes attributed. Discipline
escalation pattern: directive (D-1776434533) → gate #3 hardening
(D-1776887480) → 4-step discipline (03a0dcb569 today) → mtime-
checkpoint extension (proposed).

Each escalation describes the recurrence; none diagnose it.
Per pythia #105 'fever-has-name-infection-spreads' framing: discipline
is reactive containment, not root-cause cure. CLAUDE.md amendment
03a0dcb569 honest framing paragraph correctly states this limit.

Going forward: if 7th incident occurs after mtime-checkpoint adoption,
ESCALATE to W48 NBS-suite tree-watcher feasibility study (option (a)).

---

## 4. Tier 8 Phase A resume recommendation

PIR result UNBLOCKS Tier 8 Phase A resume per supervisor 2026-04-24
T02:14:16Z 'PIR result needed before next supervisor checkpoint':

1. PIR delivered (this doc) with feasibility analysis +
   undiagnosed-recurring acknowledgment
2. RECOMMEND atomic commit: docs/2026-04-24-pre-edit-revert-pir.md
   + CLAUDE.md mtime-checkpoint extension to rule 1 (1-2L addition)
3. Push 29 lands PIR + discipline extension
4. Tier 8 Phase A resumes per (R-retry) under enhanced discipline
5. If Alex 01:25:09Z disposition lands later confirming intentional
   revert, theologian re-evaluates; meantime PIR satisfies pythia
   #105 (3) requirement

Phase 3 closure-amendment fast-path STAGED on disk per supervisor
02:23:54Z + theologian 02:26:02Z. Triggers ONLY if PIR rejected OR
Alex disposition negates Phase A resume; otherwise RETRACT amendment
on Tier 8 Phase A successful resume.

---

## 5. Cross-link

- Pythia #104 (3) 2026-04-24T01:36:43Z (recurrence-prevention plan
  request)
- Pythia #105 (1)+(3) 2026-04-24T02:12:18Z (automated-detection
  feasibility + undiagnosed-recurring class label)
- Supervisor 2026-04-24T02:14:16Z scope assignment to fixup
- Supervisor 2026-04-24T02:35:05Z redirect to theologian
- Librarian 2026-04-24T01:46:19Z institutional memory (5+ priors)
- Librarian 2026-04-24T02:34:46Z fixup operational gap note
- CLAUDE.md amendment 03a0dcb569 (theologian 2026-04-24T02:14:46Z)
- Phase 3 closure summary fa8dfef1e1 (generalist 2026-04-24T01:05:45Z;
  closure-amendment STAGED on disk per theologian 2026-04-24T02:26:02Z)
- Tier 8 spec docs/tier8-class-b-cport-migrate-arm-spec.md
