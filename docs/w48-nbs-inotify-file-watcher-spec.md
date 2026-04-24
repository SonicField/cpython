# W48 — NBS-Suite Inotify File-Watcher Spec

**Status:** FILE-NOW per supervisor 2026-04-24T02:46:15Z (b) +
theologian 2026-04-24T02:46:01Z. Escalated from PIR §2.1 (a)
DEFERRED → FILE-NOW per Tier 8 Phase A incident #7 (mtime-checkpoint
agent-side discipline exhausted).

**Owner:** TBD per supervisor (theologian sketch + generalist or
NBS-suite-owner impl).

**Estimated cost:** Multi-session NBS-suite tooling extension. ~3-5
sessions (daemon impl + agent integration + cross-cutting wiring).

**Activates when:** Tier 8 Phase A retry attempted in future session
(or any other multi-file edit class blocked by recurring file-state
revert). Currently the FOURTH escalation in undiagnosed-recurring
class (Alex directive D-1776434533 → gate #3 hardening D-1776887480 →
4-step Pre-Edit WT Integrity 03a0dcb569 → mtime-checkpoint extension
48339e23cb → W48 inotify daemon).

---

## 1. Problem statement

7 incidents on file in undiagnosed-recurring file-state-revert class
(per `docs/2026-04-24-pre-edit-revert-pir.md`). Pattern: external
process or actor reverts file content during agent multi-file edit
sequence; agent must HALT to preserve coherence.

Discipline escalation has been PROGRESSIVELY MORE DETECTIVE without
becoming PREVENTIVE:

| Escalation | Mechanism | Detection power | Prevention |
|---|---|---|---|
| Alex directive D-1776434533 | Prescriptive ('always full commit checkouts') | None — relies on actor compliance | None |
| Gate #3 hardening D-1776887480 | Build-time HEAD==binary AND tree-clean | REACTIVE at build time | None |
| 4-step Pre-Edit WT Integrity 03a0dcb569 | Agent-side baseline+HALT discipline | Mid-edit (if agent notices) | None — agent HALTs after detection |
| mtime-checkpoint extension 48339e23cb | Per-write mtime verification | Mid-edit between writes | None — agent HALTs after detection |
| **W48 inotify daemon (this spec)** | **Kernel-level file-event monitoring** | **Real-time on every file modification** | **Optional: block writes from non-agent processes** |

W48 is the next reasonable escalation. Incident #7 invalidates
'mtime-checkpoint discipline addresses risk class' (supervisor
2026-04-24T02:39:00Z reasoning; theologian 02:46:01Z empirical
re-validation).

---

## 2. Resolution

### 2.1 Mechanism

`scripts/nbs_tree_watcher.py` (or equivalent NBS-suite extension)
runs as a daemon process per agent session. Uses Linux inotify (or
fanotify for write-blocking) to monitor a working-tree subtree.

```python
# Sketch
import pyinotify

class TreeWatcherHandler(pyinotify.ProcessEvent):
    def process_IN_MODIFY(self, event):
        # Verify event source: was this the owning agent?
        # If external process modified a watched file:
        #   - Log incident with timestamp + content hash + source PID
        #   - Notify owning agent via NBS chat / event bus
        #   - Optionally: snapshot pre-revert state for forensics

watcher = pyinotify.WatchManager()
watcher.add_watch('Python/jit/hir', pyinotify.IN_MODIFY | pyinotify.IN_CLOSE_WRITE,
                  rec=True)
notifier = pyinotify.Notifier(watcher, TreeWatcherHandler())
notifier.loop()
```

### 2.2 Architecture sketch

- **Daemon lifecycle:** spawned per agent edit-session; killed on
  session end.
- **Scope:** narrow watch (e.g., `Python/jit/hir/`, `scripts/`,
  `docs/`) per agent's declared edit-set, not full tree (perf).
- **Event source identification:** correlate inotify events with
  agent's recent Write tool calls (timestamp + content hash). If
  event from non-agent source, surface to chat as `[TREE-WATCHER-
  WARNING]` with file + timestamp + content-hash diff.
- **Forensic snapshot:** on detected external write, snapshot
  pre-write content to `/tmp/nbs-tree-watcher-incident-<ts>/` for
  root-cause investigation. Closes the gap pythia #105 (3) named
  ('zero root-cause attributions').
- **Optional preventive layer:** fanotify supports BLOCK mode; non-
  agent writes can be DENIED until agent ACKs. Risk: deadlock if
  daemon crashes; needs failsafe.

### 2.3 NBS-suite integration

- Sidecar process per agent session (similar to existing nbs-bus
  sidecar pattern).
- Chat post on detection: `[TREE-WATCHER]` channel with file +
  source-PID + content-hash-diff.
- Bus event: `tree.external_write` for downstream consumers (e.g.,
  agent's edit loop can subscribe + auto-HALT before next write).

---

## 3. Test plan / falsifier strategy

### 3.1 Detection-positive test

Spawn watcher on test directory; have a separate process write to
watched file; verify watcher posts `[TREE-WATCHER-WARNING]` with
correct file + content-hash.

### 3.2 Detection-negative test (no false positive)

Spawn watcher on test directory; agent writes via Write tool;
verify watcher does NOT post warning (event correlated to agent's
recent Write).

### 3.3 Forensic snapshot test

Trigger external write incident; verify pre-write content snapshot
exists at `/tmp/nbs-tree-watcher-incident-<ts>/`; verify snapshot
matches pre-write content.

### 3.4 Performance gate

Watcher must add <100ms latency per Write tool call. inotify is
kernel-level so this is achievable; verify via benchmark.

### 3.5 Existing gates carry-over

CLAUDE.md Pre-Edit WT Integrity 4-step discipline + mtime-checkpoint
extension REMAIN ACTIVE. W48 ADDS a kernel-level layer; does not
replace agent-side discipline.

---

## 4. Acceptance criteria

W48 closure requires ALL:

1. `scripts/nbs_tree_watcher.py` (or equivalent) implemented +
   integrated as NBS-suite sidecar.
2. Detection-positive test PASSES (external write triggers warning).
3. Detection-negative test PASSES (agent writes don't trigger).
4. Forensic snapshot test PASSES (pre-write content captured for
   incident root-cause attribution).
5. Performance gate PASSES (<100ms latency per agent Write).
6. CLAUDE.md Pre-Edit WT Integrity rule 5 ADDED: 'NBS tree-watcher
   sidecar must be running for multi-file edit sequences in
   Python/jit/hir + scripts/ + docs/. Verify via nbs-bus events.'
7. Tier 8 pilot Phase A re-attempted under W48 + mtime-checkpoint;
   if recurrence (incident #8), escalate to fanotify BLOCK mode OR
   investigate non-tooling root cause (actor identification).

---

## 5. Open questions

1. **Watcher scope.** Narrow (per-edit-set) vs broad (full
   working-tree). Broad has perf cost; narrow needs agent to
   declare edit-set up front. Recommend narrow with sensible
   default subtrees.
2. **Block mode (fanotify) vs notify-only (inotify).** Block is
   stronger but risks deadlock + needs failsafe. Recommend
   notify-only for V1; block mode V2 if recurrence persists.
3. **Multi-agent contention.** When 2+ agents edit overlapping
   subtrees, watcher events from agent A may surface as 'external'
   to agent B. Need agent-PID correlation + cross-agent event
   filtering.
4. **Daemon survival across agent restart.** If agent restarts,
   does watcher restart automatically? Sidecar pattern usually
   yes; verify.

---

## 6. Cross-link

- Pythia framing: pythia #104 (3) 2026-04-24T01:36:43Z (recurrence-
  prevention plan request); pythia #105 (1)+(3) 2026-04-24T02:12:18Z
  (automated-detection feasibility + undiagnosed-recurring class)
- Supervisor authorization: 2026-04-24T02:46:15Z (FILE-NOW W48
  escalation per Tier 8 incident #7)
- Theologian disposition: 2026-04-24T02:46:01Z
- PIR: `docs/2026-04-24-pre-edit-revert-pir.md` (§2.1 (a) escalated
  from DEFERRED → FILE-NOW)
- CLAUDE.md Pre-Edit WT Integrity: 03a0dcb569 (4-step) +
  48339e23cb (mtime extension)
- Phase 3 closure-amendment: f6328cd2b9 (Tier 8 BLOCKED, pythia
  #103 RE-OPENED)
- Sibling falsifiers: W42 (refcount-correctness verifier), W44
  (DO-NOT-USE caller gate), W45 (bridge-signature drift), W45 §3.5
  (derivation drift)
- Empirical surface: 7 incidents on file (per PIR §1) + Tier 8
  Phase A incidents #6 (01:23Z) + #7 (02:45Z)
