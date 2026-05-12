# A10.1 Cold-Context Fold-In (restarted scribe + medic)

**Authored:** 2026-05-12 by generalist (per supervisor 18:09:43Z dispatch, codification
batch item 2 during c7-close ABBA wait window). Discharges resume queue item 3
(librarian 03:11Z + supervisor 14:31:23Z post-c2 codification batch).

**Class:** post-c2 codification batch item 2.

---

## Problem

A10.1 (FRESH-EYES checklist v2.1.1, pushed at a0704da7b0 2026-05-12 01:09Z) requires
that any **FABRICATION-VERDICT** class chat post (i.e., agent A accusing agent B of
fabrication, e.g. `[MEDIC-WARNING] HALLUCINATION`) inline:

1. Verbatim git command output snippet (file paths + 1-3 line excerpt) supporting the
   accusation.
2. Explicit HEAD-at-claim disambiguation: state which HEAD/commit/working-tree-state the
   original claim was about.

Routine verification claims remain governed by `feedback_no_tool_execution_citation`
(Alex 2026-04-29 directive: plain assertions; gate-side verifies). A10.1 applies SOLELY
to FABRICATION-VERDICT-class accusations.

**The cold-context gap:** scribe + medic are subject to mandatory periodic restart by
the fixup harness (every fixup cycle, ~every 60-90min). On restart, the new agent
session starts with cursor reset to a recent chat position and ingests prior chat as
flat playback. Working memory of A10.1 (and any other recently codified rule) is not
guaranteed across restart — the rule must be present in the agent's *startup context*
or in a high-prominence cited memory the new session reads.

For scribe + medic specifically, FABRICATION-VERDICT class is the substantive workload:
medic posts `[MEDIC-WARNING]` accusations; scribe records dispositions. Both agents are
A10.1's primary subjects. A restart that loses A10.1 awareness un-enforces the rule for
the very agents authoring the verdicts it governs.

## Mechanism

**Three layers, each independently sufficient as fold-in:**

**Layer 1 — Skill-file embedding.** The scribe and medic skill files
(`.nbs/workers/scribe.skill.md`, `.nbs/workers/medic.skill.md`, or equivalent) MUST
include an A10.1 stanza near the top, citing checklist v2.1.1 + the verbatim
inline-evidence requirement. Skill files are loaded on every agent session start and
are the authoritative cold-context substrate.

**Layer 2 — Memory citation.** The MEMORY.md index file
(`/home/alexturner/.claude/projects/-data-users-alexturner-phoenix/memory/MEMORY.md`)
already contains entries for related feedback (e.g., `feedback_alex_brevity`,
`feedback_chat_recheck_before_claim`). Add a one-line entry pointing to a
`feedback_a10_1_fabrication_verdict.md` memory file that captures the rule + verbatim
trigger pattern. Auto-loaded on session start.

**Layer 3 — Cursor-reset re-broadcast.** On every fixup-driven restart of scribe or
medic, the supervisor (or fixup hook) posts a single chat message tagged
`@scribe @medic A10.1 RE-BROADCAST: <one-line rule statement> + see v2.1.1` so the
restarted agent encounters the rule at the top of unread-cursor playback rather than
buried 30K lines deep.

Layer 1 is the strongest (deterministic, no human-in-the-loop). Layer 2 is the
auto-recovery. Layer 3 is the human-attention safety net for cases where Layers 1+2
were not yet in place.

## What is NOT in scope

- Modifying the fixup hook to auto-broadcast (Layer 3 mechanization). Out of scope for
  this codification doc; surfaces as a later operational item.
- Embedding the full A10/A10.1 spec into skill files. Layer 1 needs a stanza referencing
  v2.1.1, not the full text — the checklist remains canonical and is fetched via
  conventional chat search OR scribe-query when needed.
- Symmetric fold-in for theologian / supervisor / generalist / gatekeeper / pythia.
  These agents are restarted less frequently AND their FABRICATION-VERDICT exposure is
  lower (they consume the verdicts; scribe+medic author them). Periodic refresh
  acceptable; the urgency is scribe+medic.

## Open items

- Layer 1 implementation requires editing `.nbs/workers/{scribe,medic}.skill.md` files
  on this OS user. Not committed in-tree (skill files are user-config, not Phoenix
  repo content). Tracked here for traceability; actual edit is out of band.
- Layer 2 implementation: write
  `feedback_a10_1_fabrication_verdict.md` in
  `~/.claude/projects/-data-users-alexturner-phoenix/memory/`, add 1-line index entry to
  `MEMORY.md`. Also out of in-tree scope; pointer here for completeness.

## Anchors

- A10.1 codification: D-1778547973 (supervisor) + D-1778548172 (push a0704da7b0)
- librarian 03:11Z 2026-05-12: cold-context fold-in resume queue item
- supervisor 14:31:23Z 2026-05-12: post-c2 codification batch dispatch
- supervisor 18:09:43Z 2026-05-12: this codification dispatch
- pythia #345: cold-context-after-restart concern
- pythia #340 (3): detector-paperwork-vs-mechanism falsifier (related, not closed by this
  codification — measurement methodology is in `a10_measurement.md`)
- A10/A10.1 measurement methodology: `docs/methodology/a10_measurement.md` (theologian
  2026-05-12, v0.1+v0.2)
- gatekeeper-fresh-eyes-checklist-v2-2026-05-11.md: A10.1 canonical text
