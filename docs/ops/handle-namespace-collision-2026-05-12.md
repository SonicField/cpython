# NBS Handle-Namespace Collision (cross-project sidecar dedup)

**Authored:** 2026-05-12 by generalist (per supervisor 16:05:15Z dispatch, codification commit
#3 in post-c4 batch). Anchors librarian 06:16Z + pythia #350 + repeated
fixup workaround precedent (07:12Z, 08:10Z, 09:11Z, 10:12Z, 11:13Z, 12:50:32Z).

**Class:** silent cross-project work cleanup risk. Operationally workaroundable today;
proposed one-line fix below.

---

## Problem

`nbs-sidecar-restart` deduplicates running sidecars by **handle alone**. Multiple
Phoenix-style team projects on the same workstation (today: `phoenix`, `vib-jit`,
`nbsterm`) each run a `scribe` sidecar, a `medic` sidecar, and so on. Without scoping,
the dedup pass collapses `scribe-phoenix` + `scribe-vib-jit` + `scribe-nbsterm` into a
single `scribe` group and may kill the "duplicates" — i.e., other projects' live
sidecars.

The handle-strip at `nbs-sidecar-restart` line 153 (`HANDLE="${HANDLE%%-*}"`) is the
proximate driver: it intentionally collapses compound handles like `supervisor-vib` →
`supervisor` to handle a per-team-tag artefact, but the same collapse erases cross-project
distinctness when the project name follows the role.

## Trigger conditions

- Two or more Phoenix-style teams running concurrently on the same OS user.
- The fixup hook calls `nbs-sidecar-restart` (per scribe/medic mandatory-restart cycle).
- `--root=PATH` filter omitted from the call.

The default fixup invocation today omits `--root` and would silently affect cross-project
sidecars if not for the per-cycle workaround documented below.

## Current workaround

Pass `--root=<project-path>` on every invocation of `nbs-sidecar-restart` from a per-team
context:

```bash
nbs-sidecar-restart --respawn --root=/data/users/$USER/phoenix
```

Applied at every fixup cycle in this session (07:12Z, 08:10Z, 09:11Z, 10:12Z, 11:13Z,
12:50:32Z, 13:52:13Z, 14:53:48Z). Effective; not a structural fix.

The `--root=PATH` filter at `nbs-sidecar-restart` lines 43, 95, 154–164 is already
implemented and does the right thing when supplied — see `filter_root` in
`kill_sidecar_loops` (line 75) and the second filter at line 161 in the dedup pass.

## Proposed one-line fix

Make project root part of the dedup key by default (no flag required, no behavior
change for single-project setups). At `nbs-sidecar-restart` line 167:

```diff
-        echo "$pid" >> "${DEDUP_DIR}/${HANDLE}.pids"
+        echo "$pid" >> "${DEDUP_DIR}/${SC_ROOT//\//_}-${HANDLE}.pids"
```

Companion change at the args-save block immediately below (line 169 area):

```diff
-        if [[ ! -f "${DEDUP_DIR}/${HANDLE}.args" ]]; then
+        if [[ ! -f "${DEDUP_DIR}/${SC_ROOT//\//_}-${HANDLE}.args" ]]; then
```

(slash-to-underscore replacement keeps the file name path-safe.) Result: cross-project
sidecars stay in separate dedup groups; `--root=PATH` becomes an additional filter, not
the only protection.

**Tradeoffs:**

- Backward compatibility: any sidecar that omits `--root` from its own argv lands under
  an empty-string root group, which still dedupes correctly within itself but won't
  collide with project-rooted sidecars. Acceptable.
- Single-project setups (only `phoenix`): no observable change.
- Fixup hook simplicity: callers may stop passing `--root` once this fix lands; the
  behavior is correct without it. Optional follow-up: drop `--root` from project-team
  fixup hooks once the fix is in place.

This change is local to `nbs-sidecar-restart` and does not require coordination with
`nbs-sidecar` itself.

## Out of scope

- Cross-host handle collision (different OS users): out of scope; nbs-sidecar already
  refuses to operate across users.
- Handle-strip semantics for legitimate compound handles (`supervisor-vib`): preserved
  as-is. The strip is correct for tag-suffix handles; the bug is only in the dedup-key
  composition.
- Authoring of a parallel test for the fix: deferred to whoever ships the
  `nbs-sidecar-restart` change; this doc captures the operational rationale.

## Anchors

- librarian 06:16Z 2026-05-12: handle-namespace recurring per-cycle observation
- pythia #350: cross-project handle collision codification queue item
- pythia #351: per-cycle prediction confirmed by repeated fixup workarounds
- supervisor 14:31:23Z + 16:05:15Z: post-c4 codification batch dispatches
- shepard 14:48:49Z (cross-project anchor): D-1776818725 + WT-DIRTY rule
  (orthogonal class — both are cross-coordination defects but different mechanisms)
