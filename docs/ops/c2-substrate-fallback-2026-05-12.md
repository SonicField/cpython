# c2 Substrate Fallback Procedure (devgpu004 decay scenario)

**Authored:** 2026-05-12 ~13:00Z by generalist (per supervisor 12:55:33Z dispatch, pythia #353
extended-block-contingency commitment).

**Scope:** read-only investigation + procedure draft for the case where devgpu004 substrate
(HEAD 97e33890648e on `/home/alexturner/local/phoenix/cpython` + ARM64 binary
`1444034d3af09fb18ef341b1fe6e8886` + `/tmp/c2-series-1419c1261e/` patches) becomes
unavailable BEFORE Alex `data_staging` broadcast lands and c2 push completes.

**No actual spawn until decay observed.** This document is a contingency, not an active plan.

---

## (a) Alternative ARM64 host availability

**No alternative ARM64 host is documented in the Phoenix repo.**

Surveyed: `cpython/docs/`, `CLAUDE.md`. Only `devgpu004.kcm2.facebook.com` (aarch64) appears
in operational records. `devgpu009.ncg6.facebook.com` is x86_64 (per
`docs/benchmarks/abba_x86_64_20260416.md`) and not a substitute.

**Action required if decay:** escalate to Alex for alt ARM64 host provisioning. The team
does not autonomously provision new dev hosts — Phoenix-team-internal coordination needed.
Likely candidates by naming convention (UNVERIFIED, requires Alex confirmation):
`devgpu00{1,2,3,5,6,7,8}.kcm2` or any other `*.kcm2` aarch64 box she has access to.

**Pre-warmed state on alt host (none):** zero alt ARM64 substrate exists today. A fresh
spawn would start from a 524 MB bundle clone (see (b)) plus full clean build (~10–15 min
per prior precedent on devgpu004).

---

## (b) Bundle file size + transport mechanism

| Artifact | Size | Path (x86_64 origin) | Use |
|---|---|---|---|
| 4-patch series for c2 chain | 65 KB total (4 files, largest 37.6 KB) | `/tmp/c2-series-1419c1261e/` | Apply onto c1-tree to reach c2 |
| Git bundle of 4-commit chain (`56fed762a5..1419c1261e`, `--branches`) | 67 KB | `/tmp/c2-test.bundle` (sample) | Single-file alternative to patch series |
| Full-repo bundle (existing snapshot, ~3 weeks stale) | 524 MB | `/data/users/alexturner/phoenix/phoenix-arm64.bundle` | Bare-host bootstrap |

**Transport mechanism (verified working 2026-05-12):**

- Direct `scp <local> devgpu004.kcm2.facebook.com:<remote>` succeeds from the x86_64 dev
  host without a proxy — md5 round-trip on `/tmp/c2-series-1419c1261e/` matched byte-for-byte
  in this session. (CLAUDE.md "Push & Remote Access Protocol" claim that direct ssh/scp is
  blocked is stale for the patch-series case — direct works.)
- For `git push` to GitHub the canonical wrapper is still `nbs-local-run
  /data/users/alexturner/phoenix/push.sh`; that constraint is unchanged and unrelated to
  ARM64 substrate respawn.
- `nbs-ts send <session-handle> '<cmd>'` for in-session commands on devgpu004 — verified
  via session `99e77ed3` in this cycle.
- Dual recovery for c2 itself: (1) reflog/dangling commit on x86_64
  (`63718ca58b21029dd22a4f860579a7d1666500d6`) (2) format-patch
  `0004-Phase-5.B-commit-2-delete-legacy-parser.cpp-text-map.patch` (37,620 bytes,
  sha256 `226442c922e1bb4bd5fb35914ccf3d01b0b17edfcb4258cfe5ea1a7d24a8616e` per testkeeper
  Step 0 11:50:19Z; identical content re-emitted today as
  `0004-...patch`).
- Full-repo bundle is ~3 weeks stale (Apr 23) — usable as a starting point for a bare alt
  host but the c1+c2 chain still needs the 4-patch (or refreshed bundle) on top.

Patch-id of c2 (invariant across all reproductions): `8fc3493dbe2655a2d15e6aa6ec4a7901cfada9b2`.
This is the canonical content check that must hold on any spawned ARM64 substrate.

---

## (c) Saved-binary md5 (current operational reference)

| Field | Value |
|---|---|
| Path on devgpu004 | `/home/alexturner/local/phoenix/cpython/python` |
| md5 | `1444034d3af09fb18ef341b1fe6e8886` |
| Size | 56,223,088 bytes |
| Built from | ARM64 HEAD `d5fd126080c3e8f565b15b3863cd0390f717bc71` (tree-equivalent to x86_64 `1419c1261e`) |
| Build mtime | 2026-05-12 05:12 (devgpu004 local) |
| Build mode | release (no `--pydebug`, per build script defaults) |
| Build wrapper | `scripts/build_phoenix.sh --clean` |
| Verifier | `python --version` → `Python 3.12.13` |

For a respawn, parity is established when a fresh build on an alt host produces a binary
whose tree-equivalent commit has patch-id `8fc3493dbe2655a2d15e6aa6ec4a7901cfada9b2` and
whose `python --version` matches `3.12.13`. Binary md5 will NOT match across hosts (build
timestamps + machine-id differ); md5 is useful only as a same-host integrity check.

**No off-host binary backup exists today.** If devgpu004 loses its filesystem, the binary
must be rebuilt from sources. Build time on devgpu004 was ~5 minutes for clean release per
this session (12:09–12:13Z observed).

---

## (d) Build-env compatibility check

Verified on devgpu004 in this cycle (session `99e77ed3`):

| Tool | Path | Version | PATH-discovery |
|---|---|---|---|
| `cmake` | `/home/alexturner/.local/bin/cmake` | `4.3.2` | **NOT in default login PATH.** Must `export PATH="$HOME/.local/bin:$PATH"` before invoking `scripts/build_phoenix.sh`. First build attempt this cycle failed at the `cmake` configure step because of this. |
| `clang` / `clang++` | `/usr/bin/` (assumed, found by configure) | configure-detected | OK in default PATH |
| `make` | `/usr/bin/make` | OK in default PATH |
| `git` | `/usr/bin/git` | OK in default PATH |
| `module` (env modules) | shell function | available; no `cmake` modulefile present (`module avail cmake` empty) |
| `sccache` | not surveyed | unknown — configure does not require it |

**Build script entrypoint:** `scripts/build_phoenix.sh --clean` (release) /
`scripts/build_phoenix.sh --clean --pydebug` (debug). Per `feedback_build_scripts`, never
hand-roll. Per `feedback_pydebug_first_for_segv`, runtime SEGV first move = `--clean
--pydebug`.

**ARM64-specific build notes (per CLAUDE.md):**

- Do NOT use `--with-lto` on ARM64 (too slow for dev builds — older guidance in
  `docs/devgpu004-sync.md` line 69; current `build_phoenix.sh` does not enable LTO by
  default).
- `--pydebug` mandatory for ARM64 runtime correctness gates per
  `feedback_pydebug_gate`; release-build is fine for ABBA + Tier 1 but not for SEGV
  triage.

**Working-tree path on devgpu004:** `/home/alexturner/local/phoenix/cpython` — current
operational location. `docs/devgpu004-sync.md` recommends `/data/users/$USER/phoenix-cpython`
for I/O speed; that recommendation is stale (current ops use `/home`, slow but
authoritative). A respawn on a fresh alt host should follow current ops practice and use
`/home/$USER/local/phoenix/cpython`.

---

## Fallback procedure (decay-triggered)

Trigger condition (any of):

1. devgpu004 unreachable via SSH for >15 min after a previously-working session.
2. `/tmp/c2-series-1419c1261e/` removed (e.g., `/tmp` reaped by host policy) AND no
   in-session shell holds the patches.
3. ARM64 working tree at `/home/alexturner/local/phoenix/cpython` shows tree-state
   inconsistent with `d5fd126080` HEAD (e.g., uncommitted modifications, missing files).

**Step 0 — confirm decay (do not spawn on hypothesis):**

```bash
ssh devgpu004.kcm2.facebook.com 'echo OK' || echo DECAY_PROBABLE
ssh devgpu004.kcm2.facebook.com 'ls /tmp/c2-series-1419c1261e/ 2>&1; cd /home/alexturner/local/phoenix/cpython && git rev-parse HEAD && git status --short'
```

Expected on healthy substrate: `OK`, 4 patch files, HEAD `d5fd126080...`, only the 5
benchmark `.txt` files untracked.

**Step 1 — if patches missing but host healthy: re-transport from x86_64.**

```bash
# On x86_64 dev host:
scp -r /tmp/c2-series-1419c1261e devgpu004.kcm2.facebook.com:/tmp/
# Verify md5 on both ends; the 4 sha256 hashes are recorded in
# generalist 12:11Z chat post and in this cycle's nbs-ts log.
```

If `/tmp/c2-series-1419c1261e` is also missing on x86_64:

```bash
# Regenerate from canonical x86_64 HEAD (1419c1261e):
mkdir -p /tmp/c2-series-1419c1261e
git format-patch 56fed762a5..1419c1261e -o /tmp/c2-series-1419c1261e/
# Then scp as above.
```

**Step 2 — if working tree corrupted but host healthy: re-apply from c1.**

```bash
# On devgpu004 (assuming arm64-5b-c1-test still at 97e33890648e):
cd /home/alexturner/local/phoenix/cpython
git reset --hard 97e33890648e   # back to c1 ARM64 baseline
export PATH="$HOME/.local/bin:$PATH"
git am /tmp/c2-series-1419c1261e/*.patch
scripts/build_phoenix.sh --clean
# Verify: HEAD patch-id == 8fc3493dbe2655a2d15e6aa6ec4a7901cfada9b2
```

**Step 3 — if devgpu004 fully unrecoverable: alt-host spawn (requires Alex coordination).**

a. Escalate to Alex with this document, request alt ARM64 host.
b. On alt host:
   ```bash
   # Initial clone — preferred: full bundle if accessible
   scp /data/users/alexturner/phoenix/phoenix-arm64.bundle <alt-host>:/tmp/
   ssh <alt-host> 'cd /home/$USER/local/phoenix && git clone /tmp/phoenix-arm64.bundle cpython'
   # Then bring HEAD up to c1 (97e33890648e equivalent), apply 4-patch series, build.
   # Verify: cmake at ~/.local/bin or install (cmake 4.3.2 minimum tested).
   ```
c. ETA: ~30 min for clone + ~5 min build + verification.

**Step 4 — verify spawn:**

```
- HEAD patch-id:  8fc3493dbe2655a2d15e6aa6ec4a7901cfada9b2  (CANONICAL c2 content)
- python --version:  Python 3.12.13
- numstat:  6 files, 17 insertions(+), 869 deletions(-)
- Files touched (all in Python/jit/lir/):
  c_helper_translations.{c,h}, inliner_c.c, lir_impl_internal.h, parser.{cpp,h}
```

Report parity to chat with the new binary md5 + size + new HEAD SHA. Then re-enter the
sequence at supervisor's "step 4" (ARM64 Tier 1).

---

## What is NOT covered

- Alt-host provisioning — out of scope for generalist; needs Alex.
- Restoring x86_64 substrate decay — orthogonal; x86_64 substrate is at
  `/data/users/alexturner/phoenix/cpython` HEAD `1419c1261e`, with c2 reachable both as a
  current commit and as a dangling reflog entry (`63718ca58b`). Local-only loss would
  re-enter via the same patch series from any other team member's clone.
- Concurrent-write conflicts — see `feedback_phxmem_zeroinit_trap` and
  related `feedback_falsifier_restore_clobbers_unstaged` for the script-level pitfalls.
- ASGuard data_staging unblock — orthogonal, requires Alex broadcast (the original
  blocker driving this contingency).
