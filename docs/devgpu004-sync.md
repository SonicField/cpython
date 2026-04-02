# Syncing Code to devgpu004 (ARM64)

## Overview

devgpu004 is an aarch64 machine used for ARM64 build/test.
It has NO internet access — code must be transferred from devgpu009 via SSH.

## Paths

| Machine | Path | Drive | Purpose |
|---------|------|-------|---------|
| devgpu009 | /data/users/$USER/phoenix/cpython | fast (local) | Primary x86_64 development |
| devgpu004 | /data/users/$USER/phoenix-cpython | fast (local /data) | ARM64 build/test |

**Do NOT use /home/alexturner on devgpu004** — it's a network drive and builds are very slow.

## Transfer Protocol

### 1. Create git bundle on devgpu009

```bash
cd /data/users/$USER/phoenix/cpython
git bundle create /tmp/phoenix-backup.bundle phoenix-asm-integration
```

### 2. SCP bundle to devgpu004 (via nbs-local-session)

nbs-local-session has credentials including SSH access to devgpu004.

```bash
HANDLE=$(nbs-local-session)
nbs-ts send "$HANDLE" 'scp /tmp/phoenix-backup.bundle devgpu004:/tmp/phoenix-backup.bundle'
# Wait ~10 seconds for 452MB transfer
nbs-ts read-new "$HANDLE" --strip
```

### 3. Clone from bundle on devgpu004

```bash
nbs-ts send "$HANDLE" 'ssh devgpu004 "rm -rf /data/users/$USER/phoenix-cpython && cd /data/users/$USER && git clone /tmp/phoenix-backup.bundle phoenix-cpython -b phoenix-asm-integration"'
```

The `error: bundle-uri operation not supported by protocol` message is informational — the clone succeeds.

### 4. Verify

```bash
nbs-ts send "$HANDLE" 'ssh devgpu004 "cd /data/users/$USER/phoenix-cpython && git log --oneline -3"'
```

### 5. Clean up

```bash
nbs-ts send "$HANDLE" 'ssh devgpu004 "rm /tmp/phoenix-backup.bundle"'
nbs-ts kill "$HANDLE"
```

## ARM64 Build Protocol

From devgpu004 (via nbs-local-session + ssh):

```bash
cd /data/users/$USER/phoenix-cpython
cd Python/jit_build/build_asmjit && cmake . && cd ../..
./configure --without-pydebug --disable-ipv6 CC=clang CXX=clang++
make -j4
```

**Do NOT use** `--with-lto` on ARM64 (too slow for dev builds).
**Do NOT use** `build_phoenix.sh` (x86_64-only).

## Key Facts

- SSH to devgpu004 works via nbs-local-session (credentials)
- Direct `ssh` from agent Bash tool is blocked by bpfjailer
- devgpu004 has no internet — cannot clone from GitHub
- Bundle transfer takes ~10 seconds over internal network
- The `error: bundle-uri` warning is harmless
