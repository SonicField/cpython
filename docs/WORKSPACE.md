# Parallel GC Project - Directory Structure

## Overview

This project has two main directories:

### 📚 Documentation & Research
**Location:** `~/claude_docs/parallel_gc/` (Documentation drive)

```
~/claude_docs/parallel_gc/
├── docs/                      - Design documents, TDD plan, scope
├── research/                  - CinderX analysis, CPython integration strategy
├── notes/                     - Development notes
├── tests/                     - Test specifications (not actual test code)
└── implementation/            - Design specs for implementation
```

**Purpose:** Design, planning, research, and documentation

**Use for:**
- Reading design documents
- Reviewing integration strategy
- Consulting TDD plan
- Documenting findings

### 💻 Active Development Workspace
**Location:** `~/local/parallel_gc/` (**High-speed SSD**)

```
~/local/parallel_gc/
├── cpython/          - CPython source tree (git clone)
├── patches/          - Patch files for parallel GC
├── tests/            - Actual test implementation
├── benchmarks/       - Performance benchmarking
├── builds/           - Build configurations (debug, release, asan, tsan)
├── tools/            - Development scripts
└── logs/             - Build and test logs
```

**Purpose:** Active development, compilation, testing, benchmarking

**Use for:**
- Cloning CPython
- Writing code
- Building CPython
- Running tests
- Benchmarking performance
- Generating patches

---

## Quick Navigation

| Task | Location | Command |
|------|----------|---------|
| **Read design docs** | `~/claude_docs/parallel_gc/docs/` | `cd ~/claude_docs/parallel_gc/docs` |
| **Review TDD plan** | `~/claude_docs/parallel_gc/docs/tdd_plan.md` | `less ~/claude_docs/parallel_gc/docs/tdd_plan.md` |
| **Study CinderX code** | `~/claude_docs/parallel_gc/research/cinderx_source/` | `cd ~/claude_docs/parallel_gc/research/cinderx_source` |
| **Start development** | `~/local/parallel_gc/` | `cd ~/local/parallel_gc` |
| **Work on CPython** | `~/local/parallel_gc/cpython/` | `cd ~/local/parallel_gc/cpython` |
| **Build CPython** | `~/local/parallel_gc/builds/debug/` | `cd ~/local/parallel_gc/builds/debug` |
| **Run benchmarks** | `~/local/parallel_gc/benchmarks/` | `cd ~/local/parallel_gc/benchmarks` |

---

## Workflow

### 1. Design Phase (Documentation Directory)
```bash
cd ~/claude_docs/parallel_gc/

# Review design documents
less docs/tdd_plan.md
less research/cpython_integration_strategy.md

# Study CinderX implementation
less research/cinderx_analysis.md
vim research/cinderx_source/parallel_gc.c

# Take notes
vim notes/implementation_notes.md
```

### 2. Implementation Phase (Development Workspace)
```bash
cd ~/local/parallel_gc/

# Clone CPython (first time)
cd cpython/
git clone https://github.com/python/cpython.git .

# Create development branch
git checkout -b parallel-gc-dev

# Make changes
vim Python/gc_parallel.c

# Build
cd ../builds/debug/
../../cpython/configure --with-pydebug --with-parallel-gc
make -j$(nproc)

# Test
./python -m test test_gc -v
```

### 3. Documentation Phase (Back to Documentation)
```bash
cd ~/claude_docs/parallel_gc/

# Document findings
vim notes/implementation_progress.md

# Update design docs if needed
vim docs/implementation_notes.md
```

---

## Why Two Directories?

### Documentation (`~/claude_docs/parallel_gc/`)
- **Drive:** Standard storage
- **Purpose:** Long-term documentation and research
- **Size:** Small (~50 MB)
- **Access pattern:** Read-heavy, infrequent writes
- **Version control:** Can be tracked separately

### Development (`~/local/parallel_gc/`)
- **Drive:** High-speed SSD
- **Purpose:** Active development, compilation, testing
- **Size:** Large (~6 GB with all build configs)
- **Access pattern:** Heavy I/O (compilation, testing, benchmarks)
- **Performance:** SSD critical for fast iteration

**Benefits:**
- ✅ Fast compilation (SSD)
- ✅ Fast test execution (SSD)
- ✅ Fast benchmarks (SSD)
- ✅ Clean separation (docs vs code)
- ✅ Documentation persists even if workspace reset

---

## Key Documents

### Must Read
1. **TDD Plan:** `~/claude_docs/parallel_gc/docs/tdd_plan.md`
2. **Integration Strategy:** `~/claude_docs/parallel_gc/research/cpython_integration_strategy.md`
3. **GIL-Only Scope:** `~/claude_docs/parallel_gc/docs/GIL_ONLY_SCOPE.md`
4. **CinderX Analysis:** `~/claude_docs/parallel_gc/research/cinderx_analysis.md`

### Quick Reference
- **Atomic Operations Mapping:** `~/claude_docs/parallel_gc/research/atomic_operations_mapping.md`
- **CPython Coding Standards:** `~/claude_docs/parallel_gc/research/cpython_coding_standards.md`
- **Test Suite Analysis:** `~/claude_docs/parallel_gc/research/COMPREHENSIVE_REPORT.md`
- **Build Integration Guide:** `~/claude_docs/parallel_gc/research/build_integration.md`

### Development Workspace
- **Workspace README:** `~/local/parallel_gc/README.md`

---

## Environment Setup

### One-time Setup
```bash
# 1. Ensure documentation directory exists
ls -la ~/claude_docs/parallel_gc/

# 2. Ensure development workspace exists
ls -la ~/local/parallel_gc/

# 3. Clone CPython to development workspace
cd ~/local/parallel_gc/cpython/
git clone https://github.com/python/cpython.git .

# 4. Install build dependencies (Debian/Ubuntu)
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    curl \
    llvm \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libffi-dev \
    liblzma-dev \
    python3-gdbm \
    libnss3-dev \
    libgdbm-dev \
    libgdbm-compat-dev
```

### Daily Development
```bash
# Start in development workspace
cd ~/local/parallel_gc/

# Check what you're working on
git -C cpython status

# Build and test
cd builds/debug/
make -j$(nproc) && ./python -m test test_gc -v
```

---

## Aliases (Optional)

Add to `~/.bashrc`:

```bash
# Parallel GC project
alias pgc-docs='cd ~/claude_docs/parallel_gc'
alias pgc-dev='cd ~/local/parallel_gc'
alias pgc-src='cd ~/local/parallel_gc/cpython'
alias pgc-build='cd ~/local/parallel_gc/builds/debug'
alias pgc-bench='cd ~/local/parallel_gc/benchmarks'

# Quick test run
alias pgc-test='cd ~/local/parallel_gc/builds/debug && ./python -m test test_gc test_gc_parallel -v'

# Quick build
alias pgc-make='cd ~/local/parallel_gc/builds/debug && make -j$(nproc)'
```

Then:
```bash
source ~/.bashrc
pgc-dev    # Jump to development workspace
pgc-docs   # Jump to documentation
pgc-test   # Run tests
```

---

## Disk Space Management

### Monitoring
```bash
# Check development workspace size
du -sh ~/local/parallel_gc/*

# Check documentation size
du -sh ~/claude_docs/parallel_gc/*
```

### Cleanup
```bash
# Clean build artifacts
cd ~/local/parallel_gc/builds/
make clean  # In each build directory

# Clean old logs
find ~/local/parallel_gc/logs/ -mtime +30 -delete

# Clean old benchmark results
find ~/local/parallel_gc/benchmarks/results/ -mtime +90 -delete
```

---

## Backup Strategy

### What to Backup

**Critical (must backup):**
- `~/claude_docs/parallel_gc/` - All documentation and research
- `~/local/parallel_gc/patches/` - Patch files
- `~/local/parallel_gc/tools/` - Development scripts

**Can regenerate:**
- `~/local/parallel_gc/cpython/` - Git clone
- `~/local/parallel_gc/builds/` - Build artifacts
- `~/local/parallel_gc/logs/` - Logs

### Backup Command
```bash
# Backup critical files only
tar czf parallel_gc_backup_$(date +%Y%m%d).tar.gz \
    ~/claude_docs/parallel_gc/ \
    ~/local/parallel_gc/patches/ \
    ~/local/parallel_gc/tools/
```

---

## Next Steps

1. ✅ Documentation directory created (`~/claude_docs/parallel_gc/`)
2. ✅ Development workspace created (`~/local/parallel_gc/`)
3. 🔲 Clone CPython to workspace
4. 🔲 Build baseline (serial GC)
5. 🔲 Verify environment (run tests)
6. 🔲 Begin Phase 0 implementation

**Current Status:** Workspace ready for development!

---

**Last Updated:** 2025-11-27
