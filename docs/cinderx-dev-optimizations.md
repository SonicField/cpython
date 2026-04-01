# CinderX Dev Branch Optimizations Reference

Source: `devgpu004:~/local/cinderx_dev/cinderx` branch `skipped-flag-codeextra`

This document catalogs the optimizations and fixes in the cinderx_dev branch
that must be preserved or ported into Phoenix. Alex's directive: this branch
has "very many quite subtle optimisations" — do not simplify or omit.

## 1. Compilation Scheduling

### Threshold
- Default: `compile_after_n_calls = 1000` (reverted from 5000 experiment)
- The `auto()` function sets threshold to 1000 (`pyjit.cpp:1709`)
- Before `auto()` is called: `compile_after_n_calls = 0xFFFFFFFF` (never compile)
- **Why 1000**: allows CPython's adaptive interpreter to specialize opcodes
  (LOAD_ATTR → LOAD_ATTR_INSTANCE_VALUE, etc.) before JIT reads them

### Counting Trampoline (`251e5ffc`)
- `jitCountingTrampoline`: lightweight per-call check, no JIT_DCHECK overhead
- Below threshold: delegates directly to interpreter via `getInterpretedVectorcall`
- At threshold: switches `func->vectorcall` to `jitVectorcall` and delegates
- Already ported to Phoenix in `pyjit.cpp:281-301`

### Skip Heuristics (multiple commits)
Functions that should NOT be JIT-compiled:

| Skip Condition | Reason | Commit |
|---|---|---|
| Contains `IMPORT_NAME` | JIT doesn't set up frame globals correctly for `PyImport_Import` → SIGSEGV | `91baf6ba` |
| `CO_VARKEYWORDS` (`**kwargs`) | Poor JIT performance | `f0269bf6` |
| `__enter__` / `__exit__` qualname | Context manager protocol, poor JIT benefit | `c7dc24d4` |
| CodeExtra already marked `skipped` | Fast path in func watcher — avoid re-analysis | `a15ed6f3` |

- `shouldSkipCompilation` must use `specializedOpcode()` not `opcode()` (`f7b1426e`)
- @classmethod skip was tried and reverted (`0ca08ab8` → `35bd19be`)

## 2. Deopt Backoff (`105ee2c6`, `0730c07e`)

- `kDeoptBackoffThreshold = 1000` (was 100, raised to reduce false suppression)
- After 1000 runtime guard failures for a single CodeRuntime:
  1. Suppress JIT for that code object
  2. Detach compiled code, restore interpreter entry
- Prevents pathological deopt loops (e.g., polymorphic receivers)
- Critical for production stability — without this, polymorphic call sites
  cause infinite compile/deopt cycles

## 3. Codegen Optimizations

| Optimization | Description | Commit |
|---|---|---|
| Inline LOAD_ATTR_SLOT/INSTANCE_VALUE | HIR builder inlines specialized load attr | `9ee6275f` |
| Inline `__exit__`/`__aexit__` | Simplify pass inlines context manager exits | `fc4fe918` |
| PIC guard for polymorphic inlining | Polymorphic inline caching with type guards | `c564131e` |
| Skip exact-type guard for polymorphic | Reduces guard overhead on polymorphic sites | `1a0e9338` |
| kwargs dispatch optimization | 3 fixes, positional_dispatch 0.649x → 0.98x | `1fa46c9b` |
| Inline `*args` via MakeTuple | Specializes varargs function calls | `2c7e4da9` |
| Cold block marking | Infrastructure for layout optimization | `ee8e4b9c` |
| Fix kNotNegative guard (aarch64) | `tbz+b+skip` → `tbnz` (correct polarity) | `c622d6e3` |

## 4. Stability Fixes (must port)

| Fix | Description | Commit |
|---|---|---|
| TypeDeoptPatcher RAII | Destructor + unwatchType prevents crash | `a24c9225` |
| GlobalDeoptPatcher RAII | Safe destructor, move watchGlobal to link | `7434017a`, `717ad2e0` |
| unwatchType + null-check | Context cleanup for type deopt | `689f2f3c` |
| Generator SIGSEGV (aarch64) | Prevent finalizer re-entry | `f377d3d6` |
| Clear deopt patchers before shutdown GC | Prevents dangling pointer SIGSEGV | `b59e322c` |
| Guard getUnitState empty vector | Prevents crash on empty unit_frames | `de8c9140` |
| Lightweight frames + init-ordering | Enable lightweight frames safely | `517d7ac9` |
| Sub4 property setter bug | Deopt test fix | `dde9e44a` |

## 5. Frame Model

- 3.12 uses `_PyInterpreterFrame` (not shadow frames)
- `FrameMode::kLightweight` is the target mode
- PEP 523 bypass in `037bebdf` — `auto()` compilation + eval frame hook

## 6. Key Architectural Decisions

- `auto()` is the entry point for enabling auto-compilation
- Threshold tuning: 1000 calls allows specialization without excessive delay
- IMPORT_NAME exclusion is critical — JIT crashes on import without it
- Deopt backoff prevents infinite compile/deopt cycles on polymorphic code
- All skip decisions cached in CodeExtra for func watcher fast path
