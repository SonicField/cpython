# Phase 3D: C++ to C Conversion Plan

## Terminal Goal
Remove ALL C++ from the JIT. Target: zero C++ files, pure C implementation.
Per Alex's directive (2026-03-31 22:16Z): "The target is zero C++."

## Scope
- 104 C++ files, 57,473 lines across 6 subsystems
- Permanent fork from CinderX (accepted, D-1774996599)
- Phoenix is internal to CPython — use CPython internals where beneficial (D-1775026700)

## Design Principles

### 1. Use CPython Internals (Alex Directive)
| C++ Feature | CPython C Replacement |
|---|---|
| std::vector | PhxArray (PyMem_RawMalloc-based) |
| std::unique_ptr | Manual lifecycle with PyMem_RawFree |
| std::unordered_map | _Py_hashtable_* (pycore_hashtable.h) |
| std::unordered_set | _Py_hashtable_* (keys only) |
| std::atomic | _Py_atomic_* (pyatomic.h) |
| std::thread/mutex | PyThread_* |
| std::string | char* + PyMem_RawMalloc |
| std::optional | Sentinel values or nullable pointers |
| std::variant | Tagged unions |
| Templates | Function pointer tables or macros |
| Virtual dispatch | Function pointer tables |
| std::ostream | FILE* + fprintf |
| Exceptions | Return codes + goto cleanup |

### 2. RAII Replacement: __attribute__((cleanup))
```c
static void phx_array_cleanup(PhxArray *a) { phx_array_free(a); }
#define SCOPED_ARRAY(name) \
    __attribute__((cleanup(phx_array_cleanup))) PhxArray name = {0}
```
GCC+Clang only (Linux target). Gives C-level RAII without goto spaghetti.
Combined with ASAN leak detection on every conversion commit.

### 3. PhxArray Implementation
```c
typedef struct {
    void *items;
    Py_ssize_t len;
    Py_ssize_t capacity;
    size_t item_size;
} PhxArray;
```
Built on PyMem_RawMalloc/RawRealloc/RawFree (GIL-free, safe for any context).

## Conversion Order (8 Phases)

### Phase 1: Codegen Leaf Files (666 lines, 6 files) — EASY
1. `arch.cpp` (82 lines) — no external deps
2. `code_section.cpp` (44 lines) — only config.h
3. `register_preserver.cpp` (192 lines) — only arch.h, PoC exists
4. `gen_asm_utils.cpp` (93 lines) — stateless utils
5. `annotations.cpp` (110 lines) — string handling
6. `copy_graph.cpp` (145 lines) — self-contained data structure

### Phase 2: Codegen Core (4,297 lines, 3 files) — MEDIUM
7. `frame_asm.cpp` (1,191 lines) — depends on Phase 1 + hir/type.h
8. `autogen.cpp` (3,106 lines) — templates (24x), unordered_map for pattern trie
9. `gen_asm.cpp` (3,514 lines) — heaviest includes, std::vector (34x)

### Phase 3: LIR Leaf Files (3,658 lines, 14 files) — MEDIUM
Files: type, symbol_mapping, dce, verify, rewrite, cold_block_marker,
block_builder, printer, c_helper_translations, blocksorter, function,
block, instruction, operand

### Phase 4: LIR Core (6,450 lines, 6 files) — MEDIUM-HARD
Files: generator.cpp (3,708), regalloc.cpp (1,652), postalloc.cpp (1,090),
parser.cpp (587), postgen.cpp (429), inliner.cpp (368)
Note: parser.cpp uses std::regex (15 patterns) — needs C regex or manual parsing

### Phase 5: HIR Core Exports (2,666 lines, 5 files) — HARD
Files: hir.cpp (1,352), type.cpp (737), analysis.cpp (577),
function.cpp (113), cfg.cpp (175)
These are consumed by codegen/ — must maintain C-compatible interfaces.

### Phase 6: HIR Passes (17,291 lines, 25 files) — HARD
Files: builder.cpp (6,017), simplify.cpp (2,552), refcount_insertion.cpp (1,365),
parser.cpp (1,329), printer.cpp (956), pass.cpp (862), inliner.cpp (698),
preload.cpp (566), instr_effects.cpp (562), ssa.cpp (483), + 15 smaller files
Heaviest STL usage: vector (79x), unordered_map (27x), unordered_set (19x)

### Phase 7: Top-Level JIT Files (17,826 lines, 34 files) — MEDIUM-HIGH
Key files: pyjit.cpp (4,139), jit_rt.cpp (2,593), inline_cache.cpp (1,708),
generators_rt.cpp (975), context.cpp (827)
pyjit.cpp is the PUBLIC API (34 exported functions) — must maintain C linkage

### Phase 8: ELF + Wrapper Removal (1,619 lines, 9 files) — LAST
ELF: writer.cpp (473), reader.cpp (165), + 5 small files
Wrapper: asmjit_compat.h (~270 lines), phoenix_asm_wrapper.h (~1,100 lines)
Wrapper removed LAST — it cannot go until gen_asm.cpp/autogen.cpp are converted

## Prerequisites (All Met)
- [x] ARM64 crashes resolved (21/21 benchmarks pass)
- [x] ARM64 encoding corpus (25/25 tests, preserves wrapper knowledge)
- [x] Gate-before-push mechanized (CLAUDE.md rule, diff excerpts required)
- [x] Phase 3D dependency map (no circular dependencies)
- [x] CPython internals decision (PyMem_RawMalloc, __attribute__((cleanup)))
- [ ] Float unboxing fix must land BEFORE lir/generator.cpp conversion (Phase 4)

## Verification Gates (Per Phase)
1. ASAN clean (leak detection enabled)
2. x86_64 test suite: 459/480+ (zero JIT regressions)
3. ARM64 test suite: 457/483+ (zero JIT regressions)
4. ARM64 encoding corpus: 25/25 pass
5. Gatekeeper review with diff excerpts before push

## Key Risks
1. **RAII→manual cleanup**: Mitigated by __attribute__((cleanup)) + ASAN
2. **Gate throughput**: Batch reviews by phase (6 leaf files = 1 review)
3. **Template→function pointers**: autogen.cpp has 24 template uses
4. **std::regex in lir/parser.cpp**: Need C regex or manual parsing
5. **Interface propagation**: Each converted file may touch 2+ caller interfaces
