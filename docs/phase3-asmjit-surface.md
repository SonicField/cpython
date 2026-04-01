# Phase 3: asmjit API Surface for phoenix-asm Replacement

272 asmjit references across 20 files. This document maps the replacement surface.

## Abstraction Layer: arch.h

`Python/jit/codegen/arch.h` is the central abstraction point. It aliases asmjit
types to architecture-neutral names:

| arch:: alias | x86_64 (asmjit) | ARM64 (asmjit) |
|---|---|---|
| Builder | asmjit::x86::Builder | asmjit::a64::Builder |
| Emitter | asmjit::x86::Emitter | asmjit::a64::Emitter |
| Gp | asmjit::x86::Gp | asmjit::a64::Gp |
| Mem | asmjit::x86::Mem | asmjit::a64::Mem |
| Reg | asmjit::x86::Reg | asmjit::a64::Reg |
| VecD | asmjit::x86::Xmm | asmjit::a64::Vec |

**Strategy:** phoenix-asm replaces asmjit behind this abstraction layer.
Most JIT code uses `arch::Builder`, `arch::Gp`, etc. — NOT direct asmjit types.

## File-by-File Breakdown

### Tier 1: Core codegen (must replace)

| File | asmjit refs | Role |
|---|---|---|
| codegen/gen_asm.cpp | 15 | Main code generator — emits all instructions |
| codegen/autogen.cpp | 44 | Auto-generated opcode implementations |
| codegen/autogen.h | 9 | Auto-generated declarations |
| codegen/frame_asm.cpp | 27 | Frame prologue/epilogue generation |
| codegen/register_preserver.cpp | 46 | Register save/restore (push/pop/stp/ldp) |
| codegen/environ.h | 14 | Compilation environment (holds Builder ref) |
| codegen/arch.h | 35 | Architecture abstraction (type aliases) |

### Tier 2: Infrastructure (must replace)

| File | asmjit refs | Role |
|---|---|---|
| code_allocator.cpp | 26 | Allocates executable memory for JIT code |
| code_allocator.h | 9 | CodeHolder and JitRuntime management |
| code_allocator_iface.h | 4 | Interface for code allocation |

### Tier 3: Metadata (light touch)

| File | asmjit refs | Role |
|---|---|---|
| codegen/annotations.cpp | 2 | Code annotations (uses CodeHolder for offsets) |
| codegen/annotations.h | 9 | Annotation types |
| codegen/code_section.h | 3 | Code section management |
| codegen/code_section.cpp | 1 | Code section impl |
| debug_info.h | 3 | Debug info (Label references) |
| debug_info.cpp | 1 | Debug info resolution |
| codegen/gen_asm_utils.h | 1 | Utility declarations |
| codegen/gen_asm_utils.cpp | 6 | Utility implementations |
| lir/postgen.cpp | 3 | Post-generation passes |

## asmjit API Categories Used

### 1. Code Generation (Builder/Emitter pattern)
- `as_->mov(dst, src)`, `as_->add(dst, imm)`, etc.
- `as_->bind(label)`, `as_->newLabel()`
- `as_->call(target)`, `as_->ret()`
- Conditional branches: `as_->je(label)`, `as_->jne(label)`
- ARM64: `as_->ldr()`, `as_->str()`, `as_->stp()`, `as_->ldp()`

### 2. Operand Construction
- `asmjit::x86::ptr(base, offset)` — memory operands
- `asmjit::a64::ptr_pre(sp, -16)` — pre-indexed addressing
- `asmjit::Imm(value)` — immediate values
- Register references: `asmjit::x86::rax`, `asmjit::a64::x0`, etc.

### 3. Code Management
- `asmjit::CodeHolder` — holds generated code before finalization
- `asmjit::JitRuntime` — manages executable memory
- `asmjit::Environment` — target architecture description
- `asmjit::Label` — branch targets

### 4. Register Metadata
- `reg.size()`, `reg.isGp()`, `reg.isVec()`
- Register type casting: `static_cast<asmjit::x86::Gpq&>(reg)`

## Replacement Strategy

1. **arch.h is the seam.** Replace asmjit type aliases with phoenix-asm types.
   Most codegen code will need zero changes if phoenix-asm provides the same
   API shape (Builder with mov/add/sub/call/ret/bind/label methods).

2. **CodeHolder/JitRuntime replacement.** phoenix-asm needs its own code buffer
   and finalization mechanism. code_allocator.cpp is the integration point.

3. **Operand construction.** phoenix-asm must provide `ptr(base, offset)`,
   `Imm(value)`, and register constants matching asmjit's API.

4. **autogen.cpp is the largest file** (44 refs). It's auto-generated from
   opcode definitions — may need regeneration with phoenix-asm API.

## Estimated Scope

- 20 files to modify
- 272 asmjit references to replace
- arch.h abstraction means ~80% of changes are in 7 Tier 1 files
- code_allocator.cpp is the second major integration point
- Tier 3 files need mostly type renames (Label, CodeHolder)
