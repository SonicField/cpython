# phoenix-asm API Specification

Pure C assembler library replacing asmjit for the Phoenix JIT.
Dual-architecture: x86_64 and ARM64.

## Design Principle

Match asmjit's Builder API shape so codegen files need minimal changes.
The arch.h abstraction layer maps phoenix-asm types to the same aliases
the JIT already uses (arch::Builder, arch::Gp, arch::Mem, etc.).

## 1. Builder (Code Emitter)

The central object. Emits machine code into a code buffer.

```c
/* Create/destroy */
PhxBuilder* phx_builder_create(PhxCodeHolder* code);
void phx_builder_destroy(PhxBuilder* builder);

/* C++ wrapper for arch.h compatibility */
class Builder {
    PhxBuilder* impl_;
public:
    // Instruction emission (see §3 for full list)
    void mov(Gp dst, Gp src);
    void mov(Gp dst, Mem src);
    void mov(Mem dst, Gp src);
    void mov(Gp dst, uint64_t imm);
    void add(Gp dst, Gp src);
    void add(Gp dst, int32_t imm);
    void sub(Gp dst, Gp src);
    void sub(Gp dst, int32_t imm);
    void cmp(Gp a, Gp b);
    void cmp(Gp a, int32_t imm);
    void call(Gp target);
    void call(void* target);       // absolute call
    void call(Label target);
    void ret();

    // Labels and control flow
    Label newLabel();
    void bind(Label label);
    BaseNode* cursor();            // insertion point management

    // Alignment
    void align(AlignMode mode, int alignment);
};
```

## 2. Operand Types

### Gp (General Purpose Register)

```c
/* C API */
typedef struct { uint8_t id; uint8_t size; } PhxGp;

/* C++ wrapper */
class Gp {
    uint8_t id_;
    uint8_t size_;  // 1, 2, 4, or 8 bytes
public:
    uint8_t id() const;
    uint32_t size() const;
    bool isGp() const { return true; }
    bool isVec() const { return false; }
};
```

### Mem (Memory Operand)

```c
/* C API */
typedef struct {
    PhxGp base;
    PhxGp index;    // optional (x86 SIB)
    int32_t offset;
    uint8_t scale;  // 1, 2, 4, 8 (x86 SIB)
    uint8_t size;   // access size
} PhxMem;

/* Construction helpers */
// x86_64:
Mem ptr(Gp base, int32_t offset);
Mem ptr(Gp base, Gp index, int scale);
Mem dqword_ptr(Gp base);
// ARM64:
Mem ptr_pre(Gp base, int32_t offset);   // pre-indexed
Mem ptr_post(Gp base, int32_t offset);  // post-indexed
Mem ptr_offset(Gp base, int32_t offset, AccessSize size);
```

### Label

```c
typedef struct { uint32_t id; } PhxLabel;
```

### Imm (Immediate Value)

```c
/* Implicit in instruction methods — no separate type needed */
/* mov(dst, uint64_t) handles immediates directly */
```

## 3. Required Instructions

Extracted from actual JIT usage (method call counts from gen_asm.cpp + frame_asm.cpp):

### x86_64

| Instruction | Usage | Description |
|---|---|---|
| mov | 131 | Register/memory/immediate moves |
| add | 19 | Addition |
| sub | 16 | Subtraction |
| lea | 13 | Load effective address |
| cmp | 8 | Compare |
| call | 14 | Function call |
| ret | 8 | Return |
| push | 10 | Push to stack |
| pop | 3 | Pop from stack |
| jmp | 7 | Unconditional jump |
| je/jz | 6 | Jump if equal/zero |
| jne/jnz | - | Jump if not equal |
| test | 2 | Bitwise test |
| xor_ | 1 | XOR |
| movsd | 3 | SSE double move |
| movsx | 2 | Sign-extend move |
| movzx | 3 | Zero-extend move |
| leave | 4 | Stack frame teardown |
| bts | 1 | Bit test and set |
| ptest | 1 | SSE packed test |
| pcmpeqw | 1 | SSE compare |
| psrlq | 1 | SSE shift right |
| align | 5 | Code alignment |
| bind | 46 | Bind label |

### ARM64

| Instruction | Usage | Description |
|---|---|---|
| mov | (shared) | Register/immediate moves |
| add/adds | 7 | Addition (with/without flags) |
| sub | 5 | Subtraction |
| ldr | 25 | Load register |
| str | 24 | Store register |
| ldp | 4 | Load pair |
| stp | 3 | Store pair |
| blr | 10 | Branch with link (call) |
| br | 3 | Branch to register |
| b | 5 | Unconditional branch |
| b_eq | 7 | Branch if equal |
| b_ne | 1 | Branch if not equal |
| b_mi | 1 | Branch if minus |
| ret | (shared) | Return (to LR) |
| cmp | (shared) | Compare |
| cbz | 2 | Compare and branch if zero |
| cbnz | 1 | Compare and branch if not zero |
| adr | 5 | Address of label |
| fmov | 2 | Float move |
| uxtb | 2 | Unsigned extend byte |
| uxth | 1 | Unsigned extend half |
| sxtb | 1 | Signed extend byte |
| sxth | 1 | Signed extend half |
| strb | 2 | Store byte |
| mrs | 2 | Move from system register |
| ldxr | 1 | Load exclusive |
| stxr | 1 | Store exclusive |

## 4. Node-List Architecture (Deferred Assembly)

The JIT emits code non-linearly: the function body is generated first to
determine frame size (spill slots), then the prologue is inserted BEFORE the
body via setCursor(). This requires a deferred assembly model.

Evidence: gen_asm.cpp:2959-2983 — generateCode() saves prologue_cursor,
emits body, saves epilogue_cursor, then setCursor(prologue_cursor) to emit
prologue with the now-known frame size.

### PhxNode (Instruction Node)

```c
typedef struct PhxNode {
    struct PhxNode* prev;
    struct PhxNode* next;
    uint8_t opcode;        /* phoenix-asm internal opcode */
    uint8_t num_operands;
    PhxOperand operands[4]; /* max 4 operands per instruction */
    uint32_t encoded_size;  /* filled during finalize */
    uint32_t offset;        /* filled during finalize */
} PhxNode;
```

### Cursor Management

```c
/* C++ wrapper on Builder */
PhxNode* cursor();                  // current insertion point
void setCursor(PhxNode* node);      // reposition insertion point
// All subsequent emit calls insert AFTER the cursor node
```

### Finalize Pass

phx_code_finalize() linearizes the node list:
1. Walk nodes, compute instruction sizes → set offset fields
2. Resolve labels (fixup forward references using offsets)
3. Encode each node into the output byte buffer

### CodeHolder (Code Buffer)

```c
/* C API */
PhxCodeHolder* phx_code_create(PhxArch arch);
void phx_code_destroy(PhxCodeHolder* code);
void* phx_code_finalize(PhxCodeHolder* code, size_t* out_size);

/* C++ wrapper */
class CodeHolder {
    PhxCodeHolder* impl_;
public:
    void attach(Builder* builder);
    void* baseAddress() const;
    size_t codeSize() const;
};
```

## 5. Register Constants

### x86_64
rax, rbx, rcx, rdx, rsi, rdi, rbp, rsp, r8-r15
xmm0-xmm15

### ARM64
x0-x30, sp
d0-d31 (SIMD/FP)

## 6. Code Allocator Integration

phoenix-asm must provide a finalized code buffer that can be:
1. Copied to executable memory (mmap + mprotect)
2. Queried for label offsets (for debug info)

Replace asmjit::JitRuntime with:
```c
PhxRuntime* phx_runtime_create(void);
void* phx_runtime_add(PhxRuntime* rt, PhxCodeHolder* code, size_t* size);
void phx_runtime_destroy(PhxRuntime* rt);
```

## 7. cursor() / Node API

asmjit's Builder supports insertion point manipulation via cursor().
The JIT uses this to emit code out-of-order (e.g., emit prologue, save
cursor, emit body, restore cursor to insert frame setup between them).

phoenix-asm must support:
```c
PhxNode* phx_builder_cursor(PhxBuilder* b);
void phx_builder_set_cursor(PhxBuilder* b, PhxNode* node);
```

This is the most architecturally complex part — it requires a linked list
of instruction nodes rather than a flat byte buffer.

## 8. Implementation Priority

1. **Core C API** (PhxBuilder, PhxCodeHolder, PhxGp, PhxMem, PhxLabel)
2. **x86_64 backend** (~25 instructions)
3. **ARM64 backend** (~30 instructions)
4. **cursor/Node API** (out-of-order emission)
5. **Code allocator integration**
6. **arch.h adapter** (type aliases mapping phoenix-asm to arch:: names)

## 9. Testing Strategy

The existing JIT is the oracle. For each instruction:
1. Emit with asmjit, capture bytes
2. Emit with phoenix-asm, capture bytes
3. Compare byte-for-byte

Cross-arch verification on devgpu004 (ARM64) and local (x86_64).
