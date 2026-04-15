# Phase 3D: Core File Conversion Plan

## Status: DEEP ANALYSIS COMPLETE

7 sub-agents read all core files in parallel. This document contains the
mechanistic conversion recipes the generalist follows for every file.

---

## Part 0: C→C++ Bridge Wrapping Rules (LEARNED 2026-04-14)

Every C factory function passes through a C++ bridge (hir_c_api.cpp) that
calls C++ constructors. C uses raw scalars; C++ uses wrapper types. These
wrappers MUST be applied at every bridge call:

| C Type | C++ Type | Bridge Fix |
|--------|----------|------------|
| `int32_t` (bytecode offset) | `BCOffset` | `BCOffset{offset}` |
| `HirType` (16-byte bitfield) | `Type` | `*reinterpret_cast<Type*>(&hir_type)` |
| `int32_t` (opcode enum) | `Opcode` | `static_cast<Opcode>(op)` |
| `void*` (PyObject pointer) | `BorrowedRef<T>` | `BorrowedRef<T>{ptr}` |

**Include requirement:** hir_c_api.cpp must `#include "cinderx/Jit/bytecode_offsets.h"`
for BCOffset, and `#include "cinderx/Jit/hir/type.h"` for Type.

This pattern was discovered during the first factory integration (288f779698).
It will recur for every factory bridge function — apply mechanically.

---

## Part 1: The Seven Patterns

Every C++ construct in the 7 core files falls into one of seven patterns.
Each pattern has a SINGLE mechanical conversion recipe. Learn these seven
recipes and the entire conversion becomes repetitive application.

### Pattern 1: emit<T>() / appendInstr() Template Calls

**What it is:** Template-based instruction creation used 600+ times across
builder.cpp (200+), simplify.cpp (241), generator.cpp (160+).

**Three variants with identical conversion strategy:**

```cpp
// builder.cpp — TranslationContext::emit<T>()
auto instr = tc.emit<BinaryOp>(result, op_kind, left, right, tc.frame);

// simplify.cpp — Env::emit<T>()  
Register* out = env.emit<BinaryOp>(result, op_kind, left, right);

// generator.cpp — BasicBlockBuilder::appendInstr()
bbb.appendInstr(hir_reg, Instruction::kAdd, left, right);
```

**C replacement — HIR (builder + simplify):**

Step 1: Create C factory function per opcode in `hir_instr_c.h`:
```c
static inline HirInstr* hir_c_create_binary_op(
    HirBasicBlock* block, BCOffset offset,
    HirRegister* output, int op_kind,
    HirRegister* left, HirRegister* right) {
    HirInstr* instr = hir_c_alloc_instr(HIR_OP_BINARY_OP, 2 /*num_operands*/);
    instr->opcode_ = HIR_OP_BINARY_OP;
    hir_c_set_output(instr, output);
    hir_c_set_operand(instr, 0, left);
    hir_c_set_operand(instr, 1, right);
    instr->data.binary_op.op_kind = op_kind;
    hir_c_block_append_with_offset(block, instr, offset);
    return instr;
}
```

Step 2: Each `tc.emit<BinaryOp>(...)` call becomes:
```c
HirInstr* instr = hir_c_create_binary_op(
    tc.block, tc.frame.instr_offset,
    result, op_kind, left, right);
```

Step 3: `emitChecked<T>()` wraps: factory call + `hir_c_create_check_exc()`.
Step 4: `emitVariadic<T>()` wraps: allocate output + factory call + loop
setting operands from stack + push output.

**C replacement — LIR (generator.cpp):**

The LIR `appendInstr` already has C struct types. Convert to:
```c
LirInstruction* instr = lir_c_alloc_instr(block, JIT_LIR_OP_ADD, origin);
lir_c_set_output_vreg(instr, hirTypeToDataType(dest->type()));
lir_c_add_linked_input(instr, lir_c_get_def(env, left));
lir_c_add_linked_input(instr, lir_c_get_def(env, right));
env->output_map_set(env, dest, instr);
```

**Effort calibration:** The Branch factory prototype (5318f448c4) took ~30
minutes. Most opcodes are similar or simpler. Complex opcodes (VectorCall,
BeginInlinedFunction) take 1-2 hours each. Estimate: 2 hours infrastructure
+ 0.5 hours per simple factory + 1.5 hours per complex factory.

**Opcode factory inventory (from sub-agent analysis):**

| Category | Count | Avg Lines | Total Hours |
|----------|-------|-----------|-------------|
| Trivial (1 field, 0-1 operands) | 42 | 8 | 3 |
| Simple (2-3 fields, 1-2 operands) | 31 | 15 | 5 |
| Moderate (fields + deopt state) | 18 | 25 | 6 |
| Complex (variadic, FrameState, multi-block) | 9 | 50 | 9 |
| **Total** | **100** | | **23** |

### Pattern 2: static_cast to Concrete HIR/LIR Type

**What it is:** 280+ sites across all files. C++ downcasts to access
opcode-specific instruction data.

```cpp
auto* bin_op = static_cast<const BinaryOp*>(&instr);
int kind = bin_op->op();
Register* left = bin_op->left();
```

**C replacement:**
```c
assert(instr->opcode_ == HIR_OP_BINARY_OP);
int kind = instr->data.binary_op.op_kind;
HirRegister* left = hir_c_get_operand(instr, 0);
```

**Mechanical rule:** Every `static_cast<const T*>` becomes an assert on
opcode + direct union field access. The operand layout (which operand index
maps to which named field) is defined once per opcode in a comment block in
hir_instr_c.h:

```c
/* BinaryOp operand layout:
 *   operand[0] = left
 *   operand[1] = right
 *   output = result
 *   data.binary_op.op_kind = operation
 */
```

### Pattern 3: BorrowedRef<T> / Ref<T> Smart Pointers

**What it is:** ~100 sites. BorrowedRef is a zero-cost non-owning wrapper.
Ref is an owning wrapper that calls Py_INCREF/DECREF.

```cpp
BorrowedRef<PyFunctionObject> func = preloader_.code();
Ref<PyObject> result = Ref<>::steal(PyObject_Call(...));
```

**C replacement:**
```c
PyFunctionObject* func = preloader_code(preloader);    // BorrowedRef → raw ptr
PyObject* result = PyObject_Call(...);                   // Ref → raw ptr
// ... use result ...
Py_XDECREF(result);                                     // Manual cleanup
```

**Mechanical rule:**
- `BorrowedRef<T> x = expr` → `T* x = expr` (no refcount change)
- `Ref<T> x = Ref<>::steal(expr)` → `T* x = expr` + `Py_XDECREF(x)` at scope exit
- `Ref<T> x = Ref<>::create(expr)` → `T* x = expr; Py_INCREF(x)` + `Py_XDECREF(x)` at scope exit
- Use `__attribute__((cleanup))` for RAII-like scope exit:
```c
static inline void py_decref_cleanup(PyObject** p) { Py_XDECREF(*p); }
#define OWNED_REF __attribute__((cleanup(py_decref_cleanup)))
OWNED_REF PyObject* result = PyObject_Call(...);
```

### Pattern 4: std::function Callbacks

**What it is:** ~20 sites. Used for visitUses, pass dispatch, guard failure
callbacks, and operand type getters.

```cpp
bool visitUses(const std::function<bool(Register*&)>& func);
```

**C replacement:**
```c
typedef bool (*HirVisitUsesFn)(HirRegister**, void* ctx);
bool hir_c_visit_uses(HirInstr* instr, HirVisitUsesFn fn, void* ctx);
```

**Mechanical rule:** Every `std::function<R(Args...)>` becomes a C function
pointer typedef + void* context parameter. The context carries any captured
state. If the callback captures nothing, ctx is NULL.

**Specific conversions:**
- `visitUses(std::function<bool(Register*&)>)` → `hir_c_visit_uses(instr, fn, ctx)`
- `PostPassFunction` → `typedef void (*PassCallback)(HirFunction*, const char* name, size_t idx, void* ctx)`
- `GuardFailureCallback` → `typedef void (*GuardFailFn)(const DeoptMetadata*, void* ctx)`
- `Pass::Run(Function&)` → `typedef void (*PassRunFn)(HirFunction*)`

### Pattern 5: Container Replacement

**What it is:** std::vector, std::unordered_map, std::unordered_set,
std::deque, std::priority_queue — ~200 sites across all files.

**C replacements (all proven in LIR conversion):**

| C++ | C | Notes |
|-----|---|-------|
| `std::vector<T>` | `PhxArray` or `T* + count + capacity` | PhxArray for dynamic; raw for fixed-size |
| `std::unordered_map<K,V>` | `_Py_hashtable` | CPython internal, GIL-protected |
| `std::unordered_set<K>` | `_Py_hashtable` (value=1) | Same API, value ignored |
| `std::deque<T>` | Ring buffer or `T* + head + tail` | Only used in builder.cpp worklist |
| `std::priority_queue<T>` | Binary heap array | Only used in regalloc.cpp |
| `std::set<T>` | Sorted array + binary search | For ordered iteration |
| `std::optional<T>` | `T value; bool has_value;` or nullable pointer | Depends on T |
| `std::string` | `char* + PyMem_RawMalloc` | Rare; mostly for logging |
| `std::unique_ptr<T>` | `T* + manual free` | Use cleanup attribute for scope |

**Mechanical rule:** Identify the container, check if iteration order matters
(use sorted array for ordered, hashtable for unordered), allocate with
PyMem_RawMalloc (GIL-free), free at scope exit.

### Pattern 6: FrameState Management (builder.cpp only)

**What it is:** The abstract interpreter state carried through bytecode
translation. Created, cloned, and linked via parent pointers for inlining.

```cpp
struct FrameState {
    BCOffset cur_instr_offs;
    std::vector<Register*> localsplus;
    int nlocals;
    OperandStack stack;          // Stack<Register*>
    BlockStack block_stack;      // exception/loop blocks
    BorrowedRef<PyCodeObject> code;
    BorrowedRef<PyDictObject> globals, builtins;
    FrameState* parent;          // inlining chain
};
```

**C replacement:**
```c
typedef struct HirFrameState {
    int32_t cur_instr_offs;
    HirRegister** localsplus;
    int nlocals;
    int localsplus_count;
    HirRegister** stack;         // operand stack (array)
    int stack_top;
    int stack_capacity;
    HirExecBlock* block_stack;   // exception blocks (array)
    int block_stack_top;
    PyCodeObject* code;          // borrowed
    PyDictObject* globals;       // borrowed
    PyDictObject* builtins;      // borrowed
    struct HirFrameState* parent;
} HirFrameState;
```

**Operations:**
- `frame.stack.push(reg)` → `frame->stack[frame->stack_top++] = reg`
- `frame.stack.pop()` → `frame->stack[--frame->stack_top]`
- Clone: `HirFrameState* clone = hir_frame_state_clone(frame)` (deep copy of arrays)
- Free: `hir_frame_state_free(frame)` (free arrays, then struct)
- Parent chain: unchanged (raw pointer already)

**Critical invariant:** The localsplus array size is fixed at function entry
(nlocals + ncells + nfrees). Stack capacity is bounded by co_stacksize.
Block stack is bounded by exception nesting depth.

### Pattern 7: Conditional Compilation (#ifdef CINDER_X86_64 / CINDER_AARCH64)

**What it is:** Pervasive in gen_asm.cpp and generator.cpp. Architecture-
specific code gated by preprocessor conditionals.

```cpp
#if defined(CINDER_X86_64)
    phx_x86_mov_rr(as->impl(), x86::rax, x86::rbx);
#elif defined(CINDER_AARCH64)
    phx_a64_mov_rr(as->impl(), a64::x0, a64::x1);
#endif
```

**C replacement:** Keep `#ifdef` blocks as-is. C supports preprocessor
conditionals identically. The phoenix_asm C API is already pure C. Convert
`as->impl()` to a raw `PhxBuilder*` parameter:

```c
void gen_asm_c_function_entry(PhxBuilder* as) {
#if defined(CINDER_X86_64)
    phx_x86_push_r(as, PHX_RBP);
    phx_x86_mov_rr(as, PHX_RBP, PHX_RSP);
#elif defined(CINDER_AARCH64)
    phx_a64_stp_pre(as, PHX_FP, PHX_LR, PHX_SP, -16);
    phx_a64_mov_rr(as, PHX_FP, PHX_SP);
#endif
}
```

**Mechanical rule:** The C++ wrapper `as_->impl()` becomes a `PhxBuilder*`
parameter. Register namespace constants (`x86::rax`) become C defines
(`PHX_RAX`). Memory operands (`x86::ptr(base, off)`) become C structs
(`phx_mem(base, off)`). Labels stay as `PhxLabel`.

---

## Part 2: File-by-File Conversion Order and Recipes

### Phase I: Infrastructure (gates everything else)

#### I-1: hir.cpp Methods → C Functions (3-4 hours)

**File:** 1418 lines, 125 C++ features.
**Strategy:** Move method bodies to C functions in hir_instr_c.h/c.
Keep C++ class as thin wrapper delegating to C functions.

**Method inventory (from sub-agent analysis, exact line numbers):**

| Method | Lines | Pattern | Status |
|--------|-------|---------|--------|
| `Instr::NumOperands()` | 143-146 | Read preamble | DONE (958495c3c8) |
| `Instr::GetOperand()` | 148-152 | Read preamble | DONE (958495c3c8) |
| `Instr::SetOperand()` | 154-160 | Write preamble | DONE (958495c3c8) |
| `Instr::setOutput()` | 162-170 | Write field | DONE (5318f448c4) |
| `Instr::output()` | 138-141 | Read field | DONE (5318f448c4) |
| `DeoptBase::visitUsesDeopt()` | 57-74 | Pattern 4 (callback) | TODO |
| `DeoptBase::live_regs()` | 46-52 | Read field (vector) | TODO |
| `DeoptBase::setFrameState()` | 35-44 | Write field (unique_ptr) | TODO |
| `Instr::visitUses()` | 76-130 | Pattern 4 + Pattern 2 (switch) | TODO |
| `Instr::Destroy()` | 132-135 | Pattern 2 (switch on opcode) | TODO |
| `Instr::clone()` | 170-250 | Pattern 2 (switch) + alloc | TODO |
| `Instr::isTerminator()` | 252-280 | Switch on opcode | TODO |
| `Instr::isAnyBranch()` | 282-310 | Switch on opcode | TODO |

**Conversion recipe for remaining methods:**

1. `visitUses()` → C function with callback:
   ```c
   bool hir_c_visit_uses(HirInstr* instr, HirVisitUsesFn fn, void* ctx);
   ```
   Body: switch on opcode, for each operand index call `fn(&operands[i], ctx)`.
   DeoptBase variant also visits frame registers + live_regs + guilty_reg.

2. `Destroy()` → C function:
   ```c
   void hir_c_destroy(HirInstr* instr);
   ```
   Body: switch on opcode, free opcode-specific resources (FrameState, vectors).
   Then free the operand preamble + instruction memory.

3. `clone()` → C function:
   ```c
   HirInstr* hir_c_clone(const HirInstr* instr);
   ```
   Body: alloc new instruction with same opcode + operand count.
   Copy operand preamble. Switch on opcode to copy opcode-specific data.
   Deep-copy FrameState if DeoptBase.

4. `isTerminator()` / `isAnyBranch()` → lookup table:
   ```c
   static const uint8_t hir_opcode_flags[HIR_NUM_OPCODES] = {
       [HIR_OP_BRANCH] = HIR_FLAG_TERMINATOR | HIR_FLAG_BRANCH,
       [HIR_OP_COND_BRANCH] = HIR_FLAG_TERMINATOR | HIR_FLAG_BRANCH,
       [HIR_OP_RETURN] = HIR_FLAG_TERMINATOR,
       // ...
   };
   #define hir_c_is_terminator(instr) (hir_opcode_flags[(instr)->opcode_] & HIR_FLAG_TERMINATOR)
   ```

**Gate:** Each method gets an assertion wrapper (C result == C++ result)
validated by the full test suite before the C++ body is removed.

#### I-2: C Instruction Factory Functions (4-6 hours)

**What:** Create `hir_c_create_<opcode>()` for each HIR opcode used in
builder.cpp and simplify.cpp.

**Approach:** Generate factories in order of usage frequency.

**Top 20 by usage (from builder.cpp + simplify.cpp):**

| Factory | Uses | Operands | Has DeoptState |
|---------|------|----------|----------------|
| `hir_c_create_branch` | 40 | 0 + target | No |
| `hir_c_create_cond_branch` | 25 | 1 + 2 targets | No |
| `hir_c_create_binary_op` | 20 | 2 + kind | Yes |
| `hir_c_create_guard_type` | 18 | 1 + type | Yes |
| `hir_c_create_check_exc` | 15 | 1 | Yes |
| `hir_c_create_load_const` | 12 | 0 + value | No |
| `hir_c_create_compare` | 10 | 2 + op | Yes |
| `hir_c_create_load_attr` | 8 | 1 + name_idx | Yes |
| `hir_c_create_store_attr` | 6 | 2 + name_idx | Yes |
| `hir_c_create_call_static` | 6 | variadic | Yes |
| `hir_c_create_return` | 6 | 1 | No |
| `hir_c_create_snapshot` | 5 | 0 + frame | No |
| `hir_c_create_is_truthy` | 5 | 1 | Yes |
| `hir_c_create_load_field` | 5 | 1 + offset | No |
| `hir_c_create_unary_op` | 4 | 1 + kind | Yes |
| `hir_c_create_vector_call` | 4 | variadic | Yes |
| `hir_c_create_incref` | 4 | 1 | No |
| `hir_c_create_decref` | 4 | 1 | No |
| `hir_c_create_assign` | 3 | 1 | No |
| `hir_c_create_phi` | 3 | variadic | No |

**Template for factory function:**
```c
HirInstr* hir_c_create_binary_op(
    HirBasicBlock* block, BCOffset offset, HirRegister* output,
    int op_kind, HirRegister* left, HirRegister* right,
    const HirFrameState* deopt_frame)
{
    HirInstr* instr = hir_c_alloc_instr(HIR_OP_BINARY_OP, 2);
    hir_c_set_output(instr, output);
    hir_c_set_operand(instr, 0, left);
    hir_c_set_operand(instr, 1, right);
    instr->data.binary_op.op_kind = op_kind;
    if (deopt_frame) {
        hir_c_set_deopt_state(instr, deopt_frame);
    }
    hir_c_block_append_with_offset(block, instr, offset);
    return instr;
}
```

**Variadic factories** (for CALL, VECTOR_CALL, etc.):
```c
HirInstr* hir_c_create_vector_call(
    HirBasicBlock* block, BCOffset offset, HirRegister* output,
    int num_args, HirRegister** args,
    const HirFrameState* deopt_frame)
{
    HirInstr* instr = hir_c_alloc_instr(HIR_OP_VECTOR_CALL, num_args);
    hir_c_set_output(instr, output);
    for (int i = 0; i < num_args; i++) {
        hir_c_set_operand(instr, i, args[i]);
    }
    if (deopt_frame) {
        hir_c_set_deopt_state(instr, deopt_frame);
    }
    hir_c_block_append_with_offset(block, instr, offset);
    return instr;
}
```

#### I-3: Env / TranslationContext Emit Helpers → C (2-3 hours)

**What:** Convert the emit infrastructure in builder.cpp and simplify.cpp.

**builder.cpp TranslationContext:**

Current (C++):
```cpp
struct TranslationContext {
    BasicBlock* block;
    FrameState frame;
    template <typename T, typename... Args>
    T* emit(Args&&... args) {
        return block->appendWithOff<T>(frame.instrOffset(), std::forward<Args>(args)...);
    }
};
```

C replacement:
```c
typedef struct {
    HirBasicBlock* block;
    HirFrameState frame;
} HirTranslationContext;

// No generic emit — each opcode has its own factory (Pattern 1).
// The TranslationContext just carries block + frame.
```

**simplify.cpp Env:**

Current (C++):
```cpp
struct Env {
    Function& func;
    BasicBlock* block;
    Instr::List::iterator cursor;
    template <typename T, typename... Args>
    Register* emit(Args&&... args) { ... }
    template <typename T, typename... Args>
    T* emitInstr(Args&&... args) { ... }
    template <typename T, typename... Args>
    Register* emitVariadic(std::size_t n, Args&&... args) { ... }
    template <typename T, typename... Args>
    T* emitRawInstr(Args&&... args) { ... }
};
```

C replacement:
```c
typedef struct {
    HirFunction* func;
    HirBasicBlock* block;
    HirInstr* cursor;  // Insert before this instruction
    PyTypeObject* type_object;  // cached from func
} HirSimplifyEnv;

// emit → factory call + insert before cursor + set output type
// emitInstr → same but return instruction (not register)
// emitRawInstr → factory + set output type (no insert — caller does it)
```

**generator.cpp BasicBlockBuilder:**

Already partially C (LIR C API exists). Convert remaining template methods:
```c
typedef struct {
    LirFunction* func;
    LirBasicBlock* cur_bb;
    const void* cur_hir_instr;
    JitEnviron* env;
} LirBlockBuilder;

LirInstruction* lir_bb_append_instr(LirBlockBuilder* bbb,
    int opcode, ...);  // Variadic C function
LirInstruction* lir_bb_append_call(LirBlockBuilder* bbb,
    void* func_ptr, int num_args, ...);  // For appendCallInstruction
```

### Phase II: Mechanical Case Conversion (parallelizable)

After Phase I creates the factory functions and C env structs, each opcode
case in builder/simplify/generator converts independently using the same
recipe.

#### II-1: simplify.cpp Case Conversion (6-8 hours)

**37 switch cases.** Each follows one of 4 patterns:

**Pattern A — Type check + emit replacement (22 cases):**
```cpp
// C++
case Opcode::kBinaryOp: {
    auto* instr = static_cast<const BinaryOp*>(orig);
    if (instr->left()->type() <= TLongExact && ...) {
        return env.emit<LongBinaryOp>(instr->op(), instr->left(), instr->right());
    }
    return nullptr;
}
```
```c
// C
case HIR_OP_BINARY_OP: {
    assert(orig->opcode_ == HIR_OP_BINARY_OP);
    HirRegister* left = hir_c_get_operand(orig, 0);
    HirRegister* right = hir_c_get_operand(orig, 1);
    if (hir_type_le(left->type, HIR_TYPE_LONG_EXACT) && ...) {
        return hir_simplify_emit(&env, hir_c_create_long_binary_op(
            env.block, orig->bc_offset,
            hir_c_alloc_register(env.func),
            orig->data.binary_op.op_kind, left, right, NULL));
    }
    return NULL;
}
```

**Pattern B — Multi-instruction rewrite (8 cases):**
Emit 2-5 replacement instructions. Same factory calls, sequenced.

**Pattern C — Python C API type specialization (5 cases):**
Call PyType_Check, PyDict_GetItem, etc. These are already C calls — no
conversion needed for the Python API part.

**Pattern D — Complex control flow / emitCond (2 cases):**
`emitCond` creates 3 new basic blocks (true/false/merge). Convert to:
```c
HirBasicBlock* true_bb = hir_c_alloc_block(env.func);
HirBasicBlock* false_bb = hir_c_alloc_block(env.func);
HirBasicBlock* merge_bb = hir_c_alloc_block(env.func);
hir_c_create_cond_branch(env.block, ..., true_bb, false_bb);
// ... emit into true_bb, false_bb, merge_bb ...
```

**Conversion order:** Pattern A cases first (mechanical), then B, then C,
then D. Each case is one commit, tested independently.

#### II-2: builder.cpp Case Conversion (12-16 hours)

**196 switch cases** grouped by complexity tier.

**Tier 1 — Trivial cases (67 opcodes, ~4 hours):**
Single emit call, no control flow. Examples: NOP, LOAD_CONST, POP_TOP,
STORE_FAST. Convert all in batches of ~10 per commit.

**Tier 2 — Moderate cases (90 opcodes, ~6 hours):**
2-5 emit calls, simple branching. Examples: BINARY_ADD, BUILD_LIST,
COMPARE_OP, JUMP_IF_FALSE_OR_POP. Convert individually or in small groups.

**Tier 3 — Complex cases (39 opcodes, ~6 hours):**
FrameState manipulation, multiple blocks, inlining. Each needs individual
attention. The 5 hardest:

1. **CALL/CALL_FUNCTION family (9 opcodes)** — emitAnyCall: 100+ lines,
   awaited call branching, exception handling. Convert as a standalone C
   function `hir_builder_emit_any_call()`.

2. **YIELD_VALUE/YIELD_FROM/GET_AWAITABLE (7 opcodes)** — generator state
   machine. Convert as `hir_builder_emit_yield()`.

3. **FOR_ITER** — creates CondBranchIterNotDone with two FrameStates.

4. **SETUP_FINALLY/SETUP_WITH** — exception block stack.

5. **MATCH_CLASS/MATCH_KEYS** — multi-block pattern matching.

**Key builder.cpp infrastructure to convert FIRST (before any cases):**

1. `TranslationContext` struct (Pattern 6 — FrameState)
2. `TempAllocator` (register allocation) → C function
3. `BlockMap` (BC offset → IR block) → `_Py_hashtable`
4. `PendingBlock` queue (BFS worklist) → C ring buffer
5. `ExceptionTable` parsing → C array
6. `isSupportedOpcode()` switch → lookup table

#### II-3: generator.cpp Case Conversion (8-12 hours)

**162 switch cases.** Most follow the same appendInstr/appendCallInstruction
pattern.

**Trivial (80 cases):** 1-2 LIR instructions. Direct appendInstr → C call.
**Moderate (62 cases):** 3-5 LIR instructions, register juggling.
**Complex (20 cases):** Multiple blocks, deopt frames, version-gated code.

**The 5 hardest cases in generator.cpp:**

1. **kIntBinaryOp** (160 lines) — nested switch over 10+ ops, conditional
   helpers, extension instructions.
2. **kBeginInlinedFunction** (220 lines) — frame setup, version-gated,
   10+ LIR instructions.
3. **kEndInlinedFunction** (140 lines) — frame teardown.
4. **kVectorCall** (30 lines) — specialized call dispatch.
5. **kCallInd** (30 lines) — indirect call + guard setup.

**Generator-specific infrastructure to convert FIRST:**

1. `BasicBlockBuilder` struct → `LirBlockBuilder` (C struct)
2. `appendInstr` template dispatch → C function with explicit operand types
3. `appendCallInstruction` template → C function with function pointer
4. `getDefInstr(hir_reg)` → `lir_c_get_def(env, reg)` lookup
5. `MakeIncref/MakeDecref` helpers → C functions

### Phase III: Codegen + Infrastructure

#### III-1: gen_asm.cpp (8-12 hours)

**3687 lines, 37 methods.** Already partially converted (frame_asm_c,
autogen_translate_c, postgen_c, postalloc_c).

**What remains:** Prologue, epilogue, deopt exits, register save/restore,
argument validation, static entry points, generator resume.

**Strategy:** Convert per-method. Each method is an assembly emission
function that calls the phoenix_asm C API directly (already C). The only
C++ wrapper is `as_->impl()` → `PhxBuilder*`.

**Conversion order (from sub-agent analysis):**

| Priority | Methods | Lines | Hours |
|----------|---------|-------|-------|
| P0 | generateFunctionEntry/Exit | 23 | 0.5 |
| P1 | allocateHeaderAndSpill, saveCallerRegs | 85 | 1 |
| P2 | computeFrameInfo | 66 | 1 |
| P3 | generateArgcountCheck | 113 | 1.5 |
| P4 | generatePrimitiveArgsPrologue | 35 | 0.5 |
| P5 | generateStaticMethodTypeChecks | 234 | 2 |
| P6 | generateDeoptExits | 180 | 2 |
| P7 | generatePrologue | 217 | 2 |
| P8 | generateEpilogue | 203 | 2 |
| P9 | generateResumeEntry | 149 | 1.5 |
| P10 | generateBoxedReturnWrapper | 144 | 1 |
| P11 | generateStaticEntryPoint | 265 | 2 |
| P12 | linkDeoptPatchers | 30 | 0.5 |
| P13 | generateCode orchestrator | 186 | 1 |
| P14 | getVectorcallEntry orchestrator | 236 | 1 |

**Critical dep:** P6 (deopt exits) must match the stack contract expected
by deopt.cpp's global trampoline. Test with existing deopt test cases.

#### III-2: regalloc.cpp (6-8 hours)

**1652 lines, linear scan register allocation.**

**Key data structures to convert:**

1. `LiveRange` struct → C struct (trivial, 2 ints)
2. `LiveInterval` → C struct with `PhxArray<LiveRange>` + PhyLocation
3. `RegallocBlockState` → C struct
4. Priority queue → manual binary heap
5. Active/inactive sets → simple linked lists or arrays (small, <32 entries)
6. `intervals_` map → `_Py_hashtable` keyed by Operand*
7. `vreg_phy_uses_` → `_Py_hashtable` of sorted arrays

**Algorithm is pure computation** — no Python C API, no template
metaprogramming, no virtual dispatch. The C++ features are ONLY in the
container types. Once containers are replaced, the algorithm translates
line-by-line.

**Risk:** The splitting/merging logic (addRange, setFrom, splitAt) is
subtle. Must maintain binary-search invariants on the ranges vector.
Test with edge cases: empty intervals, single-point ranges, adjacent
ranges that should merge, splits at range boundaries.

#### III-3: pass.cpp + compiler.cpp + context.cpp (6-8 hours)

**pass.cpp** — `outputType()` is 450 lines of switch-on-opcode type
deduction. Mechanical Pattern 2 conversion. `reflowTypes()` is standard
fixed-point iteration. `removeTrampolineBlocks()` and
`removeUnreachableBlocks()` are standard CFG algorithms.

**compiler.cpp** — Pipeline orchestrator. Convert `runPasses()` to a C
function that calls each pass via function pointer. Convert `Compile()` to
a C function that coordinates HIR build → passes → codegen.

**context.cpp** — State management. Convert last (depends on everything
else being C). Containers → `_Py_hashtable`. Thread safety →
`PyMutex` (CPython internal). Smart pointers → raw + manual refcount.

---

## Part 3: Dependency Graph and Parallel Work

```
Phase I-1 (hir.cpp methods)
    ↓
Phase I-2 (factory functions) ←── depends on I-1 for alloc/set/append
    ↓
Phase I-3 (Env/TC helpers) ←── depends on I-2 for factories
    ↓
    ├── Phase II-1 (simplify.cpp) ←── independent per-case
    ├── Phase II-2 (builder.cpp)  ←── independent per-case
    └── Phase II-3 (generator.cpp) ←── independent per-case
         ↓ (all three must complete)
    Phase III-1 (gen_asm.cpp) ←── can start after II-3
    Phase III-2 (regalloc.cpp) ←── can start after II-3
    Phase III-3 (pass/compiler/context) ←── after all Phase II
```

**Maximum parallelism:** After Phase I completes, II-1/II-2/II-3 can run
in parallel (different files, different opcode cases, no overlap).
III-1 and III-2 can also run in parallel.

---

## Part 4: Effort Estimates (calibrated by prototype)

The Branch factory prototype (5318f448c4) provides calibration:
- Factory function definition: 15 minutes
- C allocation + field setup: 10 minutes
- Assertion wrapper + testing: 5 minutes
- Total per simple factory: 30 minutes

| Phase | Hours | Commits | Notes |
|-------|-------|---------|-------|
| I-1: hir.cpp methods | 3-4 | 5-8 | 8 remaining methods |
| I-2: Factory functions | 4-6 | 10-15 | 100 opcodes, batched |
| I-3: Env/TC helpers | 2-3 | 3-5 | 3 structs + helpers |
| II-1: simplify.cpp | 6-8 | 8-12 | 37 cases, 4 patterns |
| II-2: builder.cpp | 12-16 | 15-25 | 196 cases, 3 tiers |
| II-3: generator.cpp | 8-12 | 12-18 | 162 cases |
| III-1: gen_asm.cpp | 8-12 | 8-12 | 37 methods |
| III-2: regalloc.cpp | 6-8 | 5-8 | Algorithm + containers |
| III-3: pass/compiler/ctx | 6-8 | 5-8 | Pipeline + state |
| **Total** | **55-77** | **71-111** | |

**Pythia's calibration concern addressed:** The Branch factory prototype
validates construction patterns. Complex factories (VectorCall,
BeginInlinedFunction) will take 3-5x longer. The estimate accounts for
this via the "Complex" category (9 hours for 9 opcodes = 1 hour each).

---

## Part 5: Invariants the Generalist Must Preserve

1. **Operand preamble layout.** Operands are stored BEFORE the instruction
   struct in memory. `hir_c_alloc_instr(opcode, n)` must allocate
   `n * sizeof(HirRegister*) + sizeof(int) + sizeof(HirInstr)` and return
   a pointer past the preamble. This matches the C++ `Instr::allocate()`.

2. **FrameState deep copy.** When cloning a TranslationContext for branch
   divergence, the stack and localsplus arrays must be deep-copied. Sharing
   arrays between branches causes corruption when one branch pops.

3. **BCOffset stamping.** Every instruction must carry the bytecode offset
   from the FrameState at the time of emission. Missing offsets cause deopt
   at wrong bytecode positions.

4. **DeoptState ownership.** If an instruction has a DeoptState (FrameState
   + live registers), the instruction owns the FrameState memory. When the
   instruction is destroyed, the FrameState must be freed.

5. **Register allocation.** Registers allocated by `TempAllocator` must not
   be reused across different instructions unless the SSA property holds.
   Each `emit` that creates an output register should use a fresh register.

6. **Block terminator invariant.** Every basic block must end with exactly
   one terminator instruction (Branch, CondBranch, Return, Deopt, Raise).
   No non-terminator may follow a terminator.

7. **Assertion wrapper requirement.** Every converted C function must have
   an assertion wrapper that runs both C and C++ paths and compares results.
   The wrapper is removed only after the full test suite passes with
   assertions enabled.

8. **struct layout alignment.** C struct definitions in hir_instr_c.h must
   match C++ class layout exactly. Use `static_assert(sizeof(...))` and
   `static_assert(offsetof(...))` for every struct.

---

## Part 6: What Would Make This Plan Wrong

1. **If the operand preamble layout in C++ uses virtual dispatch or
   compiler-specific padding** that doesn't match plain C struct layout.
   Falsifier: `static_assert(sizeof(HirInstr) == sizeof(CxxInstr))` fails.
   Mitigation: Already validated by T2-C (virtual dispatch eliminated) and
   existing sizeof assertions.

2. **If FrameState has hidden dependencies** (e.g., weak references from
   optimization passes that assume C++ move semantics). Falsifier: test
   case where FrameState is cloned and original is modified — clone should
   be unaffected. Mitigation: FrameState is value-type in C++, clone is
   deep copy.

3. **If the 100 factory functions create too much header bloat** slowing
   compilation. Falsifier: compile time increases >20%. Mitigation: use
   `static inline` for trivial factories, out-of-line for complex ones.

4. **If the assertion wrapper overhead** makes the test suite too slow.
   Falsifier: test suite time doubles. Mitigation: assertion wrappers are
   compiled out in release builds; only enabled during conversion.

5. **If container replacement introduces hash collision patterns** different
   from C++ unordered_map, causing performance regression. Falsifier:
   benchmark geo-mean drops below 1.0x. Mitigation: `_Py_hashtable` is
   battle-tested in CPython; PhxArray is already proven in LIR.
