# Phase 3D: HIR Instruction Hierarchy Conversion Strategy

## Scope

hir.h: 4132 lines, 100+ instruction classes via INSTR_CLASS macro
hir.cpp: 1418 lines (instruction method implementations)

This is the largest single conversion target in Phase 3D.

## Key Insight: The 100+ Classes Are Mechanical

All concrete instruction classes use the `INSTR_CLASS` macro:
```cpp
class INSTR_CLASS(BinaryOp, (TObject, TObject), HasOutput, Operands<2>, DeoptBase) {
    BinaryOpKind op_;
    // ...
};
```

This expands to:
- A class inheriting from `InstrT<name, Opcode::kName, ...>`
- A `GetOperandTypeImpl` method returning a static type vector
- Per-opcode data fields (op_, target_, etc.)

The inheritance chain: `ConcreteInstr → InstrT → DeoptBase → Instr`

## C Struct Design

### Flattened Instruction

```c
typedef struct HirInstr {
    /* Intrusive list node (for BasicBlock ownership) */
    IntrusiveListNode block_node;
    
    /* Core fields (from Instr base) */
    HirOpcode opcode;
    uint16_t num_operands;
    uint16_t flags;          /* has_output, is_deopt_base, etc. */
    
    /* Output register (if has_output) */
    HirRegister *output;
    
    /* Operand array (variable-length, slab-allocated) */
    HirRegister **operands;
    
    /* DeoptBase fields (if is_deopt_base) */
    HirFrameState *deopt_state;    /* NULL if not deopt */
    
    /* Per-opcode data (tagged union) */
    union {
        struct { int kind; } binary_op;
        struct { int kind; } unary_op;
        struct { int kind; } compare_op;
        struct { HirBasicBlock *target; } branch;
        struct { HirBasicBlock *true_bb; HirBasicBlock *false_bb; } cond_branch;
        struct { int idx; } load_const;
        struct { int name_idx; } load_attr;
        /* ... one entry per opcode that has per-opcode data */
        /* Opcodes with no extra data use no union member */
    } data;
} HirInstr;
```

### Why This Works

1. **sizeof**: All instructions are the same size (the union covers the largest variant). Slab allocation works the same way.

2. **Opcode dispatch replaces inheritance**: Instead of `dynamic_cast<BinaryOp*>(instr)`, use `instr->opcode == HIR_OP_BINARY_OP`. Already done for Destroy() (T2-C6).

3. **GetOperandType replaces virtual dispatch**: Already devirtualized (T2-C3) — dispatches via function pointer table indexed by opcode. The C version is a static array of arrays.

4. **visitUses replaces virtual dispatch**: Already devirtualized (T2-C4) — dispatches to DeoptBase/Snapshot extensions via opcode check.

## Conversion Approach: Macro-Mechanical

### Step 1: Define C Opcode Data Structs (one per opcode with data)

```c
/* Generated from FOREACH_OPCODE or manually */
typedef struct { int kind; } HirBinaryOpData;
typedef struct { int kind; } HirUnaryOpData;
typedef struct { HirBasicBlock *target; } HirBranchData;
/* ... */
```

Most opcodes (~70%) have NO per-opcode data — they're defined via `DEFINE_SIMPLE_INSTR`. Only ~30 opcodes have custom fields.

### Step 2: Enumerate Per-Opcode Data in the Union

The union contains one field per opcode that has data. Simple opcodes use no union member.

### Step 3: C Accessors Replace Cast-Based Access

```c
/* Instead of: static_cast<BinaryOp*>(instr)->op() */
static inline int hir_instr_binary_op_kind(const HirInstr *i) {
    assert(i->opcode == HIR_OP_BINARY_OP);
    return i->data.binary_op.kind;
}
```

### Step 4: FOREACH_OPCODE Generates Boilerplate

The existing `FOREACH_OPCODE` macro already lists all opcodes. Use it to generate:
- C enum values (already done: hir_opcode_c.h)
- Per-opcode metadata table (already done: hir_instr_info_c.c)
- Accessor functions (new: one per opcode with data)
- Destroy dispatch table (already done: T2-C6)

## Migration Path for Callers

### Pattern: switch-on-opcode replaces dynamic_cast

```cpp
// BEFORE (C++):
if (auto* binop = dynamic_cast<BinaryOp*>(instr)) {
    handle_binop(binop->op());
}

// AFTER (C):
if (instr->opcode == HIR_OP_BINARY_OP) {
    handle_binop(hir_instr_binary_op_kind(instr));
}
```

This pattern is ALREADY used in many places (T2-C eliminated most dynamic_cast).

### Pattern: static_cast to concrete type → direct field access

```cpp
// BEFORE:
auto* branch = static_cast<Branch*>(instr);
branch->target();

// AFTER:
assert(instr->opcode == HIR_OP_BRANCH);
instr->data.branch.target;
```

## Phased Execution

### Phase H1a: Audit Per-Opcode Data Fields (~2 hours)
- Read every INSTR_CLASS definition in hir.h
- List which opcodes have custom data fields vs DEFINE_SIMPLE_INSTR
- Design the union entries

### Phase H1b: Create C Struct + Accessors (~4 hours)
- HirInstr C struct in hir_instr_c.h (extend existing)
- Per-opcode accessor functions
- Static_assert layout compatibility with C++ Instr

### Phase H1c: Assertion-Verify Accessors (~2 hours)
- Same methodology as type.cpp: C accessor == C++ method for every field
- Run 975/975 Phoenix with assertion wrappers

### Phase H1d: Wire Callers (~8-16 hours)
- Each HIR pass that uses dynamic_cast or static_cast to concrete types
  gets migrated to the C accessor pattern
- One pass at a time, assertion-verified

### Phase H1e: Delete C++ Classes (~4 hours)
- Once all callers use C accessors, the InstrT/INSTR_CLASS hierarchy
  becomes dead code
- Delete the class definitions, keep the C struct

## What Makes This Mechanical

1. **FOREACH_OPCODE** lists all opcodes — code generation possible
2. **T2-C already eliminated vtable** — no virtual methods remain
3. **Per-opcode data fields are small** — most are 1-2 fields (int kind, BasicBlock* target)
4. **dynamic_cast already replaced** — T2-C converted 7/7 to static_cast
5. **Metadata tables already in C** — hir_instr_info_c.c has 86 opcode entries

## Estimated Total: 20-30 hours of conversion work

This is ~4-6 working sessions. The majority is Phase H1d (wiring callers),
which is mechanical but touches many files.
