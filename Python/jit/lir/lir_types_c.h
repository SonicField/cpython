/*
 * lir_types_c.h -- C-compatible struct definitions for LIR core types
 *
 * Phase 3D: SINGLE SOURCE OF TRUTH for all LIR struct layouts.
 * Both C and C++ files include this header.
 *
 * IMPORTANT: Field names, types, and ORDER must match the corresponding
 * C++ struct definitions EXACTLY. Any mismatch causes undefined behavior
 * when C code accesses objects created by C++.
 *
 * C++ struct → C struct mapping:
 *   codegen::PhyLocation     → LirPhyLocation      (arch/x86_64.h or aarch64.h)
 *   lir::MemoryIndirect      → LirMemoryIndirect    (operand.h)
 *   lir::OperandBase/Operand → LirOperand           (operand.h)
 *   lir::Instruction         → LirInstruction       (instruction.h)
 *   lir::BasicBlock          → LirBasicBlock        (block.h)
 *   lir::Function            → LirFunction          (function.h)
 */

#ifndef JIT_LIR_TYPES_C_H
#define JIT_LIR_TYPES_C_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Forward declarations ---- */
typedef struct LirOperand LirOperand;
typedef struct LirMemoryIndirect LirMemoryIndirect;
typedef struct LirInstruction LirInstruction;
typedef struct LirBasicBlock LirBasicBlock;

/* ---- PhyLocation ----
 * Matches codegen::PhyLocation (arch/x86_64.h:143-144, aarch64.h:157-158).
 * C++ uses int32_t loc + uint32_t bitSize = 8 bytes total. */
typedef struct {
    int32_t loc;
    uint32_t bit_size;   /* C++ name: bitSize */
} LirPhyLocation;

#define LIR_REG_INVALID (-1)

/* ---- MemoryIndirect ----
 * Matches lir::MemoryIndirect (operand.h:129-167).
 * [base + index * (2^multiplier) + offset] */
struct LirMemoryIndirect {
    LirInstruction *parent_;
    LirOperand *base_reg_;       /* owned, may be NULL */
    LirOperand *index_reg_;      /* owned, may be NULL */
    uint8_t multiplier_;
    /* 3 bytes padding */
    int32_t offset_;
};

/* ---- OperandBase/Operand/LinkedOperand ----
 * Matches lir::OperandBase (operand.h:105-126).
 * Phase B5: Devirtualized — all data in OperandBase, dispatch via is_linked_.
 * Operand and LinkedOperand are thin subclasses with NO additional data. */
struct LirOperand {
    LirInstruction *parent_instr_;
    uint8_t last_use_;       /* C++: bool last_use_ — bool is 1 byte on all targets */
    uint8_t is_linked_;      /* C++: bool is_linked_ */
    uint8_t type_;           /* C++: OperandType (enum class : uint8_t) */
    uint8_t data_type_;      /* C++: DataType (enum class : uint8_t) */
    /* 4 bytes padding (align union to 8) */
    union {
        uint64_t constant;           /* kImm: integer constant (or bit-cast double) */
        void *address;               /* kMem: memory address */
        LirBasicBlock *block;        /* kLabel: BasicBlock* */
        LirMemoryIndirect *indirect; /* kInd: memory indirect (owned when !is_linked) */
        LirPhyLocation phy_loc;      /* kReg, kStack: physical location */
    } value_;                        /* 8 bytes (PhyLocation is now 8, not 16) */
    LirOperand *def_opnd_;           /* LinkedOperand: defining operand (separate from value_) */
};

/* ---- Instruction ----
 * Matches lir::Instruction (instruction.h:434-447).
 * Phase B4a: All fields public. */
struct LirInstruction {
    int id_;
    int opcode_;                     /* Instruction::Opcode (int enum) */
    LirOperand output_;              /* embedded output operand (32 bytes) */
    LirBasicBlock *basic_block_;
    const void *origin_;             /* hir::Instr* (opaque in C) */
    LirOperand **inputs_;            /* owned array of input operand pointers */
    size_t num_inputs_;
    size_t inputs_capacity_;
    LirInstruction *prev_;           /* intrusive list link */
    LirInstruction *next_;           /* intrusive list link */
};

/* ---- CodeSection enum ----
 * Matches codegen::CodeSection in code_section.h */
typedef enum {
    LIR_SECTION_HOT = 0,
    LIR_SECTION_COLD = 1
} LirCodeSection;

/* ---- BasicBlock ----
 * Matches lir::BasicBlock (block.h:175-192).
 * Phase B4b: struct with public fields. */
struct LirBasicBlock {
    int id_;
    /* 4 bytes padding (align func_ to 8) */
    struct LirFunction *func_;

    LirBasicBlock **successors_;
    size_t num_succs_;
    size_t succs_capacity_;

    LirBasicBlock **predecessors_;
    size_t num_preds_;
    size_t preds_capacity_;

    /* Intrusive doubly-linked list of instructions */
    LirInstruction *instr_head_;
    LirInstruction *instr_tail_;
    size_t num_instrs_;

    LirCodeSection section_;
    /* 4 bytes padding (struct alignment) */
};

/* ---- Function ----
 * Matches lir::Function (function.h:63-74).
 * Phase B4c: struct with public fields. */
typedef struct LirFunction {
    const void *hir_func_;   /* const hir::Function* — opaque in C */

    LirBasicBlock **blocks_;
    size_t num_blocks_;
    size_t blocks_capacity_;

    int next_id_;
    /* 4 bytes padding */
} LirFunction;

/* ---- Static size assertions ----
 * Compile-time check that C struct sizes match expectations.
 * These sizes MUST match the corresponding C++ struct sizes.
 * If a static_assert fails, the C and C++ layouts have diverged. */
#ifdef __cplusplus
static_assert(sizeof(LirPhyLocation) == 8, "LirPhyLocation size mismatch with PhyLocation");
static_assert(sizeof(LirMemoryIndirect) == 32, "LirMemoryIndirect size mismatch");
static_assert(sizeof(LirOperand) == 32, "LirOperand size mismatch with OperandBase");
static_assert(sizeof(LirInstruction) == 96, "LirInstruction size mismatch with Instruction");
static_assert(sizeof(LirBasicBlock) == 96, "LirBasicBlock size mismatch with BasicBlock");
static_assert(sizeof(LirFunction) == 40, "LirFunction size mismatch with Function");
#else
_Static_assert(sizeof(LirPhyLocation) == 8, "LirPhyLocation size mismatch with PhyLocation");
_Static_assert(sizeof(LirMemoryIndirect) == 32, "LirMemoryIndirect size mismatch");
_Static_assert(sizeof(LirOperand) == 32, "LirOperand size mismatch with OperandBase");
_Static_assert(sizeof(LirInstruction) == 96, "LirInstruction size mismatch with Instruction");
_Static_assert(sizeof(LirBasicBlock) == 96, "LirBasicBlock size mismatch with BasicBlock");
_Static_assert(sizeof(LirFunction) == 40, "LirFunction size mismatch with Function");
#endif

/* ---- Enum constants ---- */

/* Operand type constants (must match lir::OperandType enum) */
#define JIT_LIR_OPTYPE_NONE  0
#define JIT_LIR_OPTYPE_VREG  1
#define JIT_LIR_OPTYPE_REG   2
#define JIT_LIR_OPTYPE_STACK 3
#define JIT_LIR_OPTYPE_MEM   4
#define JIT_LIR_OPTYPE_IND   5
#define JIT_LIR_OPTYPE_IMM   6
#define JIT_LIR_OPTYPE_LABEL 7

/* DataType constants (must match lir::DataType enum) */
#define JIT_LIR_DT_8BIT   0
#define JIT_LIR_DT_16BIT  1
#define JIT_LIR_DT_32BIT  2
#define JIT_LIR_DT_64BIT  3
#define JIT_LIR_DT_DOUBLE 4
#define JIT_LIR_DT_OBJECT 5

/* FlagEffects constants (must match lir::FlagEffects enum) */
#define JIT_LIR_FLAG_NONE       0
#define JIT_LIR_FLAG_SET        1
#define JIT_LIR_FLAG_INVALIDATE 2

/* Code section constants */
#define JIT_LIR_SECTION_HOT  0
#define JIT_LIR_SECTION_COLD 1

/* ---- Runtime assertion macros ---- */
#define LIR_ASSERT_OPERAND_TYPE(op, expected) \
    assert((op)->type_ == (expected) && "operand type mismatch")

#define LIR_ASSERT_NOT_LINKED(op) \
    assert(!(op)->is_linked_ && "expected direct operand, got linked")

#define LIR_ASSERT_IS_LINKED(op) \
    assert((op)->is_linked_ && "expected linked operand, got direct")

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* JIT_LIR_TYPES_C_H */
