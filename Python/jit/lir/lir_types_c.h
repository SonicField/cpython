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

#include "cinderx/Jit/codegen/arch/detection.h"

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

/* LIR opcode constants — must match Instruction::Opcode enum in instruction.h */
/* Generated from FOREACH_INSTR_TYPE, 86 opcodes */
#define JIT_LIR_OP_NONE (-1)
#define JIT_LIR_OP_BIND 0
#define JIT_LIR_OP_NOP 1
#define JIT_LIR_OP_UNREACHABLE 2
#define JIT_LIR_OP_CALL 3
#define JIT_LIR_OP_VECTORCALL 4
#define JIT_LIR_OP_VARARGCALL 5
#define JIT_LIR_OP_GUARD 6
#define JIT_LIR_OP_DEOPTPATCHPOINT 7
#define JIT_LIR_OP_SEXT 8
#define JIT_LIR_OP_ZEXT 9
#define JIT_LIR_OP_NEGATE 10
#define JIT_LIR_OP_INVERT 11
#define JIT_LIR_OP_ADD 12
#define JIT_LIR_OP_SUB 13
#define JIT_LIR_OP_AND 14
#define JIT_LIR_OP_XOR 15
#define JIT_LIR_OP_DIV 16
#define JIT_LIR_OP_DIVUN 17
#define JIT_LIR_OP_MUL 18
#define JIT_LIR_OP_OR 19
#define JIT_LIR_OP_FADD 20
#define JIT_LIR_OP_FSUB 21
#define JIT_LIR_OP_FMUL 22
#define JIT_LIR_OP_FDIV 23
#define JIT_LIR_OP_LSHIFT 24
#define JIT_LIR_OP_RSHIFT 25
#define JIT_LIR_OP_RSHIFTUN 26
#define JIT_LIR_OP_TEST 27
#define JIT_LIR_OP_TEST32 28
#define JIT_LIR_OP_EQUAL 29
#define JIT_LIR_OP_NOTEQUAL 30
#define JIT_LIR_OP_GREATERTHANSIGNED 31
#define JIT_LIR_OP_LESSTHANSIGNED 32
#define JIT_LIR_OP_GREATERTHANEQUALSIGNED 33
#define JIT_LIR_OP_LESSTHANEQUALSIGNED 34
#define JIT_LIR_OP_GREATERTHANUNSIGNED 35
#define JIT_LIR_OP_LESSTHANUNSIGNED 36
#define JIT_LIR_OP_GREATERTHANEQUALUNSIGNED 37
#define JIT_LIR_OP_LESSTHANEQUALUNSIGNED 38
#define JIT_LIR_OP_CMP 39
#define JIT_LIR_OP_LEA 40
#define JIT_LIR_OP_LOADARG 41
#define JIT_LIR_OP_LOADSECONDCALLRESULT 42
#define JIT_LIR_OP_EXCHANGE 43
#define JIT_LIR_OP_MOVE 44
#define JIT_LIR_OP_MOVERELAXED 45
#define JIT_LIR_OP_PUSH 46
#define JIT_LIR_OP_POP 47
#define JIT_LIR_OP_CDQ 48
#define JIT_LIR_OP_CWD 49
#define JIT_LIR_OP_CQO 50
#define JIT_LIR_OP_BRANCH 51
#define JIT_LIR_OP_BRANCHNZ 52
#define JIT_LIR_OP_BRANCHZ 53
#define JIT_LIR_OP_BRANCHA 54
#define JIT_LIR_OP_BRANCHB 55
#define JIT_LIR_OP_BRANCHAE 56
#define JIT_LIR_OP_BRANCHBE 57
#define JIT_LIR_OP_BRANCHG 58
#define JIT_LIR_OP_BRANCHL 59
#define JIT_LIR_OP_BRANCHGE 60
#define JIT_LIR_OP_BRANCHLE 61
#define JIT_LIR_OP_BRANCHC 62
#define JIT_LIR_OP_BRANCHNC 63
#define JIT_LIR_OP_BRANCHO 64
#define JIT_LIR_OP_BRANCHNO 65
#define JIT_LIR_OP_BRANCHS 66
#define JIT_LIR_OP_BRANCHNS 67
#define JIT_LIR_OP_BRANCHE 68
#define JIT_LIR_OP_BRANCHNE 69
#define JIT_LIR_OP_BITTEST 70
#define JIT_LIR_OP_INC 71
#define JIT_LIR_OP_DEC 72
#define JIT_LIR_OP_CONDBRANCH 73
#define JIT_LIR_OP_SELECT 74
#define JIT_LIR_OP_PHI 75
#define JIT_LIR_OP_RETURN 76
#define JIT_LIR_OP_MOVZX 77
#define JIT_LIR_OP_MOVSX 78
#define JIT_LIR_OP_MOVSXD 79
#define JIT_LIR_OP_INTTOBOOL 80
#define JIT_LIR_OP_YIELDINITIAL 81
#define JIT_LIR_OP_YIELDFROM 82
#define JIT_LIR_OP_YIELDFROMSKIPINITIALSEND 83
#define JIT_LIR_OP_YIELDFROMHANDLESTOPASYNCITERATION 84
#define JIT_LIR_OP_YIELDVALUE 85
#define JIT_LIR_NUM_OPCODES 86

/* InstrGuardKind constants (must match lir::InstrGuardKind enum) */
#define JIT_GUARD_ALWAYS_FAIL 0
#define JIT_GUARD_HAS_TYPE    1
#define JIT_GUARD_IS          2
#define JIT_GUARD_NOT_NEGATIVE 3
#define JIT_GUARD_NOT_ZERO    4
#define JIT_GUARD_ZERO        5

/* FlagEffects constants (must match lir::FlagEffects enum) */
#define JIT_LIR_FLAG_NONE       0
#define JIT_LIR_FLAG_SET        1
#define JIT_LIR_FLAG_INVALIDATE 2

/* Code section constants */
#define JIT_LIR_SECTION_HOT  0
#define JIT_LIR_SECTION_COLD 1

/* ---- Utility functions ---- */

/* C replacement for fitsSignedInt<32>(constant) */
static inline int
lir_fits_signed_int32(int64_t v) {
    return v >= INT32_MIN && v <= INT32_MAX;
}

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
