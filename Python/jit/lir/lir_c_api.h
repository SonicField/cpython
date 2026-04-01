/*
 * lir_c_api.h -- C-compatible accessor API for LIR types
 *
 * Phase 3D: Provides opaque pointer accessors for LIR Function, BasicBlock,
 * Instruction, and OperandBase so that algorithm files can be written in pure C.
 *
 * C++ callers: include this header and use the C functions directly,
 * or continue using C++ class methods. The C API is additive.
 *
 * C callers: include this header for opaque pointer access to LIR types.
 */

#ifndef JIT_LIR_C_API_H
#define JIT_LIR_C_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handles for LIR types */
typedef void* JitLirFunc;
typedef void* JitLirBlock;
typedef void* JitLirInstr;
typedef void* JitLirOperand;

/*
 * ---- Phase B1: C struct definitions for LIR core types ----
 *
 * These structs will eventually REPLACE the C++ class hierarchy.
 * For now they coexist — C files use these structs via the C API,
 * C++ files continue using the C++ classes.
 *
 * When all consumers are converted to C, the C++ classes are deleted
 * and these structs become the sole implementation.
 */

/* Forward declarations for struct cross-references */
typedef struct LirOperand LirOperand;
typedef struct LirMemoryIndirect LirMemoryIndirect;
typedef struct LirInstruction LirInstruction;

/* PhyLocation: physical register or stack slot.
 * Matches codegen::PhyLocation layout (int loc + size_t bitSize). */
typedef struct {
    int loc;
    size_t bit_size;
} LirPhyLocation;

#define LIR_REG_INVALID (-1)

/* Memory indirect reference: [base + index * (2^multiplier) + offset] */
struct LirMemoryIndirect {
    LirInstruction *parent;
    LirOperand *base_reg;       /* owned, may be NULL */
    LirOperand *index_reg;      /* owned, may be NULL */
    uint8_t multiplier;
    int32_t offset;
};

/* Unified operand: replaces OperandBase, Operand, and LinkedOperand.
 * When is_linked=1, value.def_opnd points to the defining instruction's
 * output operand — all getters delegate through it.
 * When is_linked=0, type/data_type/value hold the operand's own data. */
struct LirOperand {
    LirInstruction *parent_instr;
    uint8_t is_linked;
    uint8_t last_use;
    uint8_t type;       /* OperandType enum (kNone..kLabel) */
    uint8_t data_type;  /* DataType enum (k8bit..kObject) */
    union {
        uint64_t imm;              /* kImm: integer constant (or bit-cast double) */
        void *mem_addr;            /* kMem: memory address */
        void *label;               /* kLabel: BasicBlock* */
        LirMemoryIndirect *indirect; /* kInd: memory indirect (owned) */
        LirPhyLocation phy_loc;    /* kReg, kStack: physical location */
        LirOperand *def_opnd;      /* is_linked=1: defining operand */
    } value;
};

/* Instruction: the basic unit of LIR. Has one output operand (embedded)
 * and a growable array of input operands (owned pointers). */
struct LirInstruction {
    int id;
    int opcode;                  /* Instruction::Opcode as int */
    LirOperand output;           /* embedded output operand */
    void *basic_block;           /* BasicBlock* (opaque) */
    const void *origin;          /* hir::Instr* (opaque) */
    LirOperand **inputs;         /* owned array of input operand pointers */
    size_t num_inputs;
    size_t inputs_capacity;
};

/* Code section constants (must match codegen::CodeSection enum) */
#define JIT_LIR_SECTION_HOT  0
#define JIT_LIR_SECTION_COLD 1

/* ---- Function accessors ---- */
size_t jit_lir_func_num_blocks(JitLirFunc func);
JitLirBlock jit_lir_func_get_block(JitLirFunc func, size_t index);
JitLirBlock jit_lir_func_entry_block(JitLirFunc func);

/* ---- BasicBlock accessors ---- */
size_t jit_lir_block_num_preds(JitLirBlock block);
JitLirBlock jit_lir_block_get_pred(JitLirBlock block, size_t index);
size_t jit_lir_block_num_succs(JitLirBlock block);
JitLirBlock jit_lir_block_get_succ(JitLirBlock block, size_t index);
JitLirInstr jit_lir_block_get_last_instr(JitLirBlock block);
JitLirInstr jit_lir_block_get_first_instr(JitLirBlock block);
size_t jit_lir_block_num_instrs(JitLirBlock block);
JitLirBlock jit_lir_block_get_false_succ(JitLirBlock block);
int jit_lir_block_get_section(JitLirBlock block);
void jit_lir_block_set_section(JitLirBlock block, int section);
int jit_lir_block_get_id(JitLirBlock block);
JitLirInstr jit_lir_block_get_instr_at(JitLirBlock block, size_t index);

/* ---- Instruction accessors ---- */
int jit_lir_instr_opcode(JitLirInstr instr);
int jit_lir_instr_is_branch(JitLirInstr instr);
int jit_lir_instr_is_branch_cc(JitLirInstr instr);
int jit_lir_instr_is_any_branch(JitLirInstr instr);
int jit_lir_instr_is_terminator(JitLirInstr instr);
size_t jit_lir_instr_num_inputs(JitLirInstr instr);
JitLirOperand jit_lir_instr_get_input(JitLirInstr instr, size_t index);
JitLirOperand jit_lir_instr_output(JitLirInstr instr);

/* ---- Operand accessors ---- */
int jit_lir_operand_type(JitLirOperand op);
JitLirBlock jit_lir_operand_get_basic_block(JitLirOperand op);

/* Operand type constants (must match lir::OperandType enum) */
#define JIT_LIR_OPTYPE_NONE  0
#define JIT_LIR_OPTYPE_VREG  1
#define JIT_LIR_OPTYPE_REG   2
#define JIT_LIR_OPTYPE_STACK 3
#define JIT_LIR_OPTYPE_MEM   4
#define JIT_LIR_OPTYPE_IND   5
#define JIT_LIR_OPTYPE_IMM   6
#define JIT_LIR_OPTYPE_LABEL 7

/* Opcode constants */
int jit_lir_opcode_guard(void);

/* ---- Extended instruction accessors (Phase 3D Step 10: DCE) ---- */
int jit_lir_instr_id(JitLirInstr instr);
int jit_lir_instr_is_essential(JitLirInstr instr);
int jit_lir_instr_flag_effects(JitLirInstr instr);

/* Iterate all input operands of an instruction. */
void jit_lir_instr_foreach_input(
    JitLirInstr instr,
    void (*cb)(JitLirOperand operand, void *ctx),
    void *ctx);

/* ---- Extended operand accessors ---- */
int jit_lir_operand_is_reg(JitLirOperand op);
int jit_lir_operand_is_stack(JitLirOperand op);
int jit_lir_operand_is_mem(JitLirOperand op);
int jit_lir_operand_is_ind(JitLirOperand op);
int jit_lir_operand_is_linked(JitLirOperand op);

/* Get the defining instruction of a LinkedOperand. */
JitLirInstr jit_lir_operand_get_linked_instr(JitLirOperand op);

/* ---- MemoryIndirect accessors ---- */
typedef void* JitLirIndirect;
JitLirIndirect jit_lir_operand_get_indirect(JitLirOperand op);
JitLirOperand jit_lir_indirect_base_reg(JitLirIndirect ind);
JitLirOperand jit_lir_indirect_index_reg(JitLirIndirect ind);

/* ---- Block instruction removal ---- */
/* Remove all instructions for which is_live(instr, ctx) returns 0.
 * Handles C++ iterator invalidation internally. */
void jit_lir_block_remove_dead_instrs(
    JitLirBlock block,
    int (*is_live)(JitLirInstr instr, void *ctx),
    void *ctx);

/* FlagEffects constants (must match lir::FlagEffects enum) */
#define JIT_LIR_FLAG_NONE       0
#define JIT_LIR_FLAG_SET        1
#define JIT_LIR_FLAG_INVALIDATE 2

/* DataType constants (must match lir::DataType enum) */
#define JIT_LIR_DT_8BIT   0
#define JIT_LIR_DT_16BIT  1
#define JIT_LIR_DT_32BIT  2
#define JIT_LIR_DT_64BIT  3
#define JIT_LIR_DT_DOUBLE 4
#define JIT_LIR_DT_OBJECT 5

/* ---- Phase A: Extended operand getters ---- */
int jit_lir_operand_data_type(JitLirOperand op);
int jit_lir_operand_is_fp(JitLirOperand op);
int jit_lir_operand_is_last_use(JitLirOperand op);
uint64_t jit_lir_operand_get_constant(JitLirOperand op);
double jit_lir_operand_get_fp_constant(JitLirOperand op);
int jit_lir_operand_get_phy_register(JitLirOperand op);
int jit_lir_operand_get_stack_slot(JitLirOperand op);
void* jit_lir_operand_get_mem_address(JitLirOperand op);
JitLirOperand jit_lir_operand_get_define(JitLirOperand op);

/* ---- Phase A: Extended instruction getters ---- */
JitLirBlock jit_lir_instr_basic_block(JitLirInstr instr);
const void* jit_lir_instr_origin(JitLirInstr instr);
int jit_lir_instr_is_compare(JitLirInstr instr);
int jit_lir_instr_is_any_yield(JitLirInstr instr);
int jit_lir_instr_inputs_live_across(JitLirInstr instr);
int jit_lir_instr_output_phy_use(JitLirInstr instr);
int jit_lir_instr_input_phy_use(JitLirInstr instr, size_t index);

/* ---- Phase A: MemoryIndirect getters ---- */
int jit_lir_indirect_multiplier(JitLirIndirect ind);
int32_t jit_lir_indirect_offset(JitLirIndirect ind);

/* ---- Phase A: Branch CC statics ---- */
int jit_lir_negate_branch_cc(int opcode);
int jit_lir_flip_branch_cc_direction(int opcode);
int jit_lir_compare_to_branch_cc(int opcode);

/* ---- Phase A: Operand setters ---- */
void jit_lir_operand_set_constant(JitLirOperand op, uint64_t val, int data_type);
void jit_lir_operand_set_fp_constant(JitLirOperand op, double val);
void jit_lir_operand_set_phy_register(JitLirOperand op, int loc);
void jit_lir_operand_set_stack_slot(JitLirOperand op, int loc);
void jit_lir_operand_set_virtual_register(JitLirOperand op);
void jit_lir_operand_set_data_type(JitLirOperand op, int dt);
void jit_lir_operand_set_basic_block(JitLirOperand op, JitLirBlock block);
void jit_lir_operand_set_mem_address(JitLirOperand op, void* addr);
void jit_lir_operand_set_last_use(JitLirOperand op);

/* ---- Phase A: Instruction mutation ---- */
void jit_lir_instr_set_opcode(JitLirInstr instr, int opcode);
void jit_lir_instr_set_num_inputs(JitLirInstr instr, size_t n);
const char *jit_lir_instr_opname(JitLirInstr instr);

/* ---- Phase A: Instruction operand allocation ---- */
JitLirOperand jit_lir_instr_alloc_imm_input(JitLirInstr instr,
    uint64_t val, int data_type);
JitLirOperand jit_lir_instr_alloc_fp_imm_input(JitLirInstr instr, double val);
JitLirOperand jit_lir_instr_alloc_linked_input(JitLirInstr instr,
    JitLirInstr def_instr);
JitLirOperand jit_lir_instr_alloc_phyreg_input(JitLirInstr instr, int loc);
JitLirOperand jit_lir_instr_alloc_stack_input(JitLirInstr instr, int loc);
JitLirOperand jit_lir_instr_alloc_label_input(JitLirInstr instr,
    JitLirBlock block);
JitLirOperand jit_lir_instr_alloc_addr_input(JitLirInstr instr, void* addr);

/* ---- Phase A: Block/Function allocation ---- */
JitLirBlock jit_lir_func_alloc_block(JitLirFunc func);
JitLirInstr jit_lir_block_alloc_instr(JitLirBlock block, int opcode,
    const void* hir_origin);
void jit_lir_block_add_successor(JitLirBlock block, JitLirBlock succ);

/* ---- Phase B1: LirOperand C struct operations ---- */
/* These operate directly on LirOperand/LirMemoryIndirect structs.
 * They coexist with the jit_lir_operand_* wrappers above (which cast
 * through C++ classes). New C code should use these directly. */
LirOperand *lir_operand_new(LirInstruction *parent);
LirOperand *lir_operand_new_linked(LirInstruction *parent,
                                   LirInstruction *def_instr);
void lir_operand_free(LirOperand *op);

LirMemoryIndirect *lir_memind_new(LirInstruction *parent);
void lir_memind_free(LirMemoryIndirect *mi);

uint8_t lir_operand_type(const LirOperand *op);
uint8_t lir_operand_data_type(const LirOperand *op);
int lir_operand_is_linked(const LirOperand *op);
int lir_operand_is_fp(const LirOperand *op);
int lir_operand_is_last_use(const LirOperand *op);
size_t lir_operand_size_in_bits(const LirOperand *op);
LirInstruction *lir_operand_instr(const LirOperand *op);

uint64_t lir_operand_get_constant(const LirOperand *op);
double lir_operand_get_fp_constant(const LirOperand *op);
LirPhyLocation lir_operand_get_phy_register(const LirOperand *op);
LirPhyLocation lir_operand_get_stack_slot(const LirOperand *op);
void *lir_operand_get_mem_address(const LirOperand *op);
void *lir_operand_get_basic_block(const LirOperand *op);
LirMemoryIndirect *lir_operand_get_indirect(const LirOperand *op);
LirOperand *lir_operand_get_define(LirOperand *op);
uint64_t lir_operand_get_constant_or_address(const LirOperand *op);
LirInstruction *lir_operand_get_linked_instr(const LirOperand *op);

void lir_operand_set_constant(LirOperand *op, uint64_t val, uint8_t dt);
void lir_operand_set_fp_constant(LirOperand *op, double val);
void lir_operand_set_phy_register(LirOperand *op, LirPhyLocation reg);
void lir_operand_set_stack_slot(LirOperand *op, LirPhyLocation slot);
void lir_operand_set_mem_address(LirOperand *op, void *addr);
void lir_operand_set_basic_block(LirOperand *op, void *block);
void lir_operand_set_virtual_register(LirOperand *op);
void lir_operand_set_data_type(LirOperand *op, uint8_t dt);
void lir_operand_set_last_use(LirOperand *op);
void lir_operand_set_none(LirOperand *op);
void lir_operand_set_linked_instr(LirOperand *op, LirInstruction *def);

void lir_memind_set(LirMemoryIndirect *mi,
                    LirPhyLocation base, LirPhyLocation index,
                    uint8_t multiplier, int32_t offset);
void lir_memind_set_linked(LirMemoryIndirect *mi,
                           LirInstruction *base, LirInstruction *index,
                           uint8_t multiplier, int32_t offset);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* JIT_LIR_C_API_H */
