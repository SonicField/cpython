/*
 * lir_types_c.h -- C-compatible struct definitions for LIR core types
 *
 * Phase B5: Provides C-visible struct layouts that match the C++ struct
 * memory layout exactly. C files include this header to access LIR struct
 * fields directly, without going through the C++ headers.
 *
 * IMPORTANT: Field names, types, and ORDER must match the corresponding
 * C++ struct definitions in block.h and function.h exactly. Any mismatch
 * causes undefined behavior when C code accesses objects created by C++.
 *
 * Currently covers: LirBasicBlock (block.h), LirFunction (function.h).
 * Instruction and OperandBase are accessed via opaque pointers (void*)
 * through lir_c_api.h until their C++ internals are converted.
 */

#ifndef JIT_LIR_TYPES_C_H
#define JIT_LIR_TYPES_C_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Forward declarations for types that remain opaque in C */
typedef struct LirInstruction LirInstruction;

/* CodeSection enum — must match codegen::CodeSection in code_section.h */
typedef enum {
    LIR_SECTION_HOT = 0,
    LIR_SECTION_COLD = 1
} LirCodeSection;

/*
 * LirBasicBlock — C-visible layout of jit::lir::BasicBlock (block.h)
 *
 * Field order MUST match block.h lines 175-191 exactly.
 * No vtable (no virtual methods).
 */
typedef struct LirBasicBlock {
    int id_;
    struct LirFunction* func_;

    struct LirBasicBlock** successors_;
    size_t num_succs_;
    size_t succs_capacity_;

    struct LirBasicBlock** predecessors_;
    size_t num_preds_;
    size_t preds_capacity_;

    /* Intrusive doubly-linked list of instructions */
    LirInstruction* instr_head_;
    LirInstruction* instr_tail_;
    size_t num_instrs_;

    LirCodeSection section_;
} LirBasicBlock;

/*
 * LirFunction — C-visible layout of jit::lir::Function (function.h)
 *
 * Field order MUST match function.h lines 63-74 exactly.
 * No vtable (no virtual methods).
 *
 * Note: hir_func_ is an opaque pointer to the HIR Function (C++ only).
 * C code should not dereference it.
 */
typedef struct LirFunction {
    const void* hir_func_;  /* const hir::Function* — opaque in C */

    struct LirBasicBlock** blocks_;
    size_t num_blocks_;
    size_t blocks_capacity_;

    int next_id_;
} LirFunction;

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* JIT_LIR_TYPES_C_H */
