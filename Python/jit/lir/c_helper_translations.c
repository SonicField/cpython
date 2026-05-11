/*
 * c_helper_translations.c -- C helper to LIR mapping (pure C)
 *
 * Phase 3D conversion: c_helper_translations.cpp -> c_helper_translations.c
 * Maps C helper function addresses to their LIR implementations.
 *
 * The JITRT_Cast address is set from C++ via jit_lir_set_cast_addr()
 * because JITRT_Cast has C++ linkage (name-mangled).
 *
 * Phase 5.B PIVOT (commits 1+2): the JITRT_Cast LirFunction* is built
 * programmatically via existing C bridges in build_cast_lir_function()
 * and exposed via jit_lir_map_c_helper_to_lir_func(). The legacy
 * snprintf-text + parser.cpp path was retained in commit 1 alongside a
 * byte-match equivalence falsifier and removed in commit 2.
 */

#include "Python.h"

#include "cinderx/Jit/lir/lir_c_api.h"
#include "cinderx/Jit/lir/lir_impl_internal.h"

#include <stddef.h>
#include <stdint.h>

/* JITRT_Cast address, set from C++ at init time */
static uint64_t jitrt_cast_addr = 0;

/* Phase 5.B PIVOT: programmatic JITRT_Cast LirFunction* singleton. */
static LirFunction *cast_lir_func = NULL;
static int cast_lir_function_initialized = 0;

/* JITRT_Cast PyErr_Format format string — referenced as a constant
 * operand by the third Call in BB 3 (matches snprintf line 47). */
static const char CAST_FORMAT_STR[] = "expected '%s', got '%s'";

/* External C-runtime addresses used as constant operands in the LIR.
 * These are the addresses of C entry points; the JIT inliner consumes
 * them as immediate-constant inputs to Call instructions. */
extern int PyType_IsSubtype(PyTypeObject *a, PyTypeObject *b);
extern PyObject *PyErr_Format(PyObject *exception, const char *format, ...);
/* PyExc_TypeError is an extern PyObject* declared in pyerrors.h */

void
jit_lir_set_cast_addr(uint64_t addr) {
    jitrt_cast_addr = addr;
}

/* ====================================================================
 * Phase 5.B PIVOT: programmatic JITRT_Cast LirFunction* construction
 * ==================================================================== */

/* Helpers (all static; mirror the snprintf source-order conventions). */

static LirInstruction *
cast_emit_loadarg(LirBasicBlock *bb, uint64_t arg_index)
{
    LirInstruction *inst =
        lir_block_alloc_instr(bb, JIT_LIR_OP_LOADARG, NULL);
    lir_operand_set_data_type(&inst->output_, JIT_LIR_DT_OBJECT);
    lir_instruction_alloc_imm_input(inst, arg_index, JIT_LIR_DT_OBJECT);
    return inst;
}

static LirInstruction *
cast_emit_move_memind(LirBasicBlock *bb, LirInstruction *base, int32_t offset)
{
    LirInstruction *inst =
        lir_block_alloc_instr(bb, JIT_LIR_OP_MOVE, NULL);
    lir_operand_set_data_type(&inst->output_, JIT_LIR_DT_OBJECT);
    LirOperand *in =
        lir_instruction_alloc_imm_input(inst, 0, JIT_LIR_DT_OBJECT);
    lir_operand_set_memory_indirect_instr(in, base, offset);
    return inst;
}

static LirInstruction *
cast_emit_move_const(LirBasicBlock *bb, uint64_t val)
{
    LirInstruction *inst =
        lir_block_alloc_instr(bb, JIT_LIR_OP_MOVE, NULL);
    lir_operand_set_data_type(&inst->output_, JIT_LIR_DT_OBJECT);
    lir_instruction_alloc_imm_input(inst, val, JIT_LIR_DT_OBJECT);
    return inst;
}

/* Build the JITRT_Cast LirFunction* programmatically — equivalent to
 * the legacy snprintf-text shape (deleted in commit 2 along with
 * parser.cpp). Per spec §1 + theologian 16:07:45Z Q1-Q3 confirmations:
 *   Q1 CondBranch successor order: [0]=true, [1]=false. BB0/BB1 wired
 *      bb2 then bb1 / bb2 then bb3 to match.
 *   Q2 Call function-pointer input position: FIRST input.
 *   Q3 Move memory-indirect: kInd lives on the input operand.
 *
 * Instruction-id assignment mirrors the legacy parser's two-pass scheme:
 * output-bearing instructions take their explicit text id (5, 6, 7, 8,
 * 10, 13, 14, 16); control-flow instructions without an explicit id
 * (CondBranch / Call-no-output / Return) get auto-assigned at end via
 * next_id_ counter, in BB iteration order: 17, 18, 19, 20, 21. */
static LirFunction *
build_cast_lir_function(void)
{
    LirFunction *func = lir_function_new(NULL);
    LirBasicBlock *bb0 = lir_function_alloc_block(func);
    LirBasicBlock *bb1 = lir_function_alloc_block(func);
    LirBasicBlock *bb2 = lir_function_alloc_block(func);
    LirBasicBlock *bb3 = lir_function_alloc_block(func);
    LirBasicBlock *bb4 = lir_function_alloc_block(func);

    /* ---- BB 0 ---- */
    LirInstruction *v5 = cast_emit_loadarg(bb0, 0);
    LirInstruction *v6 = cast_emit_loadarg(bb0, 1);
    LirInstruction *v7 = cast_emit_move_memind(
        bb0, v5, (int32_t)offsetof(PyObject, ob_type));
    LirInstruction *v8 = lir_block_alloc_instr(bb0, JIT_LIR_OP_EQUAL, NULL);
    lir_operand_set_data_type(&v8->output_, JIT_LIR_DT_OBJECT);
    lir_instruction_alloc_linked_input(v8, v7);
    lir_instruction_alloc_linked_input(v8, v6);
    LirInstruction *cb0 =
        lir_block_alloc_instr(bb0, JIT_LIR_OP_CONDBRANCH, NULL);
    lir_instruction_alloc_linked_input(cb0, v8);
    lir_block_add_successor(bb0, bb2);
    lir_block_add_successor(bb0, bb1);

    /* ---- BB 1 ---- */
    LirInstruction *v10 = lir_block_alloc_instr(bb1, JIT_LIR_OP_CALL, NULL);
    lir_operand_set_data_type(&v10->output_, JIT_LIR_DT_OBJECT);
    lir_instruction_alloc_imm_input(
        v10, (uint64_t)(uintptr_t)&PyType_IsSubtype, JIT_LIR_DT_OBJECT);
    lir_instruction_alloc_linked_input(v10, v7);
    lir_instruction_alloc_linked_input(v10, v6);
    LirInstruction *cb1 =
        lir_block_alloc_instr(bb1, JIT_LIR_OP_CONDBRANCH, NULL);
    lir_instruction_alloc_linked_input(cb1, v10);
    lir_block_add_successor(bb1, bb2);
    lir_block_add_successor(bb1, bb3);

    /* ---- BB 2: Return %5 ---- */
    LirInstruction *r2 = lir_block_alloc_instr(bb2, JIT_LIR_OP_RETURN, NULL);
    lir_instruction_alloc_linked_input(r2, v5);
    lir_block_add_successor(bb2, bb4);

    /* ---- BB 3 ---- */
    LirInstruction *v13 = cast_emit_move_memind(
        bb3, v7, (int32_t)offsetof(PyTypeObject, tp_name));
    LirInstruction *v14 = cast_emit_move_memind(
        bb3, v6, (int32_t)offsetof(PyTypeObject, tp_name));
    LirInstruction *call_perr =
        lir_block_alloc_instr(bb3, JIT_LIR_OP_CALL, NULL);
    lir_instruction_alloc_imm_input(
        call_perr, (uint64_t)(uintptr_t)&PyErr_Format, JIT_LIR_DT_OBJECT);
    lir_instruction_alloc_imm_input(
        call_perr, (uint64_t)(uintptr_t)PyExc_TypeError, JIT_LIR_DT_OBJECT);
    lir_instruction_alloc_imm_input(
        call_perr, (uint64_t)(uintptr_t)CAST_FORMAT_STR, JIT_LIR_DT_OBJECT);
    lir_instruction_alloc_linked_input(call_perr, v14);
    lir_instruction_alloc_linked_input(call_perr, v13);
    LirInstruction *v16 = cast_emit_move_const(bb3, 0);
    LirInstruction *r3 = lir_block_alloc_instr(bb3, JIT_LIR_OP_RETURN, NULL);
    lir_instruction_alloc_linked_input(r3, v16);
    lir_block_add_successor(bb3, bb4);

    /* ---- BB 4: empty exit ---- */

    /* Mirror the legacy parser's two-pass id assignment:
     * (1) explicit text ids on output-bearing instructions; (2) auto-
     * assign control-flow instructions in BB iteration order from
     * largest_id+1 = 17. */
    v5->id_ = 5;
    v6->id_ = 6;
    v7->id_ = 7;
    v8->id_ = 8;
    cb0->id_ = 17;       /* BB0 CondBranch — auto-assigned post-fixup */
    v10->id_ = 10;
    cb1->id_ = 18;       /* BB1 CondBranch */
    r2->id_ = 19;        /* BB2 Return */
    v13->id_ = 13;
    v14->id_ = 14;
    call_perr->id_ = 20; /* BB3 PyErr_Format Call */
    v16->id_ = 16;
    r3->id_ = 21;        /* BB3 Return */
    func->next_id_ = 22;

    return func;
}

static void
init_cast_lir_function(void)
{
    LirFunction *pivot = build_cast_lir_function();
    cast_lir_func = pivot;
    cast_lir_function_initialized = 1;
}

LirFunction *
jit_lir_map_c_helper_to_lir_func(uint64_t addr)
{
    if (jitrt_cast_addr == 0) {
        return NULL;
    }
    if (addr != jitrt_cast_addr) {
        return NULL;
    }
    if (!cast_lir_function_initialized) {
        init_cast_lir_function();
    }
    return cast_lir_func;
}
