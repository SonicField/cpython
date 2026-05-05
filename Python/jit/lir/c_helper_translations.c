/*
 * c_helper_translations.c -- C helper to LIR mapping (pure C)
 *
 * Phase 3D conversion: c_helper_translations.cpp -> c_helper_translations.c
 * Maps C helper function addresses to their LIR implementations.
 *
 * The JITRT_Cast address is set from C++ via jit_lir_set_cast_addr()
 * because JITRT_Cast has C++ linkage (name-mangled).
 *
 * Phase 5.B commit 1 (parser.cpp ELIMINATE PIVOT): adds
 * init_cast_lir_function + jit_lir_map_c_helper_to_lir_func that
 * construct the JITRT_Cast LirFunction* programmatically via existing
 * C bridges. The legacy snprintf-text path (init_cast_lir +
 * jit_lir_map_c_helper_to_lir + cast_lir_buf) is RETAINED in commit 1
 * for the byte-match equivalence falsifier; commit 2 deletes it along
 * with parser.cpp.
 */

#include "Python.h"

#include "cinderx/Common/jit_log_c.h"
#include "cinderx/Jit/lir/lir_c_api.h"
#include "cinderx/Jit/lir/lir_impl_internal.h"
#include "cinderx/Jit/lir/printer_c.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* JITRT_Cast address, set from C++ at init time */
static uint64_t jitrt_cast_addr = 0;

/* Buffer for the formatted JITRT_Cast LIR string */
static char cast_lir_buf[2048];
static int cast_lir_initialized = 0;

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

static void
init_cast_lir(void) {
    snprintf(cast_lir_buf, sizeof(cast_lir_buf),
        "Function:\n"
        "BB %%0 - succs: %%2 %%1\n"
        "       %%5:Object = LoadArg 0(0x0):Object\n"
        "       %%6:Object = LoadArg 1(0x1):Object\n"
        "       %%7:Object = Move [%%5:Object + %#zx]:Object\n"
        "       %%8:Object = Equal %%7:Object, %%6:Object\n"
        "                   CondBranch %%8:Object\n"
        "\n"
        "BB %%1 - preds: %%0 - succs: %%2 %%3\n"
        "      %%10:Object = Call PyType_IsSubtype, %%7:Object, %%6:Object\n"
        "                   CondBranch %%10:Object\n"
        "\n"
        "BB %%2 - preds: %%0 %%1 - succs: %%4\n"
        "                   Return %%5:Object\n"
        "\n"
        "BB %%3 - preds: %%1 - succs: %%4\n"
        "      %%13:Object = Move [%%7:Object + %#zx]:Object\n"
        "      %%14:Object = Move [%%6:Object + %#zx]:Object\n"
        "                   Call PyErr_Format, PyExc_TypeError, "
            "\"expected '%%s', got '%%s'\", %%14:Object, %%13:Object\n"
        "      %%16:Object = Move 0(0x0):Object\n"
        "                   Return %%16:Object\n"
        "\n"
        "BB %%4 - preds: %%2 %%3\n",
        offsetof(PyObject, ob_type),
        offsetof(PyTypeObject, tp_name),
        offsetof(PyTypeObject, tp_name));
    cast_lir_initialized = 1;
}

void
jit_lir_set_cast_addr(uint64_t addr) {
    jitrt_cast_addr = addr;
}

const char*
jit_lir_map_c_helper_to_lir(uint64_t addr) {
    if (!cast_lir_initialized) {
        init_cast_lir();
    }
    if (jitrt_cast_addr != 0 && addr == jitrt_cast_addr) {
        return cast_lir_buf;
    }
    return NULL;
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
 * the snprintf-text shape in init_cast_lir. Per spec §1 + theologian
 * 16:07:45Z Q1-Q3 confirmations:
 *   Q1 CondBranch successor order: [0]=true, [1]=false. BB0/BB1 wired
 *      bb2 then bb1 / bb2 then bb3 to match.
 *   Q2 Call function-pointer input position: FIRST input.
 *   Q3 Move memory-indirect: kInd lives on the input operand.
 *
 * Instruction-id assignment mirrors parser.cpp's two-pass scheme
 * (parser.cpp:575-583): output-bearing instructions take their explicit
 * text id (5, 6, 7, 8, 10, 13, 14, 16); control-flow instructions
 * without an explicit id (CondBranch / Call-no-output / Return) get
 * auto-assigned at end via next_id_ counter, in BB iteration order:
 * 17, 18, 19, 20, 21. The byte-match equivalence falsifier below
 * catches any divergence (id misorder, operand-shape mismatch, etc.)
 * from the parsed text path. */
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

    /* Match parser's two-pass id assignment (parser.cpp:575-583):
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

/* Falsifier (commit 1 only — deleted alongside legacy text path in
 * commit 2): build BOTH paths at first init, serialize via printer_c
 * into char buffers, JIT_CHECK_C strcmp byte-match. Aborts on any
 * divergence. Cost: one parse + two serializations + one strcmp at
 * Phoenix process startup (lazy, on first JITRT_Cast inline lookup).
 *
 * On c2 deletion, the legacy snprintf+parse path goes away and so
 * does this falsifier; the only remaining caller-facing function is
 * jit_lir_map_c_helper_to_lir_func returning the singleton built via
 * build_cast_lir_function. */
static void
falsifier_assert_pivot_matches_parsed(LirFunction *pivot_func)
{
    /* Build reference via legacy text+parse path. Requires
     * cast_lir_initialized (trips init_cast_lir if needed) — caller
     * is expected to have set jitrt_cast_addr already; without it the
     * snprintf path is still valid (offsets baked from offsetof). */
    if (!cast_lir_initialized) {
        init_cast_lir();
    }
    void *parsed = NULL;
    int rc = lir_parser_parse(cast_lir_buf, &parsed);
    JIT_CHECK_C(rc == 0 && parsed != NULL,
                "falsifier_assert_pivot_matches_parsed: "
                "lir_parser_parse failed (rc=%d)", rc);
    LirFunction *ref_func = (LirFunction *)parsed;

    /* Serialize both via printer_c into in-memory char buffers. */
    char pivot_buf[8192];
    char ref_buf[8192];
    FILE *pf = fmemopen(pivot_buf, sizeof(pivot_buf), "w");
    JIT_CHECK_C(pf != NULL, "fmemopen pivot_buf");
    lir_print_function(pf, pivot_func);
    fclose(pf);
    FILE *rf = fmemopen(ref_buf, sizeof(ref_buf), "w");
    JIT_CHECK_C(rf != NULL, "fmemopen ref_buf");
    lir_print_function(rf, ref_func);
    fclose(rf);

    /* Byte-match — strcmp tolerates the implicit NUL terminator written
     * by fmemopen-backed FILE close. */
    int eq = strcmp(pivot_buf, ref_buf);
    JIT_CHECK_C(eq == 0,
                "Phase 5.B falsifier: programmatic JITRT_Cast LirFunction* "
                "diverges from text+parse reference\n"
                "--- pivot ---\n%s\n"
                "--- ref ---\n%s\n",
                pivot_buf, ref_buf);

    lir_function_free(ref_func);
}

static void
init_cast_lir_function(void)
{
    LirFunction *pivot = build_cast_lir_function();
    falsifier_assert_pivot_matches_parsed(pivot);
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
