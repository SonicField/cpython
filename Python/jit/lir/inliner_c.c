/*
 * inliner_c.c -- C implementation of LIR inliner
 *
 * Phase 3D: Full C rewrite of inliner.cpp (377 lines).
 * Uses extern C wrappers for Parser::parse and Function::copyFrom.
 * The parsed function cache uses a simple fixed-size array with
 * pthread_mutex protection.
 */

#include "cinderx/Jit/lir/lir_impl_internal.h"

#include "Python.h"

#include <assert.h>
#include <pthread.h>
#include <string.h>

/* ---- C helper translation lookup (extern C from c_helper_translations.c) ---- */
extern const char* jit_lir_map_c_helper_to_lir(uint64_t addr);

/* Use opcode constants from lir_types_c.h (included via lir_impl_internal.h) */

/* ---- Parsed function cache ---- */

#define MAX_CACHED_FUNCTIONS 64

typedef struct {
    uint64_t addr;
    LirFunction *func;  /* NULL = parse failed or no LIR text */
    int valid;
} CachedFunction;

static CachedFunction s_func_cache[MAX_CACHED_FUNCTIONS];
static size_t s_func_cache_count = 0;
static pthread_mutex_t s_func_cache_mutex = PTHREAD_MUTEX_INITIALIZER;

static LirFunction *
cache_lookup(uint64_t addr, int *found) {
    for (size_t i = 0; i < s_func_cache_count; i++) {
        if (s_func_cache[i].valid && s_func_cache[i].addr == addr) {
            *found = 1;
            return s_func_cache[i].func;
        }
    }
    *found = 0;
    return NULL;
}

static void
cache_insert(uint64_t addr, LirFunction *func) {
    if (s_func_cache_count < MAX_CACHED_FUNCTIONS) {
        s_func_cache[s_func_cache_count].addr = addr;
        s_func_cache[s_func_cache_count].func = func;
        s_func_cache[s_func_cache_count].valid = 1;
        s_func_cache_count++;
    }
    /* If cache is full, silently don't cache — inlining will just
     * re-parse each time, which is acceptable for a small number of helpers */
}

/* ---- parseFunction ---- */

static LirFunction *
parse_function(uint64_t addr) {
    int found;

    pthread_mutex_lock(&s_func_cache_mutex);
    LirFunction *cached = cache_lookup(addr, &found);
    pthread_mutex_unlock(&s_func_cache_mutex);

    if (found) {
        return cached;
    }

    /* Try to get LIR text for this address */
    const char *lir_text = jit_lir_map_c_helper_to_lir(addr);
    if (lir_text == NULL) {
        pthread_mutex_lock(&s_func_cache_mutex);
        cache_insert(addr, NULL);
        pthread_mutex_unlock(&s_func_cache_mutex);
        return NULL;
    }

    /* Parse the LIR text */
    void *parsed = NULL;
    int rc = lir_parser_parse(lir_text, &parsed);
    if (rc != 0) {
        pthread_mutex_lock(&s_func_cache_mutex);
        cache_insert(addr, NULL);
        pthread_mutex_unlock(&s_func_cache_mutex);
        return NULL;
    }

    pthread_mutex_lock(&s_func_cache_mutex);
    /* Check again in case another thread parsed it */
    LirFunction *existing = cache_lookup(addr, &found);
    if (found) {
        /* Another thread beat us — free our copy and use theirs */
        pthread_mutex_unlock(&s_func_cache_mutex);
        lir_function_free((LirFunction *)parsed);
        return existing;
    }
    cache_insert(addr, (LirFunction *)parsed);
    pthread_mutex_unlock(&s_func_cache_mutex);

    return (LirFunction *)parsed;
}

/* ---- findCalleeFunction ---- */

static LirFunction *
find_callee_function(LirInstruction *call_instr) {
    if (call_instr->num_inputs_ < 1) {
        return NULL;
    }
    LirOperand *dest = call_instr->inputs_[0];
    if (dest->type_ != JIT_LIR_OPTYPE_IMM) {
        return NULL;
    }
    uint64_t addr = dest->value_.constant;
    return parse_function(addr);
}

/* ---- checkEntryExitReturn ---- */

static int
check_entry_exit_return(const LirFunction *callee) {
    if (callee->num_blocks_ == 0) return 0;

    LirBasicBlock *entry_block = callee->blocks_[0];
    if (entry_block->num_preds_ > 0) return 0;

    LirBasicBlock *exit_block = callee->blocks_[callee->num_blocks_ - 1];
    if (exit_block->num_succs_ > 0) return 0;

    for (size_t i = 0; i < callee->num_blocks_; i++) {
        LirBasicBlock *bb = callee->blocks_[i];
        if (bb->num_preds_ == 0 && bb != entry_block) return 0;
        if (bb->num_succs_ == 0 && bb != exit_block) return 0;

        for (LirInstruction *instr = bb->instr_head_; instr; instr = instr->next_) {
            if (instr->opcode_ == JIT_LIR_OP_RETURN) {
                if (instr != bb->instr_tail_ ||
                    bb->num_succs_ != 1 ||
                    bb->successors_[0] != exit_block) {
                    return 0;
                }
            }
        }
    }

    if (exit_block->instr_head_ != NULL) return 0;

    return 1;
}

/* ---- checkArguments ---- */

static int
check_arguments(LirInstruction *call_instr,
                LirOperand ***out_args, size_t *out_num_args) {
    size_t num_inputs = call_instr->num_inputs_;
    size_t num_args = num_inputs > 1 ? num_inputs - 1 : 0;

    LirOperand **args = NULL;
    if (num_args > 0) {
        args = (LirOperand **)PyMem_RawCalloc(num_args, sizeof(LirOperand *));
        for (size_t i = 0; i < num_args; i++) {
            LirOperand *input = call_instr->inputs_[i + 1];
            if (input->type_ != JIT_LIR_OPTYPE_IMM &&
                input->type_ != JIT_LIR_OPTYPE_VREG) {
                PyMem_RawFree(args);
                return 0;
            }
            args[i] = input;
        }
    }

    *out_args = args;
    *out_num_args = num_args;
    return 1;
}

/* ---- checkLoadArg ---- */

static int
check_load_arg(LirInstruction *call_instr, const LirFunction *callee) {
    size_t num_inputs = call_instr->num_inputs_ - 1;
    int checking = 1;

    for (size_t i = 0; i < callee->num_blocks_; i++) {
        LirBasicBlock *bb = callee->blocks_[i];
        for (LirInstruction *instr = bb->instr_head_; instr; instr = instr->next_) {
            if (checking) {
                if (instr->opcode_ == JIT_LIR_OP_LOADARG) {
                    if (instr->num_inputs_ < 1) return 0;
                    LirOperand *input = instr->inputs_[0];
                    if (input->type_ != JIT_LIR_OPTYPE_IMM) return 0;
                    if (input->value_.constant >= num_inputs) return 0;
                } else {
                    checking = 0;
                }
            } else {
                if (instr->opcode_ == JIT_LIR_OP_LOADARG) return 0;
            }
        }
    }
    return 1;
}

/* ---- isInlineable ---- */

static int
is_inlineable(LirInstruction *call_instr, const LirFunction *callee,
              LirOperand ***out_args, size_t *out_num_args) {
    if (!check_entry_exit_return(callee)) return 0;
    if (!check_arguments(call_instr, out_args, out_num_args)) return 0;
    if (!check_load_arg(call_instr, callee)) {
        PyMem_RawFree(*out_args);
        *out_args = NULL;
        return 0;
    }
    return 1;
}

/* ---- vreg map (small fixed-size for argument resolution) ---- */

#define MAX_VREG_MAP 32

typedef struct {
    LirOperand *key;    /* output operand of LoadArg */
    LirOperand *value;  /* linked operand from caller's arguments */
} VregMapEntry;

typedef struct {
    VregMapEntry entries[MAX_VREG_MAP];
    size_t count;
} VregMap;

static LirOperand *
vreg_map_get(const VregMap *map, LirOperand *key) {
    for (size_t i = 0; i < map->count; i++) {
        if (map->entries[i].key == key) {
            return map->entries[i].value;
        }
    }
    return NULL;
}

static void
vreg_map_insert(VregMap *map, LirOperand *key, LirOperand *value) {
    assert(map->count < MAX_VREG_MAP);
    map->entries[map->count].key = key;
    map->entries[map->count].value = value;
    map->count++;
}

/* ---- resolveLinkedArgumentsUses ---- */

static void
resolve_linked_uses(VregMap *vreg_map, LirInstruction *instr) {
    for (size_t i = 0; i < instr->num_inputs_; i++) {
        LirOperand *input = instr->inputs_[i];
        if (input->is_linked_) {
            /* Check if this operand's define is in the vreg_map */
            LirOperand *new_def = vreg_map_get(vreg_map, input->def_opnd_);
            if (new_def != NULL) {
                /* Relink: point to the linked operand's definition */
                LirOperand *linked_def = new_def->def_opnd_;
                if (linked_def != NULL) {
                    input->def_opnd_ = linked_def;
                }
            }
        } else if (input->type_ == JIT_LIR_OPTYPE_IND) {
            /* For indirect operands, check base and index */
            LirMemoryIndirect *mi = input->value_.indirect;
            if (mi != NULL) {
                LirOperand *base = mi->base_reg_;
                LirOperand *index = mi->index_reg_;
                if (base != NULL && base->is_linked_) {
                    LirOperand *new_def = vreg_map_get(vreg_map, base->def_opnd_);
                    if (new_def != NULL && new_def->def_opnd_ != NULL) {
                        base->def_opnd_ = new_def->def_opnd_;
                    }
                }
                if (index != NULL && index->is_linked_) {
                    LirOperand *new_def = vreg_map_get(vreg_map, index->def_opnd_);
                    if (new_def != NULL && new_def->def_opnd_ != NULL) {
                        index->def_opnd_ = new_def->def_opnd_;
                    }
                }
            }
        }
    }
}

/* ---- resolveLoadArg ---- */

static void
resolve_load_arg(VregMap *vreg_map, LirBasicBlock *bb,
                 LirInstruction **instr_it,
                 LirOperand **arguments, size_t num_arguments) {
    LirInstruction *instr = *instr_it;
    assert(instr->num_inputs_ > 0 && instr->inputs_[0]->type_ == JIT_LIR_OPTYPE_IMM);

    LirOperand *argument = instr->inputs_[0];
    uint64_t arg_index = argument->value_.constant;
    assert(arg_index < num_arguments);
    LirOperand *param = arguments[arg_index];

    if (param->type_ == JIT_LIR_OPTYPE_IMM) {
        /* For immediate values, change LoadArg to Move */
        instr->opcode_ = JIT_LIR_OP_MOVE;
        /* Create a new operand with the constant value */
        LirOperand *param_copy = lir_operand_new(instr);
        lir_operand_set_constant(param_copy, param->value_.constant, param->data_type_);
        lir_instruction_set_input(instr, 0, param_copy);
        *instr_it = instr->next_;
    } else {
        /* For virtual registers (linked), delete LoadArg and record mapping */
        assert(param->is_linked_);
        vreg_map_insert(vreg_map, &instr->output_, param);
        LirInstruction *next = instr->next_;
        LirInstruction *removed = lir_block_remove_instr(bb, instr);
        lir_instruction_free(removed);
        *instr_it = next;
    }
}

/* ---- resolveArguments ---- */

static int
resolve_arguments(LirFunction *caller, int callee_start, int callee_end,
                  LirOperand **arguments, size_t num_arguments) {
    VregMap vreg_map;
    vreg_map.count = 0;

    for (int i = callee_start; i < callee_end; i++) {
        LirBasicBlock *bb = caller->blocks_[i];
        LirInstruction *it = bb->instr_head_;
        while (it != NULL) {
            if (it->opcode_ == JIT_LIR_OP_LOADARG) {
                resolve_load_arg(&vreg_map, bb, &it, arguments, num_arguments);
            } else {
                resolve_linked_uses(&vreg_map, it);
                it = it->next_;
            }
        }
    }
    return 1;
}

/* ---- resolveReturnValue ---- */

static void
resolve_return_value(LirFunction *caller, LirInstruction *call_instr,
                     int callee_end) {
    LirBasicBlock *epilogue = caller->blocks_[callee_end - 1];

    /* Create phi instruction */
    LirInstruction *phi_instr = lir_block_alloc_instr(epilogue, JIT_LIR_OP_PHI, NULL);
    phi_instr->output_.type_ = JIT_LIR_OPTYPE_VREG;
    phi_instr->output_.data_type_ = JIT_LIR_DT_OBJECT;

    /* Find return instructions from predecessors of epilogue */
    for (size_t p = 0; p < epilogue->num_preds_; p++) {
        LirBasicBlock *pred = epilogue->predecessors_[p];
        LirInstruction *last = pred->instr_tail_;
        if (last != NULL && last->opcode_ == JIT_LIR_OP_RETURN) {
            lir_instruction_alloc_label_input(phi_instr, pred);
            assert(last->num_inputs_ > 0);
            LirOperand *released = lir_instruction_release_input(last, 0);
            lir_instruction_append_input(phi_instr, released);
            LirInstruction *removed = lir_block_remove_instr(pred, last);
            lir_instruction_free(removed);
        }
    }

    if (phi_instr->num_inputs_ == 0) {
        /* Callee has no return statements — remove phi, nop the call */
        LirInstruction *removed = lir_block_remove_instr(epilogue, phi_instr);
        lir_instruction_free(removed);
        call_instr->opcode_ = JIT_LIR_OP_NOP;
    } else {
        call_instr->opcode_ = JIT_LIR_OP_MOVE;
        /* Remove all inputs from call */
        while (call_instr->num_inputs_ > 0) {
            LirOperand *removed = lir_instruction_remove_input(
                call_instr, call_instr->num_inputs_ - 1);
            lir_operand_free(removed);
        }
        lir_instruction_alloc_linked_input(call_instr, phi_instr);
    }
}

/* ---- inlineCall ---- */

static int
inline_call(LirFunction *caller, LirInstruction *call_instr) {
    /* Find callee function */
    LirFunction *callee = find_callee_function(call_instr);
    if (callee == NULL) return 0;

    /* Check if inlineable and collect arguments */
    LirOperand **arguments = NULL;
    size_t num_arguments = 0;
    if (!is_inlineable(call_instr, callee, &arguments, &num_arguments)) {
        return 0;
    }

    /* Split basic block at call site */
    LirBasicBlock *block1 = call_instr->basic_block_;
    LirBasicBlock *block2 = lir_block_split_before(block1, call_instr);

    /* Copy callee into caller */
    int callee_start, callee_end;
    lir_function_copy_from(caller, callee, block1, block2,
                           call_instr->origin_, &callee_start, &callee_end);

    /* Resolve arguments */
    resolve_arguments(caller, callee_start, callee_end,
                      arguments, num_arguments);

    /* Resolve return value */
    resolve_return_value(caller, call_instr, callee_end);

    PyMem_RawFree(arguments);
    return 1;
}

/* ---- inlineCalls (top-level entry point) ---- */

int
lir_inliner_inline_calls(void *func_ptr) {
    LirFunction *func = (LirFunction *)func_ptr;
    int changed = 0;

    for (size_t i = 0; i < func->num_blocks_; i++) {
        LirBasicBlock *bb = func->blocks_[i];

        for (LirInstruction *instr = bb->instr_head_; instr; instr = instr->next_) {
            if (instr->opcode_ == JIT_LIR_OP_CALL) {
                if (inline_call(func, instr)) {
                    changed = 1;
                    /* Block has been split — nothing left to process */
                    break;
                }
            }
        }
    }

    return changed;
}
