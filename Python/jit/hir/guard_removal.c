/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * GuardTypeRemoval pass — pure C implementation.
 * Removes GuardType instructions whose output type refinement is not
 * needed by any downstream consumer.
 */

#include "cinderx/Jit/hir/guard_removal_c.h"
#include "cinderx/Jit/hir/hir_c_api.h"
#include "cinderx/Jit/hir/hir_instr_c.h"
#include "cinderx/Jit/hir/hir_type_c.h"
#include "cinderx/Jit/hir/copy_propagation_c.h"

#include <stdlib.h>
#include <string.h>

/* ---- Simple dynamic array for worklist ---- */

typedef struct {
    HirRegister reg;
    HirType type;
} WorkItem;

typedef struct {
    WorkItem *data;
    size_t head;
    size_t tail;
    size_t capacity;
} WorkQueue;

static void wq_init(WorkQueue *q) {
    q->capacity = 32;
    q->data = (WorkItem *)malloc(q->capacity * sizeof(WorkItem));
    q->head = 0;
    q->tail = 0;
}

static void wq_destroy(WorkQueue *q) {
    free(q->data);
}

static int wq_empty(const WorkQueue *q) {
    return q->head == q->tail;
}

static void wq_push(WorkQueue *q, HirRegister reg, const HirType *type) {
    if (q->tail >= q->capacity) {
        /* Compact or grow */
        if (q->head > 0) {
            size_t count = q->tail - q->head;
            memmove(q->data, q->data + q->head, count * sizeof(WorkItem));
            q->tail = count;
            q->head = 0;
        } else {
            q->capacity *= 2;
            q->data = (WorkItem *)realloc(q->data,
                                          q->capacity * sizeof(WorkItem));
        }
    }
    q->data[q->tail].reg = reg;
    q->data[q->tail].type = *type;
    q->tail++;
}

static WorkItem wq_pop(WorkQueue *q) {
    return q->data[q->head++];
}

/* ---- Seen state: flat array of (Register*, Type) pairs ---- */

typedef struct {
    HirRegister reg;
    HirType type;
} SeenEntry;

typedef struct {
    SeenEntry *data;
    size_t count;
    size_t capacity;
} SeenSet;

static void seen_init(SeenSet *s) {
    s->capacity = 32;
    s->data = (SeenEntry *)malloc(s->capacity * sizeof(SeenEntry));
    s->count = 0;
}

static void seen_destroy(SeenSet *s) {
    free(s->data);
}

/* Returns 1 if the entry was newly inserted, 0 if already present. */
static int seen_insert(SeenSet *s, HirRegister reg, const HirType *type) {
    for (size_t i = 0; i < s->count; i++) {
        if (s->data[i].reg == reg && hir_type_equal(&s->data[i].type, type)) {
            return 0;
        }
    }
    if (s->count >= s->capacity) {
        s->capacity *= 2;
        s->data = (SeenEntry *)realloc(s->data,
                                       s->capacity * sizeof(SeenEntry));
    }
    s->data[s->count].reg = reg;
    s->data[s->count].type = *type;
    s->count++;
    return 1;
}

/* ---- guardNeeded ---- */

static int guard_needed(HirRegUses uses, HirRegister new_reg,
                        HirType relaxed_type) {
    if (!hir_reg_uses_contains(uses, new_reg)) {
        return 0;
    }

    WorkQueue worklist;
    SeenSet seen;
    wq_init(&worklist);
    seen_init(&seen);

    wq_push(&worklist, new_reg, &relaxed_type);
    seen_insert(&seen, new_reg, &relaxed_type);

    int result = 0;

    while (!wq_empty(&worklist)) {
        WorkItem item = wq_pop(&worklist);
        HirRegister cur_reg = item.reg;
        HirType cur_type = item.type;

        if (!hir_reg_uses_contains(uses, cur_reg)) {
            continue;
        }

        size_t use_count = hir_reg_uses_count(uses, cur_reg);
        for (size_t u = 0; u < use_count; u++) {
            HirInstr instr = hir_reg_uses_get(uses, cur_reg, u);
            size_t num_ops = hir_instr_num_operands(instr);

            for (size_t i = 0; i < num_ops; i++) {
                if (hir_instr_get_operand(instr, i) != cur_reg) {
                    continue;
                }

                /* If this is a passthrough or Phi with output, propagate. */
                HirRegister output = hir_c_output(instr);
                if (output != NULL &&
                    (hir_c_is_phi(instr) || hir_is_passthrough(instr))) {
                    HirType passthrough_type =
                        hir_output_type_with_override(instr, i, &cur_type);
                    if (seen_insert(&seen, output, &passthrough_type)) {
                        wq_push(&worklist, output, &passthrough_type);
                    }
                }

                /* Check operand constraint with the relaxed type,
                 * not the register's current type. */
                if (hir_operands_must_match(instr, i)) {
                    result = 1;
                    goto done;
                }
                if (!hir_type_matches_operand(instr, i, &cur_type)) {
                    result = 1;
                    goto done;
                }
            }
        }
    }

done:
    wq_destroy(&worklist);
    seen_destroy(&seen);
    return result;
}

/* ---- Main pass ---- */

void hir_guard_type_removal_run(HirFunction func) {
    HirRegUses reg_uses = hir_collect_reg_uses(func);

    /* Collect guards to remove. */
    HirInstr *removed = NULL;
    size_t removed_count = 0;
    size_t removed_cap = 0;

    HirCFG cfg = hir_func_cfg(func);
    for (HirBasicBlock block = hir_cfg_blocks_first(cfg);
         block != NULL;
         block = hir_cfg_blocks_next(cfg, block)) {

        HirInstr instr = hir_block_first(block);
        while (instr != NULL) {
            HirInstr next = hir_block_next(block, instr);

            if (!hir_c_is_guard_type(instr)) {
                instr = next;
                continue;
            }

            HirRegister guard_out = hir_c_output(instr);
            HirRegister guard_in = hir_instr_get_operand(instr, 0);
            HirType guard_in_type = hir_register_type(guard_in);

            if (!guard_needed(reg_uses, guard_out, guard_in_type)) {
                HirInstr assign = hir_assign_create(guard_out, guard_in);
                hir_c_copy_bytecode_offset(assign, instr);
                hir_instr_replace_with(instr, assign);

                /* Track for deferred destruction. */
                if (removed_count >= removed_cap) {
                    removed_cap = removed_cap ? removed_cap * 2 : 16;
                    removed = (HirInstr *)realloc(
                        removed, removed_cap * sizeof(HirInstr));
                }
                removed[removed_count++] = instr;
            }

            instr = next;
        }
    }

    hir_reg_uses_destroy(reg_uses);

    /* Run copy propagation and reflow types. */
    hir_copy_propagation_run(func);
    hir_reflow_types(func);

    /* Destroy removed guards AFTER passes complete (T2-C6). */
    for (size_t i = 0; i < removed_count; i++) {
        hir_instr_delete(removed[i]);
    }
    free(removed);
}
