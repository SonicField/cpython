/*
 * cold_block_marker.c -- Mark cold basic blocks in LIR (pure C)
 *
 * Phase 3D conversion: cold_block_marker.cpp -> cold_block_marker.c
 *
 * Uses static heuristics to identify blocks unlikely to execute on the hot
 * path. Cold blocks are moved to a separate code section by the code
 * generator, improving I-cache utilisation.
 *
 * Heuristics:
 *   H1: Guard failure targets — blocks reached only via guard failure
 *   H2: Deopt stubs — single-instruction blocks with a Guard
 *   H3: Transitive closure — blocks reachable only from cold blocks
 */

#include "cinderx/Jit/lir/lir_c_api.h"

#include "Python.h"  /* for PyMem_RawMalloc/RawFree */

/* Simple growable queue for BFS traversal */
typedef struct {
    JitLirBlock *items;
    size_t head;
    size_t tail;
    size_t capacity;
} BlockQueue;

static void
queue_init(BlockQueue *q, size_t initial_cap) {
    q->items = (JitLirBlock *)PyMem_RawMalloc(
        initial_cap * sizeof(JitLirBlock));
    q->head = 0;
    q->tail = 0;
    q->capacity = initial_cap;
}

static void
queue_free(BlockQueue *q) {
    PyMem_RawFree(q->items);
    q->items = NULL;
}

static int
queue_empty(const BlockQueue *q) {
    return q->head == q->tail;
}

static void
queue_push(BlockQueue *q, JitLirBlock block) {
    if (q->tail >= q->capacity) {
        size_t new_cap = q->capacity * 2;
        q->items = (JitLirBlock *)PyMem_RawRealloc(
            q->items, new_cap * sizeof(JitLirBlock));
        q->capacity = new_cap;
    }
    q->items[q->tail++] = block;
}

static JitLirBlock
queue_pop(BlockQueue *q) {
    return q->items[q->head++];
}

/*
 * H1: A block is a guard failure target if:
 *   - It has exactly one predecessor
 *   - That predecessor's last instruction is a Guard
 *   - This block is the guard's failure target (false successor)
 */
static int
is_guard_failure_target(JitLirBlock block) {
    int kGuard = jit_lir_opcode_guard();

    if (jit_lir_block_num_preds(block) != 1) {
        return 0;
    }

    JitLirBlock pred = jit_lir_block_get_pred(block, 0);
    JitLirInstr last = jit_lir_block_get_last_instr(pred);
    if (last == NULL || jit_lir_instr_opcode(last) != kGuard) {
        return 0;
    }

    /* Guard's false successor is the failure path */
    return jit_lir_block_get_false_succ(pred) == block;
}

/*
 * H2: A block is a deopt stub if it contains only a single Guard
 * instruction (which encodes the deopt exit inline).
 */
static int
is_deopt_stub(JitLirBlock block) {
    int kGuard = jit_lir_opcode_guard();

    if (jit_lir_block_num_instrs(block) == 0) {
        return 0;
    }

    if (jit_lir_block_num_instrs(block) == 1) {
        JitLirInstr first = jit_lir_block_get_first_instr(block);
        if (jit_lir_instr_opcode(first) == kGuard) {
            return 1;
        }
    }

    return 0;
}

void
jit_lir_mark_cold_blocks(JitLirFunc func) {
    size_t num_blocks = jit_lir_func_num_blocks(func);
    if (num_blocks == 0) {
        return;
    }

    JitLirBlock entry = jit_lir_func_entry_block(func);

    /* Phase 1: Seed cold blocks using H1 and H2 heuristics */
    BlockQueue worklist;
    queue_init(&worklist, num_blocks > 16 ? num_blocks : 16);

    for (size_t i = 0; i < num_blocks; i++) {
        JitLirBlock block = jit_lir_func_get_block(func, i);
        if (block == entry) {
            continue;
        }

        if (is_guard_failure_target(block) || is_deopt_stub(block)) {
            jit_lir_block_set_section(block, JIT_LIR_SECTION_COLD);
            /* Add successors to worklist for transitive closure */
            size_t num_succs = jit_lir_block_num_succs(block);
            for (size_t s = 0; s < num_succs; s++) {
                queue_push(&worklist,
                           jit_lir_block_get_succ(block, s));
            }
        }
    }

    /* Phase 2: Transitive closure (H3).
     * A block is cold if ALL its predecessors are cold.
     * Monotonic forward dataflow — only transition hot->cold. */
    while (!queue_empty(&worklist)) {
        JitLirBlock block = queue_pop(&worklist);

        /* Skip entry block and already-cold blocks */
        if (block == entry ||
            jit_lir_block_get_section(block) == JIT_LIR_SECTION_COLD) {
            continue;
        }

        /* Check if ALL predecessors are cold */
        size_t num_preds = jit_lir_block_num_preds(block);
        int all_preds_cold = 1;
        for (size_t p = 0; p < num_preds; p++) {
            if (jit_lir_block_get_section(
                    jit_lir_block_get_pred(block, p)) !=
                JIT_LIR_SECTION_COLD) {
                all_preds_cold = 0;
                break;
            }
        }

        if (all_preds_cold && num_preds > 0) {
            jit_lir_block_set_section(block, JIT_LIR_SECTION_COLD);
            size_t num_succs = jit_lir_block_num_succs(block);
            for (size_t s = 0; s < num_succs; s++) {
                JitLirBlock succ = jit_lir_block_get_succ(block, s);
                if (jit_lir_block_get_section(succ) !=
                    JIT_LIR_SECTION_COLD) {
                    queue_push(&worklist, succ);
                }
            }
        }
    }

    queue_free(&worklist);
}
