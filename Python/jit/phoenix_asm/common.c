/*
 * common.c -- Core infrastructure for the Phoenix assembler
 *
 * Node-list management, cursor positioning, label tracking, and operand
 * constructors.  No instruction encoding -- that lives in the arch
 * backends (x86_64.c, arm64.c).
 *
 * C11, no C++ dependencies.
 */

#include "phoenix_asm.h"
#include "x86_64.h"
#include "arm64.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/*  Default capacities                                                 */
/* ------------------------------------------------------------------ */

#define PHX_NODE_BLOCK_SIZE         256
#define PHX_DEFAULT_LABEL_CAP       64

/* ------------------------------------------------------------------ */
/*  Node block allocator                                               */
/*  Nodes are allocated in fixed-size blocks that are never moved.     */
/*  This guarantees stable PhxNode* pointers for the lifetime of the   */
/*  builder.                                                           */
/* ------------------------------------------------------------------ */

typedef struct PhxNodeBlock {
    struct PhxNodeBlock *next;
    PhxNode nodes[PHX_NODE_BLOCK_SIZE];
} PhxNodeBlock;
#define PHX_DEFAULT_FIXUP_CAP       32
#define PHX_DEFAULT_CODE_CAP        4096

/* ------------------------------------------------------------------ */
/*  CodeHolder                                                         */
/* ------------------------------------------------------------------ */

PhxCodeHolder *phx_code_create(PhxArch arch) {
    PhxCodeHolder *c = (PhxCodeHolder *)calloc(1, sizeof(PhxCodeHolder));
    if (!c) {
        return NULL;
    }
    c->arch = arch;
    c->buffer = (uint8_t *)malloc(PHX_DEFAULT_CODE_CAP);
    if (!c->buffer) {
        free(c);
        return NULL;
    }
    c->buffer_size = 0;
    c->buffer_capacity = PHX_DEFAULT_CODE_CAP;
    return c;
}

void phx_code_destroy(PhxCodeHolder *code) {
    if (!code) {
        return;
    }
    free(code->buffer);
    free(code);
}

/* ------------------------------------------------------------------ */
/*  Builder lifecycle                                                  */
/* ------------------------------------------------------------------ */

PhxBuilder *phx_builder_create(PhxCodeHolder *code) {
    assert(code != NULL);

    PhxBuilder *b = (PhxBuilder *)calloc(1, sizeof(PhxBuilder));
    if (!b) {
        return NULL;
    }
    b->code = code;

    /* Node block allocator */
    b->node_blocks = (PhxNodeBlock *)calloc(1, sizeof(PhxNodeBlock));
    if (!b->node_blocks) {
        free(b);
        return NULL;
    }
    b->node_blocks->next = NULL;
    b->node_block_used = 0;

    /* Label table */
    b->label_capacity = PHX_DEFAULT_LABEL_CAP;
    b->label_nodes = (PhxNode **)calloc(b->label_capacity, sizeof(PhxNode *));
    if (!b->label_nodes) {
        free(b->node_blocks);
        free(b);
        return NULL;
    }
    b->next_label_id = 0;

    /* Fixup list */
    b->fixup_capacity = PHX_DEFAULT_FIXUP_CAP;
    b->fixups = (PhxFixup *)calloc(b->fixup_capacity, sizeof(PhxFixup));
    if (!b->fixups) {
        free(b->label_nodes);
        free(b->node_blocks);
        free(b);
        return NULL;
    }
    b->fixup_count = 0;

    /* Node list starts empty */
    b->head = NULL;
    b->tail = NULL;
    b->cursor = NULL;

    return b;
}

void phx_builder_destroy(PhxBuilder *builder) {
    if (!builder) {
        return;
    }

    /* Walk the node list and free any embed_data allocations */
    PhxNode *n = builder->head;
    while (n) {
        if (n->embed_data) {
            free(n->embed_data);
        }
        n = n->next;
    }

    free(builder->fixups);
    free(builder->label_nodes);

    /* Free all node blocks */
    PhxNodeBlock *blk = builder->node_blocks;
    while (blk) {
        PhxNodeBlock *next_blk = blk->next;
        free(blk);
        blk = next_blk;
    }

    free(builder);
}

/* ------------------------------------------------------------------ */
/*  Node allocation                                                    */
/* ------------------------------------------------------------------ */

PhxNode *phx_builder_alloc_node(PhxBuilder *b) {
    assert(b != NULL);

    /* Allocate a new block if the current one is full */
    if (b->node_block_used >= PHX_NODE_BLOCK_SIZE) {
        PhxNodeBlock *new_blk = (PhxNodeBlock *)calloc(1, sizeof(PhxNodeBlock));
        if (!new_blk) {
            return NULL;
        }
        /* Prepend new block (current block is always first) */
        new_blk->next = b->node_blocks;
        b->node_blocks = new_blk;
        b->node_block_used = 0;
    }

    PhxNode *node = &b->node_blocks->nodes[b->node_block_used++];
    memset(node, 0, sizeof(PhxNode));
    return node;
}

/* ------------------------------------------------------------------ */
/*  Node insertion                                                     */
/*                                                                     */
/*  Critical invariant: new nodes are inserted AFTER the current       */
/*  cursor.  After insertion the cursor advances to the new node.      */
/*                                                                     */
/*  Falsifier scenario:                                                */
/*    emit A  ->  list: [A],       cursor = A                          */
/*    save C1 = cursor (= A)                                           */
/*    emit B  ->  list: [A, B],    cursor = B                          */
/*    setCursor(C1)  ->            cursor = A                          */
/*    emit X  ->  list: [A, X, B], cursor = X                          */
/*                                                                     */
/*  Walking head->next must give: A -> X -> B.                         */
/* ------------------------------------------------------------------ */

PhxNode *phx_builder_append_node(PhxBuilder *b, PhxNode *node) {
    assert(b != NULL);
    assert(node != NULL);

    if (b->head == NULL) {
        /* First node in the list */
        node->prev = NULL;
        node->next = NULL;
        b->head = node;
        b->tail = node;
    } else if (b->cursor == NULL) {
        /* No cursor: prepend before head */
        node->prev = NULL;
        node->next = b->head;
        b->head->prev = node;
        b->head = node;
    } else {
        /* Insert after cursor */
        PhxNode *after = b->cursor->next;
        node->prev = b->cursor;
        node->next = after;
        b->cursor->next = node;
        if (after) {
            after->prev = node;
        } else {
            /* cursor was the tail */
            b->tail = node;
        }
    }

    /* Advance cursor to the newly inserted node */
    b->cursor = node;
    return node;
}

/* ------------------------------------------------------------------ */
/*  Cursor management                                                  */
/* ------------------------------------------------------------------ */

PhxNode *phx_builder_cursor(PhxBuilder *b) {
    assert(b != NULL);
    return b->cursor;
}

void phx_builder_set_cursor(PhxBuilder *b, PhxNode *node) {
    assert(b != NULL);
    b->cursor = node;
}

/* ------------------------------------------------------------------ */
/*  Label management                                                   */
/* ------------------------------------------------------------------ */

PhxLabel phx_builder_new_label(PhxBuilder *b) {
    assert(b != NULL);

    /* Grow label table if needed */
    if (b->next_label_id >= b->label_capacity) {
        uint32_t new_cap = b->label_capacity * 2;
        PhxNode **new_tbl = (PhxNode **)realloc(
            b->label_nodes, (size_t)new_cap * sizeof(PhxNode *));
        if (!new_tbl) {
            /* Out of memory -- return an invalid label */
            PhxLabel bad = { UINT32_MAX };
            return bad;
        }
        /* Zero the new portion */
        memset(new_tbl + b->label_capacity, 0,
               (size_t)(new_cap - b->label_capacity) * sizeof(PhxNode *));
        b->label_nodes = new_tbl;
        b->label_capacity = new_cap;
    }

    PhxLabel label = { b->next_label_id++ };
    return label;
}

void phx_builder_bind(PhxBuilder *b, PhxLabel label) {
    assert(b != NULL);
    assert(label.id < b->next_label_id);

    /* Allocate a label node and insert it at the cursor */
    PhxNode *node = phx_builder_alloc_node(b);
    if (!node) {
        return;
    }
    node->node_type = PHX_NODE_LABEL;
    node->label_id = label.id;
    node->encoded_size = 0; /* label nodes emit no bytes */

    phx_builder_append_node(b, node);

    /* Record the binding so fixups can resolve it later */
    b->label_nodes[label.id] = node;
}

/* ------------------------------------------------------------------ */
/*  Fixup tracking                                                     */
/* ------------------------------------------------------------------ */

void phx_builder_add_fixup(PhxBuilder *b, PhxNode *node,
                           uint32_t label_id, uint8_t operand_idx) {
    assert(b != NULL);
    assert(node != NULL);

    /* Grow fixup array if needed */
    if (b->fixup_count >= b->fixup_capacity) {
        uint32_t new_cap = b->fixup_capacity * 2;
        PhxFixup *new_arr = (PhxFixup *)realloc(
            b->fixups, (size_t)new_cap * sizeof(PhxFixup));
        if (!new_arr) {
            return; /* silent failure -- finalize will catch unresolved refs */
        }
        b->fixups = new_arr;
        b->fixup_capacity = new_cap;
    }

    PhxFixup *f = &b->fixups[b->fixup_count++];
    f->node = node;
    f->label_id = label_id;
    f->operand_idx = operand_idx;
}

/* ------------------------------------------------------------------ */
/*  Alignment                                                          */
/* ------------------------------------------------------------------ */

void phx_builder_align(PhxBuilder *b, int alignment) {
    assert(b != NULL);
    assert(alignment > 0);
    /* alignment should be a power of two */
    assert((alignment & (alignment - 1)) == 0);

    PhxNode *node = phx_builder_alloc_node(b);
    if (!node) {
        return;
    }
    node->node_type = PHX_NODE_ALIGN;
    /* Store the alignment value in the first operand as an immediate */
    node->operands[0].type = PHX_OP_IMM;
    node->operands[0].imm = alignment;
    node->num_operands = 1;
    node->encoded_size = 0; /* determined during finalize */

    phx_builder_append_node(b, node);
}

/* ------------------------------------------------------------------ */
/*  Raw data embedding                                                 */
/* ------------------------------------------------------------------ */

void phx_builder_embed(PhxBuilder *b, const void *data, size_t size) {
    assert(b != NULL);
    if (size == 0) {
        return;
    }

    PhxNode *node = phx_builder_alloc_node(b);
    if (!node) {
        return;
    }
    node->node_type = PHX_NODE_EMBED;

    if (size <= PHX_MAX_ENCODED) {
        /* Small data fits inline */
        memcpy(node->encoded, data, size);
        node->encoded_size = (uint32_t)size;
    } else {
        /* Large data goes to a heap allocation */
        node->embed_data = (uint8_t *)malloc(size);
        if (!node->embed_data) {
            /* Roll back pool allocation -- best-effort */
            b->node_block_used--;
            return;
        }
        memcpy(node->embed_data, data, size);
        node->embed_size = (uint32_t)size;
        node->encoded_size = 0; /* signal: use embed_data instead */
    }

    phx_builder_append_node(b, node);
}

/* ------------------------------------------------------------------ */
/*  Operand constructors                                               */
/* ------------------------------------------------------------------ */

PhxMem phx_ptr(PhxGp base, int32_t offset) {
    PhxMem m;
    memset(&m, 0, sizeof(m));
    m.base = base;
    m.offset = offset;
    m.scale = 1;
    m.size = base.size;
    m.has_index = 0;
    return m;
}

PhxMem phx_ptr_index(PhxGp base, PhxGp index, uint8_t scale,
                     int32_t offset) {
    PhxMem m;
    memset(&m, 0, sizeof(m));
    m.base = base;
    m.index = index;
    m.offset = offset;
    m.scale = scale;
    m.size = base.size;
    m.has_index = 1;
    return m;
}

PhxOperand phx_op_gp(PhxGp gp) {
    PhxOperand op;
    memset(&op, 0, sizeof(op));
    op.type = PHX_OP_GP;
    op.gp = gp;
    return op;
}

PhxOperand phx_op_mem(PhxMem mem) {
    PhxOperand op;
    memset(&op, 0, sizeof(op));
    op.type = PHX_OP_MEM;
    op.mem = mem;
    return op;
}

PhxOperand phx_op_imm(int64_t imm) {
    PhxOperand op;
    memset(&op, 0, sizeof(op));
    op.type = PHX_OP_IMM;
    op.imm = imm;
    return op;
}

PhxOperand phx_op_label(PhxLabel label) {
    PhxOperand op;
    memset(&op, 0, sizeof(op));
    op.type = PHX_OP_LABEL;
    op.label = label;
    return op;
}

/* ------------------------------------------------------------------ */
/*  Finalize (common entry point)                                      */
/* ------------------------------------------------------------------ */

int phx_finalize(PhxBuilder *b) {
    assert(b != NULL);
    assert(b->code != NULL);

    if (b->code->arch == PHX_ARCH_X86_64) {
        return phx_x86_finalize(b);
    } else {
        return phx_a64_finalize(b);
    }
}
