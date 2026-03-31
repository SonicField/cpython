/*
 * phoenix_asm.h -- Public API for the Phoenix assembler
 *
 * A pure-C, node-list assembler that supports deferred code generation
 * via cursor repositioning (setCursor). This is the infrastructure layer;
 * instruction encoding is handled by arch-specific backends (x86_64.c,
 * arm64.c) built on top of this API.
 *
 * C11, no C++ dependencies.
 */

#ifndef PHOENIX_ASM_H
#define PHOENIX_ASM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/*  Operand types                                                      */
/* ------------------------------------------------------------------ */

/* Register operand */
typedef struct {
    uint8_t id;     /* register number (arch-specific) */
    uint8_t size;   /* operand size in bytes: 1, 2, 4, 8 */
} PhxGp;

/* Memory operand */
typedef struct {
    PhxGp    base;
    PhxGp    index;       /* optional (x86 SIB addressing) */
    int32_t  offset;
    uint8_t  scale;       /* 1, 2, 4, or 8 (x86 SIB) */
    uint8_t  size;        /* access size in bytes */
    uint8_t  has_index;   /* nonzero if index register is used */
    uint8_t  is_label_rel;/* nonzero for RIP-relative label addressing */
    uint8_t  segment;     /* segment override: 0=none, 4=FS, 5=GS */
    uint32_t label_id;    /* target label (when is_label_rel != 0) */
} PhxMem;

/* Label (opaque handle) */
typedef struct {
    uint32_t id;
} PhxLabel;

/* Generic operand (tagged union) */
typedef enum {
    PHX_OP_NONE  = 0,
    PHX_OP_GP    = 1,
    PHX_OP_MEM   = 2,
    PHX_OP_LABEL = 3,
    PHX_OP_IMM   = 4
} PhxOpType;

typedef struct {
    PhxOpType type;
    union {
        PhxGp    gp;
        PhxMem   mem;
        PhxLabel label;
        int64_t  imm;
    };
} PhxOperand;

/* ------------------------------------------------------------------ */
/*  Architecture                                                       */
/* ------------------------------------------------------------------ */

typedef enum {
    PHX_ARCH_X86_64 = 0,
    PHX_ARCH_ARM64  = 1
} PhxArch;

/* ------------------------------------------------------------------ */
/*  Instruction node (doubly-linked list element)                      */
/* ------------------------------------------------------------------ */

/* Node types */
#define PHX_NODE_INST   0
#define PHX_NODE_LABEL  1
#define PHX_NODE_ALIGN  2
#define PHX_NODE_EMBED  3

/* Maximum encoded bytes per node (x86 instructions are at most 15 bytes;
   ARM64 is always 4, but embed nodes can be larger -- they use the
   embedded data pointer in the operand area instead). */
#define PHX_MAX_ENCODED 16

typedef struct PhxNode {
    struct PhxNode *prev;
    struct PhxNode *next;

    uint16_t   opcode;          /* phoenix-asm internal opcode */
    uint8_t    num_operands;
    uint8_t    node_type;       /* PHX_NODE_INST / LABEL / ALIGN / EMBED */

    PhxOperand operands[4];     /* max 4 operands per instruction */

    uint8_t    encoded[PHX_MAX_ENCODED]; /* filled during finalize/emit */
    uint32_t   encoded_size;    /* number of valid bytes in encoded[] */
    uint32_t   offset;          /* byte offset in final output buffer */

    uint32_t   label_id;        /* for label nodes: which label this binds */

    /* embed-node payload (for data longer than PHX_MAX_ENCODED) */
    uint8_t   *embed_data;      /* heap-allocated when embed_size > 0 */
    uint32_t   embed_size;
} PhxNode;

/* ------------------------------------------------------------------ */
/*  Label fixup (forward references resolved during finalize)          */
/* ------------------------------------------------------------------ */

typedef struct {
    PhxNode  *node;         /* instruction node containing the reference */
    uint32_t  label_id;     /* target label */
    uint8_t   operand_idx;  /* which operand holds the label ref */
} PhxFixup;

/* ------------------------------------------------------------------ */
/*  Code holder (output buffer)                                        */
/* ------------------------------------------------------------------ */

typedef struct {
    PhxArch   arch;
    uint8_t  *buffer;          /* finalized code bytes */
    size_t    buffer_size;
    size_t    buffer_capacity;
} PhxCodeHolder;

/* ------------------------------------------------------------------ */
/*  Builder (the central assembler object)                             */
/* ------------------------------------------------------------------ */

typedef struct {
    PhxCodeHolder *code;

    /* Doubly-linked node list */
    PhxNode *head;
    PhxNode *tail;
    PhxNode *cursor;       /* current insertion point */

    /* Label management */
    uint32_t  next_label_id;
    PhxNode **label_nodes;     /* array indexed by label_id */
    uint32_t  label_capacity;

    /* Forward-reference fixups */
    PhxFixup *fixups;
    uint32_t  fixup_count;
    uint32_t  fixup_capacity;

    /* Node block allocator (stable addresses -- never realloc) */
    struct PhxNodeBlock *node_blocks;  /* linked list of blocks */
    uint32_t  node_block_used;        /* used count in current block */
} PhxBuilder;

/* ------------------------------------------------------------------ */
/*  CodeHolder API                                                     */
/* ------------------------------------------------------------------ */

PhxCodeHolder *phx_code_create(PhxArch arch);
void           phx_code_destroy(PhxCodeHolder *code);

/* ------------------------------------------------------------------ */
/*  Builder lifecycle                                                  */
/* ------------------------------------------------------------------ */

PhxBuilder *phx_builder_create(PhxCodeHolder *code);
void        phx_builder_destroy(PhxBuilder *builder);

/* ------------------------------------------------------------------ */
/*  Cursor management                                                  */
/* ------------------------------------------------------------------ */

PhxNode *phx_builder_cursor(PhxBuilder *b);
void     phx_builder_set_cursor(PhxBuilder *b, PhxNode *node);

/* ------------------------------------------------------------------ */
/*  Label management                                                   */
/* ------------------------------------------------------------------ */

PhxLabel phx_builder_new_label(PhxBuilder *b);
void     phx_builder_bind(PhxBuilder *b, PhxLabel label);

/* ------------------------------------------------------------------ */
/*  Alignment and raw data embedding                                   */
/* ------------------------------------------------------------------ */

void phx_builder_align(PhxBuilder *b, int alignment);
void phx_builder_embed(PhxBuilder *b, const void *data, size_t size);

/* ------------------------------------------------------------------ */
/*  Internal helpers (exposed for arch backends)                       */
/* ------------------------------------------------------------------ */

PhxNode *phx_builder_alloc_node(PhxBuilder *b);
PhxNode *phx_builder_append_node(PhxBuilder *b, PhxNode *node);
void     phx_builder_add_fixup(PhxBuilder *b, PhxNode *node,
                               uint32_t label_id, uint8_t operand_idx);

/* ------------------------------------------------------------------ */
/*  Finalize (common entry point)                                      */
/* ------------------------------------------------------------------ */

/* Resolve all label fixups and linearize the node list into the code
 * holder's buffer.  Dispatches to the arch-specific finalize function
 * (phx_x86_finalize or phx_a64_finalize) based on code->arch.
 * Returns 0 on success, nonzero on error. */
int phx_finalize(PhxBuilder *b);

/* ------------------------------------------------------------------ */
/*  Operand constructors                                               */
/* ------------------------------------------------------------------ */

PhxMem     phx_ptr(PhxGp base, int32_t offset);
PhxMem     phx_ptr_index(PhxGp base, PhxGp index, uint8_t scale,
                         int32_t offset);
PhxOperand phx_op_gp(PhxGp gp);
PhxOperand phx_op_mem(PhxMem mem);
PhxOperand phx_op_imm(int64_t imm);
PhxOperand phx_op_label(PhxLabel label);

#ifdef __cplusplus
}
#endif

#endif /* PHOENIX_ASM_H */
