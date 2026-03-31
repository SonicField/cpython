/*
 * x86_64.c -- x86_64 instruction encoding backend for phoenix-asm
 *
 * Encodes 52 x86_64 instructions into PhxNode encoded[] buffers.
 * Each instruction function allocates a node, encodes the machine code bytes,
 * and appends the node to the builder's instruction list.
 *
 * x86_64 encoding reference:
 *   REX prefix: 0100 WRXB
 *     W = 64-bit operand size
 *     R = extends ModR/M reg field (bit 3)
 *     X = extends SIB index field (bit 3)
 *     B = extends ModR/M r/m or SIB base field (bit 3)
 *   ModR/M: mod(2) + reg(3) + r/m(3)
 *   SIB: scale(2) + index(3) + base(3)
 *
 * C11, no C++ dependencies.
 */

#include "x86_64.h"
#include "phoenix_asm.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ================================================================== */
/*  Internal encoding helpers                                          */
/* ================================================================== */

/* Extract low 3 bits of register id (for ModR/M / SIB fields) */
static inline uint8_t reg3(PhxGp r) {
    return r.id & 0x07;
}

/* Check if register requires REX.B/R extension (id >= 8) */
static inline int reg_ext(PhxGp r) {
    return (r.id >> 3) & 1;
}

/* Check if a displacement fits in a signed 8-bit value */
static inline int fits_i8(int32_t v) {
    return v >= -128 && v <= 127;
}

/* Check if an immediate fits in a signed 32-bit value */
static inline int fits_i32(int64_t v) {
    return v >= (int64_t)INT32_MIN && v <= (int64_t)INT32_MAX;
}

/* Check if an unsigned immediate fits in 32 bits */
static inline int fits_u32(int64_t v) {
    return v >= 0 && v <= (int64_t)UINT32_MAX;
}

/* ------------------------------------------------------------------ */
/*  REX prefix builders                                                */
/* ------------------------------------------------------------------ */

/*
 * Build a REX byte for a reg-reg operation.
 *   w:   1 for 64-bit operand size, 0 otherwise
 *   reg: the register in the ModR/M reg field
 *   rm:  the register in the ModR/M r/m field
 *
 * Returns the REX byte, or 0 if no REX is needed.
 */
static uint8_t make_rex(int w, PhxGp reg, PhxGp rm) {
    uint8_t rex = 0x40;
    if (w) rex |= 0x08;
    if (reg_ext(reg)) rex |= 0x04;  /* REX.R */
    if (reg_ext(rm))  rex |= 0x01;  /* REX.B */
    return (rex != 0x40 || w) ? rex : 0;
}

/*
 * Build a REX byte for a reg-mem operation.
 * Accounts for the memory operand's base and optional index register.
 */
static uint8_t make_rex_mem(int w, PhxGp reg, PhxMem mem) {
    uint8_t rex = 0x40;
    if (w) rex |= 0x08;
    if (reg_ext(reg))      rex |= 0x04;  /* REX.R */
    if (!mem.is_abs_addr && !mem.is_label_rel) {
        if (mem.has_index && reg_ext(mem.index)) rex |= 0x02;  /* REX.X */
        if (reg_ext(mem.base)) rex |= 0x01;  /* REX.B */
    }
    return (rex != 0x40) ? rex : 0;
}

/*
 * Build a REX byte for a single-register operation (e.g., push, pop, inc).
 * The register occupies the r/m or opcode-embedded position.
 */
static uint8_t make_rex_single(int w, PhxGp rm) {
    uint8_t rex = 0x40;
    if (w) rex |= 0x08;
    if (reg_ext(rm)) rex |= 0x01;  /* REX.B */
    return (rex != 0x40) ? rex : 0;
}

/*
 * Build REX for operations that need REX for byte register access (SPL, etc.)
 * Byte registers with id >= 4 need REX prefix even without the W bit.
 */
static inline int needs_rex_for_byte(PhxGp r) {
    return r.size == 1 && r.id >= 4;
}

/* ------------------------------------------------------------------ */
/*  ModR/M + SIB + displacement encoding                               */
/* ------------------------------------------------------------------ */

/*
 * Determine the SIB scale field bits from a scale multiplier (1,2,4,8).
 */
static uint8_t scale_bits(uint8_t scale) {
    switch (scale) {
        case 1: return 0x00;
        case 2: return 0x40;
        case 4: return 0x80;
        case 8: return 0xC0;
        default: assert(0 && "invalid SIB scale"); return 0;
    }
}

/*
 * Encode ModR/M byte + optional SIB + displacement for a reg-memory operation.
 *
 *   out: output buffer (must have room for at least 6 bytes)
 *   reg_field: the 3-bit value for the ModR/M reg field (register id or opcode extension)
 *   mem: the memory operand
 *
 * Returns the number of bytes written.
 *
 * Key special cases handled:
 *   - RSP/R12 (id & 7 == 4) as base: always needs SIB byte
 *   - RBP/R13 (id & 7 == 5) with no displacement: must use mod=01, disp8=0
 *   - SIB index addressing
 */
/* Emit segment override prefix if set on the memory operand.
 * FS = 0x64, GS = 0x65. Returns number of bytes emitted (0 or 1). */
static int emit_segment_prefix(uint8_t *out, PhxMem mem) {
    if (mem.segment == 4) { out[0] = 0x64; return 1; }  /* FS */
    if (mem.segment == 5) { out[0] = 0x65; return 1; }  /* GS */
    return 0;
}

/* Encode RIP-relative addressing: [RIP + disp32]
 * mod=00, r/m=101, followed by 4-byte displacement (placeholder).
 * The displacement is patched during finalize via a label fixup.
 * If mem has an index register, encode as [RIP + disp32 + index*scale]
 * using SIB with base=101 (disp32). */
static int encode_modrm_rip_rel(uint8_t *out, uint8_t reg_field, PhxMem mem) {
    int pos = 0;
    if (mem.has_index) {
        /* SIB form: mod=00, r/m=100, SIB base=101 (disp32) */
        out[pos++] = 0x00 | (reg_field << 3) | 0x04;
        out[pos++] = scale_bits(mem.scale) | (reg3(mem.index) << 3) | 0x05;
    } else {
        /* Simple RIP-relative: mod=00, r/m=101 */
        out[pos++] = 0x00 | (reg_field << 3) | 0x05;
    }
    /* 4-byte displacement placeholder (filled by finalize) */
    int32_t disp = mem.offset;
    memcpy(out + pos, &disp, 4);
    pos += 4;
    return pos;
}

static int encode_modrm_mem(uint8_t *out, uint8_t reg_field, PhxMem mem) {
    /* Label-relative → RIP-relative encoding */
    if (mem.is_label_rel) {
        return encode_modrm_rip_rel(out, reg_field, mem);
    }

    /* Absolute address → SIB disp32 encoding: [disp32] with no base/index.
       ModR/M: mod=00, r/m=100 (SIB follows)
       SIB: scale=00, index=100 (none), base=101 (disp32 only)
       This works for addresses that fit in signed 32-bit. */
    if (mem.is_abs_addr) {
        int32_t addr32 = (int32_t)(mem.abs_addr & 0xFFFFFFFF);
        out[0] = 0x00 | (reg_field << 3) | 0x04; /* ModR/M: mod=00, r/m=100 */
        out[1] = 0x25; /* SIB: scale=00, index=100(none), base=101(disp32) */
        memcpy(out + 2, &addr32, 4);
        return 6;
    }

    int pos = 0;
    uint8_t base3 = reg3(mem.base);
    int32_t disp = mem.offset;

    /* Determine mod field based on displacement */
    uint8_t mod;
    if (disp == 0 && base3 != 5) {
        /* RBP/R13 (base3==5) with disp==0 would be RIP-relative, so we
           force mod=01 for them.  For all other bases, mod=00 is fine. */
        mod = 0x00;
    } else if (fits_i8(disp)) {
        mod = 0x40;
    } else {
        mod = 0x80;
    }

    if (mem.has_index) {
        /* SIB addressing: [base + index * scale + disp] */
        out[pos++] = mod | (reg_field << 3) | 0x04;  /* r/m = 100 => SIB follows */
        out[pos++] = scale_bits(mem.scale) | (reg3(mem.index) << 3) | base3;
    } else if (base3 == 4) {
        /* RSP/R12 as base: must use SIB byte with index = RSP (none) */
        out[pos++] = mod | (reg_field << 3) | 0x04;
        out[pos++] = 0x00 | (0x04 << 3) | base3;  /* scale=1, index=RSP(none), base */
    } else {
        /* Simple [base + disp] */
        out[pos++] = mod | (reg_field << 3) | base3;
    }

    /* Emit displacement */
    if (mod == 0x40) {
        out[pos++] = (uint8_t)(disp & 0xFF);
    } else if (mod == 0x80) {
        memcpy(out + pos, &disp, 4);
        pos += 4;
    }

    return pos;
}

/*
 * Encode ModR/M byte for a register-register operation.
 *
 *   out: output buffer (must have room for 1 byte)
 *   reg_field: 3-bit value for the ModR/M reg field
 *   rm: the register in the r/m field
 *
 * Returns 1 (always exactly one byte for reg-reg).
 */
static int encode_modrm_rr(uint8_t *out, uint8_t reg_field, PhxGp rm) {
    out[0] = 0xC0 | (reg_field << 3) | reg3(rm);
    return 1;
}

/* ------------------------------------------------------------------ */
/*  Operand size prefix                                                */
/* ------------------------------------------------------------------ */

/*
 * Determine if a 66h operand-size override prefix is needed.
 * Returns 1 if the register operand is 16-bit, 0 otherwise.
 */
static inline int needs_66h(PhxGp r) {
    return r.size == 2;
}

/* ------------------------------------------------------------------ */
/*  Common emit pattern: alloc node, encode, append                    */
/*                                                                     */
/*  Most instruction functions follow this pattern:                    */
/*    1. Allocate node from builder                                    */
/*    2. Set metadata (opcode, operands)                               */
/*    3. Encode bytes into node->encoded[]                             */
/*    4. Append node to builder list                                   */
/* ------------------------------------------------------------------ */

/* Determine if an operation should use REX.W based on register size */
static inline int want_rexw(PhxGp r) {
    return r.size == 8;
}

/* Determine if an operation should use REX.W based on memory access size */
static inline int want_rexw_mem(PhxMem m) {
    return m.size == 8;
}

/* ================================================================== */
/*  ALU instruction family helper                                      */
/*                                                                     */
/*  ADD, SUB, CMP, AND, OR, XOR share the same encoding structure:    */
/*    - opcode byte = base + direction bits                            */
/*    - /r for reg-reg and reg-mem                                     */
/*    - /digit + imm for immediate forms                               */
/*  We factor out the common encoding logic.                           */
/* ================================================================== */

/*
 * ALU opcode encoding table:
 *   ADD: base=0x00, imm_ext=0
 *   OR:  base=0x08, imm_ext=1
 *   AND: base=0x20, imm_ext=4
 *   SUB: base=0x28, imm_ext=5
 *   XOR: base=0x30, imm_ext=6
 *   CMP: base=0x38, imm_ext=7
 */

/* ALU reg, reg: opcode_base+1 /r (for 32/64-bit) */
static void emit_alu_rr(PhxBuilder *b, uint16_t phx_op, uint8_t base,
                         PhxGp dst, PhxGp src) {
    PhxNode *n = phx_builder_alloc_node(b);
    if (!n) return;
    n->node_type = PHX_NODE_INST;
    n->opcode = phx_op;
    n->operands[0] = phx_op_gp(dst);
    n->operands[1] = phx_op_gp(src);
    n->num_operands = 2;

    int pos = 0;
    if (needs_66h(dst)) n->encoded[pos++] = 0x66;
    uint8_t rex = make_rex(want_rexw(dst), src, dst);
    if (rex || needs_rex_for_byte(dst) || needs_rex_for_byte(src))
        n->encoded[pos++] = rex ? rex : 0x40;
    if (dst.size == 1)
        n->encoded[pos++] = base;       /* 8-bit */
    else
        n->encoded[pos++] = base + 1;   /* 16/32/64-bit */
    pos += encode_modrm_rr(n->encoded + pos, reg3(src), dst);
    n->encoded_size = pos;
    phx_builder_append_node(b, n);
}

/* ALU reg, mem: opcode_base+3 /r (load direction) */
static void emit_alu_rm(PhxBuilder *b, uint16_t phx_op, uint8_t base,
                         PhxGp dst, PhxMem src) {
    PhxNode *n = phx_builder_alloc_node(b);
    if (!n) return;
    n->node_type = PHX_NODE_INST;
    n->opcode = phx_op;
    n->operands[0] = phx_op_gp(dst);
    n->operands[1] = phx_op_mem(src);
    n->num_operands = 2;

    int pos = 0;
    if (needs_66h(dst)) n->encoded[pos++] = 0x66;
    uint8_t rex = make_rex_mem(want_rexw(dst), dst, src);
    if (rex || needs_rex_for_byte(dst))
        n->encoded[pos++] = rex ? rex : 0x40;
    if (dst.size == 1)
        n->encoded[pos++] = base + 2;   /* 8-bit load */
    else
        n->encoded[pos++] = base + 3;   /* 16/32/64-bit load */
    pos += encode_modrm_mem(n->encoded + pos, reg3(dst), src);
    n->encoded_size = pos;
    phx_builder_append_node(b, n);
}

/* ALU mem, reg: opcode_base+1 /r (store direction, 32/64) or base+0 (8-bit) */
static void emit_alu_mr(PhxBuilder *b, uint16_t phx_op, uint8_t base,
                         PhxMem dst, PhxGp src) {
    PhxNode *n = phx_builder_alloc_node(b);
    if (!n) return;
    n->node_type = PHX_NODE_INST;
    n->opcode = phx_op;
    n->operands[0] = phx_op_mem(dst);
    n->operands[1] = phx_op_gp(src);
    n->num_operands = 2;

    int pos = 0;
    if (needs_66h(src)) n->encoded[pos++] = 0x66;
    uint8_t rex = make_rex_mem(want_rexw(src), src, dst);
    if (rex || needs_rex_for_byte(src))
        n->encoded[pos++] = rex ? rex : 0x40;
    if (src.size == 1)
        n->encoded[pos++] = base;       /* 8-bit store */
    else
        n->encoded[pos++] = base + 1;   /* 16/32/64-bit store */
    pos += encode_modrm_mem(n->encoded + pos, reg3(src), dst);
    n->encoded_size = pos;
    phx_builder_append_node(b, n);
}

/* ALU reg, imm: 0x83 /ext imm8  or  0x81 /ext imm32  (or 0x80/0x82 for 8-bit) */
static void emit_alu_ri(PhxBuilder *b, uint16_t phx_op, uint8_t imm_ext,
                         PhxGp dst, int32_t imm) {
    PhxNode *n = phx_builder_alloc_node(b);
    if (!n) return;
    n->node_type = PHX_NODE_INST;
    n->opcode = phx_op;
    n->operands[0] = phx_op_gp(dst);
    n->operands[1] = phx_op_imm(imm);
    n->num_operands = 2;

    int pos = 0;
    if (needs_66h(dst)) n->encoded[pos++] = 0x66;
    uint8_t rex = make_rex_single(want_rexw(dst), dst);
    if (rex || needs_rex_for_byte(dst))
        n->encoded[pos++] = rex ? rex : 0x40;

    if (dst.size == 1) {
        /* 8-bit: 0x80 /ext imm8 */
        n->encoded[pos++] = 0x80;
        pos += encode_modrm_rr(n->encoded + pos, imm_ext, dst);
        n->encoded[pos++] = (uint8_t)(imm & 0xFF);
    } else if (fits_i8(imm)) {
        /* sign-extended imm8 to 16/32/64: 0x83 /ext imm8 */
        n->encoded[pos++] = 0x83;
        pos += encode_modrm_rr(n->encoded + pos, imm_ext, dst);
        n->encoded[pos++] = (uint8_t)(imm & 0xFF);
    } else if (dst.id == 0) {
        /* Accumulator short form: opcode = imm_ext*8 + 5, imm32
         * ADD EAX/RAX: 05, OR: 0D, AND: 25, SUB: 2D, XOR: 35, CMP: 3D */
        n->encoded[pos++] = (uint8_t)(imm_ext * 8 + 5);
        memcpy(n->encoded + pos, &imm, 4);
        pos += 4;
    } else {
        /* General form: 0x81 /ext imm32 */
        n->encoded[pos++] = 0x81;
        pos += encode_modrm_rr(n->encoded + pos, imm_ext, dst);
        memcpy(n->encoded + pos, &imm, 4);
        pos += 4;
    }
    n->encoded_size = pos;
    phx_builder_append_node(b, n);
}

/* ALU mem, imm: 0x83 /ext imm8  or  0x81 /ext imm32 */
static void emit_alu_mi(PhxBuilder *b, uint16_t phx_op, uint8_t imm_ext,
                         PhxMem dst, int32_t imm) {
    PhxNode *n = phx_builder_alloc_node(b);
    if (!n) return;
    n->node_type = PHX_NODE_INST;
    n->opcode = phx_op;
    n->operands[0] = phx_op_mem(dst);
    n->operands[1] = phx_op_imm(imm);
    n->num_operands = 2;

    int pos = 0;
    if (dst.size == 2) n->encoded[pos++] = 0x66;
    uint8_t rex = make_rex_mem(want_rexw_mem(dst), (PhxGp){0, 0}, dst);
    /* For memory-immediate, the "reg" field is the opcode extension, no REX.R */
    /* Recalculate: only base and index contribute */
    {
        uint8_t r = 0x40;
        if (want_rexw_mem(dst)) r |= 0x08;
        if (dst.has_index && reg_ext(dst.index)) r |= 0x02;
        if (reg_ext(dst.base)) r |= 0x01;
        rex = (r != 0x40) ? r : 0;
    }
    if (rex) n->encoded[pos++] = rex;

    if (dst.size == 1) {
        n->encoded[pos++] = 0x80;
        pos += encode_modrm_mem(n->encoded + pos, imm_ext, dst);
        n->encoded[pos++] = (uint8_t)(imm & 0xFF);
    } else if (fits_i8(imm)) {
        n->encoded[pos++] = 0x83;
        pos += encode_modrm_mem(n->encoded + pos, imm_ext, dst);
        n->encoded[pos++] = (uint8_t)(imm & 0xFF);
    } else {
        n->encoded[pos++] = 0x81;
        pos += encode_modrm_mem(n->encoded + pos, imm_ext, dst);
        memcpy(n->encoded + pos, &imm, 4);
        pos += 4;
    }
    n->encoded_size = pos;
    phx_builder_append_node(b, n);
}

/* ================================================================== */
/*  Unary instruction family helper (NEG, NOT, INC, DEC, IDIV, DIV)   */
/* ================================================================== */

/* Unary reg: 0xF7 /ext (32/64-bit) or 0xF6 /ext (8-bit) */
static void emit_unary_r(PhxBuilder *b, uint16_t phx_op, uint8_t ext,
                          PhxGp dst) {
    PhxNode *n = phx_builder_alloc_node(b);
    if (!n) return;
    n->node_type = PHX_NODE_INST;
    n->opcode = phx_op;
    n->operands[0] = phx_op_gp(dst);
    n->num_operands = 1;

    int pos = 0;
    if (needs_66h(dst)) n->encoded[pos++] = 0x66;
    uint8_t rex = make_rex_single(want_rexw(dst), dst);
    if (rex) n->encoded[pos++] = rex;
    if (dst.size == 1)
        n->encoded[pos++] = 0xF6;
    else
        n->encoded[pos++] = 0xF7;
    pos += encode_modrm_rr(n->encoded + pos, ext, dst);
    n->encoded_size = pos;
    phx_builder_append_node(b, n);
}

/* Unary mem: 0xF7 /ext (32/64-bit) or 0xF6 /ext (8-bit) */
static void emit_unary_m(PhxBuilder *b, uint16_t phx_op, uint8_t ext,
                          PhxMem dst) {
    PhxNode *n = phx_builder_alloc_node(b);
    if (!n) return;
    n->node_type = PHX_NODE_INST;
    n->opcode = phx_op;
    n->operands[0] = phx_op_mem(dst);
    n->num_operands = 1;

    int pos = 0;
    if (dst.size == 2) n->encoded[pos++] = 0x66;
    {
        uint8_t r = 0x40;
        if (want_rexw_mem(dst)) r |= 0x08;
        if (dst.has_index && reg_ext(dst.index)) r |= 0x02;
        if (reg_ext(dst.base)) r |= 0x01;
        if (r != 0x40) n->encoded[pos++] = r;
    }
    if (dst.size == 1)
        n->encoded[pos++] = 0xF6;
    else
        n->encoded[pos++] = 0xF7;
    pos += encode_modrm_mem(n->encoded + pos, ext, dst);
    n->encoded_size = pos;
    phx_builder_append_node(b, n);
}

/* INC/DEC reg: 0xFF /0 (INC) or 0xFF /1 (DEC) for 32/64-bit
   8-bit: 0xFE /0 or /1 */
static void emit_incdec_r(PhxBuilder *b, uint16_t phx_op, uint8_t ext,
                            PhxGp dst) {
    PhxNode *n = phx_builder_alloc_node(b);
    if (!n) return;
    n->node_type = PHX_NODE_INST;
    n->opcode = phx_op;
    n->operands[0] = phx_op_gp(dst);
    n->num_operands = 1;

    int pos = 0;
    if (needs_66h(dst)) n->encoded[pos++] = 0x66;
    uint8_t rex = make_rex_single(want_rexw(dst), dst);
    if (rex || needs_rex_for_byte(dst))
        n->encoded[pos++] = rex ? rex : 0x40;
    if (dst.size == 1)
        n->encoded[pos++] = 0xFE;
    else
        n->encoded[pos++] = 0xFF;
    pos += encode_modrm_rr(n->encoded + pos, ext, dst);
    n->encoded_size = pos;
    phx_builder_append_node(b, n);
}

/* INC/DEC mem: 0xFF /0 or /1 (32/64-bit), 0xFE /0 or /1 (8-bit) */
static void emit_incdec_m(PhxBuilder *b, uint16_t phx_op, uint8_t ext,
                            PhxMem dst) {
    PhxNode *n = phx_builder_alloc_node(b);
    if (!n) return;
    n->node_type = PHX_NODE_INST;
    n->opcode = phx_op;
    n->operands[0] = phx_op_mem(dst);
    n->num_operands = 1;

    int pos = 0;
    if (dst.size == 2) n->encoded[pos++] = 0x66;
    {
        uint8_t r = 0x40;
        if (want_rexw_mem(dst)) r |= 0x08;
        if (dst.has_index && reg_ext(dst.index)) r |= 0x02;
        if (reg_ext(dst.base)) r |= 0x01;
        if (r != 0x40) n->encoded[pos++] = r;
    }
    if (dst.size == 1)
        n->encoded[pos++] = 0xFE;
    else
        n->encoded[pos++] = 0xFF;
    pos += encode_modrm_mem(n->encoded + pos, ext, dst);
    n->encoded_size = pos;
    phx_builder_append_node(b, n);
}

/* ================================================================== */
/*  SSE instruction family helper                                      */
/*  Many SSE instructions share prefix + 0F + opcode + ModR/M format  */
/* ================================================================== */

/* SSE reg-reg: [prefix] 0F opcode ModR/M */
static void emit_sse_rr(PhxBuilder *b, uint16_t phx_op,
                         uint8_t prefix, uint8_t opcode,
                         PhxGp dst, PhxGp src) {
    PhxNode *n = phx_builder_alloc_node(b);
    if (!n) return;
    n->node_type = PHX_NODE_INST;
    n->opcode = phx_op;
    n->operands[0] = phx_op_gp(dst);
    n->operands[1] = phx_op_gp(src);
    n->num_operands = 2;

    int pos = 0;
    if (prefix) n->encoded[pos++] = prefix;
    /* REX for extended XMM registers (XMM8-15) */
    uint8_t rex = make_rex(0, dst, src);
    if (rex) n->encoded[pos++] = rex;
    n->encoded[pos++] = 0x0F;
    n->encoded[pos++] = opcode;
    pos += encode_modrm_rr(n->encoded + pos, reg3(dst), src);
    n->encoded_size = pos;
    phx_builder_append_node(b, n);
}

/* SSE reg-mem: [prefix] [REX] 0F opcode ModR/M [SIB] [disp] */
static void emit_sse_rm(PhxBuilder *b, uint16_t phx_op,
                         uint8_t prefix, uint8_t opcode,
                         PhxGp dst, PhxMem src) {
    PhxNode *n = phx_builder_alloc_node(b);
    if (!n) return;
    n->node_type = PHX_NODE_INST;
    n->opcode = phx_op;
    n->operands[0] = phx_op_gp(dst);
    n->operands[1] = phx_op_mem(src);
    n->num_operands = 2;

    int pos = 0;
    if (prefix) n->encoded[pos++] = prefix;
    uint8_t rex = make_rex_mem(0, dst, src);
    if (rex) n->encoded[pos++] = rex;
    n->encoded[pos++] = 0x0F;
    n->encoded[pos++] = opcode;
    pos += encode_modrm_mem(n->encoded + pos, reg3(dst), src);
    n->encoded_size = pos;
    phx_builder_append_node(b, n);
}

/* SSE mem-reg: [prefix] [REX] 0F opcode ModR/M [SIB] [disp] */
static void emit_sse_mr(PhxBuilder *b, uint16_t phx_op,
                         uint8_t prefix, uint8_t opcode,
                         PhxMem dst, PhxGp src) {
    PhxNode *n = phx_builder_alloc_node(b);
    if (!n) return;
    n->node_type = PHX_NODE_INST;
    n->opcode = phx_op;
    n->operands[0] = phx_op_mem(dst);
    n->operands[1] = phx_op_gp(src);
    n->num_operands = 2;

    int pos = 0;
    if (prefix) n->encoded[pos++] = prefix;
    uint8_t rex = make_rex_mem(0, src, dst);
    if (rex) n->encoded[pos++] = rex;
    n->encoded[pos++] = 0x0F;
    n->encoded[pos++] = opcode;
    pos += encode_modrm_mem(n->encoded + pos, reg3(src), dst);
    n->encoded_size = pos;
    phx_builder_append_node(b, n);
}

/* SSE with 66 0F 38 or 66 0F 3A prefix family (3-byte opcode) */
static void emit_sse38_rr(PhxBuilder *b, uint16_t phx_op,
                           uint8_t opcode, PhxGp dst, PhxGp src) {
    PhxNode *n = phx_builder_alloc_node(b);
    if (!n) return;
    n->node_type = PHX_NODE_INST;
    n->opcode = phx_op;
    n->operands[0] = phx_op_gp(dst);
    n->operands[1] = phx_op_gp(src);
    n->num_operands = 2;

    int pos = 0;
    n->encoded[pos++] = 0x66;
    uint8_t rex = make_rex(0, dst, src);
    if (rex) n->encoded[pos++] = rex;
    n->encoded[pos++] = 0x0F;
    n->encoded[pos++] = 0x38;
    n->encoded[pos++] = opcode;
    pos += encode_modrm_rr(n->encoded + pos, reg3(dst), src);
    n->encoded_size = pos;
    phx_builder_append_node(b, n);
}

/* ================================================================== */
/*                                                                     */
/*  PUBLIC INSTRUCTION FUNCTIONS                                       */
/*                                                                     */
/* ================================================================== */

/* ================================================================== */
/*  MOV                                                                */
/* ================================================================== */

/* MOV reg, reg:  89 /r (store direction) or 8B /r (load direction)
   We use 89 /r: src -> dst, ModR/M reg=src, r/m=dst */
void phx_x86_mov_rr(PhxBuilder *b, PhxGp dst, PhxGp src) {
    PhxNode *n = phx_builder_alloc_node(b);
    if (!n) return;
    n->node_type = PHX_NODE_INST;
    n->opcode = PHX_OP_MOV;
    n->operands[0] = phx_op_gp(dst);
    n->operands[1] = phx_op_gp(src);
    n->num_operands = 2;

    int pos = 0;
    if (needs_66h(dst)) n->encoded[pos++] = 0x66;
    uint8_t rex = make_rex(want_rexw(dst), src, dst);
    if (rex || needs_rex_for_byte(dst) || needs_rex_for_byte(src))
        n->encoded[pos++] = rex ? rex : 0x40;
    if (dst.size == 1)
        n->encoded[pos++] = 0x88;
    else
        n->encoded[pos++] = 0x89;
    pos += encode_modrm_rr(n->encoded + pos, reg3(src), dst);
    n->encoded_size = pos;
    phx_builder_append_node(b, n);
}

/* MOV reg, mem:  8B /r (32/64-bit) or 8A /r (8-bit) */
void phx_x86_mov_rm(PhxBuilder *b, PhxGp dst, PhxMem src) {
    PhxNode *n = phx_builder_alloc_node(b);
    if (!n) return;
    n->node_type = PHX_NODE_INST;
    n->opcode = PHX_OP_MOV;
    n->operands[0] = phx_op_gp(dst);
    n->operands[1] = phx_op_mem(src);
    n->num_operands = 2;

    int pos = 0;
    pos += emit_segment_prefix(n->encoded + pos, src);
    if (needs_66h(dst)) n->encoded[pos++] = 0x66;
    uint8_t rex = make_rex_mem(want_rexw(dst), dst, src);
    if (rex || needs_rex_for_byte(dst))
        n->encoded[pos++] = rex ? rex : 0x40;
    if (dst.size == 1)
        n->encoded[pos++] = 0x8A;
    else
        n->encoded[pos++] = 0x8B;
    pos += encode_modrm_mem(n->encoded + pos, reg3(dst), src);
    n->encoded_size = pos;
    phx_builder_append_node(b, n);
    if (src.is_label_rel)
        phx_builder_add_fixup(b, n, src.label_id, 1);
}

/* MOV mem, reg:  89 /r (32/64-bit) or 88 /r (8-bit) */
void phx_x86_mov_mr(PhxBuilder *b, PhxMem dst, PhxGp src) {
    PhxNode *n = phx_builder_alloc_node(b);
    if (!n) return;
    n->node_type = PHX_NODE_INST;
    n->opcode = PHX_OP_MOV;
    n->operands[0] = phx_op_mem(dst);
    n->operands[1] = phx_op_gp(src);
    n->num_operands = 2;

    int pos = 0;
    if (needs_66h(src)) n->encoded[pos++] = 0x66;
    uint8_t rex = make_rex_mem(want_rexw(src), src, dst);
    if (rex || needs_rex_for_byte(src))
        n->encoded[pos++] = rex ? rex : 0x40;
    if (src.size == 1)
        n->encoded[pos++] = 0x88;
    else
        n->encoded[pos++] = 0x89;
    pos += encode_modrm_mem(n->encoded + pos, reg3(src), dst);
    n->encoded_size = pos;
    phx_builder_append_node(b, n);
}

/* MOV reg, imm64:  REX.W B8+rd io (movabs for 64-bit immediate)
   MOV reg, imm32:  B8+rd id (32-bit, zero-extends to 64-bit)
   MOV reg, imm8: not used (use imm32 form)
   If imm fits in uint32, use 32-bit form (shorter encoding).
   If imm fits in int32, use C7 /0 with sign-extension. */
void phx_x86_mov_ri(PhxBuilder *b, PhxGp dst, int64_t imm) {
    PhxNode *n = phx_builder_alloc_node(b);
    if (!n) return;
    n->node_type = PHX_NODE_INST;
    n->opcode = PHX_OP_MOV;
    n->operands[0] = phx_op_gp(dst);
    n->operands[1] = phx_op_imm(imm);
    n->num_operands = 2;

    int pos = 0;

    if (dst.size == 1) {
        /* MOV r8, imm8: B0+rb ib */
        uint8_t rex = make_rex_single(0, dst);
        if (rex || needs_rex_for_byte(dst))
            n->encoded[pos++] = rex ? rex : 0x40;
        n->encoded[pos++] = 0xB0 + reg3(dst);
        n->encoded[pos++] = (uint8_t)(imm & 0xFF);
    } else if (dst.size == 2) {
        /* MOV r16, imm16: 66 B8+rw iw */
        n->encoded[pos++] = 0x66;
        if (reg_ext(dst)) n->encoded[pos++] = 0x41;
        n->encoded[pos++] = 0xB8 + reg3(dst);
        uint16_t imm16 = (uint16_t)(imm & 0xFFFF);
        memcpy(n->encoded + pos, &imm16, 2);
        pos += 2;
    } else if (dst.size == 8 && !fits_i32(imm) && !fits_u32(imm)) {
        /* 64-bit immediate (movabs): REX.W B8+rd io */
        uint8_t rex = 0x48;  /* REX.W */
        if (reg_ext(dst)) rex |= 0x01;
        n->encoded[pos++] = rex;
        n->encoded[pos++] = 0xB8 + reg3(dst);
        memcpy(n->encoded + pos, &imm, 8);
        pos += 8;
    } else if (dst.size == 8 && fits_i32(imm)) {
        /* 64-bit register, imm fits in sign-extended int32:
           REX.W C7 /0 id (sign-extends to 64 bits) */
        uint8_t rex = 0x48;
        if (reg_ext(dst)) rex |= 0x01;
        n->encoded[pos++] = rex;
        n->encoded[pos++] = 0xC7;
        pos += encode_modrm_rr(n->encoded + pos, 0, dst);
        int32_t imm32 = (int32_t)imm;
        memcpy(n->encoded + pos, &imm32, 4);
        pos += 4;
    } else {
        /* 32-bit register or 64-bit with uint32 value: B8+rd id
           (writing to 32-bit register zero-extends to 64) */
        uint8_t rex = make_rex_single(0, dst);
        if (rex) n->encoded[pos++] = rex;
        n->encoded[pos++] = 0xB8 + reg3(dst);
        int32_t imm32 = (int32_t)(imm & 0xFFFFFFFF);
        memcpy(n->encoded + pos, &imm32, 4);
        pos += 4;
    }
    n->encoded_size = pos;
    phx_builder_append_node(b, n);
}

/* MOV mem, imm32: C7 /0 id (32/64-bit) or C6 /0 ib (8-bit) */
void phx_x86_mov_mi(PhxBuilder *b, PhxMem dst, int32_t imm) {
    PhxNode *n = phx_builder_alloc_node(b);
    if (!n) return;
    n->node_type = PHX_NODE_INST;
    n->opcode = PHX_OP_MOV;
    n->operands[0] = phx_op_mem(dst);
    n->operands[1] = phx_op_imm(imm);
    n->num_operands = 2;

    int pos = 0;
    if (dst.size == 2) n->encoded[pos++] = 0x66;
    {
        uint8_t r = 0x40;
        if (want_rexw_mem(dst)) r |= 0x08;
        if (dst.has_index && reg_ext(dst.index)) r |= 0x02;
        if (reg_ext(dst.base)) r |= 0x01;
        if (r != 0x40) n->encoded[pos++] = r;
    }
    if (dst.size == 1) {
        n->encoded[pos++] = 0xC6;
        pos += encode_modrm_mem(n->encoded + pos, 0, dst);
        n->encoded[pos++] = (uint8_t)(imm & 0xFF);
    } else if (dst.size == 2) {
        n->encoded[pos++] = 0xC7;
        pos += encode_modrm_mem(n->encoded + pos, 0, dst);
        uint16_t imm16 = (uint16_t)(imm & 0xFFFF);
        memcpy(n->encoded + pos, &imm16, 2);
        pos += 2;
    } else {
        n->encoded[pos++] = 0xC7;
        pos += encode_modrm_mem(n->encoded + pos, 0, dst);
        memcpy(n->encoded + pos, &imm, 4);
        pos += 4;
    }
    n->encoded_size = pos;
    phx_builder_append_node(b, n);
}

/* ================================================================== */
/*  LEA                                                                */
/* ================================================================== */

/* LEA reg, mem:  8D /r */
void phx_x86_lea(PhxBuilder *b, PhxGp dst, PhxMem src) {
    PhxNode *n = phx_builder_alloc_node(b);
    if (!n) return;
    n->node_type = PHX_NODE_INST;
    n->opcode = PHX_OP_LEA;
    n->operands[0] = phx_op_gp(dst);
    n->operands[1] = phx_op_mem(src);
    n->num_operands = 2;

    int pos = 0;
    if (needs_66h(dst)) n->encoded[pos++] = 0x66;
    uint8_t rex = make_rex_mem(want_rexw(dst), dst, src);
    if (rex) n->encoded[pos++] = rex;
    n->encoded[pos++] = 0x8D;
    pos += encode_modrm_mem(n->encoded + pos, reg3(dst), src);
    n->encoded_size = pos;
    phx_builder_append_node(b, n);
    if (src.is_label_rel)
        phx_builder_add_fixup(b, n, src.label_id, 1);
}

/* ================================================================== */
/*  MOVSD (scalar double - F2 0F 10/11)                                */
/* ================================================================== */

/* MOVSD xmm, xmm:  F2 0F 10 /r */
void phx_x86_movsd_rr(PhxBuilder *b, PhxGp dst, PhxGp src) {
    emit_sse_rr(b, PHX_OP_MOVSD, 0xF2, 0x10, dst, src);
}

/* MOVSD xmm, mem:  F2 0F 10 /r */
void phx_x86_movsd_rm(PhxBuilder *b, PhxGp dst, PhxMem src) {
    emit_sse_rm(b, PHX_OP_MOVSD, 0xF2, 0x10, dst, src);
}

/* MOVSD mem, xmm:  F2 0F 11 /r */
void phx_x86_movsd_mr(PhxBuilder *b, PhxMem dst, PhxGp src) {
    emit_sse_mr(b, PHX_OP_MOVSD, 0xF2, 0x11, dst, src);
}

/* ================================================================== */
/*  MOVSX (sign-extend 8/16 to 32/64)                                 */
/* ================================================================== */

/* MOVSX reg, reg:
   8->32/64:  0F BE /r
   16->32/64: 0F BF /r */
void phx_x86_movsx_rr(PhxBuilder *b, PhxGp dst, PhxGp src) {
    PhxNode *n = phx_builder_alloc_node(b);
    if (!n) return;
    n->node_type = PHX_NODE_INST;
    n->opcode = PHX_OP_MOVSX;
    n->operands[0] = phx_op_gp(dst);
    n->operands[1] = phx_op_gp(src);
    n->num_operands = 2;

    int pos = 0;
    uint8_t rex = make_rex(want_rexw(dst), dst, src);
    if (rex || needs_rex_for_byte(src))
        n->encoded[pos++] = rex ? rex : 0x40;
    n->encoded[pos++] = 0x0F;
    if (src.size == 1)
        n->encoded[pos++] = 0xBE;
    else
        n->encoded[pos++] = 0xBF;  /* 16-bit source */
    pos += encode_modrm_rr(n->encoded + pos, reg3(dst), src);
    n->encoded_size = pos;
    phx_builder_append_node(b, n);
}

/* MOVSX reg, mem: same encoding, mem operand */
void phx_x86_movsx_rm(PhxBuilder *b, PhxGp dst, PhxMem src) {
    PhxNode *n = phx_builder_alloc_node(b);
    if (!n) return;
    n->node_type = PHX_NODE_INST;
    n->opcode = PHX_OP_MOVSX;
    n->operands[0] = phx_op_gp(dst);
    n->operands[1] = phx_op_mem(src);
    n->num_operands = 2;

    int pos = 0;
    uint8_t rex = make_rex_mem(want_rexw(dst), dst, src);
    if (rex) n->encoded[pos++] = rex;
    n->encoded[pos++] = 0x0F;
    if (src.size == 1)
        n->encoded[pos++] = 0xBE;
    else
        n->encoded[pos++] = 0xBF;
    pos += encode_modrm_mem(n->encoded + pos, reg3(dst), src);
    n->encoded_size = pos;
    phx_builder_append_node(b, n);
}

/* ================================================================== */
/*  MOVSXD (sign-extend 32 to 64)                                     */
/* ================================================================== */

/* MOVSXD r64, r32:  REX.W 63 /r */
void phx_x86_movsxd_rr(PhxBuilder *b, PhxGp dst, PhxGp src) {
    PhxNode *n = phx_builder_alloc_node(b);
    if (!n) return;
    n->node_type = PHX_NODE_INST;
    n->opcode = PHX_OP_MOVSXD;
    n->operands[0] = phx_op_gp(dst);
    n->operands[1] = phx_op_gp(src);
    n->num_operands = 2;

    int pos = 0;
    uint8_t rex = make_rex(1, dst, src);  /* always REX.W */
    n->encoded[pos++] = rex;
    n->encoded[pos++] = 0x63;
    pos += encode_modrm_rr(n->encoded + pos, reg3(dst), src);
    n->encoded_size = pos;
    phx_builder_append_node(b, n);
}

/* MOVSXD r64, m32:  REX.W 63 /r */
void phx_x86_movsxd_rm(PhxBuilder *b, PhxGp dst, PhxMem src) {
    PhxNode *n = phx_builder_alloc_node(b);
    if (!n) return;
    n->node_type = PHX_NODE_INST;
    n->opcode = PHX_OP_MOVSXD;
    n->operands[0] = phx_op_gp(dst);
    n->operands[1] = phx_op_mem(src);
    n->num_operands = 2;

    int pos = 0;
    uint8_t rex = make_rex_mem(1, dst, src);  /* always REX.W */
    n->encoded[pos++] = rex;
    n->encoded[pos++] = 0x63;
    pos += encode_modrm_mem(n->encoded + pos, reg3(dst), src);
    n->encoded_size = pos;
    phx_builder_append_node(b, n);
}

/* ================================================================== */
/*  MOVZX (zero-extend 8/16 to 32/64)                                 */
/* ================================================================== */

/* MOVZX reg, reg:
   8->32/64:  0F B6 /r
   16->32/64: 0F B7 /r */
void phx_x86_movzx_rr(PhxBuilder *b, PhxGp dst, PhxGp src) {
    PhxNode *n = phx_builder_alloc_node(b);
    if (!n) return;
    n->node_type = PHX_NODE_INST;
    n->opcode = PHX_OP_MOVZX;
    n->operands[0] = phx_op_gp(dst);
    n->operands[1] = phx_op_gp(src);
    n->num_operands = 2;

    int pos = 0;
    uint8_t rex = make_rex(want_rexw(dst), dst, src);
    if (rex || needs_rex_for_byte(src))
        n->encoded[pos++] = rex ? rex : 0x40;
    n->encoded[pos++] = 0x0F;
    if (src.size == 1)
        n->encoded[pos++] = 0xB6;
    else
        n->encoded[pos++] = 0xB7;
    pos += encode_modrm_rr(n->encoded + pos, reg3(dst), src);
    n->encoded_size = pos;
    phx_builder_append_node(b, n);
}

/* MOVZX reg, mem */
void phx_x86_movzx_rm(PhxBuilder *b, PhxGp dst, PhxMem src) {
    PhxNode *n = phx_builder_alloc_node(b);
    if (!n) return;
    n->node_type = PHX_NODE_INST;
    n->opcode = PHX_OP_MOVZX;
    n->operands[0] = phx_op_gp(dst);
    n->operands[1] = phx_op_mem(src);
    n->num_operands = 2;

    int pos = 0;
    uint8_t rex = make_rex_mem(want_rexw(dst), dst, src);
    if (rex) n->encoded[pos++] = rex;
    n->encoded[pos++] = 0x0F;
    if (src.size == 1)
        n->encoded[pos++] = 0xB6;
    else
        n->encoded[pos++] = 0xB7;
    pos += encode_modrm_mem(n->encoded + pos, reg3(dst), src);
    n->encoded_size = pos;
    phx_builder_append_node(b, n);
}

/* ================================================================== */
/*  MOVQ (move quadword to/from XMM)                                  */
/* ================================================================== */

/* MOVQ xmm, xmm: F3 0F 7E /r   (load form)
   MOVQ xmm, r64: 66 REX.W 0F 6E /r
   MOVQ r64, xmm: 66 REX.W 0F 7E /r */
void phx_x86_movq_rr(PhxBuilder *b, PhxGp dst, PhxGp src) {
    PhxNode *n = phx_builder_alloc_node(b);
    if (!n) return;
    n->node_type = PHX_NODE_INST;
    n->opcode = PHX_OP_MOVQ;
    n->operands[0] = phx_op_gp(dst);
    n->operands[1] = phx_op_gp(src);
    n->num_operands = 2;

    int pos = 0;

    if (dst.size == 16 && src.size == 16) {
        /* XMM to XMM: F3 0F 7E /r */
        n->encoded[pos++] = 0xF3;
        uint8_t rex = make_rex(0, dst, src);
        if (rex) n->encoded[pos++] = rex;
        n->encoded[pos++] = 0x0F;
        n->encoded[pos++] = 0x7E;
        pos += encode_modrm_rr(n->encoded + pos, reg3(dst), src);
    } else if (dst.size == 16 && src.size == 8) {
        /* GP r64 to XMM: 66 REX.W 0F 6E /r */
        n->encoded[pos++] = 0x66;
        uint8_t rex = make_rex(1, dst, src);
        n->encoded[pos++] = rex;
        n->encoded[pos++] = 0x0F;
        n->encoded[pos++] = 0x6E;
        pos += encode_modrm_rr(n->encoded + pos, reg3(dst), src);
    } else if (dst.size == 8 && src.size == 16) {
        /* XMM to GP r64: 66 REX.W 0F 7E /r */
        n->encoded[pos++] = 0x66;
        uint8_t rex = make_rex(1, src, dst);
        n->encoded[pos++] = rex;
        n->encoded[pos++] = 0x0F;
        n->encoded[pos++] = 0x7E;
        pos += encode_modrm_rr(n->encoded + pos, reg3(src), dst);
    } else {
        assert(0 && "unsupported MOVQ operand combination");
    }
    n->encoded_size = pos;
    phx_builder_append_node(b, n);
}

/* ================================================================== */
/*  PUSH                                                               */
/* ================================================================== */

/* PUSH r64: 50+rd (no REX.W needed; push always 64-bit in long mode) */
void phx_x86_push_r(PhxBuilder *b, PhxGp src) {
    PhxNode *n = phx_builder_alloc_node(b);
    if (!n) return;
    n->node_type = PHX_NODE_INST;
    n->opcode = PHX_OP_PUSH;
    n->operands[0] = phx_op_gp(src);
    n->num_operands = 1;

    int pos = 0;
    if (reg_ext(src)) n->encoded[pos++] = 0x41;  /* REX.B */
    n->encoded[pos++] = 0x50 + reg3(src);
    n->encoded_size = pos;
    phx_builder_append_node(b, n);
}

/* PUSH m64: FF /6 */
void phx_x86_push_m(PhxBuilder *b, PhxMem src) {
    PhxNode *n = phx_builder_alloc_node(b);
    if (!n) return;
    n->node_type = PHX_NODE_INST;
    n->opcode = PHX_OP_PUSH;
    n->operands[0] = phx_op_mem(src);
    n->num_operands = 1;

    int pos = 0;
    {
        uint8_t r = 0x40;
        if (reg_ext(src.base)) r |= 0x01;
        if (src.has_index && reg_ext(src.index)) r |= 0x02;
        if (r != 0x40) n->encoded[pos++] = r;
    }
    n->encoded[pos++] = 0xFF;
    pos += encode_modrm_mem(n->encoded + pos, 6, src);
    n->encoded_size = pos;
    phx_builder_append_node(b, n);
}

/* PUSH imm32: 68 id  (sign-extended to 64-bit)
   PUSH imm8:  6A ib  (sign-extended to 64-bit) */
void phx_x86_push_i(PhxBuilder *b, int32_t imm) {
    PhxNode *n = phx_builder_alloc_node(b);
    if (!n) return;
    n->node_type = PHX_NODE_INST;
    n->opcode = PHX_OP_PUSH;
    n->operands[0] = phx_op_imm(imm);
    n->num_operands = 1;

    int pos = 0;
    if (fits_i8(imm)) {
        n->encoded[pos++] = 0x6A;
        n->encoded[pos++] = (uint8_t)(imm & 0xFF);
    } else {
        n->encoded[pos++] = 0x68;
        memcpy(n->encoded + pos, &imm, 4);
        pos += 4;
    }
    n->encoded_size = pos;
    phx_builder_append_node(b, n);
}

/* ================================================================== */
/*  POP                                                                */
/* ================================================================== */

/* POP r64: 58+rd */
void phx_x86_pop_r(PhxBuilder *b, PhxGp dst) {
    PhxNode *n = phx_builder_alloc_node(b);
    if (!n) return;
    n->node_type = PHX_NODE_INST;
    n->opcode = PHX_OP_POP;
    n->operands[0] = phx_op_gp(dst);
    n->num_operands = 1;

    int pos = 0;
    if (reg_ext(dst)) n->encoded[pos++] = 0x41;
    n->encoded[pos++] = 0x58 + reg3(dst);
    n->encoded_size = pos;
    phx_builder_append_node(b, n);
}

/* POP m64: 8F /0 */
void phx_x86_pop_m(PhxBuilder *b, PhxMem dst) {
    PhxNode *n = phx_builder_alloc_node(b);
    if (!n) return;
    n->node_type = PHX_NODE_INST;
    n->opcode = PHX_OP_POP;
    n->operands[0] = phx_op_mem(dst);
    n->num_operands = 1;

    int pos = 0;
    {
        uint8_t r = 0x40;
        if (reg_ext(dst.base)) r |= 0x01;
        if (dst.has_index && reg_ext(dst.index)) r |= 0x02;
        if (r != 0x40) n->encoded[pos++] = r;
    }
    n->encoded[pos++] = 0x8F;
    pos += encode_modrm_mem(n->encoded + pos, 0, dst);
    n->encoded_size = pos;
    phx_builder_append_node(b, n);
}

/* ================================================================== */
/*  XCHG                                                               */
/* ================================================================== */

/* XCHG r/m, r:  87 /r (32/64-bit) or 86 /r (8-bit)
   Special case: XCHG rax, r  =  90+rd */
void phx_x86_xchg_rr(PhxBuilder *b, PhxGp a, PhxGp c) {
    PhxNode *n = phx_builder_alloc_node(b);
    if (!n) return;
    n->node_type = PHX_NODE_INST;
    n->opcode = PHX_OP_XCHG;
    n->operands[0] = phx_op_gp(a);
    n->operands[1] = phx_op_gp(c);
    n->num_operands = 2;

    int pos = 0;
    if (needs_66h(a)) n->encoded[pos++] = 0x66;
    /* Accumulator short form: XCHG RAX/EAX, reg → 0x90+rd */
    if (a.size >= 4 && a.id == 0 && c.id != 0) {
        uint8_t rex = make_rex_single(want_rexw(a), c);
        if (rex) n->encoded[pos++] = rex;
        n->encoded[pos++] = 0x90 + reg3(c);
    } else if (a.size >= 4 && c.id == 0 && a.id != 0) {
        uint8_t rex = make_rex_single(want_rexw(a), a);
        if (rex) n->encoded[pos++] = rex;
        n->encoded[pos++] = 0x90 + reg3(a);
    } else {
        uint8_t rex = make_rex(want_rexw(a), c, a);
        if (rex) n->encoded[pos++] = rex;
        if (a.size == 1)
            n->encoded[pos++] = 0x86;
        else
            n->encoded[pos++] = 0x87;
        pos += encode_modrm_rr(n->encoded + pos, reg3(c), a);
    }
    n->encoded_size = pos;
    phx_builder_append_node(b, n);
}

/* ================================================================== */
/*  MOVDQU (move unaligned double quadword - SSE2)                     */
/* ================================================================== */

/* MOVDQU xmm, m128: F3 0F 6F /r */
void phx_x86_movdqu_rm(PhxBuilder *b, PhxGp dst, PhxMem src) {
    emit_sse_rm(b, PHX_OP_MOVDQU, 0xF3, 0x6F, dst, src);
}

/* MOVDQU m128, xmm: F3 0F 7F /r */
void phx_x86_movdqu_mr(PhxBuilder *b, PhxMem dst, PhxGp src) {
    emit_sse_mr(b, PHX_OP_MOVDQU, 0xF3, 0x7F, dst, src);
}

/* ================================================================== */
/*  LEAVE                                                              */
/* ================================================================== */

/* LEAVE: C9 */
void phx_x86_leave(PhxBuilder *b) {
    PhxNode *n = phx_builder_alloc_node(b);
    if (!n) return;
    n->node_type = PHX_NODE_INST;
    n->opcode = PHX_OP_LEAVE;
    n->num_operands = 0;

    n->encoded[0] = 0xC9;
    n->encoded_size = 1;
    phx_builder_append_node(b, n);
}

/* ================================================================== */
/*  CMOVNZ (0F 45 /r)                                                 */
/* ================================================================== */

void phx_x86_cmovnz_rr(PhxBuilder *b, PhxGp dst, PhxGp src) {
    PhxNode *n = phx_builder_alloc_node(b);
    if (!n) return;
    n->node_type = PHX_NODE_INST;
    n->opcode = PHX_OP_CMOVNZ;
    n->operands[0] = phx_op_gp(dst);
    n->operands[1] = phx_op_gp(src);
    n->num_operands = 2;

    int pos = 0;
    if (needs_66h(dst)) n->encoded[pos++] = 0x66;
    uint8_t rex = make_rex(want_rexw(dst), dst, src);
    if (rex) n->encoded[pos++] = rex;
    n->encoded[pos++] = 0x0F;
    n->encoded[pos++] = 0x45;
    pos += encode_modrm_rr(n->encoded + pos, reg3(dst), src);
    n->encoded_size = pos;
    phx_builder_append_node(b, n);
}

void phx_x86_cmovnz_rm(PhxBuilder *b, PhxGp dst, PhxMem src) {
    PhxNode *n = phx_builder_alloc_node(b);
    if (!n) return;
    n->node_type = PHX_NODE_INST;
    n->opcode = PHX_OP_CMOVNZ;
    n->operands[0] = phx_op_gp(dst);
    n->operands[1] = phx_op_mem(src);
    n->num_operands = 2;

    int pos = 0;
    if (needs_66h(dst)) n->encoded[pos++] = 0x66;
    uint8_t rex = make_rex_mem(want_rexw(dst), dst, src);
    if (rex) n->encoded[pos++] = rex;
    n->encoded[pos++] = 0x0F;
    n->encoded[pos++] = 0x45;
    pos += encode_modrm_mem(n->encoded + pos, reg3(dst), src);
    n->encoded_size = pos;
    phx_builder_append_node(b, n);
}

/* ================================================================== */
/*  ADD, SUB, CMP, AND, OR, XOR (via ALU helpers)                     */
/* ================================================================== */

/* ADD: base=0x00, /0 */
void phx_x86_add_rr(PhxBuilder *b, PhxGp dst, PhxGp src) {
    emit_alu_rr(b, PHX_OP_ADD, 0x00, dst, src);
}
void phx_x86_add_rm(PhxBuilder *b, PhxGp dst, PhxMem src) {
    emit_alu_rm(b, PHX_OP_ADD, 0x00, dst, src);
}
void phx_x86_add_mr(PhxBuilder *b, PhxMem dst, PhxGp src) {
    emit_alu_mr(b, PHX_OP_ADD, 0x00, dst, src);
}
void phx_x86_add_ri(PhxBuilder *b, PhxGp dst, int32_t imm) {
    emit_alu_ri(b, PHX_OP_ADD, 0, dst, imm);
}
void phx_x86_add_mi(PhxBuilder *b, PhxMem dst, int32_t imm) {
    emit_alu_mi(b, PHX_OP_ADD, 0, dst, imm);
}

/* SUB: base=0x28, /5 */
void phx_x86_sub_rr(PhxBuilder *b, PhxGp dst, PhxGp src) {
    emit_alu_rr(b, PHX_OP_SUB, 0x28, dst, src);
}
void phx_x86_sub_rm(PhxBuilder *b, PhxGp dst, PhxMem src) {
    emit_alu_rm(b, PHX_OP_SUB, 0x28, dst, src);
}
void phx_x86_sub_mr(PhxBuilder *b, PhxMem dst, PhxGp src) {
    emit_alu_mr(b, PHX_OP_SUB, 0x28, dst, src);
}
void phx_x86_sub_ri(PhxBuilder *b, PhxGp dst, int32_t imm) {
    emit_alu_ri(b, PHX_OP_SUB, 5, dst, imm);
}
void phx_x86_sub_mi(PhxBuilder *b, PhxMem dst, int32_t imm) {
    emit_alu_mi(b, PHX_OP_SUB, 5, dst, imm);
}

/* CMP: base=0x38, /7 */
void phx_x86_cmp_rr(PhxBuilder *b, PhxGp a, PhxGp c) {
    emit_alu_rr(b, PHX_OP_CMP, 0x38, a, c);
}
void phx_x86_cmp_rm(PhxBuilder *b, PhxGp a, PhxMem c) {
    emit_alu_rm(b, PHX_OP_CMP, 0x38, a, c);
}
void phx_x86_cmp_mr(PhxBuilder *b, PhxMem a, PhxGp c) {
    emit_alu_mr(b, PHX_OP_CMP, 0x38, a, c);
}
void phx_x86_cmp_ri(PhxBuilder *b, PhxGp a, int32_t imm) {
    emit_alu_ri(b, PHX_OP_CMP, 7, a, imm);
}
void phx_x86_cmp_mi(PhxBuilder *b, PhxMem a, int32_t imm) {
    emit_alu_mi(b, PHX_OP_CMP, 7, a, imm);
}

/* AND: base=0x20, /4 */
void phx_x86_and_rr(PhxBuilder *b, PhxGp dst, PhxGp src) {
    emit_alu_rr(b, PHX_OP_AND, 0x20, dst, src);
}
void phx_x86_and_rm(PhxBuilder *b, PhxGp dst, PhxMem src) {
    emit_alu_rm(b, PHX_OP_AND, 0x20, dst, src);
}
void phx_x86_and_ri(PhxBuilder *b, PhxGp dst, int32_t imm) {
    emit_alu_ri(b, PHX_OP_AND, 4, dst, imm);
}
void phx_x86_and_mi(PhxBuilder *b, PhxMem dst, int32_t imm) {
    emit_alu_mi(b, PHX_OP_AND, 4, dst, imm);
}

/* OR: base=0x08, /1 */
void phx_x86_or_rr(PhxBuilder *b, PhxGp dst, PhxGp src) {
    emit_alu_rr(b, PHX_OP_OR, 0x08, dst, src);
}
void phx_x86_or_rm(PhxBuilder *b, PhxGp dst, PhxMem src) {
    emit_alu_rm(b, PHX_OP_OR, 0x08, dst, src);
}
void phx_x86_or_ri(PhxBuilder *b, PhxGp dst, int32_t imm) {
    emit_alu_ri(b, PHX_OP_OR, 1, dst, imm);
}
void phx_x86_or_mi(PhxBuilder *b, PhxMem dst, int32_t imm) {
    emit_alu_mi(b, PHX_OP_OR, 1, dst, imm);
}

/* XOR: base=0x30, /6 */
void phx_x86_xor_rr(PhxBuilder *b, PhxGp dst, PhxGp src) {
    emit_alu_rr(b, PHX_OP_XOR, 0x30, dst, src);
}
void phx_x86_xor_rm(PhxBuilder *b, PhxGp dst, PhxMem src) {
    emit_alu_rm(b, PHX_OP_XOR, 0x30, dst, src);
}
void phx_x86_xor_ri(PhxBuilder *b, PhxGp dst, int32_t imm) {
    emit_alu_ri(b, PHX_OP_XOR, 6, dst, imm);
}

/* ================================================================== */
/*  NEG, NOT (via unary helper)                                        */
/* ================================================================== */

/* NEG: F7 /3 */
void phx_x86_neg_r(PhxBuilder *b, PhxGp dst) {
    emit_unary_r(b, PHX_OP_NEG, 3, dst);
}
void phx_x86_neg_m(PhxBuilder *b, PhxMem dst) {
    emit_unary_m(b, PHX_OP_NEG, 3, dst);
}

/* NOT: F7 /2 */
void phx_x86_not_r(PhxBuilder *b, PhxGp dst) {
    emit_unary_r(b, PHX_OP_NOT, 2, dst);
}
void phx_x86_not_m(PhxBuilder *b, PhxMem dst) {
    emit_unary_m(b, PHX_OP_NOT, 2, dst);
}

/* ================================================================== */
/*  INC, DEC (via inc/dec helper)                                      */
/* ================================================================== */

/* INC: FF /0 (or FE /0 for 8-bit) */
void phx_x86_inc_r(PhxBuilder *b, PhxGp dst) {
    emit_incdec_r(b, PHX_OP_INC, 0, dst);
}
void phx_x86_inc_m(PhxBuilder *b, PhxMem dst) {
    emit_incdec_m(b, PHX_OP_INC, 0, dst);
}

/* DEC: FF /1 (or FE /1 for 8-bit) */
void phx_x86_dec_r(PhxBuilder *b, PhxGp dst) {
    emit_incdec_r(b, PHX_OP_DEC, 1, dst);
}
void phx_x86_dec_m(PhxBuilder *b, PhxMem dst) {
    emit_incdec_m(b, PHX_OP_DEC, 1, dst);
}

/* ================================================================== */
/*  IMUL                                                               */
/* ================================================================== */

/* IMUL r, r/m: 0F AF /r */
void phx_x86_imul_rr(PhxBuilder *b, PhxGp dst, PhxGp src) {
    PhxNode *n = phx_builder_alloc_node(b);
    if (!n) return;
    n->node_type = PHX_NODE_INST;
    n->opcode = PHX_OP_IMUL;
    n->operands[0] = phx_op_gp(dst);
    n->operands[1] = phx_op_gp(src);
    n->num_operands = 2;

    int pos = 0;
    if (needs_66h(dst)) n->encoded[pos++] = 0x66;
    uint8_t rex = make_rex(want_rexw(dst), dst, src);
    if (rex) n->encoded[pos++] = rex;
    n->encoded[pos++] = 0x0F;
    n->encoded[pos++] = 0xAF;
    pos += encode_modrm_rr(n->encoded + pos, reg3(dst), src);
    n->encoded_size = pos;
    phx_builder_append_node(b, n);
}

void phx_x86_imul_rm(PhxBuilder *b, PhxGp dst, PhxMem src) {
    PhxNode *n = phx_builder_alloc_node(b);
    if (!n) return;
    n->node_type = PHX_NODE_INST;
    n->opcode = PHX_OP_IMUL;
    n->operands[0] = phx_op_gp(dst);
    n->operands[1] = phx_op_mem(src);
    n->num_operands = 2;

    int pos = 0;
    if (needs_66h(dst)) n->encoded[pos++] = 0x66;
    uint8_t rex = make_rex_mem(want_rexw(dst), dst, src);
    if (rex) n->encoded[pos++] = rex;
    n->encoded[pos++] = 0x0F;
    n->encoded[pos++] = 0xAF;
    pos += encode_modrm_mem(n->encoded + pos, reg3(dst), src);
    n->encoded_size = pos;
    phx_builder_append_node(b, n);
}

/* IMUL r, r/m, imm: 6B /r ib (imm8) or 69 /r id (imm32) */
void phx_x86_imul_rri(PhxBuilder *b, PhxGp dst, PhxGp src, int32_t imm) {
    PhxNode *n = phx_builder_alloc_node(b);
    if (!n) return;
    n->node_type = PHX_NODE_INST;
    n->opcode = PHX_OP_IMUL;
    n->operands[0] = phx_op_gp(dst);
    n->operands[1] = phx_op_gp(src);
    n->operands[2] = phx_op_imm(imm);
    n->num_operands = 3;

    int pos = 0;
    if (needs_66h(dst)) n->encoded[pos++] = 0x66;
    uint8_t rex = make_rex(want_rexw(dst), dst, src);
    if (rex) n->encoded[pos++] = rex;

    if (fits_i8(imm)) {
        n->encoded[pos++] = 0x6B;
        pos += encode_modrm_rr(n->encoded + pos, reg3(dst), src);
        n->encoded[pos++] = (uint8_t)(imm & 0xFF);
    } else {
        n->encoded[pos++] = 0x69;
        pos += encode_modrm_rr(n->encoded + pos, reg3(dst), src);
        memcpy(n->encoded + pos, &imm, 4);
        pos += 4;
    }
    n->encoded_size = pos;
    phx_builder_append_node(b, n);
}

/* ================================================================== */
/*  IDIV, DIV (via unary helper - F7 /7 and F7 /6)                    */
/* ================================================================== */

void phx_x86_idiv_r(PhxBuilder *b, PhxGp src) {
    emit_unary_r(b, PHX_OP_IDIV, 7, src);
}
void phx_x86_idiv_m(PhxBuilder *b, PhxMem src) {
    emit_unary_m(b, PHX_OP_IDIV, 7, src);
}
void phx_x86_div_r(PhxBuilder *b, PhxGp src) {
    emit_unary_r(b, PHX_OP_DIV, 6, src);
}
void phx_x86_div_m(PhxBuilder *b, PhxMem src) {
    emit_unary_m(b, PHX_OP_DIV, 6, src);
}

/* ================================================================== */
/*  TEST                                                               */
/* ================================================================== */

/* TEST r/m, r:  85 /r (32/64-bit) or 84 /r (8-bit) */
void phx_x86_test_rr(PhxBuilder *b, PhxGp a, PhxGp c) {
    PhxNode *n = phx_builder_alloc_node(b);
    if (!n) return;
    n->node_type = PHX_NODE_INST;
    n->opcode = PHX_OP_TEST;
    n->operands[0] = phx_op_gp(a);
    n->operands[1] = phx_op_gp(c);
    n->num_operands = 2;

    int pos = 0;
    if (needs_66h(a)) n->encoded[pos++] = 0x66;
    uint8_t rex = make_rex(want_rexw(a), c, a);
    if (rex || needs_rex_for_byte(a) || needs_rex_for_byte(c))
        n->encoded[pos++] = rex ? rex : 0x40;
    if (a.size == 1)
        n->encoded[pos++] = 0x84;
    else
        n->encoded[pos++] = 0x85;
    pos += encode_modrm_rr(n->encoded + pos, reg3(c), a);
    n->encoded_size = pos;
    phx_builder_append_node(b, n);
}

/* TEST r/m, imm: F7 /0 id (32/64-bit) or F6 /0 ib (8-bit) */
void phx_x86_test_ri(PhxBuilder *b, PhxGp a, int32_t imm) {
    PhxNode *n = phx_builder_alloc_node(b);
    if (!n) return;
    n->node_type = PHX_NODE_INST;
    n->opcode = PHX_OP_TEST;
    n->operands[0] = phx_op_gp(a);
    n->operands[1] = phx_op_imm(imm);
    n->num_operands = 2;

    int pos = 0;
    if (needs_66h(a)) n->encoded[pos++] = 0x66;
    uint8_t rex = make_rex_single(want_rexw(a), a);
    if (rex || needs_rex_for_byte(a))
        n->encoded[pos++] = rex ? rex : 0x40;

    if (a.size == 1 && a.id == 0) {
        /* TEST AL, imm8: 0xA8 ib */
        n->encoded[pos++] = 0xA8;
        n->encoded[pos++] = (uint8_t)(imm & 0xFF);
    } else if (a.size == 1) {
        n->encoded[pos++] = 0xF6;
        pos += encode_modrm_rr(n->encoded + pos, 0, a);
        n->encoded[pos++] = (uint8_t)(imm & 0xFF);
    } else if (a.id == 0) {
        /* TEST EAX/RAX, imm32: 0xA9 id */
        n->encoded[pos++] = 0xA9;
        memcpy(n->encoded + pos, &imm, 4);
        pos += 4;
    } else {
        n->encoded[pos++] = 0xF7;
        pos += encode_modrm_rr(n->encoded + pos, 0, a);
        memcpy(n->encoded + pos, &imm, 4);
        pos += 4;
    }
    n->encoded_size = pos;
    phx_builder_append_node(b, n);
}

/* TEST mem, imm: F7 /0 id (32/64-bit) or F6 /0 ib (8-bit) */
void phx_x86_test_mi(PhxBuilder *b, PhxMem a, int32_t imm) {
    PhxNode *n = phx_builder_alloc_node(b);
    if (!n) return;
    n->node_type = PHX_NODE_INST;
    n->opcode = PHX_OP_TEST;
    n->operands[0] = phx_op_mem(a);
    n->operands[1] = phx_op_imm(imm);
    n->num_operands = 2;

    int pos = 0;
    if (a.size == 2) n->encoded[pos++] = 0x66;
    {
        uint8_t r = 0x40;
        if (want_rexw_mem(a)) r |= 0x08;
        if (a.has_index && reg_ext(a.index)) r |= 0x02;
        if (reg_ext(a.base)) r |= 0x01;
        if (r != 0x40) n->encoded[pos++] = r;
    }
    if (a.size == 1) {
        n->encoded[pos++] = 0xF6;
        pos += encode_modrm_mem(n->encoded + pos, 0, a);
        n->encoded[pos++] = (uint8_t)(imm & 0xFF);
    } else {
        n->encoded[pos++] = 0xF7;
        pos += encode_modrm_mem(n->encoded + pos, 0, a);
        memcpy(n->encoded + pos, &imm, 4);
        pos += 4;
    }
    n->encoded_size = pos;
    phx_builder_append_node(b, n);
}

/* ================================================================== */
/*  BT (bit test)                                                      */
/* ================================================================== */

/* BT r/m, r:  0F A3 /r */
void phx_x86_bt_rr(PhxBuilder *b, PhxGp a, PhxGp bit) {
    PhxNode *n = phx_builder_alloc_node(b);
    if (!n) return;
    n->node_type = PHX_NODE_INST;
    n->opcode = PHX_OP_BT;
    n->operands[0] = phx_op_gp(a);
    n->operands[1] = phx_op_gp(bit);
    n->num_operands = 2;

    int pos = 0;
    if (needs_66h(a)) n->encoded[pos++] = 0x66;
    uint8_t rex = make_rex(want_rexw(a), bit, a);
    if (rex) n->encoded[pos++] = rex;
    n->encoded[pos++] = 0x0F;
    n->encoded[pos++] = 0xA3;
    pos += encode_modrm_rr(n->encoded + pos, reg3(bit), a);
    n->encoded_size = pos;
    phx_builder_append_node(b, n);
}

/* BT r/m, imm8:  0F BA /4 ib */
void phx_x86_bt_ri(PhxBuilder *b, PhxGp a, uint8_t bit) {
    PhxNode *n = phx_builder_alloc_node(b);
    if (!n) return;
    n->node_type = PHX_NODE_INST;
    n->opcode = PHX_OP_BT;
    n->operands[0] = phx_op_gp(a);
    n->operands[1] = phx_op_imm(bit);
    n->num_operands = 2;

    int pos = 0;
    if (needs_66h(a)) n->encoded[pos++] = 0x66;
    uint8_t rex = make_rex_single(want_rexw(a), a);
    if (rex) n->encoded[pos++] = rex;
    n->encoded[pos++] = 0x0F;
    n->encoded[pos++] = 0xBA;
    pos += encode_modrm_rr(n->encoded + pos, 4, a);
    n->encoded[pos++] = bit;
    n->encoded_size = pos;
    phx_builder_append_node(b, n);
}

/* ================================================================== */
/*  COMISD (compare scalar double, set EFLAGS)                         */
/* ================================================================== */

/* COMISD xmm, xmm: 66 0F 2F /r */
void phx_x86_comisd(PhxBuilder *b, PhxGp a, PhxGp c) {
    emit_sse_rr(b, PHX_OP_COMISD, 0x66, 0x2F, a, c);
}

/* ================================================================== */
/*  SETcc                                                              */
/* ================================================================== */

/* SETcc r/m8: 0F 9x /0  where x = condition code */
void phx_x86_setcc_r(PhxBuilder *b, PhxCondCode cc, PhxGp dst) {
    PhxNode *n = phx_builder_alloc_node(b);
    if (!n) return;
    n->node_type = PHX_NODE_INST;
    n->opcode = PHX_OP_SETCC;
    n->operands[0] = phx_op_gp(dst);
    n->operands[1] = phx_op_imm(cc);
    n->num_operands = 2;

    int pos = 0;
    /* SETcc always operates on 8-bit register, but we need REX for
       accessing SPL/BPL/SIL/DIL (id >= 4) or extended regs */
    uint8_t rex = make_rex_single(0, dst);
    if (rex || needs_rex_for_byte(dst))
        n->encoded[pos++] = rex ? rex : 0x40;
    n->encoded[pos++] = 0x0F;
    n->encoded[pos++] = 0x90 + (uint8_t)cc;
    pos += encode_modrm_rr(n->encoded + pos, 0, dst);
    n->encoded_size = pos;
    phx_builder_append_node(b, n);
}

void phx_x86_setcc_m(PhxBuilder *b, PhxCondCode cc, PhxMem dst) {
    PhxNode *n = phx_builder_alloc_node(b);
    if (!n) return;
    n->node_type = PHX_NODE_INST;
    n->opcode = PHX_OP_SETCC;
    n->operands[0] = phx_op_mem(dst);
    n->operands[1] = phx_op_imm(cc);
    n->num_operands = 2;

    int pos = 0;
    {
        uint8_t r = 0x40;
        if (reg_ext(dst.base)) r |= 0x01;
        if (dst.has_index && reg_ext(dst.index)) r |= 0x02;
        if (r != 0x40) n->encoded[pos++] = r;
    }
    n->encoded[pos++] = 0x0F;
    n->encoded[pos++] = 0x90 + (uint8_t)cc;
    pos += encode_modrm_mem(n->encoded + pos, 0, dst);
    n->encoded_size = pos;
    phx_builder_append_node(b, n);
}

/* ================================================================== */
/*  JMP (unconditional)                                                */
/* ================================================================== */

/* JMP rel32 (to label): E9 cd
   The displacement is not known until finalize; we emit a placeholder
   and register a fixup. */
void phx_x86_jmp_label(PhxBuilder *b, PhxLabel target) {
    PhxNode *n = phx_builder_alloc_node(b);
    if (!n) return;
    n->node_type = PHX_NODE_INST;
    n->opcode = PHX_OP_JMP;
    n->operands[0] = phx_op_label(target);
    n->num_operands = 1;

    int pos = 0;
    n->encoded[pos++] = 0xE9;
    /* Placeholder for rel32 displacement */
    memset(n->encoded + pos, 0, 4);
    pos += 4;
    n->encoded_size = pos;

    phx_builder_append_node(b, n);
    phx_builder_add_fixup(b, n, target.id, 0);
}

/* JMP r/m64: FF /4 */
void phx_x86_jmp_r(PhxBuilder *b, PhxGp target) {
    PhxNode *n = phx_builder_alloc_node(b);
    if (!n) return;
    n->node_type = PHX_NODE_INST;
    n->opcode = PHX_OP_JMP;
    n->operands[0] = phx_op_gp(target);
    n->num_operands = 1;

    int pos = 0;
    /* No REX.W needed for jmp r64 in long mode */
    if (reg_ext(target)) n->encoded[pos++] = 0x41;
    n->encoded[pos++] = 0xFF;
    pos += encode_modrm_rr(n->encoded + pos, 4, target);
    n->encoded_size = pos;
    phx_builder_append_node(b, n);
}

/* JMP m64: FF /4 */
void phx_x86_jmp_m(PhxBuilder *b, PhxMem target) {
    PhxNode *n = phx_builder_alloc_node(b);
    if (!n) return;
    n->node_type = PHX_NODE_INST;
    n->opcode = PHX_OP_JMP;
    n->operands[0] = phx_op_mem(target);
    n->num_operands = 1;

    int pos = 0;
    {
        uint8_t r = 0x40;
        if (reg_ext(target.base)) r |= 0x01;
        if (target.has_index && reg_ext(target.index)) r |= 0x02;
        if (r != 0x40) n->encoded[pos++] = r;
    }
    n->encoded[pos++] = 0xFF;
    pos += encode_modrm_mem(n->encoded + pos, 4, target);
    n->encoded_size = pos;
    phx_builder_append_node(b, n);
}

/* ================================================================== */
/*  Jcc (conditional jump)                                             */
/* ================================================================== */

/* Jcc rel32: 0F 8x cd  where x = condition code
   Always uses near (rel32) encoding for simplicity. The finalize pass
   resolves the placeholder displacement. */
void phx_x86_jcc(PhxBuilder *b, PhxCondCode cc, PhxLabel target) {
    PhxNode *n = phx_builder_alloc_node(b);
    if (!n) return;
    n->node_type = PHX_NODE_INST;
    n->opcode = PHX_OP_JCC;
    n->operands[0] = phx_op_label(target);
    n->operands[1] = phx_op_imm(cc);
    n->num_operands = 2;

    int pos = 0;
    n->encoded[pos++] = 0x0F;
    n->encoded[pos++] = 0x80 + (uint8_t)cc;
    /* Placeholder for rel32 displacement */
    memset(n->encoded + pos, 0, 4);
    pos += 4;
    n->encoded_size = pos;

    phx_builder_append_node(b, n);
    phx_builder_add_fixup(b, n, target.id, 0);
}

/* ================================================================== */
/*  CALL                                                               */
/* ================================================================== */

/* CALL rel32 (label): E8 cd */
void phx_x86_call_label(PhxBuilder *b, PhxLabel target) {
    PhxNode *n = phx_builder_alloc_node(b);
    if (!n) return;
    n->node_type = PHX_NODE_INST;
    n->opcode = PHX_OP_CALL;
    n->operands[0] = phx_op_label(target);
    n->num_operands = 1;

    int pos = 0;
    n->encoded[pos++] = 0xE8;
    memset(n->encoded + pos, 0, 4);
    pos += 4;
    n->encoded_size = pos;

    phx_builder_append_node(b, n);
    phx_builder_add_fixup(b, n, target.id, 0);
}

/* CALL r/m64: FF /2 */
void phx_x86_call_r(PhxBuilder *b, PhxGp target) {
    PhxNode *n = phx_builder_alloc_node(b);
    if (!n) return;
    n->node_type = PHX_NODE_INST;
    n->opcode = PHX_OP_CALL;
    n->operands[0] = phx_op_gp(target);
    n->num_operands = 1;

    int pos = 0;
    if (reg_ext(target)) n->encoded[pos++] = 0x41;
    n->encoded[pos++] = 0xFF;
    pos += encode_modrm_rr(n->encoded + pos, 2, target);
    n->encoded_size = pos;
    phx_builder_append_node(b, n);
}

/* CALL m64: FF /2 */
void phx_x86_call_m(PhxBuilder *b, PhxMem target) {
    PhxNode *n = phx_builder_alloc_node(b);
    if (!n) return;
    n->node_type = PHX_NODE_INST;
    n->opcode = PHX_OP_CALL;
    n->operands[0] = phx_op_mem(target);
    n->num_operands = 1;

    int pos = 0;
    {
        uint8_t r = 0x40;
        if (reg_ext(target.base)) r |= 0x01;
        if (target.has_index && reg_ext(target.index)) r |= 0x02;
        if (r != 0x40) n->encoded[pos++] = r;
    }
    n->encoded[pos++] = 0xFF;
    pos += encode_modrm_mem(n->encoded + pos, 2, target);
    n->encoded_size = pos;
    phx_builder_append_node(b, n);
}

/* ================================================================== */
/*  RET, UD2                                                           */
/* ================================================================== */

/* RET: C3 */
void phx_x86_ret(PhxBuilder *b) {
    PhxNode *n = phx_builder_alloc_node(b);
    if (!n) return;
    n->node_type = PHX_NODE_INST;
    n->opcode = PHX_OP_RET;
    n->num_operands = 0;

    n->encoded[0] = 0xC3;
    n->encoded_size = 1;
    phx_builder_append_node(b, n);
}

/* UD2: 0F 0B */
void phx_x86_ud2(PhxBuilder *b) {
    PhxNode *n = phx_builder_alloc_node(b);
    if (!n) return;
    n->node_type = PHX_NODE_INST;
    n->opcode = PHX_OP_UD2;
    n->num_operands = 0;

    n->encoded[0] = 0x0F;
    n->encoded[1] = 0x0B;
    n->encoded_size = 2;
    phx_builder_append_node(b, n);
}

/* ================================================================== */
/*  FP Arithmetic (scalar double)                                      */
/*  All use the F2 0F xx /r encoding pattern                           */
/* ================================================================== */

/* ADDSD: F2 0F 58 /r */
void phx_x86_addsd_rr(PhxBuilder *b, PhxGp dst, PhxGp src) {
    emit_sse_rr(b, PHX_OP_ADDSD, 0xF2, 0x58, dst, src);
}
void phx_x86_addsd_rm(PhxBuilder *b, PhxGp dst, PhxMem src) {
    emit_sse_rm(b, PHX_OP_ADDSD, 0xF2, 0x58, dst, src);
}

/* SUBSD: F2 0F 5C /r */
void phx_x86_subsd_rr(PhxBuilder *b, PhxGp dst, PhxGp src) {
    emit_sse_rr(b, PHX_OP_SUBSD, 0xF2, 0x5C, dst, src);
}
void phx_x86_subsd_rm(PhxBuilder *b, PhxGp dst, PhxMem src) {
    emit_sse_rm(b, PHX_OP_SUBSD, 0xF2, 0x5C, dst, src);
}

/* MULSD: F2 0F 59 /r */
void phx_x86_mulsd_rr(PhxBuilder *b, PhxGp dst, PhxGp src) {
    emit_sse_rr(b, PHX_OP_MULSD, 0xF2, 0x59, dst, src);
}
void phx_x86_mulsd_rm(PhxBuilder *b, PhxGp dst, PhxMem src) {
    emit_sse_rm(b, PHX_OP_MULSD, 0xF2, 0x59, dst, src);
}

/* DIVSD: F2 0F 5E /r */
void phx_x86_divsd_rr(PhxBuilder *b, PhxGp dst, PhxGp src) {
    emit_sse_rr(b, PHX_OP_DIVSD, 0xF2, 0x5E, dst, src);
}
void phx_x86_divsd_rm(PhxBuilder *b, PhxGp dst, PhxMem src) {
    emit_sse_rm(b, PHX_OP_DIVSD, 0xF2, 0x5E, dst, src);
}

/* ================================================================== */
/*  SSE Misc                                                           */
/* ================================================================== */

/* PTEST xmm, xmm: 66 0F 38 17 /r */
void phx_x86_ptest_rr(PhxBuilder *b, PhxGp a, PhxGp c) {
    emit_sse38_rr(b, PHX_OP_PTEST, 0x17, a, c);
}

/* PCMPEQW xmm, xmm: 66 0F 75 /r */
void phx_x86_pcmpeqw_rr(PhxBuilder *b, PhxGp dst, PhxGp src) {
    emit_sse_rr(b, PHX_OP_PCMPEQW, 0x66, 0x75, dst, src);
}

/* PSRLQ xmm, imm8: 66 0F 73 /2 ib */
void phx_x86_psrlq_ri(PhxBuilder *b, PhxGp dst, uint8_t imm) {
    PhxNode *n = phx_builder_alloc_node(b);
    if (!n) return;
    n->node_type = PHX_NODE_INST;
    n->opcode = PHX_OP_PSRLQ;
    n->operands[0] = phx_op_gp(dst);
    n->operands[1] = phx_op_imm(imm);
    n->num_operands = 2;

    int pos = 0;
    n->encoded[pos++] = 0x66;
    /* REX for extended XMM (xmm8-15 uses REX.B since dst is in r/m field) */
    if (reg_ext(dst)) n->encoded[pos++] = 0x41;
    n->encoded[pos++] = 0x0F;
    n->encoded[pos++] = 0x73;
    pos += encode_modrm_rr(n->encoded + pos, 2, dst);
    n->encoded[pos++] = imm;
    n->encoded_size = pos;
    phx_builder_append_node(b, n);
}

/* PXOR xmm, xmm: 66 0F EF /r */
void phx_x86_pxor_rr(PhxBuilder *b, PhxGp dst, PhxGp src) {
    emit_sse_rr(b, PHX_OP_PXOR, 0x66, 0xEF, dst, src);
}

/* ================================================================== */
/*  CDQ / CQO (sign-extend EAX/RAX into EDX:EAX / RDX:RAX)           */
/* ================================================================== */

/* CDQ: 99 (sign-extend EAX into EDX:EAX) */
void phx_x86_cdq(PhxBuilder *b) {
    PhxNode *n = phx_builder_alloc_node(b);
    if (!n) return;
    n->node_type = PHX_NODE_INST;
    n->opcode = PHX_OP_CDQ;
    n->num_operands = 0;

    n->encoded[0] = 0x99;
    n->encoded_size = 1;
    phx_builder_append_node(b, n);
}

/* CQO: REX.W 99 (sign-extend RAX into RDX:RAX) */
void phx_x86_cqo(PhxBuilder *b) {
    PhxNode *n = phx_builder_alloc_node(b);
    if (!n) return;
    n->node_type = PHX_NODE_INST;
    n->opcode = PHX_OP_CQO;
    n->num_operands = 0;

    n->encoded[0] = 0x48;  /* REX.W */
    n->encoded[1] = 0x99;
    n->encoded_size = 2;
    phx_builder_append_node(b, n);
}

/* ================================================================== */
/*  BTS (bit test and set)                                             */
/* ================================================================== */

/* BTS r/m, imm8:  0F BA /5 ib */
void phx_x86_bts_ri(PhxBuilder *b, PhxGp dst, uint8_t bit) {
    PhxNode *n = phx_builder_alloc_node(b);
    if (!n) return;
    n->node_type = PHX_NODE_INST;
    n->opcode = PHX_OP_BTS;
    n->operands[0] = phx_op_gp(dst);
    n->operands[1] = phx_op_imm(bit);
    n->num_operands = 2;

    int pos = 0;
    if (needs_66h(dst)) n->encoded[pos++] = 0x66;
    uint8_t rex = make_rex_single(want_rexw(dst), dst);
    if (rex) n->encoded[pos++] = rex;
    n->encoded[pos++] = 0x0F;
    n->encoded[pos++] = 0xBA;
    pos += encode_modrm_rr(n->encoded + pos, 5, dst);  /* /5 for BTS */
    n->encoded[pos++] = bit;
    n->encoded_size = pos;
    phx_builder_append_node(b, n);
}

/* ================================================================== */
/*  LOCK ADD (atomic add, memory-immediate)                            */
/* ================================================================== */

/* LOCK ADD [mem], imm:  F0 83 /0 ib  or  F0 81 /0 id */
void phx_x86_lock_add_mi(PhxBuilder *b, PhxMem dst, int32_t imm) {
    PhxNode *n = phx_builder_alloc_node(b);
    if (!n) return;
    n->node_type = PHX_NODE_INST;
    n->opcode = PHX_OP_LOCK_ADD;
    n->operands[0] = phx_op_mem(dst);
    n->operands[1] = phx_op_imm(imm);
    n->num_operands = 2;

    int pos = 0;
    /* LOCK prefix */
    n->encoded[pos++] = 0xF0;

    if (dst.size == 2) n->encoded[pos++] = 0x66;
    {
        uint8_t r = 0x40;
        if (want_rexw_mem(dst)) r |= 0x08;
        if (dst.has_index && reg_ext(dst.index)) r |= 0x02;
        if (reg_ext(dst.base)) r |= 0x01;
        if (r != 0x40) n->encoded[pos++] = r;
    }

    if (dst.size == 1) {
        n->encoded[pos++] = 0x80;
        pos += encode_modrm_mem(n->encoded + pos, 0, dst);
        n->encoded[pos++] = (uint8_t)(imm & 0xFF);
    } else if (fits_i8(imm)) {
        n->encoded[pos++] = 0x83;
        pos += encode_modrm_mem(n->encoded + pos, 0, dst);
        n->encoded[pos++] = (uint8_t)(imm & 0xFF);
    } else {
        n->encoded[pos++] = 0x81;
        pos += encode_modrm_mem(n->encoded + pos, 0, dst);
        memcpy(n->encoded + pos, &imm, 4);
        pos += 4;
    }
    n->encoded_size = pos;
    phx_builder_append_node(b, n);
}

/* ================================================================== */
/*  Finalize: resolve fixups and linearize to code buffer              */
/* ================================================================== */

/*
 * phx_x86_finalize -- 3-pass finalize for x86_64
 *
 * Pass 1: Walk the node list head->tail, computing the byte offset of each
 *         node in the final output.  Alignment nodes get filled with 0x90
 *         (NOP) bytes.
 *
 * Pass 2: Resolve all label fixups.  For each fixup, compute the rel32
 *         displacement (target_offset - (source_offset + source_encoded_size))
 *         and patch the last 4 bytes of the source node's encoded[] buffer.
 *
 * Pass 3: Linearize all encoded bytes into the code holder's buffer.
 *
 * Returns 0 on success, negative on error:
 *   -1  alignment padding exceeds PHX_MAX_ENCODED
 *   -2  unbound label referenced by a fixup
 *   -3  unbound label (created but never bound)
 *   -4  buffer allocation failure
 */
int phx_x86_finalize(PhxBuilder *b) {
    assert(b != NULL);
    PhxCodeHolder *code = b->code;
    assert(code != NULL);

    /* ---- Pass 1: compute byte offsets for all nodes ---- */
    uint32_t running_offset = 0;
    for (PhxNode *n = b->head; n != NULL; n = n->next) {
        n->offset = running_offset;

        if (n->node_type == PHX_NODE_ALIGN) {
            /* Compute padding needed to reach the requested alignment */
            int align = (int)n->operands[0].imm;
            uint32_t aligned = (running_offset + (uint32_t)align - 1)
                             & ~((uint32_t)align - 1);
            uint32_t padding = aligned - running_offset;
            if (padding > PHX_MAX_ENCODED) {
                return -1;
            }
            /* Fill with single-byte NOP (0x90) */
            memset(n->encoded, 0x90, padding);
            n->encoded_size = padding;
            running_offset += padding;
        } else if (n->node_type == PHX_NODE_EMBED) {
            if (n->embed_data) {
                running_offset += n->embed_size;
            } else {
                running_offset += n->encoded_size;
            }
        } else if (n->node_type == PHX_NODE_LABEL) {
            /* Labels emit no bytes; offset stays the same */
        } else {
            /* Instruction node -- already fully encoded */
            running_offset += n->encoded_size;
        }
    }

    /* ---- Pass 2: resolve label fixups ---- */
    for (uint32_t i = 0; i < b->fixup_count; i++) {
        PhxFixup *f = &b->fixups[i];
        PhxNode *inst_node = f->node;
        uint32_t label_id = f->label_id;

        /* Look up the target label node */
        assert(label_id < b->next_label_id);
        PhxNode *label_node = b->label_nodes[label_id];
        if (!label_node) {
            /* Unbound label referenced by fixup */
            return -2;
        }

        /* x86 relative addressing: displacement is measured from the END
           of the instruction (i.e., from inst_offset + inst_encoded_size) */
        int32_t disp = (int32_t)((int64_t)label_node->offset
                     - ((int64_t)inst_node->offset
                        + (int64_t)inst_node->encoded_size));

        if (inst_node->encoded_size == 2 && inst_node->encoded[0] == 0xEB) {
            /* Short jmp (EB rel8): patch last byte with rel8 */
            if (disp < -128 || disp > 127) return -9; /* out of range for short jmp */
            inst_node->encoded[1] = (uint8_t)(disp & 0xFF);
        } else {
            /* Near jmp/call/jcc (rel32): patch last 4 bytes */
            uint32_t patch_off = inst_node->encoded_size - 4;
            inst_node->encoded[patch_off + 0] = (uint8_t)(disp);
            inst_node->encoded[patch_off + 1] = (uint8_t)(disp >> 8);
            inst_node->encoded[patch_off + 2] = (uint8_t)(disp >> 16);
            inst_node->encoded[patch_off + 3] = (uint8_t)(disp >> 24);
        }
    }

    /* ---- Verify referenced labels are bound ---- */
    /* Note: unreferenced unbound labels are harmless — the JIT creates
       labels speculatively that may not be bound if a code path is optimized
       away. Only labels referenced by fixups must be bound (checked in Pass 2). */

    /* ---- Pass 3: linearize into code buffer ---- */
    size_t total_size = (size_t)running_offset;
    if (total_size > code->buffer_capacity) {
        size_t new_cap = code->buffer_capacity;
        while (new_cap < total_size) {
            new_cap *= 2;
        }
        uint8_t *new_buf = (uint8_t *)realloc(code->buffer, new_cap);
        if (!new_buf) {
            return -4;
        }
        code->buffer = new_buf;
        code->buffer_capacity = new_cap;
    }

    uint8_t *out = code->buffer;
    for (PhxNode *n = b->head; n != NULL; n = n->next) {
        uint32_t off = n->offset;

        if (n->node_type == PHX_NODE_EMBED && n->embed_data) {
            memcpy(out + off, n->embed_data, n->embed_size);
        } else if (n->encoded_size > 0) {
            memcpy(out + off, n->encoded, n->encoded_size);
        }
        /* Label nodes have encoded_size == 0, nothing to copy */
    }

    code->buffer_size = total_size;
    return 0;
}
