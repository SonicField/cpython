/*
 * arm64.c -- ARM64 (AArch64) instruction encoding backend for phoenix-asm
 *
 * Encodes all 55 ARM64 instructions used by the Phoenix JIT into 32-bit
 * instruction words, stored as PhxNode entries in the builder's node list.
 *
 * ARM64 encoding is very regular: every instruction is exactly 4 bytes,
 * little-endian.  Register fields are 5 bits each:
 *   Rd  [4:0]   -- destination register
 *   Rn  [9:5]   -- first source register
 *   Rm  [20:16] -- second source register
 *
 * C11, no C++ dependencies.
 */

#include "arm64.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/*  Internal helpers                                                   */
/* ------------------------------------------------------------------ */

/* Extract the hardware register number (0..31), masking off the FP flag. */
static inline uint32_t hw_reg(PhxGp r) {
    return r.id & 0x1Fu;
}

/* Is this register 64-bit? */
static inline int is64(PhxGp r) {
    return r.size == 8;
}

/* Is this an FP/SIMD register? */
static inline int is_fp(PhxGp r) {
    return (r.id & PHX_FP_FLAG) != 0;
}

/* Store a 32-bit little-endian instruction into a node's encoded[] buffer. */
static inline void store_inst(PhxNode *node, uint32_t inst) {
    node->encoded[0] = (uint8_t)(inst);
    node->encoded[1] = (uint8_t)(inst >> 8);
    node->encoded[2] = (uint8_t)(inst >> 16);
    node->encoded[3] = (uint8_t)(inst >> 24);
    node->encoded_size = 4;
}

/* Read back a 32-bit LE instruction from a node's encoded[] buffer. */
static inline uint32_t load_inst(const PhxNode *node) {
    return (uint32_t)node->encoded[0]
         | ((uint32_t)node->encoded[1] << 8)
         | ((uint32_t)node->encoded[2] << 16)
         | ((uint32_t)node->encoded[3] << 24);
}

/* Allocate a node, set it as an instruction node, and append it.
 * Returns the node, or NULL on allocation failure. */
static PhxNode *emit_node(PhxBuilder *b, uint16_t opcode) {
    PhxNode *n = phx_builder_alloc_node(b);
    if (!n) {
        return NULL;
    }
    n->node_type = PHX_NODE_INST;
    n->opcode = opcode;
    phx_builder_append_node(b, n);
    return n;
}

/* ------------------------------------------------------------------ */
/*  Immediate validation                                               */
/* ------------------------------------------------------------------ */

int phx_arm64_is_add_sub_imm(uint64_t imm) {
    /* 12-bit unsigned immediate, optionally shifted left by 12 */
    if (imm <= 0xFFFu) {
        return 1;
    }
    if ((imm & 0xFFFu) == 0 && (imm >> 12) <= 0xFFFu) {
        return 1;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/*  Logical immediate encoding                                         */
/*                                                                     */
/*  ARM64 logical immediates are described by three fields:             */
/*    N     (1 bit)  -- element size flag                               */
/*    immr  (6 bits) -- right rotate amount                            */
/*    imms  (6 bits) -- number of set bits minus one                   */
/*                                                                     */
/*  The encoding works by defining a repeating bitmask pattern.  The   */
/*  element size is a power of two (2, 4, 8, 16, 32, or 64 bits).     */
/*  Within each element, a contiguous run of 1-bits is rotated right   */
/*  by immr positions.                                                 */
/*                                                                     */
/*  Not all 64-bit values are encodable.  In particular:               */
/*    - 0x0000000000000000 is not encodable                             */
/*    - 0xFFFFFFFFFFFFFFFF is not encodable                             */
/* ------------------------------------------------------------------ */

/* Population count -- portable C11 */
static int popcount64(uint64_t x) {
    x = x - ((x >> 1) & 0x5555555555555555ull);
    x = (x & 0x3333333333333333ull) + ((x >> 2) & 0x3333333333333333ull);
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0Full;
    return (int)((x * 0x0101010101010101ull) >> 56);
}

/* Rotate right by k positions within width-bit value */
static uint64_t ror64(uint64_t val, int k, int width) {
    k &= (width - 1);
    if (k == 0) return val;
    uint64_t mask = (width == 64) ? ~0ull : ((1ull << width) - 1);
    val &= mask;
    return ((val >> k) | (val << (width - k))) & mask;
}

int phx_arm64_is_logical_imm(uint64_t val, uint32_t width) {
    assert(width == 32 || width == 64);

    if (width == 32) {
        /* For 32-bit, value must replicate in both halves */
        val = (val & 0xFFFFFFFFull) | ((val & 0xFFFFFFFFull) << 32);
    }

    /* All-zeros and all-ones are not encodable */
    if (val == 0 || val == ~0ull) {
        return 0;
    }

    /* Try each element size: 2, 4, 8, 16, 32, 64 */
    for (int esz = 2; esz <= 64; esz <<= 1) {
        uint64_t mask = (esz == 64) ? ~0ull : ((1ull << esz) - 1);
        uint64_t elem = val & mask;

        /* Check that val is a repeating pattern of this element */
        int valid = 1;
        for (int i = esz; i < 64; i += esz) {
            if (((val >> i) & mask) != elem) {
                valid = 0;
                break;
            }
        }
        if (!valid) continue;

        /* Within the element, the pattern must be a contiguous run of 1s,
         * possibly rotated. */
        /* Rotate to find the canonical form: all 1s at the bottom */
        int ones = popcount64(elem);
        if (ones == 0 || ones == esz) {
            continue; /* all-0 or all-1 element is not encodable */
        }

        /* Find the rotation: rotate until the pattern is 0..01..1 */
        for (int r = 0; r < esz; r++) {
            uint64_t rotated = ror64(elem, r, esz);
            uint64_t run_mask = (1ull << ones) - 1;
            if (rotated == run_mask) {
                return 1;
            }
        }
    }

    return 0;
}

uint32_t phx_arm64_encode_logical_imm(uint64_t val, uint32_t width) {
    assert(width == 32 || width == 64);

    if (width == 32) {
        val = (val & 0xFFFFFFFFull) | ((val & 0xFFFFFFFFull) << 32);
    }

    /* Try each element size */
    for (int esz = 2; esz <= 64; esz <<= 1) {
        uint64_t mask = (esz == 64) ? ~0ull : ((1ull << esz) - 1);
        uint64_t elem = val & mask;

        /* Verify repeating pattern */
        int valid = 1;
        for (int i = esz; i < 64; i += esz) {
            if (((val >> i) & mask) != elem) {
                valid = 0;
                break;
            }
        }
        if (!valid) continue;

        int ones = popcount64(elem);
        if (ones == 0 || ones == esz) continue;

        /* Find the rotation amount */
        for (int r = 0; r < esz; r++) {
            uint64_t rotated = ror64(elem, r, esz);
            uint64_t run_mask = (1ull << ones) - 1;
            if (rotated == run_mask) {
                /* Found it: r is the rotation, ones is the run length.
                 *
                 * Encoding:
                 *   N    = 1 if esz == 64, else 0
                 *   imms = (esz==64 ? 0 : ~(esz*2 - 1)) | (ones - 1)
                 *          masked to 6 bits
                 *   immr = r
                 *
                 * The imms field encodes both the element size and the
                 * number of set bits.  The upper bits (inverted) of imms
                 * indicate the element size:
                 *   esz=2:  imms = 0b1111xx
                 *   esz=4:  imms = 0b1110xx
                 *   esz=8:  imms = 0b110xxx
                 *   esz=16: imms = 0b10xxxx
                 *   esz=32: imms = 0b0xxxxx
                 *   esz=64: N=1, imms = 0bxxxxxx (no size encoding needed)
                 */
                uint32_t N = (esz == 64) ? 1u : 0u;
                /* r is how much we rotated the pattern RIGHT to reach
                 * the canonical form (run at bit 0).  The hardware does
                 * the INVERSE: it rotates the canonical form RIGHT by
                 * immr.  So immr = esz - r (mod esz). */
                uint32_t immr = (uint32_t)((esz - r) % esz);
                uint32_t imms;

                if (esz == 64) {
                    imms = (uint32_t)(ones - 1);
                } else {
                    /* Size indicator: the bits above the ones-1 field
                     * are set to indicate sizes smaller than 64. */
                    uint32_t size_mask = (uint32_t)(~(esz * 2 - 1)) & 0x3Fu;
                    imms = size_mask | (uint32_t)(ones - 1);
                }

                /* Return the 13-bit field: N:immr:imms positioned for
                 * bits [22:10] of the instruction word. */
                return (N << 12) | (immr << 6) | imms;
            }
        }
    }

    /* Should never reach here if caller verified is_logical_imm first */
    assert(0 && "not a valid logical immediate");
    return 0;
}

/* ------------------------------------------------------------------ */
/*  Encoding helpers: ARM64 instruction word construction               */
/* ------------------------------------------------------------------ */

/*
 * ADD/SUB immediate format:
 *   [31]    sf (1=64-bit, 0=32-bit)
 *   [30:29] op (00=ADD, 01=ADDS, 10=SUB, 11=SUBS)
 *   [28:24] 10001
 *   [23:22] shift (00=none, 01=LSL#12)
 *   [21:10] imm12
 *   [9:5]   Rn
 *   [4:0]   Rd
 */
static uint32_t encode_add_sub_imm(int sf, int op, uint32_t rd, uint32_t rn,
                                   uint64_t imm) {
    uint32_t sh = 0;
    uint32_t imm12;

    if (imm <= 0xFFFu) {
        imm12 = (uint32_t)imm;
    } else {
        assert((imm & 0xFFFu) == 0 && (imm >> 12) <= 0xFFFu);
        imm12 = (uint32_t)(imm >> 12);
        sh = 1;
    }

    return ((uint32_t)sf << 31)
         | ((uint32_t)op << 29)
         | (0x11u << 24)         /* 10001 */
         | (sh << 22)
         | (imm12 << 10)
         | (rn << 5)
         | rd;
}

/*
 * ADD/SUB shifted register format:
 *   [31]    sf
 *   [30:29] op (00=ADD, 01=ADDS, 10=SUB, 11=SUBS)
 *   [28:24] 01011
 *   [23:22] shift (00=LSL)
 *   [21]    0
 *   [20:16] Rm
 *   [15:10] imm6 (shift amount, 0 for unshifted)
 *   [9:5]   Rn
 *   [4:0]   Rd
 */
static uint32_t encode_add_sub_reg_shifted(int sf, int op, uint32_t rd,
                                           uint32_t rn, uint32_t rm,
                                           uint32_t shift_type,
                                           uint32_t shift_amount) {
    return ((uint32_t)sf << 31)
         | ((uint32_t)op << 29)
         | (0x0Bu << 24)         /* 01011 */
         | ((shift_type & 3u) << 22)
         | (0u << 21)
         | (rm << 16)
         | ((shift_amount & 0x3Fu) << 10)
         | (rn << 5)
         | rd;
}

static uint32_t encode_add_sub_reg(int sf, int op, uint32_t rd, uint32_t rn,
                                   uint32_t rm) {
    return encode_add_sub_reg_shifted(sf, op, rd, rn, rm, 0, 0);
}

/*
 * Logical shifted register:
 *   [31]    sf
 *   [30:29] opc (00=AND, 01=ORR, 10=EOR, 11=ANDS)
 *   [28:24] 01010
 *   [23:22] shift (00=LSL)
 *   [21]    N (0=normal, 1=invert Rm: ORN/BIC/EON/BICS)
 *   [20:16] Rm
 *   [15:10] imm6 (shift amount)
 *   [9:5]   Rn
 *   [4:0]   Rd
 */
static uint32_t encode_logical_reg(int sf, int opc, int n_invert,
                                   uint32_t rd, uint32_t rn, uint32_t rm) {
    return ((uint32_t)sf << 31)
         | ((uint32_t)opc << 29)
         | (0x0Au << 24)         /* 01010 */
         | (0u << 22)            /* shift = LSL */
         | ((uint32_t)n_invert << 21)
         | (rm << 16)
         | (0u << 10)            /* shift amount = 0 */
         | (rn << 5)
         | rd;
}

/*
 * Logical immediate:
 *   [31]    sf
 *   [30:29] opc (00=AND, 01=ORR, 10=EOR, 11=ANDS)
 *   [28:23] 100100
 *   [22]    N
 *   [21:16] immr
 *   [15:10] imms
 *   [9:5]   Rn
 *   [4:0]   Rd
 */
static uint32_t encode_logical_imm_inst(int sf, int opc, uint32_t rd,
                                        uint32_t rn, uint32_t nimms_field) {
    /* nimms_field is the 13-bit N:immr:imms from encode_logical_imm */
    return ((uint32_t)sf << 31)
         | ((uint32_t)opc << 29)
         | (0x24u << 23)         /* 100100 */
         | (nimms_field << 10)
         | (rn << 5)
         | rd;
}

/*
 * Load/store unsigned offset:
 *   [31:30] size (00=byte, 01=halfword, 10=word, 11=doubleword)
 *   [29:27] 111
 *   [26]    V (0=GP, 1=SIMD/FP)
 *   [25:24] 01
 *   [23:22] opc (load/store variant)
 *   [21:10] imm12 (scaled by access size)
 *   [9:5]   Rn (base)
 *   [4:0]   Rt (data register)
 */
static uint32_t encode_ldst_uoff(uint32_t size_bits, int v, uint32_t opc,
                                 uint32_t rt, uint32_t rn, uint32_t imm12) {
    return (size_bits << 30)
         | (0x7u << 27)          /* 111 */
         | ((uint32_t)v << 26)
         | (0x1u << 24)          /* 01 */
         | (opc << 22)
         | (imm12 << 10)
         | (rn << 5)
         | rt;
}

/*
 * Load/store register offset (pre/post-index):
 *   [31:30] size
 *   [29:27] 111
 *   [26]    V
 *   [25:24] 00
 *   [23:22] opc
 *   [21]    0
 *   [20:12] imm9 (signed)
 *   [11:10] index type (01=post-index, 11=pre-index, 00=unscaled)
 *   [9:5]   Rn
 *   [4:0]   Rt
 */
static uint32_t encode_ldst_signed(uint32_t size_bits, int v, uint32_t opc,
                                   uint32_t rt, uint32_t rn, int32_t imm9,
                                   uint32_t idx_type) {
    return (size_bits << 30)
         | (0x7u << 27)
         | ((uint32_t)v << 26)
         | (0x0u << 24)          /* 00 */
         | (opc << 22)
         | (0u << 21)
         | (((uint32_t)imm9 & 0x1FFu) << 12)
         | (idx_type << 10)
         | (rn << 5)
         | rt;
}

/*
 * Load/store register (register offset):
 *   [31:30] size
 *   [29:27] 111
 *   [26]    V
 *   [25:24] 00
 *   [23:22] opc
 *   [21]    1
 *   [20:16] Rm
 *   [15:13] option (011=LSL for Xm, 010=UXTW for Wm)
 *   [12]    S (shift amount: 0 or log2(size))
 *   [11:10] 10
 *   [9:5]   Rn
 *   [4:0]   Rt
 */
static uint32_t encode_ldst_reg_off(uint32_t size_bits, int v, uint32_t opc,
                                    uint32_t rt, uint32_t rn, uint32_t rm,
                                    int shift) {
    return (size_bits << 30)
         | (0x7u << 27)
         | ((uint32_t)v << 26)
         | (0x0u << 24)
         | (opc << 22)
         | (1u << 21)
         | (rm << 16)
         | (0x3u << 13)         /* option = 011 (LSL, Xm) */
         | ((uint32_t)(shift != 0) << 12)
         | (0x2u << 10)         /* 10 */
         | (rn << 5)
         | rt;
}

/*
 * Load/store pair (signed offset):
 *   [31:30] opc (00=32-bit, 10=64-bit)
 *   [29:27] 101
 *   [26]    V (0=GP, 1=FP)
 *   [25:23] 010 (signed offset)
 *   [22]    L (1=load, 0=store)
 *   [21:15] imm7 (signed, scaled by access size)
 *   [14:10] Rt2
 *   [9:5]   Rn
 *   [4:0]   Rt1
 */
/* addr_mode: 1 = post-indexed, 2 = signed offset, 3 = pre-indexed */
static uint32_t encode_ldst_pair(uint32_t opc, int v, int load,
                                 uint32_t rt1, uint32_t rt2, uint32_t rn,
                                 int32_t imm7, uint32_t addr_mode) {
    return (opc << 30)
         | (0x5u << 27)         /* 101 */
         | ((uint32_t)v << 26)
         | (addr_mode << 23)
         | ((uint32_t)load << 22)
         | (((uint32_t)imm7 & 0x7Fu) << 15)
         | (rt2 << 10)
         | (rn << 5)
         | rt1;
}

/*
 * Data processing (3 source):
 *   [31]    sf
 *   [30:29] op54
 *   [28:24] 11011
 *   [23:21] op31
 *   [20:16] Rm
 *   [15]    o0
 *   [14:10] Ra
 *   [9:5]   Rn
 *   [4:0]   Rd
 */
static uint32_t encode_dp3(int sf, uint32_t op54, uint32_t op31, uint32_t o0,
                           uint32_t rd, uint32_t rn, uint32_t rm,
                           uint32_t ra) {
    return ((uint32_t)sf << 31)
         | (op54 << 29)
         | (0x1Bu << 24)        /* 11011 */
         | (op31 << 21)
         | (rm << 16)
         | (o0 << 15)
         | (ra << 10)
         | (rn << 5)
         | rd;
}

/*
 * Data processing (2 source):
 *   [31]    sf
 *   [30]    0
 *   [29]    S
 *   [28:21] 11010110
 *   [20:16] Rm
 *   [15:10] opcode2
 *   [9:5]   Rn
 *   [4:0]   Rd
 */
static uint32_t encode_dp2(int sf, uint32_t opcode2,
                           uint32_t rd, uint32_t rn, uint32_t rm) {
    return ((uint32_t)sf << 31)
         | (0u << 30)
         | (0u << 29)            /* S=0 */
         | (0xD6u << 21)        /* 11010110 */
         | (rm << 16)
         | (opcode2 << 10)
         | (rn << 5)
         | rd;
}

/*
 * Conditional select:
 *   [31]    sf
 *   [30]    op (0=CSEL, 1=CSINV/CSINC)
 *   [29]    S (0)
 *   [28:21] 11010100
 *   [20:16] Rm
 *   [15:12] cond
 *   [11:10] op2 (00=CSEL, 01=CSINC, 00=CSINV with op=1, 01=CSNEG)
 *   [9:5]   Rn
 *   [4:0]   Rd
 */
static uint32_t encode_cond_sel(int sf, int op, int op2, uint32_t rd,
                                uint32_t rn, uint32_t rm, uint32_t cond) {
    return ((uint32_t)sf << 31)
         | ((uint32_t)op << 30)
         | (0u << 29)
         | (0xD4u << 21)        /* 11010100 */
         | (rm << 16)
         | (cond << 12)
         | ((uint32_t)op2 << 10)
         | (rn << 5)
         | rd;
}

/*
 * Bitfield (SBFM, UBFM, BFM):
 *   [31]    sf
 *   [30:29] opc (00=SBFM, 01=BFM, 10=UBFM)
 *   [28:23] 100110
 *   [22]    N (=sf for well-formed instructions)
 *   [21:16] immr
 *   [15:10] imms
 *   [9:5]   Rn
 *   [4:0]   Rd
 */
static uint32_t encode_bitfield(int sf, int opc, uint32_t rd, uint32_t rn,
                                uint32_t immr, uint32_t imms) {
    return ((uint32_t)sf << 31)
         | ((uint32_t)opc << 29)
         | (0x26u << 23)        /* 100110 */
         | ((uint32_t)sf << 22) /* N = sf */
         | (immr << 16)
         | (imms << 10)
         | (rn << 5)
         | rd;
}

/*
 * FP data-processing (2 source):
 *   [31]    M (0)
 *   [30]    0
 *   [29]    S (0)
 *   [28:24] 11110
 *   [23:22] ftype (00=single, 01=double)
 *   [21]    1
 *   [20:16] Rm
 *   [15:12] opcode
 *   [11:10] 10
 *   [9:5]   Rn
 *   [4:0]   Rd
 */
static uint32_t encode_fp_dp2(int ftype, uint32_t fp_op,
                              uint32_t rd, uint32_t rn, uint32_t rm) {
    return (0u << 31)
         | (0u << 30)
         | (0u << 29)
         | (0x1Eu << 24)        /* 11110 */
         | ((uint32_t)ftype << 22)
         | (1u << 21)
         | (rm << 16)
         | (fp_op << 12)
         | (0x2u << 10)         /* 10 */
         | (rn << 5)
         | rd;
}

/* ------------------------------------------------------------------ */
/*  Load/store offset computation                                      */
/*                                                                     */
/*  Decides between unsigned-offset, pre-index, or unscaled forms.     */
/* ------------------------------------------------------------------ */

/*
 * Materialise an absolute address into X16 (IP0) via MOVZ/MOVK, then
 * rewrite mem to [X16] with offset=0.  ARM64 has no disp32-absolute
 * addressing mode, so this is required for any `ptr(uint64_t)` operand.
 *
 * X16 (IP0) is the primary assembler scratch register, used by
 * resolve_abs_addr and other memory operand materialization.
 * X17 (IP1) is the secondary scratch, used by add/sub/cmp immediate
 * fallback paths.  Both are in DISALLOWED_REGISTERS.
 *
 * X16 and X17 must NOT be used in the same role: X16 handles memory
 * operand resolution (resolve_abs_addr), X17 handles large immediate
 * fallback (CMP/ADD/SUB with oversized immediates).  This prevents
 * clobbering during compound emissions where both are needed.
 */
static PhxMem resolve_abs_addr(PhxBuilder *b, PhxMem mem) {
    if (!mem.is_abs_addr) return mem;

    /* Emit MOVZ/MOVK sequence to load abs_addr into X16 */
    PhxGp x16 = PHX_X16;
    phx_a64_mov_ri(b, x16, mem.abs_addr);

    /* Rewrite mem to [X16, #0] */
    PhxMem resolved = {};
    resolved.base = x16;
    resolved.offset = 0;
    resolved.size = mem.size;
    return resolved;
}

/*
 * Compute the load/store encoding for a memory operand.  Returns the
 * full 32-bit instruction word.
 *
 * Parameters:
 *   size_bits  - [31:30] size field (00=byte, 01=half, 10=word, 11=dword)
 *   v          - 1 if FP/SIMD register, 0 if GP
 *   opc        - [23:22] opc field for this particular load/store variant
 *   rt         - data register number
 *   mem        - memory operand
 *   access_sz  - byte count of the access (1, 2, 4, or 8)
 */
static uint32_t encode_ldst(uint32_t size_bits, int v, uint32_t opc,
                            uint32_t rt, PhxMem mem, int access_sz) {
    uint32_t rn = hw_reg(mem.base);
    int32_t off = mem.offset;

    /* Pre-indexed: [base, #off]!  idx_type=11 */
    if (mem.is_pre_index) {
        assert(off >= -256 && off <= 255);
        return encode_ldst_signed(size_bits, v, opc, rt, rn, off, 0x3u);
    }

    /* Post-indexed: [base], #off  idx_type=01 */
    if (mem.is_post_index) {
        assert(off >= -256 && off <= 255);
        return encode_ldst_signed(size_bits, v, opc, rt, rn, off, 0x1u);
    }

    if (mem.has_index) {
        /* Register offset form */
        return encode_ldst_reg_off(size_bits, v, opc, rt, rn,
                                   hw_reg(mem.index), /*shift=*/0);
    }

    /* Try unsigned offset form: offset must be non-negative and aligned */
    if (off >= 0 && (off % access_sz) == 0) {
        uint32_t scaled = (uint32_t)off / (uint32_t)access_sz;
        if (scaled <= 0xFFFu) {
            return encode_ldst_uoff(size_bits, v, opc, rt, rn, scaled);
        }
    }

    /* Fall back to unscaled signed 9-bit form (LDUR/STUR) */
    assert(off >= -256 && off <= 255);
    return encode_ldst_signed(size_bits, v, opc, rt, rn, off,
                              0x0u); /* idx_type=00 = unscaled */
}

/* ------------------------------------------------------------------ */
/*  MOV (16-bit immediate MOVZ/MOVK sequence)                          */
/* ------------------------------------------------------------------ */

/*
 * MOVZ: move 16-bit immediate with zero to other bits
 *   [31]    sf
 *   [30:29] 10 (MOVZ)
 *   [28:23] 100101
 *   [22:21] hw (shift: 0=0, 1=16, 2=32, 3=48)
 *   [20:5]  imm16
 *   [4:0]   Rd
 */
static uint32_t encode_movz(int sf, uint32_t rd, uint16_t imm16, int hw) {
    return ((uint32_t)sf << 31)
         | (0x2u << 29)         /* 10 = MOVZ */
         | (0x25u << 23)        /* 100101 */
         | ((uint32_t)hw << 21)
         | ((uint32_t)imm16 << 5)
         | rd;
}

/*
 * MOVK: move 16-bit immediate, keep other bits
 *   [31]    sf
 *   [30:29] 11 (MOVK)
 *   [28:23] 100101
 *   [22:21] hw
 *   [20:5]  imm16
 *   [4:0]   Rd
 */
static uint32_t encode_movk(int sf, uint32_t rd, uint16_t imm16, int hw) {
    return ((uint32_t)sf << 31)
         | (0x3u << 29)         /* 11 = MOVK */
         | (0x25u << 23)        /* 100101 */
         | ((uint32_t)hw << 21)
         | ((uint32_t)imm16 << 5)
         | rd;
}

/*
 * MOVN: move wide with NOT
 *   [31]    sf
 *   [30:29] 00 (MOVN)
 *   [28:23] 100101
 *   [22:21] hw
 *   [20:5]  imm16
 *   [4:0]   Rd
 */
static uint32_t encode_movn(int sf, uint32_t rd, uint16_t imm16, int hw) {
    return ((uint32_t)sf << 31)
         | (0x0u << 29)         /* 00 = MOVN */
         | (0x25u << 23)        /* 100101 */
         | ((uint32_t)hw << 21)
         | ((uint32_t)imm16 << 5)
         | rd;
}

/* ================================================================== */
/*  INSTRUCTION EMISSION FUNCTIONS                                     */
/* ================================================================== */

/* ------------------------------------------------------------------ */
/*  Data Movement                                                      */
/* ------------------------------------------------------------------ */

void phx_a64_mov_rr(PhxBuilder *b, PhxGp dst, PhxGp src) {
    assert(dst.size == src.size);
    PhxNode *n = emit_node(b, PHX_A64_MOV);
    if (!n) return;

    int sf = is64(dst);
    uint32_t rd = hw_reg(dst);
    uint32_t rm = hw_reg(src);

    /* ARM64 register 31 is SP in ADD/SUB but XZR in logical ops (ORR).
     * MOV Xd, Xm is normally ORR Xd, XZR, Xm — but if src or dst is
     * register 31 (SP), we must use ADD Xd, Xn, #0 instead. */
    if (rm == 31 || rd == 31) {
        /* ADD Xd, Xn, #0 — treats reg 31 as SP */
        uint32_t inst = ((uint32_t)sf << 31)
                      | (0x11u << 24)   /* ADD immediate */
                      | (0u << 22)      /* shift=0 */
                      | (0u << 10)      /* imm12=0 */
                      | (rm << 5)       /* Rn */
                      | rd;             /* Rd */
        store_inst(n, inst);
    } else {
        /* MOV Xd, Xm is encoded as ORR Xd, XZR, Xm */
        uint32_t inst = encode_logical_reg(sf, 0x1 /*ORR*/, 0 /*N*/,
                                           rd, 31 /*XZR*/, rm);
        store_inst(n, inst);
    }

    n->operands[0] = phx_op_gp(dst);
    n->operands[1] = phx_op_gp(src);
    n->num_operands = 2;
}

void phx_a64_mov_ri(PhxBuilder *b, PhxGp dst, uint64_t imm) {
    int sf = is64(dst);
    uint32_t rd = hw_reg(dst);

    /* Special case: MOV to zero register or imm == 0 -> MOVZ */
    if (imm == 0) {
        PhxNode *n = emit_node(b, PHX_A64_MOV);
        if (!n) return;
        store_inst(n, encode_movz(sf, rd, 0, 0));
        n->operands[0] = phx_op_gp(dst);
        n->operands[1] = phx_op_imm((int64_t)imm);
        n->num_operands = 2;
        return;
    }

    /* For 32-bit, truncate */
    if (!sf) {
        imm &= 0xFFFFFFFFull;
    }

    /* Check if we can use MOVN (for values close to all-ones) */
    uint64_t inv = sf ? ~imm : (~imm & 0xFFFFFFFFull);
    int max_hw = sf ? 4 : 2;

    /* Count how many 16-bit chunks are non-zero for imm and ~imm */
    int nz_imm = 0, nz_inv = 0;
    for (int hw = 0; hw < max_hw; hw++) {
        if ((imm >> (hw * 16)) & 0xFFFF) nz_imm++;
        if ((inv >> (hw * 16)) & 0xFFFF) nz_inv++;
    }

    /* Use MOVN if the inverted form has fewer non-zero chunks */
    if (nz_inv < nz_imm) {
        int first = 1;
        for (int hw = 0; hw < max_hw; hw++) {
            uint16_t chunk = (uint16_t)(inv >> (hw * 16));
            if (first) {
                PhxNode *n = emit_node(b, PHX_A64_MOV);
                if (!n) return;
                store_inst(n, encode_movn(sf, rd, chunk, hw));
                n->operands[0] = phx_op_gp(dst);
                n->operands[1] = phx_op_imm((int64_t)imm);
                n->num_operands = 2;
                first = 0;
            } else if (chunk != 0xFFFF) {
                /* We need MOVK to fix up chunks that aren't all-ones
                 * in the original (all-zeros in the inverted form) */
                uint16_t orig_chunk = (uint16_t)(imm >> (hw * 16));
                PhxNode *n = emit_node(b, PHX_A64_MOV);
                if (!n) return;
                store_inst(n, encode_movk(sf, rd, orig_chunk, hw));
                n->operands[0] = phx_op_gp(dst);
                n->operands[1] = phx_op_imm((int64_t)orig_chunk);
                n->num_operands = 2;
            }
        }
        return;
    }

    /* Standard MOVZ + MOVK sequence */
    int first = 1;
    for (int hw = 0; hw < max_hw; hw++) {
        uint16_t chunk = (uint16_t)(imm >> (hw * 16));
        if (chunk == 0 && !first) {
            continue; /* skip zero chunks after the first MOVZ */
        }
        if (first) {
            PhxNode *n = emit_node(b, PHX_A64_MOV);
            if (!n) return;
            store_inst(n, encode_movz(sf, rd, chunk, hw));
            n->operands[0] = phx_op_gp(dst);
            n->operands[1] = phx_op_imm((int64_t)imm);
            n->num_operands = 2;
            first = 0;
        } else {
            PhxNode *n = emit_node(b, PHX_A64_MOV);
            if (!n) return;
            store_inst(n, encode_movk(sf, rd, chunk, hw));
            n->operands[0] = phx_op_gp(dst);
            n->operands[1] = phx_op_imm((int64_t)chunk);
            n->num_operands = 2;
        }
    }
}

void phx_a64_ldr(PhxBuilder *b, PhxGp dst, PhxMem mem) {
    mem = resolve_abs_addr(b, mem);

    PhxNode *n = emit_node(b, PHX_A64_LDR);
    if (!n) return;

    int sf = is64(dst);
    /* size_bits: 11 for 64-bit, 10 for 32-bit */
    uint32_t size_bits = sf ? 0x3u : 0x2u;
    int access_sz = sf ? 8 : 4;

    uint32_t inst = encode_ldst(size_bits, 0, 0x1 /*opc=01=LDR*/,
                                hw_reg(dst), mem, access_sz);
    store_inst(n, inst);

    n->operands[0] = phx_op_gp(dst);
    n->operands[1] = phx_op_mem(mem);
    n->num_operands = 2;
}

void phx_a64_ldr_fp(PhxBuilder *b, PhxGp dst, PhxMem mem) {
    mem = resolve_abs_addr(b, mem);

    PhxNode *n = emit_node(b, PHX_A64_LDR);
    if (!n) return;

    /* 64-bit FP/SIMD load: size=11, V=1, opc=01 (LDR Dt, [...]) */
    uint32_t inst = encode_ldst(0x3u, 1, 0x1 /*opc=01=LDR*/,
                                hw_reg(dst), mem, 8);
    store_inst(n, inst);

    n->operands[0] = phx_op_gp(dst);
    n->operands[1] = phx_op_mem(mem);
    n->num_operands = 2;
}

void phx_a64_ldrb(PhxBuilder *b, PhxGp dst, PhxMem mem) {
    mem = resolve_abs_addr(b, mem);

    PhxNode *n = emit_node(b, PHX_A64_LDRB);
    if (!n) return;

    /* LDRB: size=00, opc=01 */
    uint32_t inst = encode_ldst(0x0u, 0, 0x1u, hw_reg(dst), mem, 1);
    store_inst(n, inst);

    n->operands[0] = phx_op_gp(dst);
    n->operands[1] = phx_op_mem(mem);
    n->num_operands = 2;
}

void phx_a64_ldrh(PhxBuilder *b, PhxGp dst, PhxMem mem) {
    mem = resolve_abs_addr(b, mem);

    PhxNode *n = emit_node(b, PHX_A64_LDRH);
    if (!n) return;

    /* LDRH: size=01, opc=01 */
    uint32_t inst = encode_ldst(0x1u, 0, 0x1u, hw_reg(dst), mem, 2);
    store_inst(n, inst);

    n->operands[0] = phx_op_gp(dst);
    n->operands[1] = phx_op_mem(mem);
    n->num_operands = 2;
}

void phx_a64_ldrsb(PhxBuilder *b, PhxGp dst, PhxMem mem) {
    mem = resolve_abs_addr(b, mem);

    PhxNode *n = emit_node(b, PHX_A64_LDRSB);
    if (!n) return;

    /* LDRSB: size=00, opc depends on destination:
     *   opc=10 -> LDRSB to 64-bit (Xt)
     *   opc=11 -> LDRSB to 32-bit (Wt) */
    uint32_t opc = is64(dst) ? 0x2u : 0x3u;
    uint32_t inst = encode_ldst(0x0u, 0, opc, hw_reg(dst), mem, 1);
    store_inst(n, inst);

    n->operands[0] = phx_op_gp(dst);
    n->operands[1] = phx_op_mem(mem);
    n->num_operands = 2;
}

void phx_a64_ldrsh(PhxBuilder *b, PhxGp dst, PhxMem mem) {
    mem = resolve_abs_addr(b, mem);

    PhxNode *n = emit_node(b, PHX_A64_LDRSH);
    if (!n) return;

    /* LDRSH: size=01, opc=10 (64-bit) or opc=11 (32-bit) */
    uint32_t opc = is64(dst) ? 0x2u : 0x3u;
    uint32_t inst = encode_ldst(0x1u, 0, opc, hw_reg(dst), mem, 2);
    store_inst(n, inst);

    n->operands[0] = phx_op_gp(dst);
    n->operands[1] = phx_op_mem(mem);
    n->num_operands = 2;
}

void phx_a64_ldrsw(PhxBuilder *b, PhxGp dst, PhxMem mem) {
    mem = resolve_abs_addr(b, mem);

    PhxNode *n = emit_node(b, PHX_A64_LDRSW);
    if (!n) return;

    /* LDRSW: size=10, opc=10 (always 64-bit destination) */
    uint32_t inst = encode_ldst(0x2u, 0, 0x2u, hw_reg(dst), mem, 4);
    store_inst(n, inst);

    n->operands[0] = phx_op_gp(dst);
    n->operands[1] = phx_op_mem(mem);
    n->num_operands = 2;
}

void phx_a64_ldp(PhxBuilder *b, PhxGp rt1, PhxGp rt2, PhxMem mem) {
    /* Dispatch to pre/post-indexed variants if flags are set */
    if (mem.is_pre_index) {
        phx_a64_ldp_pre(b, rt1, rt2, mem.base, mem.offset);
        return;
    }
    if (mem.is_post_index) {
        phx_a64_ldp_post(b, rt1, rt2, mem.base, mem.offset);
        return;
    }

    assert(rt1.size == rt2.size);
    PhxNode *n = emit_node(b, PHX_A64_LDP);
    if (!n) return;

    int sf = is64(rt1);
    /* opc: 00 for 32-bit, 10 for 64-bit */
    uint32_t opc = sf ? 0x2u : 0x0u;
    int access_sz = sf ? 8 : 4;
    int32_t imm7 = mem.offset / access_sz;
    assert(mem.offset % access_sz == 0);
    assert(imm7 >= -64 && imm7 <= 63);

    uint32_t inst = encode_ldst_pair(opc, 0, 1 /*load*/,
                                     hw_reg(rt1), hw_reg(rt2),
                                     hw_reg(mem.base), imm7, 2 /*signed offset*/);
    store_inst(n, inst);

    n->operands[0] = phx_op_gp(rt1);
    n->operands[1] = phx_op_gp(rt2);
    n->operands[2] = phx_op_mem(mem);
    n->num_operands = 3;
}

/* LDP pre-indexed: ldp rt1, rt2, [rn, #imm]! */
void phx_a64_ldp_pre(PhxBuilder *b, PhxGp rt1, PhxGp rt2, PhxGp base, int32_t offset) {
    assert(rt1.size == rt2.size);
    PhxNode *n = emit_node(b, PHX_A64_LDP);
    if (!n) return;

    int sf = is64(rt1);
    uint32_t opc = sf ? 0x2u : 0x0u;
    int access_sz = sf ? 8 : 4;
    int32_t imm7 = offset / access_sz;
    assert(offset % access_sz == 0);
    assert(imm7 >= -64 && imm7 <= 63);

    uint32_t inst = encode_ldst_pair(opc, 0, 1 /*load*/,
                                     hw_reg(rt1), hw_reg(rt2),
                                     hw_reg(base), imm7, 3 /*pre-indexed*/);
    store_inst(n, inst);

    n->operands[0] = phx_op_gp(rt1);
    n->operands[1] = phx_op_gp(rt2);
    n->operands[2] = phx_op_gp(base);
    n->num_operands = 3;
}

/* LDP post-indexed: ldp rt1, rt2, [rn], #imm */
void phx_a64_ldp_post(PhxBuilder *b, PhxGp rt1, PhxGp rt2, PhxGp base, int32_t offset) {
    assert(rt1.size == rt2.size);
    PhxNode *n = emit_node(b, PHX_A64_LDP);
    if (!n) return;

    int sf = is64(rt1);
    uint32_t opc = sf ? 0x2u : 0x0u;
    int access_sz = sf ? 8 : 4;
    int32_t imm7 = offset / access_sz;
    assert(offset % access_sz == 0);
    assert(imm7 >= -64 && imm7 <= 63);

    uint32_t inst = encode_ldst_pair(opc, 0, 1 /*load*/,
                                     hw_reg(rt1), hw_reg(rt2),
                                     hw_reg(base), imm7, 1 /*post-indexed*/);
    store_inst(n, inst);

    n->operands[0] = phx_op_gp(rt1);
    n->operands[1] = phx_op_gp(rt2);
    n->operands[2] = phx_op_gp(base);
    n->num_operands = 3;
}

void phx_a64_str(PhxBuilder *b, PhxGp src, PhxMem mem) {
    mem = resolve_abs_addr(b, mem);

    PhxNode *n = emit_node(b, PHX_A64_STR);
    if (!n) return;

    int sf = is64(src);
    uint32_t size_bits = sf ? 0x3u : 0x2u;
    int access_sz = sf ? 8 : 4;

    uint32_t inst = encode_ldst(size_bits, 0, 0x0 /*opc=00=STR*/,
                                hw_reg(src), mem, access_sz);
    store_inst(n, inst);

    n->operands[0] = phx_op_gp(src);
    n->operands[1] = phx_op_mem(mem);
    n->num_operands = 2;
}

void phx_a64_str_fp(PhxBuilder *b, PhxGp src, PhxMem mem) {
    mem = resolve_abs_addr(b, mem);

    PhxNode *n = emit_node(b, PHX_A64_STR);
    if (!n) return;

    /* 64-bit FP/SIMD store: size=11, V=1, opc=00 (STR Dt, [...]) */
    uint32_t inst = encode_ldst(0x3u, 1, 0x0 /*opc=00=STR*/,
                                hw_reg(src), mem, 8);
    store_inst(n, inst);

    n->operands[0] = phx_op_gp(src);
    n->operands[1] = phx_op_mem(mem);
    n->num_operands = 2;
}

void phx_a64_strb(PhxBuilder *b, PhxGp src, PhxMem mem) {
    mem = resolve_abs_addr(b, mem);

    PhxNode *n = emit_node(b, PHX_A64_STRB);
    if (!n) return;

    /* STRB: size=00, opc=00 */
    uint32_t inst = encode_ldst(0x0u, 0, 0x0u, hw_reg(src), mem, 1);
    store_inst(n, inst);

    n->operands[0] = phx_op_gp(src);
    n->operands[1] = phx_op_mem(mem);
    n->num_operands = 2;
}

void phx_a64_strh(PhxBuilder *b, PhxGp src, PhxMem mem) {
    mem = resolve_abs_addr(b, mem);

    PhxNode *n = emit_node(b, PHX_A64_STRH);
    if (!n) return;

    /* STRH: size=01, opc=00 */
    uint32_t inst = encode_ldst(0x1u, 0, 0x0u, hw_reg(src), mem, 2);
    store_inst(n, inst);

    n->operands[0] = phx_op_gp(src);
    n->operands[1] = phx_op_mem(mem);
    n->num_operands = 2;
}

void phx_a64_stp(PhxBuilder *b, PhxGp rt1, PhxGp rt2, PhxMem mem) {
    /* Dispatch to pre/post-indexed variants if flags are set */
    if (mem.is_pre_index) {
        phx_a64_stp_pre(b, rt1, rt2, mem.base, mem.offset);
        return;
    }
    if (mem.is_post_index) {
        phx_a64_stp_post(b, rt1, rt2, mem.base, mem.offset);
        return;
    }

    assert(rt1.size == rt2.size);
    PhxNode *n = emit_node(b, PHX_A64_STP);
    if (!n) return;

    int sf = is64(rt1);
    uint32_t opc = sf ? 0x2u : 0x0u;
    int access_sz = sf ? 8 : 4;
    int32_t imm7 = mem.offset / access_sz;
    assert(mem.offset % access_sz == 0);
    assert(imm7 >= -64 && imm7 <= 63);

    uint32_t inst = encode_ldst_pair(opc, 0, 0 /*store*/,
                                     hw_reg(rt1), hw_reg(rt2),
                                     hw_reg(mem.base), imm7, 2 /*signed offset*/);
    store_inst(n, inst);

    n->operands[0] = phx_op_gp(rt1);
    n->operands[1] = phx_op_gp(rt2);
    n->operands[2] = phx_op_mem(mem);
    n->num_operands = 3;
}

/* STP pre-indexed: stp rt1, rt2, [rn, #imm]! */
void phx_a64_stp_pre(PhxBuilder *b, PhxGp rt1, PhxGp rt2, PhxGp base, int32_t offset) {
    assert(rt1.size == rt2.size);
    PhxNode *n = emit_node(b, PHX_A64_STP);
    if (!n) return;

    int sf = is64(rt1);
    uint32_t opc = sf ? 0x2u : 0x0u;
    int access_sz = sf ? 8 : 4;
    int32_t imm7 = offset / access_sz;
    assert(offset % access_sz == 0);
    assert(imm7 >= -64 && imm7 <= 63);

    uint32_t inst = encode_ldst_pair(opc, 0, 0 /*store*/,
                                     hw_reg(rt1), hw_reg(rt2),
                                     hw_reg(base), imm7, 3 /*pre-indexed*/);
    store_inst(n, inst);

    n->operands[0] = phx_op_gp(rt1);
    n->operands[1] = phx_op_gp(rt2);
    n->operands[2] = phx_op_gp(base);
    n->num_operands = 3;
}

/* STP post-indexed: stp rt1, rt2, [rn], #imm */
void phx_a64_stp_post(PhxBuilder *b, PhxGp rt1, PhxGp rt2, PhxGp base, int32_t offset) {
    assert(rt1.size == rt2.size);
    PhxNode *n = emit_node(b, PHX_A64_STP);
    if (!n) return;

    int sf = is64(rt1);
    uint32_t opc = sf ? 0x2u : 0x0u;
    int access_sz = sf ? 8 : 4;
    int32_t imm7 = offset / access_sz;
    assert(offset % access_sz == 0);
    assert(imm7 >= -64 && imm7 <= 63);

    uint32_t inst = encode_ldst_pair(opc, 0, 0 /*store*/,
                                     hw_reg(rt1), hw_reg(rt2),
                                     hw_reg(base), imm7, 1 /*post-indexed*/);
    store_inst(n, inst);

    n->operands[0] = phx_op_gp(rt1);
    n->operands[1] = phx_op_gp(rt2);
    n->operands[2] = phx_op_gp(base);
    n->num_operands = 3;
}

void phx_a64_fmov(PhxBuilder *b, PhxGp dst, PhxGp src) {
    PhxNode *n = emit_node(b, PHX_A64_FMOV);
    if (!n) return;

    uint32_t rd = hw_reg(dst);
    uint32_t rn = hw_reg(src);
    int dst_fp = is_fp(dst);
    int src_fp = is_fp(src);
    /* Determine the ftype from the FP register's size */
    int ftype;
    if (dst_fp) {
        ftype = (dst.size == 8) ? 1 : 0; /* 01=double, 00=single */
    } else {
        ftype = (src.size == 8) ? 1 : 0;
    }

    uint32_t inst;

    if (dst_fp && src_fp) {
        /* FMOV Dd, Dn (FP to FP)
         * 0001 1110 xx1 00000 01 0000 nnnnn ddddd
         * [31:24] = 0x1E, [23:22]=ftype, [21]=1, [20:17]=0, [16:15]=01
         * Actually: 0001 1110 TT1 0000 0 0100 00 nnnnn ddddd */
        inst = (0x1Eu << 24)
             | ((uint32_t)ftype << 22)
             | (1u << 21)
             | (0u << 17)
             | (0x10u << 10)    /* opcode=000000, 01, 0000 */
             | (rn << 5)
             | rd;
    } else if (dst_fp && !src_fp) {
        /* FMOV Dd, Xn (GP to FP)
         * Scalar: sf=1 for 64-bit GP, ftype=01(double)/00(single)
         * 0x9E670000 for FMOV Dd, Xn (64-bit)
         * [31] sf, [30:24] 0011110, [23:22] ftype, [21] 1, [20:19] 00,
         * [18:16] 111, [15:10] 000000, [9:5] Rn, [4:0] Rd */
        int sf = (ftype == 1) ? 1 : 0;
        inst = ((uint32_t)sf << 31)
             | (0x1Eu << 24)
             | ((uint32_t)ftype << 22)
             | (1u << 21)
             | (0x7u << 16)     /* rmode=00, opcode=111 */
             | (0u << 10)
             | (rn << 5)
             | rd;
    } else {
        /* FMOV Xd, Dn (FP to GP) */
        int sf = (ftype == 1) ? 1 : 0;
        inst = ((uint32_t)sf << 31)
             | (0x1Eu << 24)
             | ((uint32_t)ftype << 22)
             | (1u << 21)
             | (0x6u << 16)     /* rmode=00, opcode=110 */
             | (0u << 10)
             | (rn << 5)
             | rd;
    }

    store_inst(n, inst);

    n->operands[0] = phx_op_gp(dst);
    n->operands[1] = phx_op_gp(src);
    n->num_operands = 2;
}

void phx_a64_adr(PhxBuilder *b, PhxGp dst, PhxLabel label) {
    PhxNode *n = emit_node(b, PHX_A64_ADR);
    if (!n) return;

    /* ADR Xd, label
     * Encoding: [31]=0, [30:29]=immlo, [28:24]=10000,
     *           [23:5]=immhi, [4:0]=Rd
     *
     * The immediate is a 21-bit signed PC-relative offset.
     * We emit a placeholder (offset=0) and fix it up during finalize. */
    uint32_t inst = (0u << 31)
                  | (0u << 29)       /* immlo placeholder */
                  | (0x10u << 24)    /* 10000 */
                  | (0u << 5)        /* immhi placeholder */
                  | hw_reg(dst);
    store_inst(n, inst);

    n->operands[0] = phx_op_gp(dst);
    n->operands[1] = phx_op_label(label);
    n->num_operands = 2;

    /* Register fixup for label resolution during finalize */
    phx_builder_add_fixup(b, n, label.id, 1);
}

/* ------------------------------------------------------------------ */
/*  Arithmetic                                                         */
/* ------------------------------------------------------------------ */

void phx_a64_add_rrr(PhxBuilder *b, PhxGp dst, PhxGp src1, PhxGp src2) {
    assert(dst.size == src1.size && src1.size == src2.size);
    PhxNode *n = emit_node(b, PHX_A64_ADD);
    if (!n) return;

    int sf = is64(dst);
    uint32_t inst = encode_add_sub_reg(sf, 0x0 /*ADD*/, hw_reg(dst),
                                       hw_reg(src1), hw_reg(src2));
    store_inst(n, inst);

    n->operands[0] = phx_op_gp(dst);
    n->operands[1] = phx_op_gp(src1);
    n->operands[2] = phx_op_gp(src2);
    n->num_operands = 3;
}

void phx_a64_add_rrr_shifted(PhxBuilder *b, PhxGp dst, PhxGp src1, PhxGp src2,
                              uint32_t shift_type, uint32_t shift_amount) {
    assert(dst.size == src1.size && src1.size == src2.size);
    PhxNode *n = emit_node(b, PHX_A64_ADD);
    if (!n) return;

    int sf = is64(dst);
    uint32_t inst = encode_add_sub_reg_shifted(sf, 0x0 /*ADD*/, hw_reg(dst),
                                                hw_reg(src1), hw_reg(src2),
                                                shift_type, shift_amount);
    store_inst(n, inst);

    n->operands[0] = phx_op_gp(dst);
    n->operands[1] = phx_op_gp(src1);
    n->operands[2] = phx_op_gp(src2);
    n->num_operands = 3;
}

void phx_a64_add_rri(PhxBuilder *b, PhxGp dst, PhxGp src, int64_t imm) {
    assert(dst.size == src.size);
    int sf = is64(dst);
    int op;
    uint64_t abs_imm;

    /* If imm is negative, encode as SUB with |imm| */
    if (imm < 0) {
        op = 0x2; /* SUB */
        abs_imm = (uint64_t)(-imm);
    } else {
        op = 0x0; /* ADD */
        abs_imm = (uint64_t)imm;
    }

    /* Large immediate: MOV to X17 scratch + ADD reg,reg.
     * Uses X17 (IP1) to avoid clobbering X16 (IP0) which may hold
     * a live value from resolve_abs_addr in compound emissions. */
    if (!phx_arm64_is_add_sub_imm(abs_imm)) {
        PhxGp scratch = sf ? PHX_X17 : (PhxGp)PHX_REG_GP(17, 4);
        phx_a64_mov_ri(b, scratch, (uint64_t)imm);
        phx_a64_add_rrr(b, dst, src, scratch);
        return;
    }

    PhxNode *n = emit_node(b, PHX_A64_ADD);
    if (!n) return;

    uint32_t inst = encode_add_sub_imm(sf, op, hw_reg(dst), hw_reg(src),
                                       abs_imm);
    store_inst(n, inst);

    n->operands[0] = phx_op_gp(dst);
    n->operands[1] = phx_op_gp(src);
    n->operands[2] = phx_op_imm(imm);
    n->num_operands = 3;
}

void phx_a64_adds_rrr(PhxBuilder *b, PhxGp dst, PhxGp src1, PhxGp src2) {
    assert(dst.size == src1.size && src1.size == src2.size);
    PhxNode *n = emit_node(b, PHX_A64_ADDS);
    if (!n) return;

    int sf = is64(dst);
    uint32_t inst = encode_add_sub_reg(sf, 0x1 /*ADDS*/, hw_reg(dst),
                                       hw_reg(src1), hw_reg(src2));
    store_inst(n, inst);

    n->operands[0] = phx_op_gp(dst);
    n->operands[1] = phx_op_gp(src1);
    n->operands[2] = phx_op_gp(src2);
    n->num_operands = 3;
}

void phx_a64_adds_rri(PhxBuilder *b, PhxGp dst, PhxGp src, int64_t imm) {
    assert(dst.size == src.size);
    int sf = is64(dst);
    int op;
    uint64_t abs_imm;

    if (imm < 0) {
        op = 0x3; /* SUBS */
        abs_imm = (uint64_t)(-imm);
    } else {
        op = 0x1; /* ADDS */
        abs_imm = (uint64_t)imm;
    }

    /* Large immediate: MOV to X17 scratch + ADDS reg,reg */
    if (!phx_arm64_is_add_sub_imm(abs_imm)) {
        PhxGp scratch = sf ? PHX_X17 : (PhxGp)PHX_REG_GP(17, 4);
        phx_a64_mov_ri(b, scratch, (uint64_t)imm);
        phx_a64_adds_rrr(b, dst, src, scratch);
        return;
    }

    PhxNode *n = emit_node(b, PHX_A64_ADDS);
    if (!n) return;

    uint32_t inst = encode_add_sub_imm(sf, op, hw_reg(dst), hw_reg(src),
                                       abs_imm);
    store_inst(n, inst);

    n->operands[0] = phx_op_gp(dst);
    n->operands[1] = phx_op_gp(src);
    n->operands[2] = phx_op_imm(imm);
    n->num_operands = 3;
}

void phx_a64_sub_rrr(PhxBuilder *b, PhxGp dst, PhxGp src1, PhxGp src2) {
    assert(dst.size == src1.size && src1.size == src2.size);
    PhxNode *n = emit_node(b, PHX_A64_SUB);
    if (!n) return;

    int sf = is64(dst);
    uint32_t inst = encode_add_sub_reg(sf, 0x2 /*SUB*/, hw_reg(dst),
                                       hw_reg(src1), hw_reg(src2));
    store_inst(n, inst);

    n->operands[0] = phx_op_gp(dst);
    n->operands[1] = phx_op_gp(src1);
    n->operands[2] = phx_op_gp(src2);
    n->num_operands = 3;
}

void phx_a64_sub_rrr_shifted(PhxBuilder *b, PhxGp dst, PhxGp src1, PhxGp src2,
                              uint32_t shift_type, uint32_t shift_amount) {
    assert(dst.size == src1.size && src1.size == src2.size);
    PhxNode *n = emit_node(b, PHX_A64_SUB);
    if (!n) return;

    int sf = is64(dst);
    uint32_t inst = encode_add_sub_reg_shifted(sf, 0x2 /*SUB*/, hw_reg(dst),
                                                hw_reg(src1), hw_reg(src2),
                                                shift_type, shift_amount);
    store_inst(n, inst);

    n->operands[0] = phx_op_gp(dst);
    n->operands[1] = phx_op_gp(src1);
    n->operands[2] = phx_op_gp(src2);
    n->num_operands = 3;
}

void phx_a64_sub_rri(PhxBuilder *b, PhxGp dst, PhxGp src, int64_t imm) {
    assert(dst.size == src.size);
    int sf = is64(dst);
    int op;
    uint64_t abs_imm;

    /* If imm is negative, encode as ADD with |imm| */
    if (imm < 0) {
        op = 0x0; /* ADD */
        abs_imm = (uint64_t)(-imm);
    } else {
        op = 0x2; /* SUB */
        abs_imm = (uint64_t)imm;
    }

    /* Large immediate: MOV to X17 scratch + SUB reg,reg */
    if (!phx_arm64_is_add_sub_imm(abs_imm)) {
        PhxGp scratch = sf ? PHX_X17 : (PhxGp)PHX_REG_GP(17, 4);
        phx_a64_mov_ri(b, scratch, (uint64_t)imm);
        phx_a64_sub_rrr(b, dst, src, scratch);
        return;
    }

    PhxNode *n = emit_node(b, PHX_A64_SUB);
    if (!n) return;

    uint32_t inst = encode_add_sub_imm(sf, op, hw_reg(dst), hw_reg(src),
                                       abs_imm);
    store_inst(n, inst);

    n->operands[0] = phx_op_gp(dst);
    n->operands[1] = phx_op_gp(src);
    n->operands[2] = phx_op_imm(imm);
    n->num_operands = 3;
}

void phx_a64_subs_rrr(PhxBuilder *b, PhxGp dst, PhxGp src1, PhxGp src2) {
    assert(dst.size == src1.size && src1.size == src2.size);
    PhxNode *n = emit_node(b, PHX_A64_SUBS);
    if (!n) return;

    int sf = is64(dst);
    uint32_t inst = encode_add_sub_reg(sf, 0x3 /*SUBS*/, hw_reg(dst),
                                       hw_reg(src1), hw_reg(src2));
    store_inst(n, inst);

    n->operands[0] = phx_op_gp(dst);
    n->operands[1] = phx_op_gp(src1);
    n->operands[2] = phx_op_gp(src2);
    n->num_operands = 3;
}

void phx_a64_subs_rri(PhxBuilder *b, PhxGp dst, PhxGp src, int64_t imm) {
    assert(dst.size == src.size);
    int sf = is64(dst);
    int op;
    uint64_t abs_imm;

    if (imm < 0) {
        op = 0x1; /* ADDS */
        abs_imm = (uint64_t)(-imm);
    } else {
        op = 0x3; /* SUBS */
        abs_imm = (uint64_t)imm;
    }

    /* Large immediate: MOV to X17 scratch + SUBS reg,reg */
    if (!phx_arm64_is_add_sub_imm(abs_imm)) {
        PhxGp scratch = sf ? PHX_X17 : (PhxGp)PHX_REG_GP(17, 4);
        phx_a64_mov_ri(b, scratch, (uint64_t)imm);
        phx_a64_subs_rrr(b, dst, src, scratch);
        return;
    }

    PhxNode *n = emit_node(b, PHX_A64_SUBS);
    if (!n) return;

    uint32_t inst = encode_add_sub_imm(sf, op, hw_reg(dst), hw_reg(src),
                                       abs_imm);
    store_inst(n, inst);

    n->operands[0] = phx_op_gp(dst);
    n->operands[1] = phx_op_gp(src);
    n->operands[2] = phx_op_imm(imm);
    n->num_operands = 3;
}

void phx_a64_mul(PhxBuilder *b, PhxGp dst, PhxGp src1, PhxGp src2) {
    /* MUL is MADD with Ra = XZR */
    phx_a64_madd(b, dst, src1, src2, (PhxGp){ 31, dst.size });
}

void phx_a64_madd(PhxBuilder *b, PhxGp dst, PhxGp src1, PhxGp src2,
                  PhxGp addend) {
    assert(dst.size == src1.size && src1.size == src2.size);
    PhxNode *n = emit_node(b, PHX_A64_MADD);
    if (!n) return;

    int sf = is64(dst);
    /* MADD: op54=00, op31=000, o0=0 */
    uint32_t inst = encode_dp3(sf, 0x0, 0x0, 0x0,
                               hw_reg(dst), hw_reg(src1),
                               hw_reg(src2), hw_reg(addend));
    store_inst(n, inst);

    n->operands[0] = phx_op_gp(dst);
    n->operands[1] = phx_op_gp(src1);
    n->operands[2] = phx_op_gp(src2);
    n->operands[3] = phx_op_gp(addend);
    n->num_operands = 4;
}

void phx_a64_sdiv(PhxBuilder *b, PhxGp dst, PhxGp src1, PhxGp src2) {
    assert(dst.size == src1.size && src1.size == src2.size);
    PhxNode *n = emit_node(b, PHX_A64_SDIV);
    if (!n) return;

    int sf = is64(dst);
    /* SDIV: opcode2 = 000011 (0x03) */
    uint32_t inst = encode_dp2(sf, 0x03, hw_reg(dst),
                               hw_reg(src1), hw_reg(src2));
    store_inst(n, inst);

    n->operands[0] = phx_op_gp(dst);
    n->operands[1] = phx_op_gp(src1);
    n->operands[2] = phx_op_gp(src2);
    n->num_operands = 3;
}

void phx_a64_udiv(PhxBuilder *b, PhxGp dst, PhxGp src1, PhxGp src2) {
    assert(dst.size == src1.size && src1.size == src2.size);
    PhxNode *n = emit_node(b, PHX_A64_UDIV);
    if (!n) return;

    int sf = is64(dst);
    /* UDIV: opcode2 = 000010 (0x02) */
    uint32_t inst = encode_dp2(sf, 0x02, hw_reg(dst),
                               hw_reg(src1), hw_reg(src2));
    store_inst(n, inst);

    n->operands[0] = phx_op_gp(dst);
    n->operands[1] = phx_op_gp(src1);
    n->operands[2] = phx_op_gp(src2);
    n->num_operands = 3;
}

/* ------------------------------------------------------------------ */
/*  Logic                                                              */
/* ------------------------------------------------------------------ */

void phx_a64_and_rrr(PhxBuilder *b, PhxGp dst, PhxGp src1, PhxGp src2) {
    assert(dst.size == src1.size && src1.size == src2.size);
    PhxNode *n = emit_node(b, PHX_A64_AND);
    if (!n) return;

    int sf = is64(dst);
    uint32_t inst = encode_logical_reg(sf, 0x0 /*AND*/, 0,
                                       hw_reg(dst), hw_reg(src1),
                                       hw_reg(src2));
    store_inst(n, inst);

    n->operands[0] = phx_op_gp(dst);
    n->operands[1] = phx_op_gp(src1);
    n->operands[2] = phx_op_gp(src2);
    n->num_operands = 3;
}

void phx_a64_and_rri(PhxBuilder *b, PhxGp dst, PhxGp src, uint64_t imm) {
    assert(dst.size == src.size);
    PhxNode *n = emit_node(b, PHX_A64_AND);
    if (!n) return;

    int sf = is64(dst);
    uint32_t width = sf ? 64 : 32;
    assert(phx_arm64_is_logical_imm(imm, width));
    uint32_t nimms = phx_arm64_encode_logical_imm(imm, width);
    uint32_t inst = encode_logical_imm_inst(sf, 0x0 /*AND*/, hw_reg(dst),
                                            hw_reg(src), nimms);
    store_inst(n, inst);

    n->operands[0] = phx_op_gp(dst);
    n->operands[1] = phx_op_gp(src);
    n->operands[2] = phx_op_imm((int64_t)imm);
    n->num_operands = 3;
}

void phx_a64_eor_rrr(PhxBuilder *b, PhxGp dst, PhxGp src1, PhxGp src2) {
    assert(dst.size == src1.size && src1.size == src2.size);
    PhxNode *n = emit_node(b, PHX_A64_EOR);
    if (!n) return;

    int sf = is64(dst);
    uint32_t inst = encode_logical_reg(sf, 0x2 /*EOR*/, 0,
                                       hw_reg(dst), hw_reg(src1),
                                       hw_reg(src2));
    store_inst(n, inst);

    n->operands[0] = phx_op_gp(dst);
    n->operands[1] = phx_op_gp(src1);
    n->operands[2] = phx_op_gp(src2);
    n->num_operands = 3;
}

void phx_a64_eor_rri(PhxBuilder *b, PhxGp dst, PhxGp src, uint64_t imm) {
    assert(dst.size == src.size);
    PhxNode *n = emit_node(b, PHX_A64_EOR);
    if (!n) return;

    int sf = is64(dst);
    uint32_t width = sf ? 64 : 32;
    assert(phx_arm64_is_logical_imm(imm, width));
    uint32_t nimms = phx_arm64_encode_logical_imm(imm, width);
    uint32_t inst = encode_logical_imm_inst(sf, 0x2 /*EOR*/, hw_reg(dst),
                                            hw_reg(src), nimms);
    store_inst(n, inst);

    n->operands[0] = phx_op_gp(dst);
    n->operands[1] = phx_op_gp(src);
    n->operands[2] = phx_op_imm((int64_t)imm);
    n->num_operands = 3;
}

void phx_a64_orr_rrr(PhxBuilder *b, PhxGp dst, PhxGp src1, PhxGp src2) {
    assert(dst.size == src1.size && src1.size == src2.size);
    PhxNode *n = emit_node(b, PHX_A64_ORR);
    if (!n) return;

    int sf = is64(dst);
    uint32_t inst = encode_logical_reg(sf, 0x1 /*ORR*/, 0,
                                       hw_reg(dst), hw_reg(src1),
                                       hw_reg(src2));
    store_inst(n, inst);

    n->operands[0] = phx_op_gp(dst);
    n->operands[1] = phx_op_gp(src1);
    n->operands[2] = phx_op_gp(src2);
    n->num_operands = 3;
}

void phx_a64_orr_rri(PhxBuilder *b, PhxGp dst, PhxGp src, uint64_t imm) {
    assert(dst.size == src.size);
    PhxNode *n = emit_node(b, PHX_A64_ORR);
    if (!n) return;

    int sf = is64(dst);
    uint32_t width = sf ? 64 : 32;
    assert(phx_arm64_is_logical_imm(imm, width));
    uint32_t nimms = phx_arm64_encode_logical_imm(imm, width);
    uint32_t inst = encode_logical_imm_inst(sf, 0x1 /*ORR*/, hw_reg(dst),
                                            hw_reg(src), nimms);
    store_inst(n, inst);

    n->operands[0] = phx_op_gp(dst);
    n->operands[1] = phx_op_gp(src);
    n->operands[2] = phx_op_imm((int64_t)imm);
    n->num_operands = 3;
}

void phx_a64_mvn(PhxBuilder *b, PhxGp dst, PhxGp src) {
    assert(dst.size == src.size);
    PhxNode *n = emit_node(b, PHX_A64_MVN);
    if (!n) return;

    /* MVN Xd, Xm = ORN Xd, XZR, Xm */
    int sf = is64(dst);
    uint32_t inst = encode_logical_reg(sf, 0x1 /*ORR*/, 1 /*N=invert*/,
                                       hw_reg(dst), 31 /*XZR*/,
                                       hw_reg(src));
    store_inst(n, inst);

    n->operands[0] = phx_op_gp(dst);
    n->operands[1] = phx_op_gp(src);
    n->num_operands = 2;
}

/* ------------------------------------------------------------------ */
/*  Comparison / Test                                                  */
/* ------------------------------------------------------------------ */

void phx_a64_cmp_rr(PhxBuilder *b, PhxGp src1, PhxGp src2) {
    assert(src1.size == src2.size);
    PhxNode *n = emit_node(b, PHX_A64_CMP);
    if (!n) return;

    /* CMP Xn, Xm = SUBS XZR, Xn, Xm */
    int sf = is64(src1);
    uint32_t inst = encode_add_sub_reg(sf, 0x3 /*SUBS*/, 31 /*XZR*/,
                                       hw_reg(src1), hw_reg(src2));
    store_inst(n, inst);

    n->operands[0] = phx_op_gp(src1);
    n->operands[1] = phx_op_gp(src2);
    n->num_operands = 2;
}

void phx_a64_cmp_ri(PhxBuilder *b, PhxGp src, int64_t imm) {
    int sf = is64(src);
    int op;
    uint64_t abs_imm;

    /* CMP Xn, #imm = SUBS XZR, Xn, #imm
     * If imm is negative, use ADDS XZR, Xn, #|imm| (CMN) */
    if (imm < 0) {
        op = 0x1; /* ADDS (CMN) */
        abs_imm = (uint64_t)(-imm);
    } else {
        op = 0x3; /* SUBS (CMP) */
        abs_imm = (uint64_t)imm;
    }

    /* If immediate doesn't fit 12-bit add/sub encoding, fall back to
     * MOV imm to X17 scratch + CMP reg, reg. Uses X17 (IP1) to avoid
     * clobbering X16 (IP0) which may hold a live value from
     * resolve_abs_addr in compound emissions. */
    if (!phx_arm64_is_add_sub_imm(abs_imm)) {
        PhxGp scratch = PHX_X17;
        if (!sf) {
            scratch = (PhxGp)PHX_REG_GP(17, 4); /* W17 for 32-bit */
        }
        phx_a64_mov_ri(b, scratch, (uint64_t)imm);
        phx_a64_cmp_rr(b, src, scratch);
        return;
    }

    PhxNode *n = emit_node(b, PHX_A64_CMP);
    if (!n) return;

    uint32_t inst = encode_add_sub_imm(sf, op, 31 /*XZR*/, hw_reg(src),
                                       abs_imm);
    store_inst(n, inst);

    n->operands[0] = phx_op_gp(src);
    n->operands[1] = phx_op_imm(imm);
    n->num_operands = 2;
}

void phx_a64_tst_rr(PhxBuilder *b, PhxGp src1, PhxGp src2) {
    assert(src1.size == src2.size);
    PhxNode *n = emit_node(b, PHX_A64_TST);
    if (!n) return;

    /* TST Xn, Xm = ANDS XZR, Xn, Xm */
    int sf = is64(src1);
    uint32_t inst = encode_logical_reg(sf, 0x3 /*ANDS*/, 0,
                                       31 /*XZR*/, hw_reg(src1),
                                       hw_reg(src2));
    store_inst(n, inst);

    n->operands[0] = phx_op_gp(src1);
    n->operands[1] = phx_op_gp(src2);
    n->num_operands = 2;
}

void phx_a64_tst_ri(PhxBuilder *b, PhxGp src, uint64_t imm) {
    PhxNode *n = emit_node(b, PHX_A64_TST);
    if (!n) return;

    /* TST Xn, #imm = ANDS XZR, Xn, #imm */
    int sf = is64(src);
    uint32_t width = sf ? 64 : 32;
    assert(phx_arm64_is_logical_imm(imm, width));
    uint32_t nimms = phx_arm64_encode_logical_imm(imm, width);
    uint32_t inst = encode_logical_imm_inst(sf, 0x3 /*ANDS*/, 31 /*XZR*/,
                                            hw_reg(src), nimms);
    store_inst(n, inst);

    n->operands[0] = phx_op_gp(src);
    n->operands[1] = phx_op_imm((int64_t)imm);
    n->num_operands = 2;
}

void phx_a64_fcmp(PhxBuilder *b, PhxGp src1, PhxGp src2) {
    PhxNode *n = emit_node(b, PHX_A64_FCMP);
    if (!n) return;

    uint32_t rn = hw_reg(src1);
    uint32_t rm = hw_reg(src2);
    int ftype = (src1.size == 8) ? 1 : 0;

    /* FCMP Dn, Dm:
     * [31:24] 00011110, [23:22] ftype, [21] 1, [20:16] Rm,
     * [15:10] 001000, [9:5] Rn, [4:0] 00000
     * (opcode2=00000 for normal compare, 01000 for compare-with-zero) */
    uint32_t inst = (0x1Eu << 24)
                  | ((uint32_t)ftype << 22)
                  | (1u << 21)
                  | (rm << 16)
                  | (0x08u << 10)    /* 001000 */
                  | (rn << 5)
                  | 0x00u;           /* opc=0, opcode2=00000 */
    store_inst(n, inst);

    n->operands[0] = phx_op_gp(src1);
    n->operands[1] = phx_op_gp(src2);
    n->num_operands = 2;
}

/* ------------------------------------------------------------------ */
/*  Conditional                                                        */
/* ------------------------------------------------------------------ */

void phx_a64_csel(PhxBuilder *b, PhxGp dst, PhxGp src1, PhxGp src2,
                  PhxArm64Cond cond) {
    assert(dst.size == src1.size && src1.size == src2.size);
    PhxNode *n = emit_node(b, PHX_A64_CSEL);
    if (!n) return;

    int sf = is64(dst);
    uint32_t inst = encode_cond_sel(sf, 0 /*op=CSEL*/, 0 /*op2*/,
                                    hw_reg(dst), hw_reg(src1),
                                    hw_reg(src2), (uint32_t)cond);
    store_inst(n, inst);

    n->operands[0] = phx_op_gp(dst);
    n->operands[1] = phx_op_gp(src1);
    n->operands[2] = phx_op_gp(src2);
    n->operands[3] = phx_op_imm((int64_t)cond);
    n->num_operands = 4;
}

void phx_a64_cset(PhxBuilder *b, PhxGp dst, PhxArm64Cond cond) {
    PhxNode *n = emit_node(b, PHX_A64_CSET);
    if (!n) return;

    /* CSET Xd, cond = CSINC Xd, XZR, XZR, invert(cond)
     * Invert = flip bit 0 of the condition code */
    int sf = is64(dst);
    uint32_t inv_cond = (uint32_t)cond ^ 1u;
    uint32_t inst = encode_cond_sel(sf, 0 /*op*/, 1 /*op2=01 CSINC*/,
                                    hw_reg(dst), 31 /*XZR*/, 31 /*XZR*/,
                                    inv_cond);
    store_inst(n, inst);

    n->operands[0] = phx_op_gp(dst);
    n->operands[1] = phx_op_imm((int64_t)cond);
    n->num_operands = 2;
}

/* ------------------------------------------------------------------ */
/*  Branches: unconditional                                            */
/* ------------------------------------------------------------------ */

void phx_a64_b(PhxBuilder *b, PhxLabel label) {
    PhxNode *n = emit_node(b, PHX_A64_B);
    if (!n) return;

    /* B label:
     * [31:26] 000101
     * [25:0]  imm26 (signed, word-aligned offset / 4)
     * Placeholder with offset=0 */
    uint32_t inst = 0x14000000u; /* B with imm26=0 */
    store_inst(n, inst);

    n->operands[0] = phx_op_label(label);
    n->num_operands = 1;

    phx_builder_add_fixup(b, n, label.id, 0);
}

void phx_a64_bl(PhxBuilder *b, PhxLabel label) {
    PhxNode *n = emit_node(b, PHX_A64_BL);
    if (!n) return;

    /* BL label:
     * [31:26] 100101
     * [25:0]  imm26
     * Placeholder with offset=0 */
    uint32_t inst = 0x94000000u; /* BL with imm26=0 */
    store_inst(n, inst);

    n->operands[0] = phx_op_label(label);
    n->num_operands = 1;

    phx_builder_add_fixup(b, n, label.id, 0);
}

void phx_a64_blr(PhxBuilder *b, PhxGp target) {
    PhxNode *n = emit_node(b, PHX_A64_BLR);
    if (!n) return;

    /* BLR Xn:
     * [31:25] 1101011 0
     * [24:21] 0 01 (opc=01=BLR)
     * [20:16] 11111
     * [15:10] 000000
     * [9:5]   Rn
     * [4:0]   00000
     *
     * Full: 1101 0110 0011 1111 0000 00nn nnn0 0000
     *     = 0xD63F0000 | (Rn << 5) */
    uint32_t inst = 0xD63F0000u | (hw_reg(target) << 5);
    store_inst(n, inst);

    n->operands[0] = phx_op_gp(target);
    n->num_operands = 1;
}

void phx_a64_br(PhxBuilder *b, PhxGp target) {
    PhxNode *n = emit_node(b, PHX_A64_BR);
    if (!n) return;

    /* BR Xn:
     * 1101 0110 0001 1111 0000 00nn nnn0 0000
     * = 0xD61F0000 | (Rn << 5) */
    uint32_t inst = 0xD61F0000u | (hw_reg(target) << 5);
    store_inst(n, inst);

    n->operands[0] = phx_op_gp(target);
    n->num_operands = 1;
}

/* ------------------------------------------------------------------ */
/*  Branches: conditional                                              */
/* ------------------------------------------------------------------ */

void phx_a64_b_cond(PhxBuilder *b, PhxArm64Cond cond, PhxLabel label) {
    PhxNode *n = emit_node(b, PHX_A64_B_COND);
    if (!n) return;

    /* B.cond label:
     * [31:24] 01010100
     * [23:5]  imm19 (signed, word-aligned offset / 4)
     * [4]     0
     * [3:0]   cond
     * Placeholder with imm19=0 */
    uint32_t inst = 0x54000000u | (uint32_t)cond;
    store_inst(n, inst);

    n->operands[0] = phx_op_imm((int64_t)cond);
    n->operands[1] = phx_op_label(label);
    n->num_operands = 2;

    phx_builder_add_fixup(b, n, label.id, 1);
}

/* ------------------------------------------------------------------ */
/*  Compare-and-branch                                                 */
/* ------------------------------------------------------------------ */

void phx_a64_cbz(PhxBuilder *b, PhxGp src, PhxLabel label) {
    PhxNode *n = emit_node(b, PHX_A64_CBZ);
    if (!n) return;

    /* CBZ Xt, label:
     * [31]    sf
     * [30:25] 011010 0
     * [24]    op (0=CBZ, 1=CBNZ)
     * [23:5]  imm19
     * [4:0]   Rt
     * Placeholder with imm19=0 */
    int sf = is64(src);
    uint32_t inst = ((uint32_t)sf << 31)
                  | (0x34u << 24)    /* 0110100 0 = CBZ */
                  | hw_reg(src);
    store_inst(n, inst);

    n->operands[0] = phx_op_gp(src);
    n->operands[1] = phx_op_label(label);
    n->num_operands = 2;

    phx_builder_add_fixup(b, n, label.id, 1);
}

void phx_a64_cbnz(PhxBuilder *b, PhxGp src, PhxLabel label) {
    PhxNode *n = emit_node(b, PHX_A64_CBNZ);
    if (!n) return;

    /* CBNZ Xt, label:
     * Same as CBZ but op=1 ([24]=1)
     * 0x35 << 24 = 0011 0101 = sf:0110101 */
    int sf = is64(src);
    uint32_t inst = ((uint32_t)sf << 31)
                  | (0x35u << 24)    /* 0110101 0 = CBNZ */
                  | hw_reg(src);
    store_inst(n, inst);

    n->operands[0] = phx_op_gp(src);
    n->operands[1] = phx_op_label(label);
    n->num_operands = 2;

    phx_builder_add_fixup(b, n, label.id, 1);
}

/* ------------------------------------------------------------------ */
/*  Test-and-branch                                                    */
/* ------------------------------------------------------------------ */

void phx_a64_tbnz(PhxBuilder *b, PhxGp src, uint32_t bit, PhxLabel label) {
    PhxNode *n = emit_node(b, PHX_A64_TBNZ);
    if (!n) return;

    /* TBNZ Rt, #bit, label:
     * [31]    b5       (bit number [5], also sets 64-bit if 1)
     * [30:25] 011011 1 (TBNZ fixed bits: 0110111)
     * [24]    op=1     (1=TBNZ, 0=TBZ)
     * [23:19] b40      (bit number [4:0])
     * [18:5]  imm14    (signed word offset, resolved during finalize)
     * [4:0]   Rt
     *
     * Full encoding: b5 | 0110111 | b40 | imm14 | Rt
     * Placeholder with imm14=0 */
    uint32_t b5 = (bit >> 5) & 1u;
    uint32_t b40 = bit & 0x1Fu;
    uint32_t inst = (b5 << 31)
                  | (0x37u << 24)     /* 0110111 x = TBNZ */
                  | (b40 << 19)
                  | hw_reg(src);
    store_inst(n, inst);

    n->operands[0] = phx_op_gp(src);
    n->operands[1] = phx_op_imm((int64_t)bit);
    n->operands[2] = phx_op_label(label);
    n->num_operands = 3;

    phx_builder_add_fixup(b, n, label.id, 1);
}

/* ------------------------------------------------------------------ */
/*  Sign/zero extension                                                */
/* ------------------------------------------------------------------ */

void phx_a64_sxtb(PhxBuilder *b, PhxGp dst, PhxGp src) {
    PhxNode *n = emit_node(b, PHX_A64_SXTB);
    if (!n) return;

    /* SXTB Xd, Wn = SBFM Xd, Xn, #0, #7
     * (sign extend bits [7:0]) */
    int sf = is64(dst);
    uint32_t inst = encode_bitfield(sf, 0x0 /*SBFM*/, hw_reg(dst),
                                    hw_reg(src), 0, 7);
    store_inst(n, inst);

    n->operands[0] = phx_op_gp(dst);
    n->operands[1] = phx_op_gp(src);
    n->num_operands = 2;
}

void phx_a64_sxth(PhxBuilder *b, PhxGp dst, PhxGp src) {
    PhxNode *n = emit_node(b, PHX_A64_SXTH);
    if (!n) return;

    /* SXTH Xd, Wn = SBFM Xd, Xn, #0, #15 */
    int sf = is64(dst);
    uint32_t inst = encode_bitfield(sf, 0x0 /*SBFM*/, hw_reg(dst),
                                    hw_reg(src), 0, 15);
    store_inst(n, inst);

    n->operands[0] = phx_op_gp(dst);
    n->operands[1] = phx_op_gp(src);
    n->num_operands = 2;
}

void phx_a64_sxtw(PhxBuilder *b, PhxGp dst, PhxGp src) {
    PhxNode *n = emit_node(b, PHX_A64_SXTW);
    if (!n) return;

    /* SXTW Xd, Wn = SBFM Xd, Xn, #0, #31
     * Always 64-bit destination */
    uint32_t inst = encode_bitfield(1 /*sf=64*/, 0x0 /*SBFM*/, hw_reg(dst),
                                    hw_reg(src), 0, 31);
    store_inst(n, inst);

    n->operands[0] = phx_op_gp(dst);
    n->operands[1] = phx_op_gp(src);
    n->num_operands = 2;
}

void phx_a64_uxtb(PhxBuilder *b, PhxGp dst, PhxGp src) {
    PhxNode *n = emit_node(b, PHX_A64_UXTB);
    if (!n) return;

    /* UXTB Wd, Wn = UBFM Wd, Wn, #0, #7
     * Always 32-bit (sf=0) */
    uint32_t inst = encode_bitfield(0 /*sf=32*/, 0x2 /*UBFM*/, hw_reg(dst),
                                    hw_reg(src), 0, 7);
    store_inst(n, inst);

    n->operands[0] = phx_op_gp(dst);
    n->operands[1] = phx_op_gp(src);
    n->num_operands = 2;
}

void phx_a64_uxth(PhxBuilder *b, PhxGp dst, PhxGp src) {
    PhxNode *n = emit_node(b, PHX_A64_UXTH);
    if (!n) return;

    /* UXTH Wd, Wn = UBFM Wd, Wn, #0, #15
     * Always 32-bit (sf=0) */
    uint32_t inst = encode_bitfield(0 /*sf=32*/, 0x2 /*UBFM*/, hw_reg(dst),
                                    hw_reg(src), 0, 15);
    store_inst(n, inst);

    n->operands[0] = phx_op_gp(dst);
    n->operands[1] = phx_op_gp(src);
    n->num_operands = 2;
}

/* ------------------------------------------------------------------ */
/*  Shift                                                              */
/* ------------------------------------------------------------------ */

void phx_a64_lsl(PhxBuilder *b, PhxGp dst, PhxGp src, uint32_t shift) {
    assert(dst.size == src.size);
    PhxNode *n = emit_node(b, PHX_A64_LSL);
    if (!n) return;

    /* LSL Xd, Xn, #shift = UBFM Xd, Xn, #(regwidth - shift), #(regwidth - 1 - shift)
     * For 64-bit: immr = 64 - shift, imms = 63 - shift
     * For 32-bit: immr = 32 - shift, imms = 31 - shift */
    int sf = is64(dst);
    uint32_t regwidth = sf ? 64 : 32;
    assert(shift > 0 && shift < regwidth);

    uint32_t immr = (regwidth - shift) & (regwidth - 1);
    uint32_t imms = regwidth - 1 - shift;

    uint32_t inst = encode_bitfield(sf, 0x2 /*UBFM*/, hw_reg(dst),
                                    hw_reg(src), immr, imms);
    store_inst(n, inst);

    n->operands[0] = phx_op_gp(dst);
    n->operands[1] = phx_op_gp(src);
    n->operands[2] = phx_op_imm((int64_t)shift);
    n->num_operands = 3;
}

/* ------------------------------------------------------------------ */
/*  FP arithmetic                                                      */
/* ------------------------------------------------------------------ */

void phx_a64_fadd(PhxBuilder *b, PhxGp dst, PhxGp src1, PhxGp src2) {
    PhxNode *n = emit_node(b, PHX_A64_FADD);
    if (!n) return;

    int ftype = (dst.size == 8) ? 1 : 0;
    /* FADD: opcode = 0010 */
    uint32_t inst = encode_fp_dp2(ftype, 0x2, hw_reg(dst),
                                  hw_reg(src1), hw_reg(src2));
    store_inst(n, inst);

    n->operands[0] = phx_op_gp(dst);
    n->operands[1] = phx_op_gp(src1);
    n->operands[2] = phx_op_gp(src2);
    n->num_operands = 3;
}

void phx_a64_fsub(PhxBuilder *b, PhxGp dst, PhxGp src1, PhxGp src2) {
    PhxNode *n = emit_node(b, PHX_A64_FSUB);
    if (!n) return;

    int ftype = (dst.size == 8) ? 1 : 0;
    /* FSUB: opcode = 0011 */
    uint32_t inst = encode_fp_dp2(ftype, 0x3, hw_reg(dst),
                                  hw_reg(src1), hw_reg(src2));
    store_inst(n, inst);

    n->operands[0] = phx_op_gp(dst);
    n->operands[1] = phx_op_gp(src1);
    n->operands[2] = phx_op_gp(src2);
    n->num_operands = 3;
}

void phx_a64_fmul(PhxBuilder *b, PhxGp dst, PhxGp src1, PhxGp src2) {
    PhxNode *n = emit_node(b, PHX_A64_FMUL);
    if (!n) return;

    int ftype = (dst.size == 8) ? 1 : 0;
    /* FMUL: opcode = 0000 */
    uint32_t inst = encode_fp_dp2(ftype, 0x0, hw_reg(dst),
                                  hw_reg(src1), hw_reg(src2));
    store_inst(n, inst);

    n->operands[0] = phx_op_gp(dst);
    n->operands[1] = phx_op_gp(src1);
    n->operands[2] = phx_op_gp(src2);
    n->num_operands = 3;
}

void phx_a64_fdiv(PhxBuilder *b, PhxGp dst, PhxGp src1, PhxGp src2) {
    PhxNode *n = emit_node(b, PHX_A64_FDIV);
    if (!n) return;

    int ftype = (dst.size == 8) ? 1 : 0;
    /* FDIV: opcode = 0001 */
    uint32_t inst = encode_fp_dp2(ftype, 0x1, hw_reg(dst),
                                  hw_reg(src1), hw_reg(src2));
    store_inst(n, inst);

    n->operands[0] = phx_op_gp(dst);
    n->operands[1] = phx_op_gp(src1);
    n->operands[2] = phx_op_gp(src2);
    n->num_operands = 3;
}

/* ------------------------------------------------------------------ */
/*  Return / trap                                                      */
/* ------------------------------------------------------------------ */

void phx_a64_ret(PhxBuilder *b) {
    phx_a64_ret_reg(b, PHX_LR);
}

void phx_a64_ret_reg(PhxBuilder *b, PhxGp target) {
    PhxNode *n = emit_node(b, PHX_A64_RET);
    if (!n) return;

    /* RET {Xn}:
     * 1101 0110 0101 1111 0000 00nn nnn0 0000
     * = 0xD65F0000 | (Rn << 5) */
    uint32_t inst = 0xD65F0000u | (hw_reg(target) << 5);
    store_inst(n, inst);

    n->operands[0] = phx_op_gp(target);
    n->num_operands = 1;
}

void phx_a64_udf(PhxBuilder *b, uint16_t imm) {
    PhxNode *n = emit_node(b, PHX_A64_UDF);
    if (!n) return;

    /* UDF #imm16:
     * [31:16] 0000 0000 0000 0000
     * [15:0]  imm16
     * = 0x00000000 | imm16 */
    uint32_t inst = (uint32_t)imm;
    store_inst(n, inst);

    n->operands[0] = phx_op_imm((int64_t)imm);
    n->num_operands = 1;
}

/* ------------------------------------------------------------------ */
/*  Exclusive / atomic                                                 */
/* ------------------------------------------------------------------ */

void phx_a64_ldxr(PhxBuilder *b, PhxGp dst, PhxGp base) {
    PhxNode *n = emit_node(b, PHX_A64_LDXR);
    if (!n) return;

    /* LDXR Xt, [Xn]:
     * [31:30] size (11=64-bit, 10=32-bit)
     * [29:24] 001000
     * [23]    0 (not acquire)
     * [22]    1 (load, not store)
     * [21]    0
     * [20:16] 11111 (Rs = 31)
     * [15]    0 (not release)
     * [14:10] 11111 (Rt2 = 31)
     * [9:5]   Rn
     * [4:0]   Rt
     */
    int sf = is64(dst);
    uint32_t size_bits = sf ? 0x3u : 0x2u;
    uint32_t inst = (size_bits << 30)
                  | (0x08u << 24)    /* 001000 */
                  | (0u << 23)       /* not acquire */
                  | (1u << 22)       /* load */
                  | (0u << 21)
                  | (0x1Fu << 16)    /* Rs=31 */
                  | (0u << 15)       /* not release */
                  | (0x1Fu << 10)    /* Rt2=31 */
                  | (hw_reg(base) << 5)
                  | hw_reg(dst);
    store_inst(n, inst);

    n->operands[0] = phx_op_gp(dst);
    n->operands[1] = phx_op_gp(base);
    n->num_operands = 2;
}

void phx_a64_stxr(PhxBuilder *b, PhxGp status, PhxGp src, PhxGp base) {
    PhxNode *n = emit_node(b, PHX_A64_STXR);
    if (!n) return;

    /* STXR Ws, Xt, [Xn]:
     * [31:30] size
     * [29:24] 001000
     * [23]    0
     * [22]    0 (store)
     * [21]    0
     * [20:16] Rs (status register)
     * [15]    0
     * [14:10] 11111 (Rt2=31)
     * [9:5]   Rn
     * [4:0]   Rt
     */
    int sf = is64(src);
    uint32_t size_bits = sf ? 0x3u : 0x2u;
    uint32_t inst = (size_bits << 30)
                  | (0x08u << 24)
                  | (0u << 23)
                  | (0u << 22)       /* store */
                  | (0u << 21)
                  | (hw_reg(status) << 16)
                  | (0u << 15)
                  | (0x1Fu << 10)
                  | (hw_reg(base) << 5)
                  | hw_reg(src);
    store_inst(n, inst);

    n->operands[0] = phx_op_gp(status);
    n->operands[1] = phx_op_gp(src);
    n->operands[2] = phx_op_gp(base);
    n->num_operands = 3;
}

/* ------------------------------------------------------------------ */
/*  System                                                             */
/* ------------------------------------------------------------------ */

void phx_a64_mrs(PhxBuilder *b, PhxGp dst, uint16_t sysreg) {
    PhxNode *n = emit_node(b, PHX_A64_MRS);
    if (!n) return;

    /* MRS Xt, <sysreg>:
     * [31:20] 1101 0101 0011 (0xD53)
     * [19:5]  op0:op1:CRn:CRm:op2 (15 bits from sysreg encoding)
     * [4:0]   Rt
     *
     * The sysreg value packs: o0(1):op1(3):CRn(4):CRm(4):op2(3) = 15 bits
     * But the hardware encoding uses bit [19] for the "read" flag (1=MRS).
     * Standard system register encodings (e.g. TPIDR_EL0) already include
     * the o0 bit.  The full encoding is:
     *   0xD5300000 | (sysreg << 5) | Rt
     */
    uint32_t inst = 0xD5300000u
                  | ((uint32_t)sysreg << 5)
                  | hw_reg(dst);
    store_inst(n, inst);

    n->operands[0] = phx_op_gp(dst);
    n->operands[1] = phx_op_imm((int64_t)sysreg);
    n->num_operands = 2;
}

/* ================================================================== */
/*  FINALIZE: resolve label fixups and linearize into code buffer      */
/* ================================================================== */

/*
 * Walk all nodes to compute offsets, then resolve fixups for:
 *   - B/BL:     26-bit signed offset (imm26), word-aligned
 *   - B.cond:   19-bit signed offset (imm19), word-aligned
 *   - CBZ/CBNZ: 19-bit signed offset (imm19), word-aligned
 *   - TBNZ:     14-bit signed offset (imm14), word-aligned
 *   - ADR:      21-bit signed offset (split immhi:immlo)
 *
 * Finally, linearize all encoded bytes into the code holder's buffer.
 */
int phx_a64_finalize(PhxBuilder *b) {
    assert(b != NULL);
    PhxCodeHolder *code = b->code;
    assert(code != NULL);

    /* ---- Pass 1: compute byte offsets for all nodes ---- */
    uint32_t offset = 0;
    for (PhxNode *n = b->head; n != NULL; n = n->next) {
        n->offset = offset;

        if (n->node_type == PHX_NODE_ALIGN) {
            /* Alignment node: compute padding needed */
            int align = (int)n->operands[0].imm;
            uint32_t padding = ((offset + (uint32_t)align - 1)
                                & ~((uint32_t)align - 1)) - offset;
            /* Fill with NOP (0xD503201F) instructions for ARM64 */
            if (padding > PHX_MAX_ENCODED) {
                /* Should not happen for reasonable alignments */
                return -1;
            }
            uint32_t nop = 0xD503201Fu;
            for (uint32_t i = 0; i < padding; i += 4) {
                n->encoded[i + 0] = (uint8_t)(nop);
                n->encoded[i + 1] = (uint8_t)(nop >> 8);
                n->encoded[i + 2] = (uint8_t)(nop >> 16);
                n->encoded[i + 3] = (uint8_t)(nop >> 24);
            }
            n->encoded_size = padding;
            offset += padding;
        } else if (n->node_type == PHX_NODE_EMBED) {
            if (n->embed_data) {
                offset += n->embed_size;
            } else {
                offset += n->encoded_size;
            }
        } else if (n->node_type == PHX_NODE_LABEL) {
            /* Labels emit no bytes */
            /* offset stays the same */
        } else {
            /* Instruction node: always 4 bytes on ARM64 */
            assert(n->encoded_size == 4);
            offset += 4;
        }
    }

    /* ---- Pass 2: resolve label fixups ---- */
    for (uint32_t i = 0; i < b->fixup_count; i++) {
        PhxFixup *f = &b->fixups[i];
        PhxNode *inst_node = f->node;
        uint32_t label_id = f->label_id;

        /* Lookup the label node */
        assert(label_id < b->next_label_id);
        PhxNode *label_node = b->label_nodes[label_id];
        if (!label_node) {
            /* Unresolved label */
            return -2;
        }

        int64_t pc_offset = (int64_t)label_node->offset
                          - (int64_t)inst_node->offset;

        uint32_t inst = load_inst(inst_node);
        uint16_t opc = inst_node->opcode;

        if (opc == PHX_A64_B || opc == PHX_A64_BL) {
            /* 26-bit signed word offset */
            int64_t imm26 = pc_offset >> 2;
            if (imm26 < -(1 << 25) || imm26 >= (1 << 25)) {
                return -3; /* branch offset out of range */
            }
            inst = (inst & 0xFC000000u)
                 | ((uint32_t)imm26 & 0x03FFFFFFu);
        } else if (opc == PHX_A64_B_COND) {
            /* 19-bit signed word offset, bits [23:5] */
            int64_t imm19 = pc_offset >> 2;
            if (imm19 < -(1 << 18) || imm19 >= (1 << 18)) {
                return -4;
            }
            inst = (inst & 0xFF00001Fu)
                 | (((uint32_t)imm19 & 0x7FFFFu) << 5);
        } else if (opc == PHX_A64_CBZ || opc == PHX_A64_CBNZ) {
            /* 19-bit signed word offset, bits [23:5] */
            int64_t imm19 = pc_offset >> 2;
            if (imm19 < -(1 << 18) || imm19 >= (1 << 18)) {
                return -5;
            }
            inst = (inst & 0xFF00001Fu)
                 | (((uint32_t)imm19 & 0x7FFFFu) << 5);
        } else if (opc == PHX_A64_TBNZ) {
            /* 14-bit signed word offset, bits [18:5] */
            int64_t imm14 = pc_offset >> 2;
            if (imm14 < -(1 << 13) || imm14 >= (1 << 13)) {
                return -8; /* tbnz branch offset out of range */
            }
            inst = (inst & 0xFFF8001Fu)
                 | (((uint32_t)imm14 & 0x3FFFu) << 5);
        } else if (opc == PHX_A64_ADR) {
            /* 21-bit signed byte offset, split: immhi [23:5], immlo [30:29] */
            if (pc_offset < -(1 << 20) || pc_offset >= (1 << 20)) {
                return -6;
            }
            uint32_t uoff = (uint32_t)pc_offset & 0x1FFFFFu;
            uint32_t immlo = uoff & 0x3u;
            uint32_t immhi = (uoff >> 2) & 0x7FFFFu;
            inst = (inst & 0x9F00001Fu)
                 | (immlo << 29)
                 | (immhi << 5);
        } else {
            /* Unknown fixup type */
            return -7;
        }

        store_inst(inst_node, inst);
    }

    /* ---- Pass 3: linearize into code buffer ---- */
    /* Ensure buffer is large enough */
    size_t total_size = (size_t)offset;
    if (total_size > code->buffer_capacity) {
        size_t new_cap = code->buffer_capacity;
        while (new_cap < total_size) {
            new_cap *= 2;
        }
        uint8_t *new_buf = (uint8_t *)realloc(code->buffer, new_cap);
        if (!new_buf) {
            return -8;
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
