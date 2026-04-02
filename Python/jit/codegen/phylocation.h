/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * C-compatible PhyLocation type — physical register or stack slot.
 * Replaces the C++ struct PhyLocation for Phase 3D.
 */
#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PHYLOC_REG_INVALID (-1)

/* Architecture-specific register counts and base offsets. */
#if defined(__x86_64__) || defined(_M_X64)

#define PHYLOC_VECD_REG_BASE 16
#define PHYLOC_NUM_GP_REGS   16
#define PHYLOC_NUM_VECD_REGS 16
#define PHYLOC_NUM_REGS      32

/* GP register IDs (matching RegId enum order) */
#define PHYLOC_RAX  0
#define PHYLOC_RCX  1
#define PHYLOC_RDX  2
#define PHYLOC_RBX  3
#define PHYLOC_RSP  4
#define PHYLOC_RBP  5
#define PHYLOC_RSI  6
#define PHYLOC_RDI  7
#define PHYLOC_R8   8
#define PHYLOC_R9   9
#define PHYLOC_R10  10
#define PHYLOC_R11  11
#define PHYLOC_R12  12
#define PHYLOC_R13  13
#define PHYLOC_R14  14
#define PHYLOC_R15  15

/* VECD register IDs */
#define PHYLOC_XMM0  16
#define PHYLOC_XMM1  17
#define PHYLOC_XMM2  18
#define PHYLOC_XMM3  19
#define PHYLOC_XMM4  20
#define PHYLOC_XMM5  21
#define PHYLOC_XMM6  22
#define PHYLOC_XMM7  23
#define PHYLOC_XMM8  24
#define PHYLOC_XMM9  25
#define PHYLOC_XMM10 26
#define PHYLOC_XMM11 27
#define PHYLOC_XMM12 28
#define PHYLOC_XMM13 29
#define PHYLOC_XMM14 30
#define PHYLOC_XMM15 31

/* Memory slot frame pointer name for toString */
#define PHYLOC_FP_NAME "RBP"

#elif defined(__aarch64__)

#define PHYLOC_VECD_REG_BASE 32
#define PHYLOC_NUM_GP_REGS   32
#define PHYLOC_NUM_VECD_REGS 32
#define PHYLOC_NUM_REGS      64

/* GP register IDs (matching RegId enum order) */
#define PHYLOC_X0   0
#define PHYLOC_X1   1
#define PHYLOC_X2   2
#define PHYLOC_X3   3
#define PHYLOC_X4   4
#define PHYLOC_X5   5
#define PHYLOC_X6   6
#define PHYLOC_X7   7
#define PHYLOC_X8   8
#define PHYLOC_X9   9
#define PHYLOC_X10  10
#define PHYLOC_X11  11
#define PHYLOC_X12  12
#define PHYLOC_X13  13
#define PHYLOC_X14  14
#define PHYLOC_X15  15
#define PHYLOC_X16  16
#define PHYLOC_X17  17
#define PHYLOC_X18  18
#define PHYLOC_X19  19
#define PHYLOC_X20  20
#define PHYLOC_X21  21
#define PHYLOC_X22  22
#define PHYLOC_X23  23
#define PHYLOC_X24  24
#define PHYLOC_X25  25
#define PHYLOC_X26  26
#define PHYLOC_X27  27
#define PHYLOC_X28  28
#define PHYLOC_X29  29
#define PHYLOC_X30  30
#define PHYLOC_XZR  31

/* VECD register IDs */
#define PHYLOC_D0   32
#define PHYLOC_D1   33
#define PHYLOC_D2   34
#define PHYLOC_D3   35
#define PHYLOC_D4   36
#define PHYLOC_D5   37
#define PHYLOC_D6   38
#define PHYLOC_D7   39
#define PHYLOC_D8   40
#define PHYLOC_D9   41
#define PHYLOC_D10  42
#define PHYLOC_D11  43
#define PHYLOC_D12  44
#define PHYLOC_D13  45
#define PHYLOC_D14  46
#define PHYLOC_D15  47
#define PHYLOC_D16  48
#define PHYLOC_D17  49
#define PHYLOC_D18  50
#define PHYLOC_D19  51
#define PHYLOC_D20  52
#define PHYLOC_D21  53
#define PHYLOC_D22  54
#define PHYLOC_D23  55
#define PHYLOC_D24  56
#define PHYLOC_D25  57
#define PHYLOC_D26  58
#define PHYLOC_D27  59
#define PHYLOC_D28  60
#define PHYLOC_D29  61
#define PHYLOC_D30  62
#define PHYLOC_D31  63

/* SP is a special register ID on aarch64 (not in the normal range) */
#define PHYLOC_SP   0xFFFF

/* Memory slot frame pointer name for toString */
#define PHYLOC_FP_NAME "X29"

#else
/* Unknown architecture fallback */
#define PHYLOC_VECD_REG_BASE 4
#define PHYLOC_NUM_GP_REGS   4
#define PHYLOC_NUM_VECD_REGS 4
#define PHYLOC_NUM_REGS      8

#define PHYLOC_R0   0
#define PHYLOC_R1   1
#define PHYLOC_R2   2
#define PHYLOC_R3   3
#define PHYLOC_D0   4
#define PHYLOC_D1   5
#define PHYLOC_D2   6
#define PHYLOC_D3   7

#define PHYLOC_FP_NAME "FP"

#endif /* architecture */

/* ---------------------------------------------------------------
 * PhyLoc struct — layout-compatible with C++ PhyLocation
 * (int32_t loc + uint32_t bitSize = 8 bytes)
 * --------------------------------------------------------------- */
typedef struct {
    int32_t  loc;       /* register ID or stack slot offset (negative) */
    uint32_t bit_size;  /* size in bits: 8, 16, 32, 64, 128 */
} PhyLoc;

/* Initializer macros */
#define PHYLOC_INIT        { PHYLOC_REG_INVALID, 64 }
#define PHYLOC(reg, size)  ((PhyLoc){ (reg), (size) })
#define PHYLOC64(reg)      ((PhyLoc){ (reg), 64 })
#define PHYLOC32(reg)      ((PhyLoc){ (reg), 32 })
#define PHYLOC16(reg)      ((PhyLoc){ (reg), 16 })
#define PHYLOC8(reg)       ((PhyLoc){ (reg), 8 })
#define PHYLOC128(reg)     ((PhyLoc){ (reg), 128 })

/* ---------------------------------------------------------------
 * Inline query functions
 * --------------------------------------------------------------- */
static inline bool phyloc_is_memory(PhyLoc p) {
    return p.loc < 0;
}

static inline bool phyloc_is_register(PhyLoc p) {
    return p.loc >= 0;
}

static inline bool phyloc_is_gp(PhyLoc p) {
    return p.loc >= 0 && p.loc < PHYLOC_VECD_REG_BASE;
}

static inline bool phyloc_is_fp(PhyLoc p) {
    return p.loc >= 0 && p.loc >= PHYLOC_VECD_REG_BASE;
}

/* Equality compares loc only (intentional — matches C++ operator==) */
static inline bool phyloc_eq(PhyLoc a, PhyLoc b) {
    return a.loc == b.loc;
}

/* ---------------------------------------------------------------
 * Register name lookup and formatting
 * --------------------------------------------------------------- */

/* Write the register/slot name to buf. Returns number of chars written
 * (excluding NUL), or negative on error. */
int phyloc_to_string(PhyLoc p, char *buf, size_t len);

/* Parse a register name. Returns PhyLoc with loc=PHYLOC_REG_INVALID on
 * failure. Does not support parsing stack slots. */
PhyLoc phyloc_parse(const char *name);

#ifdef __cplusplus
} /* extern "C" */
#endif
