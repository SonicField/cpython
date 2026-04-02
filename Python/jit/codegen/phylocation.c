/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * C implementation of PhyLoc register name lookup and parsing.
 */

#include "cinderx/Jit/codegen/phylocation.h"

#include <string.h>
#include <stdio.h>

/* ---------------------------------------------------------------
 * x86_64 register name tables
 * --------------------------------------------------------------- */
#if defined(__x86_64__) || defined(_M_X64)

static const char *gp_names_64[] = {
    "RAX", "RCX", "RDX", "RBX", "RSP", "RBP", "RSI", "RDI",
    "R8",  "R9",  "R10", "R11", "R12", "R13", "R14", "R15",
};

static const char *gp_names_32[] = {
    "EAX", "ECX", "EDX", "EBX", "ESP", "EBP", "ESI", "EDI",
    "R8D", "R9D", "R10D","R11D","R12D","R13D","R14D","R15D",
};

static const char *gp_names_16[] = {
    "AX",  "CX",  "DX",  "BX",  "SP",  "BP",  "SI",  "DI",
    "R8W", "R9W", "R10W","R11W","R12W","R13W","R14W","R15W",
};

static const char *gp_names_8[] = {
    "AL",  "CL",  "DL",  "BL",  "SPL", "BPL", "SIL", "DIL",
    "R8B", "R9B", "R10B","R11B","R12B","R13B","R14B","R15B",
};

static const char *vecd_names[] = {
    "XMM0", "XMM1", "XMM2",  "XMM3",  "XMM4",  "XMM5",  "XMM6",  "XMM7",
    "XMM8", "XMM9", "XMM10", "XMM11", "XMM12", "XMM13", "XMM14", "XMM15",
};

int phyloc_to_string(PhyLoc p, char *buf, size_t len) {
    if (phyloc_is_memory(p)) {
        return snprintf(buf, len, "[RBP(%d)]", p.loc);
    }
    const char *n = NULL;
    if (p.loc < PHYLOC_VECD_REG_BASE) {
        /* GP register */
        if (p.bit_size == 32) {
            n = gp_names_32[p.loc];
        } else if (p.bit_size == 16) {
            n = gp_names_16[p.loc];
        } else if (p.bit_size == 8) {
            n = gp_names_8[p.loc];
        } else {
            n = gp_names_64[p.loc];
        }
    } else {
        /* VECD register */
        n = vecd_names[p.loc - PHYLOC_VECD_REG_BASE];
    }
    return snprintf(buf, len, "%s", n);
}

PhyLoc phyloc_parse(const char *name) {
    /* Try GP registers — all 4 size variants */
    for (int i = 0; i < PHYLOC_NUM_GP_REGS; i++) {
        if (strcmp(name, gp_names_64[i]) == 0)
            return PHYLOC(i, 64);
        if (strcmp(name, gp_names_32[i]) == 0)
            return PHYLOC(i, 32);
        if (strcmp(name, gp_names_16[i]) == 0)
            return PHYLOC(i, 16);
        if (strcmp(name, gp_names_8[i]) == 0)
            return PHYLOC(i, 8);
    }
    /* Try VECD registers */
    for (int i = 0; i < PHYLOC_NUM_VECD_REGS; i++) {
        if (strcmp(name, vecd_names[i]) == 0)
            return PHYLOC(PHYLOC_VECD_REG_BASE + i, 128);
    }
    /* Not found */
    PhyLoc invalid = PHYLOC_INIT;
    return invalid;
}

/* ---------------------------------------------------------------
 * aarch64 register name tables
 * --------------------------------------------------------------- */
#elif defined(__aarch64__)

static const char *gp_names_64[] = {
    "X0",  "X1",  "X2",  "X3",  "X4",  "X5",  "X6",  "X7",
    "X8",  "X9",  "X10", "X11", "X12", "X13", "X14", "X15",
    "X16", "X17", "X18", "X19", "X20", "X21", "X22", "X23",
    "X24", "X25", "X26", "X27", "X28", "X29", "X30", "XZR",
};

static const char *gp_names_32[] = {
    "W0",  "W1",  "W2",  "W3",  "W4",  "W5",  "W6",  "W7",
    "W8",  "W9",  "W10", "W11", "W12", "W13", "W14", "W15",
    "W16", "W17", "W18", "W19", "W20", "W21", "W22", "W23",
    "W24", "W25", "W26", "W27", "W28", "W29", "W30", "WZR",
};

static const char *vecd_names[] = {
    "D0",  "D1",  "D2",  "D3",  "D4",  "D5",  "D6",  "D7",
    "D8",  "D9",  "D10", "D11", "D12", "D13", "D14", "D15",
    "D16", "D17", "D18", "D19", "D20", "D21", "D22", "D23",
    "D24", "D25", "D26", "D27", "D28", "D29", "D30", "D31",
};

int phyloc_to_string(PhyLoc p, char *buf, size_t len) {
    if (phyloc_is_memory(p)) {
        return snprintf(buf, len, "[X29(%d)]", p.loc);
    }
    const char *n = NULL;
    if (p.loc < PHYLOC_VECD_REG_BASE) {
        /* GP register */
        if (p.bit_size == 32 || p.bit_size == 16 || p.bit_size == 8) {
            n = gp_names_32[p.loc];
        } else {
            n = gp_names_64[p.loc];
        }
    } else {
        /* VECD register */
        n = vecd_names[p.loc - PHYLOC_VECD_REG_BASE];
    }
    if (p.loc == PHYLOC_SP) {
        n = "SP";
    }
    return snprintf(buf, len, "%s", n);
}

PhyLoc phyloc_parse(const char *name) {
    /* Try GP registers — 64 and 32 bit variants */
    for (int i = 0; i < PHYLOC_NUM_GP_REGS; i++) {
        if (strcmp(name, gp_names_64[i]) == 0)
            return PHYLOC(i, 64);
        if (strcmp(name, gp_names_32[i]) == 0)
            return PHYLOC(i, 32);
    }
    /* Try VECD registers */
    for (int i = 0; i < PHYLOC_NUM_VECD_REGS; i++) {
        if (strcmp(name, vecd_names[i]) == 0)
            return PHYLOC(PHYLOC_VECD_REG_BASE + i, 64);
    }
    /* Special: SP */
    if (strcmp(name, "SP") == 0) {
        return PHYLOC(PHYLOC_SP, 64);
    }
    /* Not found */
    PhyLoc invalid = PHYLOC_INIT;
    return invalid;
}

/* ---------------------------------------------------------------
 * Unknown architecture fallback
 * --------------------------------------------------------------- */
#else

static const char *gp_names[] = { "R0", "R1", "R2", "R3" };
static const char *vecd_names[] = { "D0", "D1", "D2", "D3" };

int phyloc_to_string(PhyLoc p, char *buf, size_t len) {
    if (phyloc_is_memory(p)) {
        return snprintf(buf, len, "[FP(%d)]", p.loc);
    }
    const char *n = NULL;
    if (p.loc < PHYLOC_VECD_REG_BASE) {
        n = gp_names[p.loc];
    } else {
        n = vecd_names[p.loc - PHYLOC_VECD_REG_BASE];
    }
    return snprintf(buf, len, "%s", n);
}

PhyLoc phyloc_parse(const char *name) {
    for (int i = 0; i < PHYLOC_NUM_GP_REGS; i++) {
        if (strcmp(name, gp_names[i]) == 0)
            return PHYLOC(i, 64);
    }
    for (int i = 0; i < PHYLOC_NUM_VECD_REGS; i++) {
        if (strcmp(name, vecd_names[i]) == 0)
            return PHYLOC(PHYLOC_VECD_REG_BASE + i, 64);
    }
    if (strcmp(name, "SP") == 0) {
        return PHYLOC(0xFFFF, 64);
    }
    PhyLoc invalid = PHYLOC_INIT;
    return invalid;
}

#endif /* architecture */
