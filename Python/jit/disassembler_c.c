/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * C implementation of the JIT disassembler — Phase 3D replacement for
 * disassembler.cpp. Uses capstone C API directly, outputs to FILE*.
 */

#include "cinderx/Jit/disassembler_c.h"
#include "cinderx/Jit/jit_config_c.h"

#include <inttypes.h>
#include <string.h>

void
jit_disasm_init(JitDisassembler *d, const char *buf, size_t size) {
    d->buf = buf;
    d->start = 0;
    d->size = size;
    d->addr_len = 16;
    d->print_addr = 1;
    d->print_instr_bytes = 1;
}

void jit_disasm_set_print_addr(JitDisassembler *d, int print) {
    d->print_addr = print;
}

void jit_disasm_set_print_instr_bytes(JitDisassembler *d, int print) {
    d->print_instr_bytes = print;
}

const char *jit_disasm_cursor(const JitDisassembler *d) {
    if (!d->buf) return NULL;
    return d->buf + d->start;
}

#ifndef ENABLE_DISASSEMBLER

void jit_disasm_one(JitDisassembler *d, FILE *out) { (void)d; (void)out; }
void jit_disasm_all(JitDisassembler *d, FILE *out) { (void)d; (void)out; }

#else /* ENABLE_DISASSEMBLER */

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wduplicate-enum"
#include "capstone/capstone.h"
#pragma GCC diagnostic pop

#if defined(__aarch64__)
#include "capstone/arm64.h"
#endif

static csh
open_capstone(void) {
    cs_arch arch;
    cs_mode mode;
    csh handle;

#if defined(__x86_64__)
    arch = CS_ARCH_X86;
    mode = CS_MODE_64;
#elif defined(__aarch64__)
    arch = CS_ARCH_ARM64;
    mode = CS_MODE_ARM;
#else
    return 0;
#endif

    cs_err err = cs_open(arch, mode, &handle);
    if (err != CS_ERR_OK) return 0;
    return handle;
}

static void
disasm_one_internal(JitDisassembler *d, FILE *out, csh handle) {
    const char *cursor = d->buf + d->start;

    if (d->print_addr) {
        fprintf(out, "0x%0*" PRIxPTR ":%8s",
                (int)d->addr_len,
                (uintptr_t)cursor, "");
    }

    cs_option(handle, CS_OPT_DETAIL, CS_OPT_ON);

#if defined(__x86_64__)
    {
        const JitConfig *cfg = jit_get_config();
        int syntax = (cfg && cfg->asm_syntax == JIT_ASM_INTEL)
            ? CS_OPT_SYNTAX_INTEL : CS_OPT_SYNTAX_ATT;
        cs_option(handle, CS_OPT_SYNTAX, syntax);
    }
#endif

    const uint8_t *code = (const uint8_t *)cursor;
    size_t size = d->size - d->start;
    uint64_t address = (uint64_t)(uintptr_t)code;

    cs_insn *insn = cs_malloc(handle);
    if (!insn) return;

    if (!cs_disasm_iter(handle, &code, &size, &address, insn)) {
        fprintf(out, "RAW: 0x%02x",
                (unsigned char)d->buf[d->start++]);
        cs_free(insn, 1);
        return;
    }

    fprintf(out, "%s %s", insn->mnemonic, insn->op_str);

    /* Try to symbolize call/branch targets */
    const void *symbol = NULL;

#if defined(__x86_64__)
    {
        uint8_t *opcode = insn->detail->x86.opcode;
        if (opcode[0] == 0xe8 || opcode[0] == 0x9a || opcode[0] == 0xff) {
            cs_x86_op *opnd = &insn->detail->x86.operands[0];
            if (opnd->type == X86_OP_IMM) {
                symbol = (const void *)(uintptr_t)opnd->imm;
            } else if (opnd->type == X86_OP_MEM &&
                       opnd->mem.base == X86_REG_RIP) {
                symbol = (const void *)(uintptr_t)(address + opnd->mem.disp);
            }
        }
    }
#elif defined(__aarch64__)
    if (insn->id == ARM64_INS_BR || insn->id == ARM64_INS_BLR) {
        arm64_reg reg = insn->detail->arm64.operands[0].reg;
        uintptr_t symbol_value = 0;
        int matched = 0;

        for (size_t backward = 4; !matched && backward <= d->start;
             backward += 4) {
            const uint8_t *mcode =
                (const uint8_t *)(d->buf + d->start - backward);
            size_t msize = d->size - (d->start - backward);
            uint64_t maddress = (uint64_t)(uintptr_t)mcode;

            cs_insn *minsn = cs_malloc(handle);
            if (!minsn) { symbol_value = 0; break; }

            if (!cs_disasm_iter(handle, &mcode, &msize, &maddress, minsn)) {
                cs_free(minsn, 1);
                symbol_value = 0;
                break;
            }

            cs_arm64_op *mopnds = minsn->detail->arm64.operands;

            switch (minsn->id) {
            case ARM64_INS_MOV:
            case ARM64_INS_MOVZ:
                if (mopnds[0].type == ARM64_OP_REG &&
                    mopnds[0].reg == reg &&
                    mopnds[1].type == ARM64_OP_IMM) {
                    symbol_value |= mopnds[1].imm;
                    matched = 1;
                }
                break;
            case ARM64_INS_MOVK:
                if (mopnds[0].type == ARM64_OP_REG &&
                    mopnds[0].reg == reg &&
                    mopnds[1].type == ARM64_OP_IMM) {
                    symbol_value |= ((uintptr_t)mopnds[1].imm
                                     << mopnds[1].shift.value);
                }
                break;
            default:
                symbol_value = 0;
                break;
            }

            cs_free(minsn, 1);
            if (!symbol_value) break;
        }

        if (matched) {
            symbol = (const void *)symbol_value;
        }
    }
#endif

    if (symbol) {
        char sym_name[256];
        if (jit_symbolize(symbol, sym_name, sizeof(sym_name))) {
            fprintf(out, "\t(%s)", sym_name);
        }
    }

    if (d->print_instr_bytes) {
        size_t i;
        for (i = d->start; i < d->start + 8; i++) {
            if (i < d->start + insn->size) {
                fprintf(out, "%02x ",
                        (unsigned char)d->buf[i]);
            } else {
                fprintf(out, "   ");
            }
        }
    }

    d->start += insn->size;
    cs_free(insn, 1);
}

void
jit_disasm_one(JitDisassembler *d, FILE *out) {
    csh handle = open_capstone();
    if (!handle) return;
    disasm_one_internal(d, out, handle);
    cs_close(&handle);
}

void
jit_disasm_all(JitDisassembler *d, FILE *out) {
    csh handle = open_capstone();
    if (!handle) return;
    while (d->start < d->size) {
        disasm_one_internal(d, out, handle);
        fprintf(out, "\n");
    }
    cs_close(&handle);
}

#endif /* ENABLE_DISASSEMBLER */
