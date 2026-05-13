/* Phase 5.B c22a: JitLirOperandDesc layout verifier + round-trip echo.
 * Amendments per gatekeeper 13:01:53Z BLOCK: 5a per-variant sizeof,
 * 5b nested offsetof, 6a value-preservation echo. */
#include "cinderx/Jit/lir/lir_c_api.h"
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace jit::lir {

static_assert(sizeof(JitLirOperandDescKind) == sizeof(int), "enum size");
static_assert(sizeof(JitLirInstr) == sizeof(void *), "JitLirInstr size");
static_assert(sizeof(int64_t) == 8, "int64_t");
static_assert(sizeof(uint64_t) == 8, "uint64_t");
static_assert(sizeof(JitLirOperandDesc) == 24, "OperandDesc != 24");

namespace {
using ImmIntT = decltype(JitLirOperandDesc{}.data.imm_int);
using MemImmT = decltype(JitLirOperandDesc{}.data.mem_imm);
static_assert(sizeof(ImmIntT) == 16, "imm_int variant != 16");
static_assert(sizeof(MemImmT) == 16, "mem_imm variant != 16");
}

static_assert(offsetof(JitLirOperandDesc, kind) == 0, "kind offset");
static_assert(offsetof(JitLirOperandDesc, data) == 8, "data offset");
static_assert(offsetof(JitLirOperandDesc, data.imm_int.value) == 8,
    "imm_int.value sub-offset");
static_assert(offsetof(JitLirOperandDesc, data.imm_int.width_bits) == 16,
    "imm_int.width_bits sub-offset");
static_assert(offsetof(JitLirOperandDesc, data.mem_imm.addr) == 8,
    "mem_imm.addr sub-offset");
static_assert(offsetof(JitLirOperandDesc, data.mem_imm.width_bits) == 16,
    "mem_imm.width_bits sub-offset");

}

extern "C" int
lir_invoke_bridge_roundtrip_echo(int n_args,
                                  const JitLirOperandDesc *args,
                                  JitLirOperandDesc *echoed_back,
                                  int *failed_index) {
    for (int i = 0; i < n_args; i++) {
        const JitLirOperandDesc *in = &args[i];
        JitLirOperandDesc *out = &echoed_back[i];
        out->kind = in->kind;
        switch (in->kind) {
        case JIT_LIR_OPDESC_INSTR:
            out->data.instr = in->data.instr;
            break;
        case JIT_LIR_OPDESC_IMM_INT:
            out->data.imm_int.value = in->data.imm_int.value;
            out->data.imm_int.width_bits = in->data.imm_int.width_bits;
            break;
        case JIT_LIR_OPDESC_IMM_BOOL:
            out->data.imm_bool = in->data.imm_bool;
            break;
        case JIT_LIR_OPDESC_REG_REF:
            out->data.reg = in->data.reg;
            break;
        case JIT_LIR_OPDESC_MEM_IMM:
            out->data.mem_imm.addr = in->data.mem_imm.addr;
            out->data.mem_imm.width_bits = in->data.mem_imm.width_bits;
            break;
        default:
            *failed_index = i;
            return 1;
        }
    }
    return 0;
}
