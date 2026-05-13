/* Phase 5.B c22a: JitLirOperandDesc layout verifier + round-trip receiver. */
#include "cinderx/Jit/lir/lir_c_api.h"
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace jit::lir {

static_assert(sizeof(JitLirOperandDescKind) == sizeof(int),
    "enum size != int");
static_assert(sizeof(JitLirInstr) == sizeof(void *),
    "JitLirInstr != void*");
static_assert(sizeof(int64_t) == 8, "int64_t != 8");
static_assert(sizeof(uint64_t) == 8, "uint64_t != 8");
static_assert(sizeof(JitLirOperandDesc) == 24,
    "JitLirOperandDesc != 24");
static_assert(offsetof(JitLirOperandDesc, kind) == 0, "kind offset");
static_assert(offsetof(JitLirOperandDesc, data) == 8, "data offset");

}

extern "C" int
lir_invoke_bridge_roundtrip_check(int n_args,
                                   const JitLirOperandDesc *args,
                                   int *failed_index) {
    for (int i = 0; i < n_args; i++) {
        const JitLirOperandDesc *d = &args[i];
        switch (d->kind) {
        case JIT_LIR_OPDESC_INSTR:
            if (d->data.instr == NULL) { *failed_index = i; return 1; }
            break;
        case JIT_LIR_OPDESC_IMM_INT:
            if (d->data.imm_int.width_bits != 8 &&
                d->data.imm_int.width_bits != 16 &&
                d->data.imm_int.width_bits != 32 &&
                d->data.imm_int.width_bits != 64) {
                *failed_index = i; return 2;
            }
            break;
        case JIT_LIR_OPDESC_IMM_BOOL:
            if (d->data.imm_bool != 0 && d->data.imm_bool != 1) {
                *failed_index = i; return 3;
            }
            break;
        case JIT_LIR_OPDESC_REG_REF:
            break;
        case JIT_LIR_OPDESC_MEM_IMM:
            if (d->data.mem_imm.width_bits <= 0) {
                *failed_index = i; return 4;
            }
            break;
        default:
            *failed_index = i; return 5;
        }
    }
    return 0;
}
