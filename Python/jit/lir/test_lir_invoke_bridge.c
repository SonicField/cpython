/* c22a round-trip ABI test (echo + memcmp). */
#include <stdio.h>
#include <string.h>
#include "Python.h"
#include "cinderx/Jit/lir/lir_c_api.h"

extern int lir_invoke_bridge_roundtrip_echo(int n,
    const JitLirOperandDesc *a, JitLirOperandDesc *e, int *fi);

#define RT(args, n) ({ \
    JitLirOperandDesc _e[8] = {0}; int _fi = -1; \
    int _rc = lir_invoke_bridge_roundtrip_echo((n), (args), _e, &_fi); \
    (_rc == 0 && memcmp((args), _e, (size_t)(n) * sizeof(JitLirOperandDesc)) == 0); \
})

static int test_all(void) {
    int x = 42, r = 7;
    JitLirOperandDesc a[5] = {0};
    a[0].kind = JIT_LIR_OPDESC_INSTR;     a[0].data.instr = &x;
    a[1].kind = JIT_LIR_OPDESC_IMM_INT;   a[1].data.imm_int.value = 0x123456789ABCDEFLL; a[1].data.imm_int.width_bits = 64;
    a[2].kind = JIT_LIR_OPDESC_IMM_BOOL;  a[2].data.imm_bool = 1;
    a[3].kind = JIT_LIR_OPDESC_REG_REF;   a[3].data.reg = &r;
    a[4].kind = JIT_LIR_OPDESC_MEM_IMM;   a[4].data.mem_imm.addr = 0xDEADBEEFCAFEBABELL; a[4].data.mem_imm.width_bits = 64;

    if (!RT(&a[0], 1)) { printf("FAIL instr\n"); return 1; }
    if (!RT(&a[1], 1)) { printf("FAIL imm_int\n"); return 1; }
    if (!RT(&a[2], 1)) { printf("FAIL imm_bool\n"); return 1; }
    if (!RT(&a[3], 1)) { printf("FAIL reg_ref\n"); return 1; }
    if (!RT(&a[4], 1)) { printf("FAIL mem_imm\n"); return 1; }
    if (!RT(a, 5))     { printf("FAIL all_kinds\n"); return 1; }

    JitLirOperandDesc e = {0}; int fi = -1;
    if (lir_invoke_bridge_roundtrip_echo(0, NULL, &e, &fi) != 0) { printf("FAIL empty\n"); return 1; }

    printf("PASS all 7 tests\n");
    return 0;
}

int main(void) { return test_all(); }
