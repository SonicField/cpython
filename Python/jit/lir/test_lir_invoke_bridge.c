/* test_lir_invoke_bridge.c - c22a round-trip ABI test
 * Pass-(b) item 6: C-to-C++ kind+data preservation.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "Python.h"
#include "cinderx/Jit/lir/lir_c_api.h"

extern int lir_invoke_bridge_roundtrip_check(int n_args,
    const JitLirOperandDesc *args, int *failed_index);

static int g_pass = 0, g_fail = 0;
#define ASSERT(c, m) do { if(!(c)){ printf("FAIL %s: %s\n", __func__, m); g_fail++; return; } } while(0)
#define PASS() do { printf("PASS %s\n", __func__); g_pass++; } while(0)

static void test_empty(void) {
    int idx = -1;
    ASSERT(lir_invoke_bridge_roundtrip_check(0, NULL, &idx) == 0, "empty");
    PASS();
}

static void test_instr(void) {
    int x = 42;
    JitLirOperandDesc a = { .kind = JIT_LIR_OPDESC_INSTR };
    a.data.instr = &x;
    int idx = -1;
    ASSERT(lir_invoke_bridge_roundtrip_check(1, &a, &idx) == 0, "instr");
    PASS();
}

static void test_imm_int(void) {
    JitLirOperandDesc a = { .kind = JIT_LIR_OPDESC_IMM_INT };
    a.data.imm_int.value = 0xCAFE;
    a.data.imm_int.width_bits = 64;
    int idx = -1;
    ASSERT(lir_invoke_bridge_roundtrip_check(1, &a, &idx) == 0, "imm_int");
    PASS();
}

static void test_imm_bool(void) {
    JitLirOperandDesc a = { .kind = JIT_LIR_OPDESC_IMM_BOOL };
    a.data.imm_bool = 1;
    int idx = -1;
    ASSERT(lir_invoke_bridge_roundtrip_check(1, &a, &idx) == 0, "imm_bool");
    PASS();
}

static void test_reg_ref(void) {
    int r = 7;
    JitLirOperandDesc a = { .kind = JIT_LIR_OPDESC_REG_REF };
    a.data.reg = &r;
    int idx = -1;
    ASSERT(lir_invoke_bridge_roundtrip_check(1, &a, &idx) == 0, "reg_ref");
    PASS();
}

static void test_mem_imm(void) {
    JitLirOperandDesc a = { .kind = JIT_LIR_OPDESC_MEM_IMM };
    a.data.mem_imm.addr = 0x1000;
    a.data.mem_imm.width_bits = 64;
    int idx = -1;
    ASSERT(lir_invoke_bridge_roundtrip_check(1, &a, &idx) == 0, "mem_imm");
    PASS();
}

static void test_invalid_kind(void) {
    JitLirOperandDesc a = { .kind = (JitLirOperandDescKind)999 };
    int idx = -1;
    ASSERT(lir_invoke_bridge_roundtrip_check(1, &a, &idx) == 5, "invalid kind detected");
    ASSERT(idx == 0, "failed_index = 0");
    PASS();
}

int main(void) {
    test_empty();
    test_instr();
    test_imm_int();
    test_imm_bool();
    test_reg_ref();
    test_mem_imm();
    test_invalid_kind();
    printf("\n%d pass, %d fail\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
