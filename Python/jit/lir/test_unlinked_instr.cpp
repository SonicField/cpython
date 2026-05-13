/* test_unlinked_instr.cpp - c22b-api tests for Block::allocateInstrUnlinked.
 * API-1 + API-3 + API-6 (with -DJIT_TEST_COUNTER per supervisor 14:49:03Z iii).
 * Standalone manual compile (with counter):
 *   c++ -DJIT_TEST_COUNTER -I. -I../../../Include test_unlinked_instr.cpp
 */
#include <cstdio>
#include "Python.h"
#include "cinderx/Jit/lir/function.h"
#include "cinderx/Jit/lir/block.h"
#include "cinderx/Jit/lir/instruction.h"
#include "cinderx/Jit/lir/operand.h"

using jit::lir::Function;
using jit::lir::BasicBlock;
using jit::lir::Instruction;
using jit::lir::Imm;
using jit::lir::OperandBase;

#ifdef JIT_TEST_COUNTER
extern "C" int g_jit_test_operand_free_count;
#endif

static int gp = 0, gf = 0;
#define A(c, m) do { if(!(c)){ printf("FAIL %s: %s\n", __func__, m); gf++; return; } } while(0)
#define P() do { printf("PASS %s\n", __func__); gp++; } while(0)

static void t_no_bb_pollution(void) {
    Function f; BasicBlock* bb = f.allocateBasicBlock();
    auto head = bb->getFirstInstr();
    size_t n = bb->getNumInstrs();
    Instruction* s = bb->allocateInstrUnlinked(Instruction::kCall, nullptr);
    A(s != nullptr, "null"); A(bb->getFirstInstr()==head, "head"); A(bb->getNumInstrs()==n, "size");
    delete s;
    A(bb->getFirstInstr()==head, "head post"); A(bb->getNumInstrs()==n, "size post");
    P();
}
static void t_alloc_delete(void) {
    Function f; BasicBlock* bb = f.allocateBasicBlock();
    Instruction* s = bb->allocateInstrUnlinked(Instruction::kMove, nullptr);
    A(s != nullptr, "null"); A(s->basicblock()==bb, "parent");
    delete s; P();
}
static void t_operand_cleanup(void) {
    Function f; BasicBlock* bb = f.allocateBasicBlock();
#ifdef JIT_TEST_COUNTER
    int pre = g_jit_test_operand_free_count;
#endif
    const int N = 10;
    for (int i = 0; i < N; i++) {
        Instruction* s = bb->allocateInstrUnlinked(
            Instruction::kMove, nullptr,
            Imm{(uint64_t)(i+1), OperandBase::k64bit},
            Imm{(uint64_t)(i+100), OperandBase::k32bit});
        A(s != nullptr, "alloc null");
        A(s->getNumInputs() == 2, "n_inputs != 2");
        delete s;
    }
    A(bb->getNumInstrs() == 0, "BB polluted");
#ifdef JIT_TEST_COUNTER
    int delta = g_jit_test_operand_free_count - pre;
    if (delta != 2 * N) {
        printf("FAIL %s: counter delta %d != %d (leak)\n", __func__, delta, 2*N);
        gf++; return;
    }
#endif
    P();
}

#ifdef JIT_TEST_COUNTER
/* c22b-mech (supervisor 15:29:12Z + shepard 15:28:27Z falsifier flag):
 * deliberate-leak negative test for JIT_TEST_COUNTER counter-mode gate.
 *
 * Closes pythia #379 #3: without a deliberately-broken commit driving
 * counter-mode to FAIL, JIT_TEST_COUNTER is "static-assert dressed as
 * runtime detection". This test simulates a real operand cleanup-skip
 * by leaking one instruction (NOT calling delete), then asserts the
 * counter-delta ASSERTION FIRES (delta != 2*N) — proving the gate
 * actually catches a leak, not just the absence of one.
 *
 * Expected behavior: leak path is hit, delta-check returns FAIL inside
 * the simulated leak; outer test counts that fail-path firing as PASS
 * because the gate detected the leak. */
static void t_negative_leak_detection(void) {
    Function f; BasicBlock* bb = f.allocateBasicBlock();
    int pre = g_jit_test_operand_free_count;
    const int N = 10;
    for (int i = 0; i < N; i++) {
        Instruction* s = bb->allocateInstrUnlinked(
            Instruction::kMove, nullptr,
            Imm{(uint64_t)(i+1), OperandBase::k64bit},
            Imm{(uint64_t)(i+100), OperandBase::k32bit});
        if (i < N - 1) {
            delete s;        /* clean up first N-1 */
        }
        /* Last iteration: deliberately LEAK s (no delete) — simulates
         * missed-free defect that the counter-mode gate must detect. */
    }
    int delta = g_jit_test_operand_free_count - pre;
    /* Real (clean) cycle would give delta == 2*N. The leak skips 2 frees
     * (one for each operand of the leaked instr), so delta == 2*N - 2.
     * Negative test PASSES if the assertion catches the discrepancy. */
    if (delta == 2 * N) {
        printf("FAIL %s: counter delta %d == %d — gate missed the leak\n",
               __func__, delta, 2*N);
        gf++; return;
    }
    if (delta != 2 * (N - 1)) {
        printf("FAIL %s: unexpected delta %d (expected %d after 1 leaked instr)\n",
               __func__, delta, 2*(N-1));
        gf++; return;
    }
    /* Gate detected the leak (delta < 2*N) AND identified expected size
     * (delta == 2*(N-1)). Negative-test asserts the counter-mode gate
     * actually fires on a real leak. */
    P();
}
#endif  /* JIT_TEST_COUNTER */

int main(void) {
    t_no_bb_pollution(); t_alloc_delete(); t_operand_cleanup();
#ifdef JIT_TEST_COUNTER
    t_negative_leak_detection();
#endif
    printf("\n%d pass, %d fail\n", gp, gf);
    return gf == 0 ? 0 : 1;
}
