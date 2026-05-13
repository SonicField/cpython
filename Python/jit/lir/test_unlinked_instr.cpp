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

int main(void) {
    t_no_bb_pollution(); t_alloc_delete(); t_operand_cleanup();
    printf("\n%d pass, %d fail\n", gp, gf);
    return gf == 0 ? 0 : 1;
}
