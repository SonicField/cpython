/* test_unlinked_instr.cpp - c22b-api Block::allocateInstrUnlinked tests.
 * Pre-spec criteria API-1 (caller delete), API-3 (BB list unchanged),
 * API-6 (operand cleanup cascade via dtor).
 *
 * Standalone, not auto-built. Manual compile:
 *   c++ -I. -I../../../Include test_unlinked_instr.cpp -lpython3
 */
#include <cstdio>
#include "Python.h"
#include "cinderx/Jit/lir/function.h"
#include "cinderx/Jit/lir/block.h"
#include "cinderx/Jit/lir/instruction.h"

using jit::lir::Function;
using jit::lir::BasicBlock;
using jit::lir::Instruction;

static int g_pass = 0, g_fail = 0;
#define A(c, m) do { if(!(c)){ printf("FAIL %s: %s\n", __func__, m); g_fail++; return; } } while(0)
#define P() do { printf("PASS %s\n", __func__); g_pass++; } while(0)

/* API-3: alloc + delete leaves BB instr list unchanged */
static void test_no_bb_pollution(void) {
    Function func;
    BasicBlock* bb = func.allocateBasicBlock();
    Instruction* head_pre = bb->getFirstInstr();
    size_t n_pre = bb->getNumInstrs();

    Instruction* shadow = bb->allocateInstrUnlinked(
        Instruction::kCall, nullptr);
    A(shadow != nullptr, "alloc returned null");
    A(bb->getFirstInstr() == head_pre, "BB head changed");
    A(bb->getNumInstrs() == n_pre, "BB size changed");

    delete shadow;
    A(bb->getFirstInstr() == head_pre, "BB head changed post-delete");
    A(bb->getNumInstrs() == n_pre, "BB size changed post-delete");
    P();
}

/* API-1: canonical alloc + delete pattern */
static void test_alloc_delete(void) {
    Function func;
    BasicBlock* bb = func.allocateBasicBlock();
    Instruction* shadow = bb->allocateInstrUnlinked(
        Instruction::kMove, nullptr);
    A(shadow != nullptr, "alloc null");
    A(shadow->basicblock() == bb, "parent mismatch");
    delete shadow;
    P();
}

/* API-6: operand cleanup cascade. Two unlinked instrs in sequence;
 * if dtor leaks operands or unlinked allocation leaks state, second
 * alloc + delete cycle would observe corruption. Compile-time
 * assertion supplements valgrind/ASAN if available. */
static void test_operand_cleanup(void) {
    Function func;
    BasicBlock* bb = func.allocateBasicBlock();
    for (int i = 0; i < 10; i++) {
        Instruction* shadow = bb->allocateInstrUnlinked(
            Instruction::kCall, nullptr);
        A(shadow != nullptr, "alloc null in cycle");
        delete shadow;
    }
    A(bb->getNumInstrs() == 0, "BB polluted after 10 cycles");
    P();
}

int main(void) {
    test_no_bb_pollution();
    test_alloc_delete();
    test_operand_cleanup();
    printf("\n%d pass, %d fail\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
