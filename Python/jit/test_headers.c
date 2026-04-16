/* C/C++ header compatibility gate — C compilation.
 * Includes every JIT C header from a pure C translation unit.
 * If any header pulls in C++-only constructs or _Atomic via Python.h,
 * this file will fail to compile, catching the bug LOCALLY before ARM64.
 *
 * Companion: test_headers.cpp (same headers, C++ compilation).
 */

#include "cinderx/Jit/hir/hir_type_c.h"
#include "cinderx/Jit/hir/hir_opcode_c.h"
#include "cinderx/Jit/hir/hir_operand_types_c.h"
#include "cinderx/Jit/hir/hir_instr_info_c.h"
#include "cinderx/Jit/hir/phx_ptr_array.h"

/* Verify types are usable from C */
static void test_c_types(void) {
    HirType t = {0, {0}};
    (void)hir_type_bits(&t);

    HirOpcodeOperandInfo info = {0, {{0}}};
    (void)info.count;

    const HirInstrInfo *ii = hir_instr_get_info(0);
    (void)ii;
}
