/* C/C++ header compatibility gate — C++ compilation.
 * Includes every JIT C header from a C++ translation unit.
 * If any header uses _Atomic, Py_TYPE, or other C-only constructs
 * that break C++ compilation, this file will fail to compile.
 *
 * Companion: test_headers.c (same headers, C compilation).
 */

// Include JIT C headers from C++ context
#include "cinderx/Jit/hir/hir_type_c.h"
#include "cinderx/Jit/hir/hir_opcode_c.h"
#include "cinderx/Jit/hir/hir_operand_types_c.h"
#include "cinderx/Jit/hir/hir_instr_info_c.h"
#include "cinderx/Jit/hir/phx_ptr_array.h"

// Verify types are usable from C++
namespace {
void test_cpp_types() {
    HirType t{};
    (void)hir_type_bits(&t);

    HirOpcodeOperandInfo info{};
    (void)info.count;

    const HirInstrInfo *ii = hir_instr_get_info(0);
    (void)ii;
}
} // namespace
