/* arch.c -- Architecture-specific pointer resolution (pure C)
 *
 * Phase 3D conversion: arch.cpp -> arch.c
 * Provides ARM64 ptr_offset and ptr_resolve using the phoenix-asm C API.
 */

#include "cinderx/Jit/codegen/arch/detection.h"

#if defined(PHOENIX_ASM) || defined(__aarch64__)

#include "jit/phoenix_asm/phoenix_asm.h"
#include "jit/phoenix_asm/arm64.h"

#include <assert.h>
#include <stdint.h>

#if defined(CINDER_AARCH64)

static int
jit_arch_is_add_sub_imm(uint64_t val) {
    return val <= 0xFFF || (val <= 0xFFF000 && (val & 0xFFF) == 0);
}

/* Try to build a memory operand using an offset from a base register.
 * Returns 1 on success (result written to *out), 0 if offset is out of range. */
static int
jit_arch_ptr_offset_try(PhxMem *out, PhxGp base, int32_t offset,
                        int32_t access_size) {
    if (offset >= -256 && offset < 256) {
        *out = phx_ptr(base, offset);
        return 1;
    }

    if (offset >= 0 && (offset & (access_size - 1)) == 0 &&
        offset < access_size * 4096) {
        *out = phx_ptr(base, offset);
        return 1;
    }

    return 0;
}

PhxMem
jit_arch_ptr_offset(PhxGp base, int32_t offset, int32_t access_size) {
    PhxMem result;
    int ok = jit_arch_ptr_offset_try(&result, base, offset, access_size);
    assert(ok && "jit_arch_ptr_offset: offset out of range");
    (void)ok;
    return result;
}

PhxMem
jit_arch_ptr_resolve(PhxBuilder *as, PhxGp base, int32_t offset,
                     PhxGp scratch, int32_t access_size) {
    PhxMem result;
    if (jit_arch_ptr_offset_try(&result, base, offset, access_size)) {
        return result;
    }

    if (offset >= 0 && jit_arch_is_add_sub_imm((uint64_t)offset)) {
        phx_a64_add_rri(as, scratch, base, offset);
        return phx_ptr(scratch, 0);
    }
    if (offset < 0 && jit_arch_is_add_sub_imm((uint64_t)(-offset))) {
        phx_a64_sub_rri(as, scratch, base, -offset);
        return phx_ptr(scratch, 0);
    }

    /* Worst case: load offset into scratch, then base+index */
    phx_a64_mov_ri(as, scratch, (uint64_t)(int64_t)offset);
    return phx_ptr_index(base, scratch, 1, 0);
}

#endif /* CINDER_AARCH64 */

#endif /* PHOENIX_ASM */
