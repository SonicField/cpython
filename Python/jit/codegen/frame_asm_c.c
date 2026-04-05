/*
 * frame_asm_c.c -- C implementations of FrameAsm methods
 *
 * Phase 3D: Converts frame_asm.cpp methods to standalone C functions.
 * Each function takes (void *env) instead of this->.
 */

#include "cinderx/Jit/lir/lir_c_api.h"
#include "cinderx/Jit/lir/lir_types_c.h"
#include "cinderx/Jit/codegen/phylocation.h"

#include "Python.h"
#include "internal/pycore_pystate.h"

#include "jit/phoenix_asm/phoenix_asm.h"

/* JITRT_AllocateAndLinkGenAndInterpreterFrame is a C++ function
 * whose address is loaded into a register for BLR/CALL. We only
 * need its address, not its signature, from C code. */
extern void JITRT_AllocateAndLinkGenAndInterpreterFrame(void);
#if defined(CINDER_X86_64)
#include "jit/phoenix_asm/x86_64.h"
#elif defined(CINDER_AARCH64)
#include "jit/phoenix_asm/arm64.h"
#endif

#include <assert.h>
#include <stdint.h>

/* Forward declarations for C functions defined elsewhere */
#if defined(CINDER_AARCH64)
PhxMem jit_arch_ptr_resolve(PhxBuilder *as, PhxGp base, int32_t offset,
                            PhxGp scratch, int32_t access_size);
#endif

/* ---- Helpers ---- */

static inline PhxBuilder *
get_builder(void *env) {
    return (PhxBuilder *)jit_environ_get_phx_builder(env);
}

/* ================================================================
 * initThreadStateOffset + loadTState
 *
 * Discovers the thread-local storage offset for PyThreadState
 * by introspecting _PyThreadState_GetCurrent's machine code.
 * ================================================================ */

static int tstate_offset_inited = 0;
static int32_t tstate_offset = -1;

void
frame_asm_c_init_tstate_offset(void) {
    if (tstate_offset_inited) return;

#if defined(CINDER_X86_64)
    uint8_t *ts_func = (uint8_t *)&_PyThreadState_GetCurrent;
    if (ts_func[0] == 0x55 &&
        ts_func[1] == 0x48 && ts_func[2] == 0x89 &&
        ts_func[3] == 0xe5 &&
        ts_func[4] == 0x64 && ts_func[5] == 0x48 && ts_func[6] == 0x8b &&
        ts_func[7] == 0x04 && ts_func[8] == 0x25) {
        tstate_offset = *(int32_t *)(ts_func + 9);
    }
#elif defined(CINDER_AARCH64)
    uint32_t *ts_func = (uint32_t *)&_PyThreadState_GetCurrent;
    if (ts_func[0] == 0xa9bf7bfd &&
        ts_func[1] == 0x910003fd &&
        ((ts_func[2] & ~0x1f) == 0xd53bd048)) {
        uint32_t reg = ts_func[2] & 0x1f;
        int32_t current_offset = 0;
        for (size_t index = 3; ; index++) {
            if (ts_func[index] == (0xf9400000 | (reg << 5))) {
                break;
            } else if ((ts_func[index] & ~0x7ffc00) ==
                       (0x91000000 | (reg << 5) | reg)) {
                uint32_t imm = (ts_func[index] >> 10) & 0xfff;
                if (ts_func[index] & (1 << 22)) imm <<= 12;
                current_offset += (int32_t)imm;
            } else {
                current_offset = -1;
                break;
            }
        }
        tstate_offset = current_offset;
    }
#endif

    tstate_offset_inited = 1;
}

void
frame_asm_c_load_tstate(void *env, PhxGp dst_reg) {
    PhxBuilder *pb = get_builder(env);

#if defined(CINDER_X86_64)
    if (tstate_offset != -1) {
        PhxMem tls = phx_fs_ptr(tstate_offset);
        phx_x86_mov_rm(pb, dst_reg, tls);
    } else {
        phx_x86_mov_ri(pb, PHX_R11,
            (int64_t)(uintptr_t)_PyThreadState_GetCurrent);
        phx_x86_call_r(pb, PHX_R11);
        phx_x86_mov_rr(pb, dst_reg, PHX_RAX);
    }
#elif defined(CINDER_AARCH64)
    PhxGp scratch0 = {12, 8}; /* X12 */
    PhxGp scratch_br = {16, 8}; /* X16 */
    PhxGp x0 = {0, 8};
    if (tstate_offset != -1) {
        phx_a64_mrs(pb, dst_reg, PHX_SYSREG_TPIDR_EL0);
        phx_a64_ldr(pb, dst_reg,
            jit_arch_ptr_resolve(pb, dst_reg, tstate_offset, scratch0, 8));
    } else {
        phx_a64_mov_ri(pb, scratch_br,
            (uint64_t)(uintptr_t)_PyThreadState_GetCurrent);
        phx_a64_blr(pb, scratch_br);
        phx_a64_mov_rr(pb, dst_reg, x0);
    }
#endif
}

/* ================================================================
 * linkNormalGeneratorFrame — set up generator frame
 * ================================================================ */

void
frame_asm_c_link_normal_generator_frame(
    void *env, PhxGp tstate_reg, void *code_rt_ptr)
{
    PhxBuilder *pb = get_builder(env);
    int spill_size = jit_environ_shadow_frames_and_spill_size(env);
    uint64_t full_words = (uint64_t)spill_size / sizeof(void*);

#if defined(CINDER_X86_64)
    PhxGp rsi = {6, 8}, rdx = {2, 8}, rcx = {1, 8}, r8 = {8, 8};
    PhxGp rbp = {5, 8}, rax = {0, 8};

    phx_x86_mov_ri(pb, rsi, (int64_t)full_words);
    phx_x86_mov_ri(pb, rdx, (int64_t)(uintptr_t)code_rt_ptr);
    /* lea rcx, [gen_resume_entry_label] — need label */
    /* For now, skip label-dependent code — will wire when label C API is ready */
    phx_x86_mov_rr(pb, r8, rbp);
    phx_x86_mov_ri(pb, PHX_R11,
        (int64_t)(uintptr_t)JITRT_AllocateAndLinkGenAndInterpreterFrame);
    phx_x86_call_r(pb, PHX_R11);
    phx_x86_mov_rr(pb, tstate_reg, rax);
    phx_x86_mov_rr(pb, rbp, rdx);

#elif defined(CINDER_AARCH64)
    PhxGp x1 = {1, 8}, x2 = {2, 8}, x4 = {4, 8};
    PhxGp x0 = {0, 8};
    PhxGp fp = {29, 8};
    PhxGp scratch_br = {16, 8};

    phx_a64_mov_ri(pb, x1, full_words);
    phx_a64_mov_ri(pb, x2, (uint64_t)(uintptr_t)code_rt_ptr);
    /* adr x3, gen_resume_entry_label — need label C API */
    phx_a64_mov_rr(pb, x4, fp);
    phx_a64_mov_ri(pb, scratch_br,
        (uint64_t)(uintptr_t)JITRT_AllocateAndLinkGenAndInterpreterFrame);
    phx_a64_blr(pb, scratch_br);
    phx_a64_mov_rr(pb, tstate_reg, x0);
    phx_a64_mov_rr(pb, fp, x1);
#endif
}
