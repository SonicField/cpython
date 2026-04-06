/*
 * frame_asm_c.c -- C implementations of FrameAsm methods
 *
 * Phase 3D: Converts frame_asm.cpp methods to standalone C functions.
 * Each function takes (void *env) instead of this->.
 */

#include "cinderx/Jit/codegen/arch/detection.h"
#include "cinderx/Jit/lir/lir_c_api.h"
#include "cinderx/Jit/lir/lir_types_c.h"
#include "cinderx/Jit/codegen/phylocation.h"

#include "Python.h"
#include "internal/pycore_pystate.h"
#include "internal/pycore_frame.h"

#include "cinderx/Common/py-portability.h"
#include "cinderx/Jit/jit_config_c.h"

#include "jit/phoenix_asm/phoenix_asm.h"
#include "cinderx/Jit/codegen/register_preserver_c.h"

/* Generator frame allocation — address obtained via C API bridge
 * (jit_rt_get_alloc_link_gen_frame_addr) because the C++ function
 * has C++ name mangling and a std::pair return type. */
#if defined(CINDER_X86_64)
#include "jit/phoenix_asm/x86_64.h"
#elif defined(CINDER_AARCH64)
#include "jit/phoenix_asm/arm64.h"
#endif

#include <assert.h>
#include <stdint.h>

/* FrameMode constants (must match enum class FrameMode in config.h) */
#define FRAME_MODE_NORMAL     0
#define FRAME_MODE_SHADOW     1
#define FRAME_MODE_LIGHTWEIGHT 2

/* Forward declarations for functions defined later in this file */
#if defined(ENABLE_LIGHTWEIGHT_FRAMES)
void frame_asm_c_link_lightweight_function_frame(
    void *env, PhxGp func_reg, PhxGp tstate_reg,
    const void *hir_func,
    const PhxRegPair *save_regs, int num_save_regs);
#endif

/* Forward declarations for C functions defined elsewhere */
#if defined(CINDER_AARCH64)
PhxMem jit_arch_ptr_resolve(PhxBuilder *as, PhxGp base, int32_t offset,
                            PhxGp scratch, int32_t access_size);
PhxMem jit_arch_ptr_offset(PhxGp base, int32_t offset, int32_t access_size);
#endif

/* ---- Helpers ---- */

static inline PhxBuilder *
get_builder(void *env) {
    return (PhxBuilder *)jit_environ_get_phx_builder(env);
}

#if defined(CINDER_AARCH64)
/* Construct a pre-indexed memory operand: [base, #offset]! */
static inline PhxMem
phx_a64_mem_pre(PhxGp base, int32_t offset) {
    PhxMem m = {0};
    m.base = base;
    m.offset = offset;
    m.size = 8;
    m.is_pre_index = 1;
    return m;
}

/* Construct a post-indexed memory operand: [base], #offset */
static inline PhxMem
phx_a64_mem_post(PhxGp base, int32_t offset) {
    PhxMem m = {0};
    m.base = base;
    m.offset = offset;
    m.size = 8;
    m.is_post_index = 1;
    return m;
}

/* Simple [base, #offset] memory operand */
static inline PhxMem
phx_a64_mem(PhxGp base, int32_t offset) {
    PhxMem m = {0};
    m.base = base;
    m.offset = offset;
    m.size = 8;
    return m;
}
#endif

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
    {
        /* lea rcx, [gen_resume_entry_label] */
        PhxLabel *lbl = (PhxLabel *)jit_environ_get_gen_resume_entry_label(env);
        PhxMem label_mem = {0};
        label_mem.is_label_rel = 1;
        label_mem.label_id = lbl->id;
        label_mem.size = 8;
        phx_x86_lea(pb, rcx, label_mem);
    }
    phx_x86_mov_rr(pb, r8, rbp);
    phx_x86_mov_ri(pb, PHX_R11,
        (int64_t)(uintptr_t)jit_rt_get_alloc_link_gen_frame_addr());
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
    {
        /* adr x3, gen_resume_entry_label */
        PhxLabel *lbl = (PhxLabel *)jit_environ_get_gen_resume_entry_label(env);
        PhxGp x3 = {3, 8};
        phx_a64_adr(pb, x3, *lbl);
    }
    phx_a64_mov_rr(pb, x4, fp);
    phx_a64_mov_ri(pb, scratch_br,
        (uint64_t)(uintptr_t)jit_rt_get_alloc_link_gen_frame_addr());
    phx_a64_blr(pb, scratch_br);
    phx_a64_mov_rr(pb, tstate_reg, x0);
    phx_a64_mov_rr(pb, fp, x1);
#endif
}

/* ================================================================
 * incRef — emit inline reference count increment
 *
 * GIL-enabled path only (Py_GIL_DISABLED is a separate, more complex path
 * that will be converted when the GIL-disabled build is targeted).
 * ================================================================ */

#ifndef Py_GIL_DISABLED

void
frame_asm_c_inc_ref(void *env, PhxGp obj_reg, PhxGp scratch_reg) {
    PhxBuilder *pb = get_builder(env);
    PhxLabel immortal = phx_builder_new_label(pb);

#if defined(CINDER_X86_64)
    /* Load ob_refcnt (32-bit), increment, check immortal */
    PhxGp scratch32 = {scratch_reg.id, 4};
    phx_x86_mov_rm(pb, scratch32,
        phx_dword_ptr(obj_reg, offsetof(PyObject, ob_refcnt)));
    phx_x86_inc_r(pb, scratch32);
#if PY_VERSION_HEX >= 0x030E0000
    phx_x86_js(pb, immortal);
#else
    phx_x86_je(pb, immortal);
#endif
    /* mortal — store back */
    phx_x86_mov_mr(pb,
        phx_dword_ptr(obj_reg, offsetof(PyObject, ob_refcnt)), scratch32);

    phx_builder_bind(pb, immortal);

#elif defined(CINDER_AARCH64)
    PhxGp scratch_w = {scratch_reg.id, 4};
    /* Load ob_refcnt (32-bit) */
    phx_a64_ldr(pb, scratch_w,
        jit_arch_ptr_offset(obj_reg, offsetof(PyObject, ob_refcnt), 4));
    /* adds — sets flags for immortal check */
    phx_a64_adds_rri(pb, scratch_w, scratch_w, 1);
#if PY_VERSION_HEX >= 0x030E0000
    phx_a64_b_mi(pb, immortal);
#else
    phx_a64_b_eq(pb, immortal);
#endif
    /* mortal — store back */
    phx_a64_str(pb, scratch_w,
        jit_arch_ptr_offset(obj_reg, offsetof(PyObject, ob_refcnt), 4));

    phx_builder_bind(pb, immortal);
#endif
}

#endif /* !Py_GIL_DISABLED */

/* ================================================================
 * storeConst — store a constant pointer value to [reg + offset]
 *
 * Returns 1 if scratch was NOT used (value fits in 32 bits on x86_64).
 * Returns 0 if scratch was used (ARM64 always uses scratch).
 * ================================================================ */

/* ================================================================
 * frameHeaderSize — compute frame header size for a function
 * ================================================================ */

/* Forward declaration — defined in frame_header.h as C function */
int jit_frame_header_size(PyCodeObject *code, int frame_mode_lightweight,
                          size_t frame_header_sizeof, size_t ptr_size);

/* Forward declaration — C function in frame_header.c or similar */
int jit_is_frame_mode_lightweight(void);

/* FrameHeader size — matches sizeof(jit::FrameHeader).
 * On 3.12+, FrameHeader = union { PyFunctionObject*; uintptr_t rtfs; }
 * which is just sizeof(void*). */
#define JIT_FRAME_HEADER_SIZE sizeof(void*)

/* frameHeaderSizeExcludingSpillSpace — the raw frame header size */
static int
frame_header_size_excl_spill(PyCodeObject *code) {
    int lightweight = jit_is_frame_mode_lightweight();
    return jit_frame_header_size(code, lightweight,
                                 JIT_FRAME_HEADER_SIZE, sizeof(void*));
}

int
frame_asm_c_frame_header_size(PyCodeObject *code) {
    int base = frame_header_size_excl_spill(code);
#ifdef ENABLE_SHADOW_FRAMES
    return base;
#else
    return base + (int)sizeof(void*);
#endif
}

/* ================================================================
 * storeConst — store a constant pointer value to [reg + offset]
 * ================================================================ */

int
frame_asm_c_store_const(void *env, PhxGp reg, int32_t offset,
                        void *val, PhxGp scratch0, PhxGp scratch1) {
    PhxBuilder *pb = get_builder(env);
    int64_t value = (int64_t)(uintptr_t)val;

#if defined(CINDER_X86_64)
    (void)scratch1;
    PhxMem dest = phx_qword_ptr(reg, offset);
    if (lir_fits_signed_int32(value)) {
        phx_x86_mov_mi(pb, dest, (uint32_t)value);
        return 1;
    }
    phx_x86_mov_ri(pb, scratch0, value);
    phx_x86_mov_mr(pb, dest, scratch0);
    return 0;

#elif defined(CINDER_AARCH64)
    phx_a64_mov_ri(pb, scratch0, (uint64_t)value);
    phx_a64_str(pb, scratch0,
        jit_arch_ptr_resolve(pb, reg, offset, scratch1, 8));
    return 0;
#endif
}

/* ================================================================
 * linkNormalFunctionFrame — allocate + link interpreter frame
 * (non-generator, non-lightweight)
 * ================================================================ */

void
frame_asm_c_link_normal_function_frame(
    void *env, PhxGp tstate_reg, const void *hir_func)
{
    PhxBuilder *pb = get_builder(env);
    void *code_obj = jit_hir_func_get_code(hir_func);
    void *debug_addr = jit_rt_get_alloc_link_frame_debug_addr();
    void *release_addr = jit_rt_get_alloc_link_frame_release_addr();

#if defined(CINDER_X86_64)
    PhxGp rsi = {6, 8}, rax = {0, 8};
#ifdef Py_DEBUG
    phx_x86_mov_ri(pb, rsi, (int64_t)(uintptr_t)code_obj);
    phx_x86_mov_ri(pb, PHX_R11, (int64_t)(uintptr_t)debug_addr);
    phx_x86_call_r(pb, PHX_R11);
#else
    (void)code_obj;
    phx_x86_mov_ri(pb, PHX_R11, (int64_t)(uintptr_t)release_addr);
    phx_x86_call_r(pb, PHX_R11);
#endif
    phx_x86_mov_rr(pb, tstate_reg, rax);

#elif defined(CINDER_AARCH64)
    PhxGp x0 = {0, 8}, x1 = {1, 8};
    PhxGp scratch_br = {16, 8};
#ifdef Py_DEBUG
    phx_a64_mov_ri(pb, x1, (uint64_t)(uintptr_t)code_obj);
    phx_a64_mov_ri(pb, scratch_br, (uint64_t)(uintptr_t)debug_addr);
#else
    (void)code_obj;
    phx_a64_mov_ri(pb, scratch_br, (uint64_t)(uintptr_t)release_addr);
#endif
    phx_a64_blr(pb, scratch_br);
    phx_a64_mov_rr(pb, tstate_reg, x0);
#endif
}

/* ================================================================
 * linkNormalFrame — dispatch to generator/lightweight/normal
 * (3.12+ version)
 * ================================================================ */

void
frame_asm_c_link_normal_frame(
    void *env, PhxGp func_reg, PhxGp tstate_reg,
    const void *hir_func, void *code_rt_ptr,
    const PhxRegPair *save_regs, int num_save_regs)
{
    PhxBuilder *pb = get_builder(env);
    PhxRegPreserver preserver;
    phx_reg_preserver_init(&preserver, pb, save_regs, num_save_regs);

    if (jit_hir_func_is_gen(hir_func)) {
        /* generator path — preserve/restore around the call */
        phx_reg_preserver_preserve(&preserver);
        frame_asm_c_link_normal_generator_frame(env, tstate_reg, code_rt_ptr);
        phx_reg_preserver_restore(&preserver);
    } else if (jit_hir_func_get_frame_mode(hir_func) == FRAME_MODE_LIGHTWEIGHT) {
#if defined(ENABLE_LIGHTWEIGHT_FRAMES)
        /* lightweight path manages its own preservation */
        frame_asm_c_link_lightweight_function_frame(
            env, func_reg, tstate_reg, hir_func, save_regs, num_save_regs);
#else
        assert(0 && "lightweight frames not enabled");
#endif
    } else {
        /* normal function path */
        phx_reg_preserver_preserve(&preserver);
        frame_asm_c_link_normal_function_frame(env, tstate_reg, hir_func);
        phx_reg_preserver_restore(&preserver);
    }
}

/* ================================================================
 * generateLinkFrame — top-level frame linking entry point (3.12+)
 * ================================================================ */

void
frame_asm_c_generate_link_frame(
    void *env, PhxGp func_reg, PhxGp tstate_reg,
    const void *hir_func, void *code_rt_ptr,
    const PhxRegPair *save_regs, int num_save_regs)
{
    /* 3.12+ always links a normal frame (shadow frames don't exist) */
    frame_asm_c_link_normal_frame(
        env, func_reg, tstate_reg, hir_func, code_rt_ptr,
        save_regs, num_save_regs);
}

/* ================================================================
 * generateUnlinkFrame — unlink frame on function exit
 * ================================================================ */

void
frame_asm_c_generate_unlink_frame(
    void *env, const void *hir_func)
{
    PhxBuilder *pb = get_builder(env);
    int returns_double = jit_hir_func_returns_double(hir_func);
    PyCodeObject *code = (PyCodeObject *)jit_hir_func_get_code(hir_func);
    int header_size = frame_asm_c_frame_header_size(code);
    void *unlink_addr = jit_rt_get_unlink_frame_addr();

#if defined(CINDER_X86_64)
    PhxGp rax = {0, 8};

#ifdef ENABLE_SHADOW_FRAMES
    int is_gen = jit_hir_func_is_gen(hir_func);
    phx_x86_mov_ri(pb, PHX_RDI, (int64_t)(is_gen ? 0 : 1));
    PhxMem saved_rax_ptr = phx_qword_ptr((PhxGp){5, 8}, -8);  /* rbp - 8 */
#else
    PhxGp rbp = {5, 8};
    PhxMem saved_rax_ptr = phx_qword_ptr(rbp, -header_size);
#endif

    {
        PhxGp xmm0 = {0, 16};
        if (returns_double) {
            phx_x86_movsd_mr(pb, saved_rax_ptr, xmm0);
        } else {
            phx_x86_mov_mr(pb, saved_rax_ptr, rax);
        }
        phx_x86_mov_ri(pb, PHX_R11, (int64_t)(uintptr_t)unlink_addr);
        phx_x86_call_r(pb, PHX_R11);
        if (returns_double) {
            phx_x86_movsd_rm(pb, xmm0, saved_rax_ptr);
        } else {
            phx_x86_mov_rm(pb, rax, saved_rax_ptr);
        }
    }

#elif defined(CINDER_AARCH64)
#ifdef ENABLE_SHADOW_FRAMES
    assert(0 && "shadow frames unsupported on ARM64");
#else
    PhxGp x0 = {0, 8};
    PhxGp fp = {29, 8};
    PhxGp scratch0 = {12, 8};
    PhxGp scratch_br = {16, 8};

    PhxMem saved_x0_ptr = jit_arch_ptr_resolve(pb, fp, -header_size, scratch0, 8);

    {
        PhxGp d0 = {0, 8 | 0x40};  /* FP register with PHX_FP_FLAG */
        if (returns_double) {
            phx_a64_str_fp(pb, d0, saved_x0_ptr);
        } else {
            phx_a64_str(pb, x0, saved_x0_ptr);
        }

        phx_a64_mov_ri(pb, scratch_br, (uint64_t)(uintptr_t)unlink_addr);
        phx_a64_blr(pb, scratch_br);

        /* Reload saved_x0_ptr — scratch0 may have been clobbered by the call */
        saved_x0_ptr = jit_arch_ptr_resolve(pb, fp, -header_size, scratch0, 8);

        if (returns_double) {
            phx_a64_ldr_fp(pb, d0, saved_x0_ptr);
        } else {
            phx_a64_ldr(pb, x0, saved_x0_ptr);
        }
    }
#endif
#endif
}

/* ================================================================
 * linkLightWeightFunctionFrame — set up lightweight interpreter frame
 *
 * This is the most complex frame_asm method (~400 lines in C++).
 * It initializes _PyInterpreterFrame fields on the stack and links
 * the frame into the thread state chain.
 * ================================================================ */

#if defined(ENABLE_LIGHTWEIGHT_FRAMES)

void
frame_asm_c_link_lightweight_function_frame(
    void *env, PhxGp func_reg, PhxGp tstate_reg,
    const void *hir_func,
    const PhxRegPair *save_regs, int num_save_regs)
{
    PhxBuilder *pb = get_builder(env);
    PhxRegPreserver preserver;
    phx_reg_preserver_init(&preserver, pb, save_regs, num_save_regs);

    PyCodeObject *code = (PyCodeObject *)jit_hir_func_get_code(hir_func);
    void *code_rt_ptr = jit_environ_get_code_rt(env);
    int frame_header_size = frame_header_size_excl_spill(code);

    /* Get reifier: on 3.14+ from code_rt, on 3.12 from module state */
#if PY_VERSION_HEX >= 0x030E0000
    PyObject *frame_reifier = (PyObject *)jit_code_rt_get_reifier(code_rt_ptr);
#else
    /* Pre-3.14: cinderx::getModuleState()->frameReifier() — need C++ accessor.
     * For now use the HIR function's reifier. */
    PyObject *frame_reifier = (PyObject *)jit_hir_func_get_reifier(hir_func);
#endif

    /* INITIAL_EXTRA_ARGS_REG = R10 (x86_64) / X10 (aarch64) */
    PhxGp scratch = {10, 8};

    /* Init tstate offset + load tstate */
    frame_asm_c_init_tstate_offset();

    if (tstate_offset == -1) {
        phx_reg_preserver_preserve(&preserver);
    }
    frame_asm_c_load_tstate(env, tstate_reg);

    if (tstate_offset == -1) {
        phx_reg_preserver_restore(&preserver);
#if defined(CINDER_X86_64)
        phx_x86_push_r(pb, scratch);
#elif defined(CINDER_AARCH64)
        PhxGp sp = {31, 8};
        phx_a64_str(pb, scratch, phx_a64_mem_pre(sp, -16));
#endif
    }

/* Macro for frame field offsets relative to frame pointer */
#define FRM_OFF(NAME) \
    (-frame_header_size + (int32_t)offsetof(_PyInterpreterFrame, NAME) \
     + (int32_t)JIT_FRAME_HEADER_SIZE)

    /* Zero-fill the entire frame region to prevent uninitialized field
     * crashes (frame_obj, localsplus, f_globals, etc.). The cost is
     * ~10-15 stores of zero, negligible vs function call overhead. */
    {
        int frame_words = frame_header_size / (int)sizeof(void*);
#if defined(CINDER_X86_64)
        PhxGp rbp_z = {5, 8};
        for (int i = 0; i < frame_words; i++) {
            phx_x86_mov_mi(pb,
                phx_qword_ptr(rbp_z, -frame_header_size + i * (int)sizeof(void*)),
                0);
        }
#elif defined(CINDER_AARCH64)
        PhxGp fp_z = {29, 8};
        PhxGp xzr_z = {31, 8};
        PhxGp scratch1_z = {13, 8};
        for (int i = 0; i < frame_words; i++) {
            phx_a64_str(pb, xzr_z,
                jit_arch_ptr_resolve(pb, fp_z,
                    -frame_header_size + i * (int)sizeof(void*),
                    scratch1_z, 8));
        }
#endif
    }

#if defined(CINDER_X86_64)
    PhxGp rbp = {5, 8}, rax = {0, 8};

    /* Store func/rtfs before the header */
#if PY_VERSION_HEX >= 0x030E0000
    /* Store rtfs state to 0 */
    phx_x86_mov_mi(pb, phx_qword_ptr(rbp, -frame_header_size), 0);
#else
    /* Store func before frame header */
    phx_x86_mov_mr(pb, phx_qword_ptr(rbp, -frame_header_size), func_reg);
    frame_asm_c_inc_ref(env, func_reg, rax);
#endif

    /* Set f_executable/f_code */
#if PY_VERSION_HEX >= 0x030E0000
    PyObject *executable = frame_reifier;
#else
    PyObject *executable = (PyObject *)code;
#endif
    {
        int needs_load = frame_asm_c_store_const(
            env, rbp, FRM_OFF(FRAME_EXECUTABLE), executable, scratch, scratch);
        if (!_Py_IsImmortal(executable)) {
            if (needs_load) {
                phx_x86_mov_ri(pb, scratch,
                    (int64_t)(uintptr_t)executable);
            }
            frame_asm_c_inc_ref(env, scratch, rax);
        }
    }

    /* Set f_funcobj */
#if PY_VERSION_HEX >= 0x030E0000
    phx_x86_mov_mr(pb, phx_qword_ptr(rbp, FRM_OFF(f_funcobj)), func_reg);
    frame_asm_c_inc_ref(env, func_reg, rax);
#else
    frame_asm_c_store_const(env, rbp, FRM_OFF(f_funcobj),
                            frame_reifier, scratch, scratch);
    /* frame_reifier must be immortal */
#endif

    /* Set prev_instr/instr_ptr */
    {
#if PY_VERSION_HEX >= 0x030E0000
        _Py_CODEUNIT *bytecode = _PyCode_CODE(code);
#else
        _Py_CODEUNIT *bytecode = _PyCode_CODE(code) - 1;
#endif
        frame_asm_c_store_const(env, rbp, FRM_OFF(FRAME_INSTR),
                                bytecode, scratch, scratch);
    }

#ifdef Py_GIL_DISABLED
    phx_x86_mov_mi(pb,
        phx_dword_ptr(rbp, FRM_OFF(tlbc_index)), 0);
#endif

    /* Set owner = FRAME_OWNED_BY_THREAD */
    {
        PhxMem owner_mem = phx_byte_ptr(rbp, FRM_OFF(owner));
        phx_x86_mov_mi(pb, owner_mem, FRAME_OWNED_BY_THREAD);
    }

    /* Set frame_obj = NULL (must be zeroed before frame chain walking) */
    phx_x86_mov_mi(pb, phx_qword_ptr(rbp, FRM_OFF(frame_obj)), 0);

    /* Get topmost frame from thread state */
#if PY_VERSION_HEX >= 0x030D0000
    PhxGp frame_holder = tstate_reg;
    phx_x86_mov_rm(pb, scratch,
        phx_qword_ptr(tstate_reg, offsetof(PyThreadState, current_frame)));
#else
    PhxGp frame_holder = rax;
    phx_x86_mov_rm(pb, frame_holder,
        phx_qword_ptr(tstate_reg, offsetof(PyThreadState, cframe)));
    phx_x86_mov_rm(pb, scratch,
        phx_qword_ptr(frame_holder, offsetof(_PyCFrame, current_frame)));
#endif

    /* Set previous */
    phx_x86_mov_mr(pb, phx_qword_ptr(rbp, FRM_OFF(previous)), scratch);

#if PY_VERSION_HEX >= 0x030E0000
    /* Set stackpointer = &localsplus */
    phx_x86_lea(pb, scratch,
        phx_qword_ptr(rbp, FRM_OFF(localsplus)));
    phx_x86_mov_mr(pb,
        phx_qword_ptr(rbp, FRM_OFF(stackpointer)), scratch);

    /* Set f_locals = NULL */
    phx_x86_mov_mi(pb,
        phx_qword_ptr(rbp, FRM_OFF(f_locals)), 0);
#endif

    /* Link our frame into thread state */
    {
        int frame_ptr_off = -frame_header_size + (int32_t)sizeof(PyObject*);
        phx_x86_lea(pb, scratch, phx_qword_ptr(rbp, frame_ptr_off));
#if PY_VERSION_HEX >= 0x030D0000
        phx_x86_mov_mr(pb,
            phx_qword_ptr(frame_holder,
                offsetof(PyThreadState, current_frame)),
            scratch);
#else
        phx_x86_mov_mr(pb,
            phx_qword_ptr(frame_holder,
                offsetof(_PyCFrame, current_frame)),
            scratch);
#endif
    }

    if (tstate_offset == -1) {
        phx_x86_pop_r(pb, scratch);
    } else {
        phx_reg_preserver_remap(&preserver);
    }

#elif defined(CINDER_AARCH64)
    PhxGp fp = {29, 8};
    PhxGp scratch0 = {12, 8};  /* X12 */
    PhxGp scratch1 = {13, 8};  /* X13 — reg_scratch_1 */
    PhxGp sp = {31, 8};
    PhxGp xzr = {31, 8};  /* xzr and sp share encoding, context-dependent */
    PhxGp w9 = {9, 4};     /* ref_cnt */
    PhxGp w12 = {12, 4};   /* ref_cnt_scratch */

    /* Store func/rtfs before the header */
#if PY_VERSION_HEX >= 0x030E0000
    phx_a64_sub_rri(pb, scratch0, fp, frame_header_size);
    /* str xzr, [scratch0] — store zero */
    phx_a64_str(pb, xzr, phx_a64_mem(scratch0, 0));
#else
    phx_a64_sub_rri(pb, scratch0, fp, frame_header_size);
    phx_a64_str(pb, func_reg, phx_a64_mem(scratch0, 0));
    /* incRef func_reg — ARM64 needs 2 scratch regs */
    /* For the C path, we call frame_asm_c_inc_ref which only takes 2 regs.
     * The ARM64 version in the C++ code uses 4 args (reg, scratch0, scratch1, tstate).
     * Our C inc_ref is the GIL-enabled path only. */
    frame_asm_c_inc_ref(env, func_reg, w9);
#endif

    /* Set f_executable/f_code */
#if PY_VERSION_HEX >= 0x030E0000
    PyObject *executable = frame_reifier;
#else
    PyObject *executable = (PyObject *)code;
#endif
    frame_asm_c_store_const(env, fp, FRM_OFF(FRAME_EXECUTABLE),
                            executable, scratch, scratch1);
    if (!_Py_IsImmortal(executable)) {
        frame_asm_c_inc_ref(env, scratch, w9);
    }

    /* Set f_funcobj */
#if PY_VERSION_HEX >= 0x030E0000
    phx_a64_str(pb, func_reg,
        jit_arch_ptr_resolve(pb, fp, FRM_OFF(f_funcobj), scratch0, 8));
    frame_asm_c_inc_ref(env, func_reg, w9);
#else
    frame_asm_c_store_const(env, fp, FRM_OFF(f_funcobj),
                            frame_reifier, scratch, scratch1);
#endif

    /* Set prev_instr/instr_ptr */
    {
#if PY_VERSION_HEX >= 0x030E0000
        _Py_CODEUNIT *bytecode = _PyCode_CODE(code);
#else
        _Py_CODEUNIT *bytecode = _PyCode_CODE(code) - 1;
#endif
        frame_asm_c_store_const(env, fp, FRM_OFF(FRAME_INSTR),
                                bytecode, scratch, scratch1);
    }

#ifdef Py_GIL_DISABLED
    /* str xzr, [fp, FRM_OFF(tlbc_index)] */
    phx_a64_str(pb, xzr,
        jit_arch_ptr_offset(fp, FRM_OFF(tlbc_index), 8));
#endif

    /* Set owner = FRAME_OWNED_BY_THREAD */
    {
        PhxGp w12_tmp = {12, 4};
        phx_a64_mov_ri(pb, w12_tmp, FRAME_OWNED_BY_THREAD);
        phx_a64_strb(pb, w12_tmp,
            jit_arch_ptr_resolve(pb, fp, FRM_OFF(owner), scratch1, 4));
    }

    /* Set frame_obj = NULL (must be zeroed before frame chain walking) */
    phx_a64_str(pb, xzr,
        jit_arch_ptr_resolve(pb, fp, FRM_OFF(frame_obj), scratch1, 8));

    /* Get topmost frame */
#if PY_VERSION_HEX >= 0x030D0000
    PhxGp frame_holder = tstate_reg;
    phx_a64_ldr(pb, scratch,
        jit_arch_ptr_offset(tstate_reg,
            offsetof(PyThreadState, current_frame), 8));
#else
    PhxGp frame_holder = scratch0;
    phx_a64_ldr(pb, frame_holder,
        jit_arch_ptr_offset(tstate_reg,
            offsetof(PyThreadState, cframe), 8));
    phx_a64_ldr(pb, scratch,
        jit_arch_ptr_offset(frame_holder,
            offsetof(_PyCFrame, current_frame), 8));
#endif

    /* Set previous */
    phx_a64_str(pb, scratch,
        jit_arch_ptr_resolve(pb, fp, FRM_OFF(previous), scratch1, 8));

#if PY_VERSION_HEX >= 0x030E0000
    /* Set stackpointer = &localsplus */
    {
        int localsplus_off = FRM_OFF(localsplus);
        if (localsplus_off < 0) {
            /* isAddSubImm: check if fits in 12-bit unsigned immediate */
            if ((uint32_t)(-localsplus_off) < 4096) {
                phx_a64_sub_rri(pb, scratch, fp, -localsplus_off);
            } else {
                phx_a64_mov_ri(pb, scratch, (uint64_t)(-localsplus_off));
                phx_a64_sub_rrr(pb, scratch, fp, scratch);
            }
        } else {
            phx_a64_add_rri(pb, scratch, fp, localsplus_off);
        }
    }
    phx_a64_str(pb, scratch,
        jit_arch_ptr_resolve(pb, fp, FRM_OFF(stackpointer), scratch1, 8));

    /* Set f_locals = NULL */
    phx_a64_str(pb, xzr,
        jit_arch_ptr_resolve(pb, fp, FRM_OFF(f_locals), scratch1, 8));
#endif

    /* Link our frame into thread state */
    {
        int size = -frame_header_size + (int32_t)sizeof(PyObject*);
        if (size > 0) {
            phx_a64_add_rri(pb, scratch, fp, size);
        } else {
            phx_a64_sub_rri(pb, scratch, fp, -size);
        }
#if PY_VERSION_HEX >= 0x030D0000
        phx_a64_str(pb, scratch,
            jit_arch_ptr_offset(frame_holder,
                offsetof(PyThreadState, current_frame), 8));
#else
        phx_a64_str(pb, scratch,
            jit_arch_ptr_offset(frame_holder,
                offsetof(_PyCFrame, current_frame), 8));
#endif
    }

    if (tstate_offset == -1) {
        phx_a64_ldr(pb, scratch, phx_a64_mem_post(sp, 16));
    } else {
        phx_reg_preserver_remap(&preserver);
    }
#endif

#undef FRM_OFF
}

#endif /* ENABLE_LIGHTWEIGHT_FRAMES */
