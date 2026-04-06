/*
 * frame_asm_c.h -- C declarations for frame_asm standalone functions
 *
 * Phase 3D: Replaces FrameAsm class methods with C functions.
 */

#ifndef JIT_CODEGEN_FRAME_ASM_C_H
#define JIT_CODEGEN_FRAME_ASM_C_H

#include "jit/phoenix_asm/phoenix_asm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Discover thread-local storage offset for PyThreadState */
void frame_asm_c_init_tstate_offset(void);

/* Emit code to load PyThreadState into dst_reg */
void frame_asm_c_load_tstate(void *env, PhxGp dst_reg);

/* Set up generator frame (allocate + link) */
void frame_asm_c_link_normal_generator_frame(
    void *env, PhxGp tstate_reg, void *code_rt_ptr);

/* Emit inline incRef (GIL-enabled path) */
#ifndef Py_GIL_DISABLED
void frame_asm_c_inc_ref(void *env, PhxGp obj_reg, PhxGp scratch_reg);
#endif

/* Compute frame header size for a function */
int frame_asm_c_frame_header_size(PyCodeObject *code);

/* Store constant pointer to [reg + offset]. Returns 1 if scratch unused. */
int frame_asm_c_store_const(void *env, PhxGp reg, int32_t offset,
                            void *val, PhxGp scratch0, PhxGp scratch1);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* JIT_CODEGEN_FRAME_ASM_C_H */
