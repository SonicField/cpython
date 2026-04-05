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

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* JIT_CODEGEN_FRAME_ASM_C_H */
