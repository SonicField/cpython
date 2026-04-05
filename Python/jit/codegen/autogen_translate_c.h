/*
 * autogen_translate_c.h -- C declarations for autogen translate* functions
 *
 * Phase 3D: These replace the C++ translate* functions in autogen.cpp.
 * Include from autogen.cpp to call via CALL_C() macro.
 */

#ifndef JIT_CODEGEN_AUTOGEN_TRANSLATE_C_H
#define JIT_CODEGEN_AUTOGEN_TRANSLATE_C_H

#include "cinderx/Jit/lir/lir_types_c.h"

#ifdef __cplusplus
extern "C" {
#endif

#if defined(CINDER_AARCH64)

void autogen_c_translateUnreachable(void *env, const LirInstruction *instr);
void autogen_c_translateAdd(void *env, const LirInstruction *instr);
void autogen_c_translateSub(void *env, const LirInstruction *instr);
void autogen_c_translateInc(void *env, const LirInstruction *instr);
void autogen_c_translateDec(void *env, const LirInstruction *instr);

#endif /* CINDER_AARCH64 */

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* JIT_CODEGEN_AUTOGEN_TRANSLATE_C_H */
