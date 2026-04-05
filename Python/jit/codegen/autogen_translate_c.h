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

/* Cross-architecture translate* functions */
void autogen_c_TranslateGuard(void *env, const LirInstruction *instr);
void autogen_c_TranslateCompare(void *env, const LirInstruction *instr);
void autogen_c_TranslateDeoptPatchpoint(void *env, const LirInstruction *instr);

#if defined(CINDER_AARCH64)

void autogen_c_translateUnreachable(void *env, const LirInstruction *instr);
void autogen_c_translateAdd(void *env, const LirInstruction *instr);
void autogen_c_translateSub(void *env, const LirInstruction *instr);
void autogen_c_translateInc(void *env, const LirInstruction *instr);
void autogen_c_translateDec(void *env, const LirInstruction *instr);
void autogen_c_translateAnd(void *env, const LirInstruction *instr);
void autogen_c_translateOr(void *env, const LirInstruction *instr);
void autogen_c_translateXor(void *env, const LirInstruction *instr);
void autogen_c_translateMul(void *env, const LirInstruction *instr);
void autogen_c_translateDiv(void *env, const LirInstruction *instr);
void autogen_c_translateDivUn(void *env, const LirInstruction *instr);
void autogen_c_translatePush(void *env, const LirInstruction *instr);
void autogen_c_translatePop(void *env, const LirInstruction *instr);
void autogen_c_translateExchange(void *env, const LirInstruction *instr);
void autogen_c_translateCmp(void *env, const LirInstruction *instr);
void autogen_c_translateBitTest(void *env, const LirInstruction *instr);
void autogen_c_translateMovZX(void *env, const LirInstruction *instr);
void autogen_c_translateMovSX(void *env, const LirInstruction *instr);
void autogen_c_translateMovSXD(void *env, const LirInstruction *instr);
void autogen_c_translateTst(void *env, const LirInstruction *instr);
void autogen_c_translateIntToBool(void *env, const LirInstruction *instr);
void autogen_c_translateSelect(void *env, const LirInstruction *instr);
void autogen_c_translateLea(void *env, const LirInstruction *instr);
void autogen_c_translateMove(void *env, const LirInstruction *instr);
void autogen_c_translateCall(void *env, const LirInstruction *instr);

#endif /* CINDER_AARCH64 */

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* JIT_CODEGEN_AUTOGEN_TRANSLATE_C_H */
