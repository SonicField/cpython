/*
 * register_preserver_c.h -- Pure C interface for register preservation
 *
 * Replaces the C++ RegisterPreserver class. Codegen files that have been
 * converted to C include this directly; unconverted C++ files continue
 * using register_preserver.h which wraps this C API.
 */

#ifndef REGISTER_PRESERVER_C_H
#define REGISTER_PRESERVER_C_H

#include "jit/phoenix_asm/phoenix_asm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* A register pair: source register to save, destination to restore into */
typedef struct {
    PhxGp src;
    PhxGp dst;
} PhxRegPair;

/* Register preserver state */
typedef struct {
    PhxBuilder* builder;
    const PhxRegPair* regs;
    int num_regs;
    int align_stack;  /* whether stack alignment padding was added */
} PhxRegPreserver;

/* Initialize a register preserver */
void phx_reg_preserver_init(PhxRegPreserver* rp, PhxBuilder* builder,
                            const PhxRegPair* regs, int num_regs);

/* Save all registers to the stack */
void phx_reg_preserver_preserve(PhxRegPreserver* rp);

/* Remap registers (mov src -> dst for each pair where src != dst) */
void phx_reg_preserver_remap(PhxRegPreserver* rp);

/* Restore all registers from the stack (reverse order) */
void phx_reg_preserver_restore(PhxRegPreserver* rp);

#ifdef __cplusplus
}
#endif

#endif /* REGISTER_PRESERVER_C_H */
