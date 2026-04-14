/*
 * code_patcher_c.h — Pure C code patcher for runtime code modification.
 *
 * Phase 3D: Replaces code_patcher.h (C++ CodePatcher/JumpPatcher classes).
 * Virtual dispatch → function pointer table.
 */
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Forward declaration. */
typedef struct JitCodePatcher JitCodePatcher;

/* Virtual method table — replaces C++ virtual dispatch. */
typedef struct {
    void (*on_link)(JitCodePatcher *self);
    void (*on_patch)(JitCodePatcher *self);
    void (*on_unpatch)(JitCodePatcher *self);
    void (*destroy)(JitCodePatcher *self);
} JitCodePatcherVtable;

/* Flags bitfield (matches C++ CodePatcher::Flags layout). */
typedef union {
    struct {
        uint8_t data_len : 3;
        uint8_t is_patched : 1;
        uint8_t lock : 1;
    } f;
    uint8_t byte;
} JitCodePatcherFlags;

/* Base code patcher struct — replaces CodePatcher class.
 * Subclasses (JumpPatcher, GlobalDeoptPatcher, TypeDeoptPatcher)
 * are extensions with the same prefix fields. */
struct JitCodePatcher {
    const JitCodePatcherVtable *vtable;  /* function pointer dispatch */
    uint8_t *patchpoint;                  /* where in code to patch */
    uint8_t data[7];                      /* stored bytes (capacity) */
    JitCodePatcherFlags flags;            /* data_len, is_patched, lock */
};

/* ---- Lifecycle ---- */

/* Initialize a base code patcher (no jump support). */
void jit_code_patcher_init(JitCodePatcher *p, const JitCodePatcherVtable *vt);

/* Initialize a jump patcher (pre-fills with nop bytes). */
void jit_jump_patcher_init(JitCodePatcher *p, const JitCodePatcherVtable *vt);

/* Destroy — calls vtable->destroy if set. */
void jit_code_patcher_destroy(JitCodePatcher *p);

/* ---- Operations ---- */

/* Link the patcher to a location in generated code.
 * data/data_len: bytes to write on patch. */
void jit_code_patcher_link(JitCodePatcher *p, uintptr_t patchpoint,
                           const uint8_t *data, size_t data_len);

/* Patch the code (swap stored bytes with live code). */
void jit_code_patcher_patch(JitCodePatcher *p);

/* Unpatch the code (swap back). */
void jit_code_patcher_unpatch(JitCodePatcher *p);

/* ---- Queries ---- */

int jit_code_patcher_is_linked(const JitCodePatcher *p);
int jit_code_patcher_is_patched(const JitCodePatcher *p);
uint8_t *jit_code_patcher_patchpoint(const JitCodePatcher *p);
const uint8_t *jit_code_patcher_stored_bytes(const JitCodePatcher *p,
                                              size_t *out_len);

/* ---- Jump patcher specific ---- */

/* Link a jump from patchpoint to jump_target. */
void jit_jump_patcher_link_jump(JitCodePatcher *p, uintptr_t patchpoint,
                                uintptr_t jump_target);

/* Get the jump target address. */
uint8_t *jit_jump_patcher_target(const JitCodePatcher *p);

/* ---- Default vtable (no-op callbacks) ---- */

extern const JitCodePatcherVtable jit_code_patcher_default_vtable;

#ifdef __cplusplus
}
#endif
