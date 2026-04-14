/*
 * code_patcher_c.c — Pure C code patcher implementation.
 *
 * Phase 3D: Replaces code_patcher.cpp.
 */

#include "cinderx/Jit/code_patcher_c.h"
#include "cinderx/Jit/codegen/arch/detection.h"

#include <assert.h>
#include <string.h>

/* ---- Platform-specific nop bytes ---- */

#if defined(CINDER_X86_64)
static const uint8_t kJmpNopBytes[] = {0x0f, 0x1f, 0x44, 0x00, 0x00};
#define JMP_NOP_SIZE 5
#elif defined(CINDER_AARCH64)
static const uint8_t kJmpNopBytes[] = {0x1f, 0x20, 0x03, 0xd5};
#define JMP_NOP_SIZE 4
#else
static const uint8_t kJmpNopBytes[] = {0x00};
#define JMP_NOP_SIZE 1
#endif

/* ---- Default vtable (all no-ops) ---- */

static void noop_callback(JitCodePatcher *self) { (void)self; }

const JitCodePatcherVtable jit_code_patcher_default_vtable = {
    .on_link = noop_callback,
    .on_patch = noop_callback,
    .on_unpatch = noop_callback,
    .destroy = NULL,
};

/* ---- Helpers ---- */

static uint32_t jump_displacement(uintptr_t from, uintptr_t to) {
    uintptr_t disp = to - from;
#if defined(CINDER_X86_64)
    disp -= JMP_NOP_SIZE;
#endif
    return (uint32_t)disp;
}

static uintptr_t resolve_displacement(uintptr_t from, uint32_t displacement) {
    uintptr_t disp = from + displacement;
#if defined(CINDER_X86_64)
    disp += JMP_NOP_SIZE;
#endif
    return disp;
}

/* ---- Swap (with optional atomic support for free-threaded) ---- */

static void patcher_swap(JitCodePatcher *p) {
    uint8_t temp[7];
    size_t len = p->flags.f.data_len;
    memcpy(temp, p->patchpoint, len);
    memcpy(p->patchpoint, p->data, len);
    memcpy(p->data, temp, len);

    /* Flush instruction cache after patching executable code.
     * On ARM64 the icache is not coherent with dcache.
     * On x86_64 this is a no-op (caches are coherent). */
    __builtin___clear_cache(
        (char *)p->patchpoint,
        (char *)p->patchpoint + len);
}

/* ---- Lifecycle ---- */

void jit_code_patcher_init(JitCodePatcher *p, const JitCodePatcherVtable *vt) {
    memset(p, 0, sizeof(*p));
    p->vtable = vt ? vt : &jit_code_patcher_default_vtable;
}

void jit_jump_patcher_init(JitCodePatcher *p, const JitCodePatcherVtable *vt) {
    jit_code_patcher_init(p, vt);
    memcpy(p->data, kJmpNopBytes, JMP_NOP_SIZE);
    p->flags.f.data_len = JMP_NOP_SIZE;
}

void jit_code_patcher_destroy(JitCodePatcher *p) {
    if (p->vtable && p->vtable->destroy) {
        p->vtable->destroy(p);
    }
}

/* ---- Operations ---- */

void jit_code_patcher_link(JitCodePatcher *p, uintptr_t patchpoint,
                           const uint8_t *data, size_t data_len) {
    assert(!jit_code_patcher_is_linked(p) && "Trying to re-link a patcher");
    assert(data_len <= sizeof(p->data) && "Data too large for patcher");

    p->patchpoint = (uint8_t *)patchpoint;
    memcpy(p->data, data, data_len);
    p->flags.f.data_len = (uint8_t)data_len;

    if (p->vtable->on_link) {
        p->vtable->on_link(p);
    }
}

void jit_code_patcher_patch(JitCodePatcher *p) {
    assert(jit_code_patcher_is_linked(p) && "Trying to patch unlinked patcher");
    patcher_swap(p);
    p->flags.f.is_patched = 1;
    if (p->vtable->on_patch) {
        p->vtable->on_patch(p);
    }
}

void jit_code_patcher_unpatch(JitCodePatcher *p) {
    assert(jit_code_patcher_is_linked(p) && "Trying to unpatch unlinked patcher");
    patcher_swap(p);
    p->flags.f.is_patched = 0;
    if (p->vtable->on_unpatch) {
        p->vtable->on_unpatch(p);
    }
}

/* ---- Queries ---- */

int jit_code_patcher_is_linked(const JitCodePatcher *p) {
    return p->patchpoint != NULL;
}

int jit_code_patcher_is_patched(const JitCodePatcher *p) {
    return p->flags.f.is_patched;
}

uint8_t *jit_code_patcher_patchpoint(const JitCodePatcher *p) {
    return p->patchpoint;
}

const uint8_t *jit_code_patcher_stored_bytes(const JitCodePatcher *p,
                                              size_t *out_len) {
    if (out_len) *out_len = p->flags.f.data_len;
    return p->data;
}

/* ---- Jump patcher ---- */

void jit_jump_patcher_link_jump(JitCodePatcher *p, uintptr_t patchpoint,
                                uintptr_t jump_target) {
    uint32_t disp = jump_displacement(patchpoint, jump_target);
    uint8_t buf[JMP_NOP_SIZE];
    memset(buf, 0, sizeof(buf));

#if defined(CINDER_X86_64)
    buf[0] = 0xe9;  /* 32-bit relative jump */
    memcpy(buf + 1, &disp, sizeof(uint32_t));
#elif defined(CINDER_AARCH64)
    assert(disp % 4 == 0 && "Jump displacement must be multiple of 4");
    disp /= 4;
    uint32_t insn = 0x14000000 | disp;
    memcpy(buf, &insn, sizeof(uint32_t));
#endif

    jit_code_patcher_link(p, patchpoint, buf, JMP_NOP_SIZE);
}

uint8_t *jit_jump_patcher_target(const JitCodePatcher *p) {
    assert(jit_code_patcher_is_linked(p) &&
           "Can't compute jump target before linking");

    size_t len;
    const uint8_t *bytes = jit_code_patcher_stored_bytes(p, &len);
    assert(len == JMP_NOP_SIZE);

    uint32_t disp = 0;

#if defined(CINDER_X86_64)
    memcpy(&disp, bytes + 1, len - 1);
#elif defined(CINDER_AARCH64)
    memcpy(&disp, bytes, len);
    disp &= 0x03ffffff;
    if (disp & 0x02000000) {
        disp |= 0xfc000000;  /* sign-extend 26-bit to 32-bit */
    }
    disp *= 4;
#endif

    return (uint8_t *)resolve_displacement((uintptr_t)p->patchpoint, disp);
}
