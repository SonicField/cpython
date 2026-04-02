/*
 * frame_header.c -- Frame header size calculation (pure C, 3.12+)
 *
 * Phase 3D conversion: frame_header.cpp -> frame_header.c
 * On 3.12+, only frameHeaderSize() is needed.
 * Pre-3.12 shadow frame code is dropped (PY_VERSION_HEX < 0x030C0000).
 */

#include "cinderx/Jit/frame_header.h"

#if PY_VERSION_HEX >= 0x030C0000

int
jit_frame_header_size(PyCodeObject *code, int frame_mode_lightweight,
                      size_t header_size, size_t frame_obj_size)
{
    int co_flags_any_gen =
        CO_ASYNC_GENERATOR | CO_COROUTINE | CO_GENERATOR | CO_ITERABLE_COROUTINE;

    if (code->co_flags & co_flags_any_gen) {
        return 0;
    }

    if (frame_mode_lightweight) {
        return (int)(header_size + frame_obj_size * code->co_framesize);
    }

    return 0;
}

#endif
