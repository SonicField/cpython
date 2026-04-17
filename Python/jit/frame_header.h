// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "cinderx/python.h"

/* ---- C API (static inline for zero overhead on hot paths) ---- */
#if PY_VERSION_HEX >= 0x030C0000

#define _JIT_CO_FLAGS_ANY_GEN \
    (CO_ASYNC_GENERATOR | CO_COROUTINE | CO_GENERATOR | CO_ITERABLE_COROUTINE)

static inline int
jit_frame_header_size(PyCodeObject *code, int frame_mode_lightweight,
                      size_t header_size, size_t frame_obj_size)
{
    if (code->co_flags & _JIT_CO_FLAGS_ANY_GEN) {
        return 0;
    }
    if (frame_mode_lightweight) {
        return (int)(header_size + frame_obj_size * code->co_framesize);
    }
    return 0;
}

#endif

/* ---- C++ convenience ---- */
#ifdef __cplusplus

#include "cinderx/Common/ref.h"
#include "cinderx/Jit/jit_config_c.h"

namespace jit {

#if PY_VERSION_HEX < 0x030C0000

#include "internal/pycore_shadow_frame_struct.h"

// FrameHeader lives at the beginning of the stack frame for JIT-compiled
// functions. Note these will be garbage in generator objects.
struct FrameHeader {
  JITShadowFrame shadow_frame;
};

void assertShadowCallStackConsistent(PyThreadState* tstate);

const char* shadowFrameKind(_PyShadowFrame* sf);

#else

// FrameHeader lives at the beginning of the stack frame for JIT-compiled
// functions. In 3.12+ this will be followed by the _PyInterpreterFrame.
struct FrameHeader {
  union {
    PyFunctionObject* func;
    uintptr_t rtfs;
  };
};

#define JIT_FRAME_RTFS 0x01
#define JIT_FRAME_INITIALIZED 0x02
#define JIT_FRAME_MASK 0x03

#endif

// C++ wrapper — delegates to C implementation with config values.
inline int frameHeaderSize(PyCodeObject* code) {
  return jit_frame_header_size(
      code,
      jit_get_config()->frame_mode == JIT_FRAME_LIGHTWEIGHT ? 1 : 0,
      sizeof(FrameHeader),
      sizeof(PyObject*));
}

} // namespace jit

#endif /* __cplusplus */
