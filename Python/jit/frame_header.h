// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "cinderx/python.h"

/* ---- C API (implemented in frame_header.c) ---- */
#if PY_VERSION_HEX >= 0x030C0000
#ifdef __cplusplus
extern "C" {
#endif

/*
 * Calculate frame header size for a code object.
 * frame_mode_lightweight: nonzero if FrameMode is kLightweight.
 * header_size: sizeof(FrameHeader).
 * frame_obj_size: sizeof(PyObject*).
 */
int jit_frame_header_size(PyCodeObject *code, int frame_mode_lightweight,
                          size_t header_size, size_t frame_obj_size);

#ifdef __cplusplus
} /* extern "C" */
#endif
#endif

/* ---- C++ convenience ---- */
#ifdef __cplusplus

#include "cinderx/Common/ref.h"

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
// Include config.h only in the .cpp files that call this, not here.
int frameHeaderSize(BorrowedRef<PyCodeObject> code);

} // namespace jit

#endif /* __cplusplus */
