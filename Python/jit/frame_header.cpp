// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// C++ wrapper for frameHeaderSize — delegates to pure C implementation
// with config values from getConfig().

#include "cinderx/Jit/frame_header.h"

#include "cinderx/Jit/config.h"

namespace jit {

#if PY_VERSION_HEX >= 0x030C0000

int frameHeaderSize(BorrowedRef<PyCodeObject> code) {
  return jit_frame_header_size(
      code,
      getConfig().frame_mode == FrameMode::kLightweight ? 1 : 0,
      sizeof(FrameHeader),
      sizeof(PyObject*));
}

#else

// Pre-3.12: shadow frame support — kept as C++ for legacy compatibility.
// This code is compiled out on 3.12+ (PY_VERSION_HEX >= 0x030C0000).

#include "internal/pycore_shadow_frame.h"

#include <cinderx/Common/code.h>
#include <cinderx/Common/util.h>

#include <unordered_set>
#include <vector>

int frameHeaderSize(BorrowedRef<PyCodeObject> code) {
  if (code->co_flags & kCoFlagsAnyGenerator) {
    return 0;
  }

#ifdef ENABLE_SHADOW_FRAMES
  return sizeof(FrameHeader);
#else
  return 0;
#endif
}

void assertShadowCallStackConsistent(PyThreadState* tstate) {
  // Pre-3.12 shadow frame assertion (not used on 3.12+).
  (void)tstate;
}

const char* shadowFrameKind(_PyShadowFrame* sf) {
  (void)sf;
  return "<unknown>";
}

#endif

} // namespace jit
