// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// Phase 3D: Reduced — GenYieldPoint + RuntimeFrameState methods inlined
// to code_runtime.h. Remaining: CodeRuntime methods (ThreadedCompileSerialize,
// std::vector, ThreadedRef).

#include "cinderx/Jit/code_runtime.h"

#include "cinderx/Common/util.h"

namespace jit {

CodeRuntime::CodeRuntime(BorrowedRef<PyFunctionObject> func)
    : CodeRuntime{
          BorrowedRef<PyCodeObject>{func->func_code},
          func->func_builtins,
          func->func_globals} {}

CodeRuntime::CodeRuntime(
    BorrowedRef<PyCodeObject> code,
    BorrowedRef<PyDictObject> builtins,
    BorrowedRef<PyDictObject> globals)
    : frame_state_{code, builtins, globals} {
  addReference(code);
  addReference(builtins);
  addReference(globals);
}

void CodeRuntime::addReference(BorrowedRef<> obj) {
  ThreadedCompileSerialize guard;
  references_.emplace(ThreadedRef<>::create(obj));
}

void CodeRuntime::releaseReferences() {
  ThreadedCompileSerialize guard;
  references_.clear();
#if PY_VERSION_HEX >= 0x030E0000 && defined(ENABLE_LIGHTWEIGHT_FRAMES)
  reifier_.reset(nullptr);
#endif
}

GenYieldPoint* CodeRuntime::addGenYieldPoint(GenYieldPoint&& gen_yield_point) {
  gen_yield_points_.emplace_back(std::move(gen_yield_point));
  return &gen_yield_points_.back();
}

std::size_t CodeRuntime::addDeoptMetadata(DeoptMetadata&& deopt_meta) {
  deopt_metadatas_.emplace_back(std::move(deopt_meta));
  return deopt_metadatas_.size() - 1;
}

DeoptMetadata& CodeRuntime::getDeoptMetadata(std::size_t id) {
  return deopt_metadatas_[id];
}

const DeoptMetadata& CodeRuntime::getDeoptMetadata(std::size_t id) const {
  return deopt_metadatas_[id];
}

const std::vector<DeoptMetadata>& CodeRuntime::deoptMetadatas() const {
  return deopt_metadatas_;
}

const RuntimeFrameState* CodeRuntime::frameState() const {
  return &frame_state_;
}

int CodeRuntime::frameSize() const {
  return frame_size_;
}

void CodeRuntime::setFrameSize(int size) {
  frame_size_ = size;
}

#if defined(__aarch64__)
int CodeRuntime::savedIpFpOffset() const {
  return saved_ip_fp_offset_;
}

void CodeRuntime::setSavedIpFpOffset(int offset) {
  saved_ip_fp_offset_ = offset;
}
#endif

DebugInfo* CodeRuntime::debugInfo() {
  return &debug_info_;
}

} // namespace jit
