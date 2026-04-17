// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "cinderx/Common/ref.h"
#include "cinderx/Jit/debug_info.h"
#include "cinderx/Jit/deopt.h"
#include "cinderx/Jit/threaded_compile.h"

#include <deque>
#include <limits>
#include <unordered_set>

namespace jit {

constexpr ptrdiff_t kInvalidYieldFromOffset =
    std::numeric_limits<ptrdiff_t>::max();

// Information about how a specific yield instruction should resume.
class GenYieldPoint {
 public:
  static constexpr int resumeTargetOffset() {
    return offsetof(GenYieldPoint, resume_target_);
  }

  GenYieldPoint(std::size_t deopt_idx, ptrdiff_t yield_from_offset)
      : deopt_idx_{deopt_idx}, yield_from_offset_{yield_from_offset} {}

  uintptr_t resumeTarget() const { return resume_target_; }
  void setResumeTarget(uintptr_t resume_target) { resume_target_ = resume_target; }
  std::size_t deoptIdx() const { return deopt_idx_; }
  bool isYieldFrom() const { return yield_from_offset_ != kInvalidYieldFromOffset; }
  ptrdiff_t yieldFromOffset() const { return yield_from_offset_; }

 private:
  uintptr_t resume_target_{0};
  const std::size_t deopt_idx_;
  const ptrdiff_t yield_from_offset_;
};

class alignas(16) RuntimeFrameState {
 public:
  static constexpr int64_t codeOffset() {
    return offsetof(RuntimeFrameState, code_);
  }

  RuntimeFrameState(
      PyCodeObject* code,
      PyDictObject* builtins,
      PyDictObject* globals,
      PyFunctionObject* func = nullptr)
      : code_{code}, builtins_{builtins}, globals_{globals}, func_{func} {}

  bool isGen() const { return code()->co_flags & kCoFlagsAnyGenerator; }
  PyCodeObject* code() const { return code_; }
  PyDictObject* builtins() const { return builtins_; }
  PyDictObject* globals() const { return globals_; }
  PyFunctionObject* func() const { return func_; }

 private:
  // All fields are owned by the CodeRuntime that owns this RuntimeFrameState.

  PyCodeObject* code_;
  PyDictObject* builtins_;
  PyDictObject* globals_;
  // The function is only set for inlined frames.
  PyFunctionObject* func_;
};

// Runtime data for a PyCodeObject object, containing caches and any other data
// associated with a JIT-compiled function.
class alignas(16) CodeRuntime {
 public:
  static constexpr int64_t frameStateOffset() {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Winvalid-offsetof"
    return offsetof(CodeRuntime, frame_state_);
#pragma GCC diagnostic pop
  }

  static constexpr int64_t codeOffset() {
    return CodeRuntime::frameStateOffset() + RuntimeFrameState::codeOffset();
  }

  explicit CodeRuntime(PyFunctionObject* func)
      : CodeRuntime{
            (PyCodeObject*)func->func_code,
            (PyDictObject*)func->func_builtins,
            (PyDictObject*)func->func_globals} {}

  CodeRuntime(
      PyCodeObject* code,
      PyDictObject* builtins,
      PyDictObject* globals)
      : frame_state_{code, builtins, globals} {
    addReference((PyObject*)code);
    addReference((PyObject*)builtins);
    addReference((PyObject*)globals);
  }

  template <typename... Args>
  RuntimeFrameState* allocateRuntimeFrameState(Args&&... args) {
    return inlined_frame_states_
        .emplace_back(
            std::make_unique<RuntimeFrameState>(std::forward<Args>(args)...))
        .get();
  }

  // Ensure that this CodeRuntime owns a reference to the given borrowed
  // object, keeping it alive for use by the compiled code. Make CodeRuntime a
  // new owner of the object.
  void addReference(PyObject* obj) {
    ThreadedCompileSerialize guard;
    references_.emplace(ThreadedRef<>::create(obj));
  }

  void releaseReferences() {
    ThreadedCompileSerialize guard;
    references_.clear();
#if PY_VERSION_HEX >= 0x030E0000 && defined(ENABLE_LIGHTWEIGHT_FRAMES)
    reifier_.reset(nullptr);
#endif
  }

  GenYieldPoint* addGenYieldPoint(GenYieldPoint&& gen_yield_point) {
    gen_yield_points_.emplace_back(std::move(gen_yield_point));
    return &gen_yield_points_.back();
  }

  std::size_t addDeoptMetadata(DeoptMetadata&& deopt_meta) {
    deopt_metadatas_.emplace_back(std::move(deopt_meta));
    return deopt_metadatas_.size() - 1;
  }

  DeoptMetadata& getDeoptMetadata(std::size_t id) {
    return deopt_metadatas_[id];
  }
  const DeoptMetadata& getDeoptMetadata(std::size_t id) const {
    return deopt_metadatas_[id];
  }

  const std::vector<DeoptMetadata>& deoptMetadatas() const {
    return deopt_metadatas_;
  }

  const RuntimeFrameState* frameState() const { return &frame_state_; }

  int frameSize() const { return frame_size_; }
  void setFrameSize(int size) { frame_size_ = size; }
#if defined(__aarch64__)
  int savedIpFpOffset() const { return saved_ip_fp_offset_; }
  void setSavedIpFpOffset(int offset) { saved_ip_fp_offset_ = offset; }
#endif

  DebugInfo* debugInfo() { return &debug_info_; }

#if PY_VERSION_HEX >= 0x030E0000 && defined(ENABLE_LIGHTWEIGHT_FRAMES)
  void setReifier(PyObject* reifier) {
    ThreadedCompileSerialize guard;
    reifier_ = ThreadedRef<>::create(reifier);
  }
  PyObject* reifier() {
    return reifier_;
  }
#else
  PyObject* reifier() {
    return nullptr;
  }
#endif
 private:
  RuntimeFrameState frame_state_;
  std::vector<std::unique_ptr<RuntimeFrameState>> inlined_frame_states_;

  // References owned by this CodeRuntime.
  std::unordered_set<ThreadedRef<PyObject>> references_;

  // Metadata about yield points. Deque so we can have raw pointers to content.
  std::deque<GenYieldPoint> gen_yield_points_;

  // Metadata about deopt points.  Safe to use a vector as these are always
  // accessed by index.
  std::vector<DeoptMetadata> deopt_metadatas_;

#if PY_VERSION_HEX >= 0x030E0000 && defined(ENABLE_LIGHTWEIGHT_FRAMES)
  ThreadedRef<> reifier_;
#endif

  int frame_size_{-1};
#if defined(__aarch64__)
  int saved_ip_fp_offset_{0};
#endif
  DebugInfo debug_info_;
};

} // namespace jit
