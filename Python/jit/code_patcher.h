// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <array>
#include <bit>
#include <cstdint>
#include <span>

namespace jit {

// A CodePatcher is used by the runtime to overwrite parts of compiled code.
// Often times this is used to patch in a jump to a deopt exit when an invariant
// that the compiled code relies on is invalidated. It is intended to be used in
// conjunction with the DeoptPatchpoint HIR instruction.
//
// Using a CodePatcher looks roughly like:
//   1. Allocate a CodePatcher.
//
//   2. Allocate a DeoptPatchpoint HIR instruction linked to the CodePatcher
//      from (1) and insert it into the appropriate point in the HIR
//      instruction stream.
//
//   3. Link the CodePatcher from (1) to the appropriate address in the
//      generated code after code generation is complete.
//
// A CodePatcher is only valid for as long as the compiled code to which it is
// linked is alive, so care must be taken not to call `patch()` after the code
// has been destroyed.
class CodePatcher {
 public:
  virtual ~CodePatcher() = default;

  // Link the patcher to a specific location in generated code. This is
  // intended to be called by the JIT after code has been generated but before
  // it is active.
  //
  // `patchpoint` contains the address of the first byte of the patchpoint.
  // `data` contains the bytes that will be written on patching.
  void link(uintptr_t patchpoint, std::span<const uint8_t> data);

  // Overwrite the patchpoint.
  //
  // The patcher must be linked before this can be called.
  void patch();

  // Revert the patchpoint back to a nop.
  //
  // The patcher must be linked before this can be called.
  void unpatch();

  bool isLinked() const { return patchpoint_ != nullptr; }
  bool isPatched() const { return flags_.is_patched; }
  uint8_t* patchpoint() const { return patchpoint_; }
  std::span<const uint8_t> storedBytes() const {
    return std::span{data_.data(), flags_.data_len};
  }

 protected:
  // Callback to execute after linking (e.g. subscribing to changes).
  virtual void onLink() {}

  // Callback to execute after patching (e.g. cleaning up the patcher).
  virtual void onPatch() {}

  // Callback to execute after unpatching.
  virtual void onUnpatch() {}

  // Swap data between this object and the actual patchpoint.
  void swap();

  // Where in the code we should patch.
  uint8_t* patchpoint_{nullptr};

  // Data that's written into the patch point.  This is swapped with what's
  // already there, so that this can continuously patch and unpatch.
  //
  // The size of the array here is the total capacity, not necessarily all of it
  // will be patched.
  std::array<uint8_t, 7> data_{};

  // RAII lock guard for swap operations.
  class [[nodiscard]] SwapLockGuard {
   public:
    explicit SwapLockGuard(CodePatcher& patcher);
    ~SwapLockGuard();
    SwapLockGuard(const SwapLockGuard&) = delete;
    SwapLockGuard& operator=(const SwapLockGuard&) = delete;

   private:
    CodePatcher& patcher_;
  };

  // Bit packed because we don't want to bloat CodePatchers.
  struct Flags {
    uint8_t data_len : 3;
    bool is_patched : 1;
    bool lock : 1;
  };
  union {
    Flags flags_{};
    uint8_t flags_byte_;
  };

  // Avoids hardcoding the bit-pattern to access Flags::lock.
  static constexpr uint8_t lockBit() {
    Flags f{};
    f.lock = true;
    return std::bit_cast<uint8_t>(f);
  }
};

// Subclass of a CodePatcher that is intended for patching in jumps.
class JumpPatcher : public CodePatcher {
 public:
  JumpPatcher();
  ~JumpPatcher() override = default;

  // Specific form of link() for handling jumps.
  //
  // NB: The distance between the patchpoint and the jump target must fit into a
  // signed 32-bit int.
  void linkJump(uintptr_t patchpoint, uintptr_t jump_target);

  // Get the jump target of this patcher.
  uint8_t* jumpTarget() const;
};

} // namespace jit
