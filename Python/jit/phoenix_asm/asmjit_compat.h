/*
 * asmjit_compat.h -- Namespace compatibility layer
 *
 * When PHOENIX_ASM is defined, this header provides the asmjit:: namespace
 * types that codegen files reference directly (not through arch.h aliases).
 * This allows gen_asm.cpp, autogen.cpp, etc. to compile unchanged.
 *
 * Include this INSTEAD of <asmjit/asmjit.h> when using phoenix-asm.
 */

#ifndef PHOENIX_ASM_ASMJIT_COMPAT_H
#define PHOENIX_ASM_ASMJIT_COMPAT_H

/* Ensure architecture detection macros are defined before including
 * the wrapper, which uses #if defined(CINDER_X86_64). Without this,
 * if asmjit_compat.h is included before arch.h's detection.h, the
 * architecture-specific namespaces won't be created. */
#include "cinderx/Jit/codegen/arch/detection.h"

#include "phoenix_asm_wrapper.h"
#include "alloc.h"

namespace asmjit {

/* Core types */
using Error = phx::Error;
using Label = phx::Label;
using Imm = phx::Imm;
using BaseNode = phx::BaseNode;
using CodeHolder = phx::CodeHolder;
using BaseReg = phx::Gp;
using BaseMem = phx::Mem;
using BaseBuilder = phx::Builder;
using BaseEmitter = phx::Builder;

constexpr auto kErrorOk = phx::kErrorOk;
constexpr Error kErrorNotInitialized = 1;

/* AlignMode */
using AlignMode = phx::AlignMode;

/* Environment stub — used by code_allocator for arch detection */
struct Environment {
  uint32_t arch() const { return 0; }
};

/* Section — maps to CodeHolder's nested Section type */

#if defined(CINDER_X86_64)

namespace x86 {

/* Types */
using Builder = phx::Builder;
using Emitter = phx::Builder;
using Gp = phx::Gp;
using Gpq = phx::Gp;  /* 64-bit GP register type alias */
using Mem = phx::Mem;
using Reg = phx::Gp;
using Xmm = phx::Xmm;

template <typename T>
using EmitterExplicitT = phx::EmitterExplicitT<T>;

/* Register factory functions */
using phx::gpb;
using phx::gpw;
using phx::gpd;
using phx::gpq;
using phx::xmm;
using phx::ptr;
using phx::byte_ptr;
using phx::word_ptr;
using phx::dword_ptr;
using phx::qword_ptr;
using phx::dqword_ptr;

/* Register constants */
using namespace phx::x86;

} /* namespace x86 */

#elif defined(CINDER_AARCH64)

namespace a64 {

using Builder = phx::Builder;
using Emitter = phx::Builder;
using Gp = phx::Gp;
using Mem = phx::Mem;
using Reg = phx::Gp;
using Vec = phx::Vec;

template <typename T>
using EmitterExplicitT = phx::EmitterExplicitT<T>;

/* Register factory functions */
using phx::w;
using phx::x;
using phx::ptr;

/* ARM64 ptr overloads */
inline phx::Mem ptr(const phx::Gp& base, const phx::Gp& index) {
  return phx::Mem(phx_ptr_index(base, index, 1, 0));
}

/* Pre/post-indexed addressing modes */
inline phx::Mem ptr_pre(const phx::Gp& base, int32_t offset) {
  /* Pre-indexed: [base, #offset]! — base updated before access */
  phx::Mem m = phx::ptr(base, offset);
  m.mem().is_pre_index = 1;
  return m;
}
inline phx::Mem ptr_post(const phx::Gp& base, int32_t offset) {
  /* Post-indexed: [base], #offset — base updated after access */
  phx::Mem m = phx::ptr(base, offset);
  m.mem().is_post_index = 1;
  return m;
}

/* SIMD/FP register factory + constants — Vec = Gp on ARM64 (needs id+size) */
#ifdef CINDER_AARCH64
inline phx::Vec d(uint8_t id) { return phx::Vec(id, 8); }
constexpr phx::Vec d0{0,8}, d1{1,8}, d2{2,8}, d3{3,8}, d4{4,8}, d5{5,8}, d6{6,8}, d7{7,8};
#else
inline phx::Vec d(uint8_t id) { return phx::Vec(id); }
constexpr phx::Vec d0{0}, d1{1}, d2{2}, d3{3}, d4{4}, d5{5}, d6{6}, d7{7};
#endif

/* Shift specification */
using phx::lsl;
using phx::lsr;
using phx::asr;

/* 32-bit register constants */
constexpr phx::Gp w0{0,4}, w1{1,4}, w2{2,4}, w3{3,4}, w4{4,4};
constexpr phx::Gp w8{8,4}, w9{9,4}, w10{10,4};

/* Special registers */
constexpr phx::Gp xzr{31, 8};  /* zero register (reads as 0, writes discarded) */

/* ARM64 Utils (for immediate range checks) */
struct Utils {
  static bool isAddSubImm(uint64_t val) {
    return val <= 0xFFF || (val <= 0xFFF000 && (val & 0xFFF) == 0);
  }
};

/* System register constants */
namespace Predicate { namespace SysReg {
constexpr uint16_t kTPIDR_EL0 = 0xDE82; /* Thread pointer EL0 */
}}

/* Register constants */
using namespace phx::a64;

} /* namespace a64 */

namespace arm {
using Mem = phx::Mem;
using Shift = phx::Shift;

/* Condition codes for ARM64.
 * Used as arm::CondCode::kEQ etc. in autogen.cpp.
 * Must be implicitly convertible to uint32_t for csel/cset. */
namespace CondCode {
  constexpr uint32_t kEQ = 0, kNE = 1, kCS = 2, kHS = 2, kCC = 3, kLO = 3;
  constexpr uint32_t kMI = 4, kPL = 5, kVS = 6, kVC = 7, kHI = 8, kLS = 9;
  constexpr uint32_t kGE = 10, kLT = 11, kGT = 12, kLE = 13, kAL = 14;
}

struct Utils {
  static bool isAddSubImm(uint64_t val) {
    return val <= 0xFFF || (val <= 0xFFF000 && (val & 0xFFF) == 0);
  }
  static bool isLogicalImm(uint64_t val, uint32_t width) {
    if (val == 0) return false;
    if (width == 32) val = (val & 0xFFFFFFFF) | (val << 32);
    if (val == ~(uint64_t)0) return false;
    for (uint32_t size = 2; size <= 64; size <<= 1) {
      uint64_t mask = (~(uint64_t)0) >> (64 - size);
      uint64_t elem = val & mask;
      uint64_t rep = 0;
      for (uint32_t i = 0; i < 64; i += size) rep |= (elem << i);
      if (rep != val) continue;
      if (elem == 0 || elem == mask) continue;
      uint64_t transitions = elem ^ ((elem >> 1) | ((elem & 1) << (size - 1)));
      if (__builtin_popcountll(transitions) == 2) return true;
    }
    return false;
  }
};
} /* namespace arm */

/* a64::CondCode and a64::VecD aliases — must come after arm namespace */
namespace a64 {
  namespace CondCode = arm::CondCode;
  using VecD = phx::Gp;
}

#endif /* architecture */

/* String stub — used in debug/error formatting paths */
class String {
 public:
  String() = default;
  const char* data() const { return ""; }
  size_t size() const { return 0; }
  void clear() {}
};

/* FormatOptions stub */
struct FormatOptions {
  uint32_t flags() const { return 0; }
  void setFlags(uint32_t) {}
};

/* Section type alias — maps to CodeHolder's nested Section */
using Section = phx::CodeHolder::Section;

/* FormatFlags stub */
namespace FormatFlags {
constexpr uint32_t kNone = 0;
constexpr uint32_t kHexImms = 1;
constexpr uint32_t kMachineCode = 2;
}

/* Formatter stub — debug disassembly, not critical */
struct Formatter {
  template <typename... Args>
  static Error formatNode(Args&&...) { return kErrorOk; }
  template <typename... Args>
  static Error formatNodeList(Args&&...) { return kErrorOk; }
};

/* DebugUtils stub */
struct DebugUtils {
  static const char* errorAsString(Error err) {
    return err == kErrorOk ? "Ok" : "Error";
  }
};

/* ErrorHandler stub */
class ErrorHandler {
 public:
  virtual ~ErrorHandler() = default;
  virtual void handleError(Error err, const char* msg, BaseEmitter* origin) = 0;
};

/* Support utility stubs */
struct Support {
  static size_t alignUp(size_t value, size_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
  }
};

/* JitAllocator stub — wraps phoenix-asm's alloc.c */
class JitAllocator {
 public:
  struct Span {
    void* rx;
    void* rw;
    size_t _size;
    size_t size() const { return _size; }
  };

  struct Stats {
    size_t usedSize() const { return 0; }
    size_t reservedSize() const { return 0; }
    size_t overheadSize() const { return 0; }
    size_t blockCount() const { return 0; }
  };

  Error alloc(Span& span, size_t size) {
    (void)size;
    span = {nullptr, nullptr, 0};
    return kErrorOk;
  }

  Error release(void* ptr) {
    (void)ptr;
    return kErrorOk;
  }

  Error query(Stats& stats) const {
    stats = {};
    return kErrorOk;
  }
  /* query overloads — asmjit uses multiple forms */
  Error query(Stats& stats, const void*) const { stats = {}; return kErrorOk; }
  Error query(Span& span, size_t) const { span = {}; return kErrorOk; }
  Error query(Span& span, const void*) const { span = {}; return kErrorOk; }
};

/* JitRuntime — wraps phoenix-asm's PhxRuntime (alloc.c).
 * Handles: finalize code → allocate RWX memory → copy → icache flush. */
class JitRuntime {
 public:
  JitRuntime() : runtime_(phx_runtime_create()) {}
  ~JitRuntime() { if (runtime_) phx_runtime_destroy(runtime_); }

  JitRuntime(const JitRuntime&) = delete;
  JitRuntime& operator=(const JitRuntime&) = delete;

  /* Add compiled code: finalize the builder, copy to executable memory.
   * This is the main entry point called by the JIT after code generation. */
  Error add(void** dst, CodeHolder* code) {
    if (!runtime_ || !code || !code->get()) {
      *dst = nullptr;
      return kErrorNotInitialized;
    }
    PhxCodeHolder* phx_code = code->get();
    if (!phx_code->buffer || phx_code->buffer_size == 0) {
      *dst = nullptr;
      return kErrorNotInitialized;
    }
    size_t out_size = 0;
    void* addr = phx_runtime_add(runtime_, phx_code->buffer,
                                 phx_code->buffer_size, &out_size);
    *dst = addr;
    return addr ? kErrorOk : kErrorNotInitialized;
  }

  template <typename Func>
  Error _add(Func* dst, CodeHolder* code) {
    void* addr = nullptr;
    Error err = add(&addr, code);
    if (err == kErrorOk) {
      *dst = (Func)(uintptr_t)addr;
    }
    return err;
  }

  Error release(void* ptr) {
    if (runtime_) phx_runtime_release(runtime_, ptr);
    return kErrorOk;
  }

  JitAllocator* allocator() { return &allocator_; }
  const JitAllocator* allocator() const { return &allocator_; }
  const Environment& environment() const { return env_; }

 private:
  PhxRuntime* runtime_;
  JitAllocator allocator_;
  Environment env_;
};

} /* namespace asmjit */

#endif /* PHOENIX_ASM_ASMJIT_COMPAT_H */
