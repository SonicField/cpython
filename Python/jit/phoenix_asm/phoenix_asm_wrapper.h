/*
 * phoenix_asm_wrapper.h -- C++ wrapper mapping phoenix-asm C API
 * to asmjit-compatible types for the JIT codegen layer.
 *
 * This is Chunk 6 of the phoenix-asm implementation. It provides:
 *   - Gp, Mem, Label, Imm, VecD (Xmm) operand types
 *   - Builder class with instruction methods
 *   - EmitterExplicitT<Builder> CRTP base for autogen.cpp compatibility
 *   - Cursor management (BaseNode*) for deferred assembly
 *   - Factory functions (gpb, gpw, gpd, gpq, ptr, xmm, etc.)
 *
 * The arch.h header maps arch::Builder, arch::Gp, etc. to these types,
 * so codegen files (gen_asm.cpp, frame_asm.cpp, autogen.cpp) are unchanged.
 */

#ifndef PHOENIX_ASM_WRAPPER_H
#define PHOENIX_ASM_WRAPPER_H

#include "phoenix_asm.h"

#ifdef __cplusplus

#include <cstdint>
#include <cstddef>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <type_traits>

/* ================================================================== */
/*  Forward declarations                                               */
/* ================================================================== */

namespace phx {

class Gp;
class Mem;
class Label;
class Imm;
class Xmm;
class Builder;

/* Error type (compatible with asmjit::Error) */
using Error = int;
constexpr Error kErrorOk = 0;

/* BaseNode — opaque handle for cursor management */
using BaseNode = PhxNode;

/* AlignMode */
enum class AlignMode : uint32_t {
  kCode = 0,
  kData = 1,
};

/* ================================================================== */
/*  Gp — General Purpose Register                                      */
/* ================================================================== */

class Gp {
 public:
  constexpr Gp() : gp_{0, 0} {}
  constexpr Gp(uint8_t id, uint8_t size) : gp_{id, size} {}
  constexpr explicit Gp(PhxGp gp) : gp_(gp) {}

  constexpr uint8_t id() const { return gp_.id; }
  constexpr uint32_t size() const { return gp_.size; }
  constexpr bool isGp() const { return gp_.size <= 8; }
  constexpr bool isGpW() const { return gp_.size == 4; }
  constexpr bool isGpX() const { return gp_.size == 8; }
  constexpr bool isGpq() const { return gp_.size == 8; }
  constexpr bool isVec() const { return gp_.size > 8; }
  constexpr bool isVecD() const { return gp_.size == 8; }
  constexpr bool isXmm() const { return gp_.size == 16; }

  /* Size conversion — asmjit compatibility */
  constexpr Gp r8() const { return Gp(gp_.id, 1); }
  constexpr Gp r16() const { return Gp(gp_.id, 2); }
  constexpr Gp r32() const { return Gp(gp_.id, 4); }
  constexpr Gp r64() const { return Gp(gp_.id, 8); }

  /* ARM64 aliases */
  constexpr Gp w() const { return Gp(gp_.id, 4); }
  constexpr Gp x() const { return Gp(gp_.id, 8); }

  /* Implicit conversion to PhxGp for C API calls */
  constexpr operator PhxGp() const { return gp_; }


  constexpr bool operator==(const Gp& o) const {
    return gp_.id == o.gp_.id && gp_.size == o.gp_.size;
  }
  constexpr bool operator!=(const Gp& o) const { return !(*this == o); }

 private:
  PhxGp gp_;
};

/* Factory functions (asmjit::x86 namespace compatibility) */
constexpr Gp gpb(uint8_t id) { return Gp(id, 1); }
constexpr Gp gpw(uint8_t id) { return Gp(id, 2); }
constexpr Gp gpd(uint8_t id) { return Gp(id, 4); }
constexpr Gp gpq(uint8_t id) { return Gp(id, 8); }

/* ARM64 factory functions */
constexpr Gp w(uint8_t id) { return Gp(id, 4); }
constexpr Gp x(uint8_t id) { return Gp(id, 8); }

/* ================================================================== */
/*  Mem — Memory Operand                                               */
/* ================================================================== */

class Mem {
 public:
  constexpr Mem() : mem_{} {}
  constexpr explicit Mem(PhxMem m) : mem_(m) {}
  /* Absolute offset memory operand (used for TLS access via FS segment) */
  explicit Mem(int32_t offset) : mem_{} { mem_.offset = offset; mem_.size = 8; }
  /* Base + offset constructor (ARM64 arm::Mem(Xn, offset) compatibility) */
  Mem(const Gp& base, int32_t offset) : mem_(phx_ptr(base, offset)) {}

  /* Base register */
  constexpr Gp baseReg() const { return Gp(mem_.base); }
  constexpr bool hasBase() const { return mem_.base.size != 0; }

  /* Index register (x86 SIB) */
  constexpr bool hasIndex() const { return mem_.has_index; }
  constexpr Gp indexReg() const { return Gp(mem_.index); }
  constexpr uint8_t scale() const { return mem_.scale; }

  /* Offset */
  constexpr int32_t offset() const { return mem_.offset; }
  void setOffset(int32_t off) { mem_.offset = off; }
  void addOffset(int32_t delta) { mem_.offset += delta; }

  /* Access size */
  constexpr uint8_t size() const { return mem_.size; }
  void setSize(uint8_t s) { mem_.size = s; }

  /* Segment override (x86 FS/GS for TLS access) */
  void setSegment(uint8_t seg) { mem_.segment = seg; }

  /* Implicit conversion to PhxMem for C API calls */
  constexpr operator PhxMem() const { return mem_; }

 private:
  PhxMem mem_;
};

/* Memory operand construction — asmjit::x86::ptr compatibility */
inline Mem ptr(const Gp& base) {
  return Mem(phx_ptr(base, 0));
}

inline Mem ptr(const Gp& base, int32_t offset) {
  return Mem(phx_ptr(base, offset));
}

inline Mem ptr(const Gp& base, int32_t offset, uint32_t size) {
  PhxMem m = phx_ptr(base, offset);
  m.size = static_cast<uint8_t>(size);
  return Mem(m);
}

inline Mem ptr(const Gp& base, const Gp& index, uint8_t shift) {
  return Mem(phx_ptr_index(base, index, 1u << shift, 0));
}

inline Mem ptr(const Gp& base, const Gp& index, uint8_t shift, int32_t offset) {
  return Mem(phx_ptr_index(base, index, 1u << shift, offset));
}

/* Absolute address memory operand — encoded as RIP-relative, patched
   by phx_relocate_to_base once the final code address is known. */
inline Mem ptr(uint64_t addr) {
  PhxMem m = {};
  m.offset = 0; /* placeholder — patched during relocation */
  m.size = 8;
  m.is_abs_addr = 1;
  m.abs_addr = addr;
  return Mem(m);
}

/* Sized pointer constructors */
inline Mem byte_ptr(const Gp& base, int32_t offset = 0) {
  return ptr(base, offset, 1);
}

inline Mem word_ptr(const Gp& base, int32_t offset = 0) {
  return ptr(base, offset, 2);
}

inline Mem dword_ptr(const Gp& base, int32_t offset = 0) {
  return ptr(base, offset, 4);
}

inline Mem qword_ptr(const Gp& base, int32_t offset = 0) {
  return ptr(base, offset, 8);
}

inline Mem dqword_ptr(const Gp& base, int32_t offset = 0) {
  return ptr(base, offset, 16);
}

/* ================================================================== */
/*  Label — Branch Target                                              */
/* ================================================================== */

class Label {
 public:
  constexpr Label() : label_{UINT32_MAX} {}
  constexpr explicit Label(PhxLabel l) : label_(l) {}
  constexpr explicit Label(uint32_t id) : label_{id} {}

  constexpr uint32_t id() const { return label_.id; }
  constexpr bool isValid() const { return label_.id != UINT32_MAX; }

  constexpr operator PhxLabel() const { return label_; }

 private:
  PhxLabel label_;
};

/* ptr(Label) — for label-relative addressing (x86 RIP-relative)
 * RIP-relative label addressing: [rip + label_offset + disp32]
 * Used for switch table dispatch in gen_asm.cpp:
 *   lea(r8, ptr(table_label))
 *   lea(r8, ptr(r8, rcx, 3))
 * The label reference is stored in PhxMem (is_label_rel=1, label_id)
 * and resolved during finalize via encode_modrm_rip_rel + label fixup. */
inline Mem ptr(const Label& label) {
  PhxMem m = {};
  m.is_label_rel = 1;
  m.label_id = label.id();
  m.size = 8;
  return Mem(m);
}

inline Mem ptr(const Label& label, const Gp& index, uint8_t shift) {
  PhxMem m = {};
  m.is_label_rel = 1;
  m.label_id = label.id();
  m.index = index;
  m.scale = 1u << shift;
  m.has_index = 1;
  m.size = 8;
  return Mem(m);
}

/* ================================================================== */
/*  Imm — Immediate Value                                              */
/* ================================================================== */

class Imm {
 public:
  constexpr Imm() : value_(0) {}
  constexpr Imm(int64_t v) : value_(v) {}
  Imm(const void* p) : value_(reinterpret_cast<int64_t>(p)) {}

  constexpr int64_t value() const { return value_; }

  template <typename T>
  constexpr T valueAs() const { return static_cast<T>(value_); }

 private:
  int64_t value_;
};

/* ================================================================== */
/*  Xmm / Vec — SIMD/FP Register                                      */
/* ================================================================== */

/* Xmm inherits from Gp so const Gp& can bind to Xmm (same as asmjit's
 * Xmm inheriting from BaseReg, same base as Gp). Size=16 distinguishes. */
class Xmm : public Gp {
 public:
  constexpr Xmm() : Gp(0, 16) {}
  constexpr explicit Xmm(uint8_t id) : Gp(id, 16) {}
  constexpr Xmm(const Gp& gp) : Gp(gp.id(), 16) {}
};

/* Vec type: on ARM64 the same physical register file serves GP and SIMD —
 * the instruction determines the register interpretation, not the operand
 * type. So Vec = Gp on ARM64, avoiding CRTP type collisions where codegen
 * passes Vec to methods expecting const Gp&. On x86, Vec = Xmm as usual. */
#ifdef CINDER_AARCH64
using Vec = Gp;
#else
using Vec = Xmm;
#endif

/* Factory function */
constexpr Xmm xmm(uint8_t id) { return Xmm(id); }


/* ================================================================== */
/*  ARM64 shift specification                                          */
/* ================================================================== */

struct Shift {
  enum Type { kLSL = 0, kLSR = 1, kASR = 2 };
  Type type;
  uint8_t amount;
  constexpr Shift(Type t, uint8_t a) : type(t), amount(a) {}
};

inline Shift lsl(uint8_t amount) { return Shift(Shift::kLSL, amount); }
inline Shift lsr(uint8_t amount) { return Shift(Shift::kLSR, amount); }
inline Shift asr(uint8_t amount) { return Shift(Shift::kASR, amount); }

/* ================================================================== */
/*  EmitterExplicitT — CRTP base for autogen.cpp compatibility         */
/*                                                                     */
/*  autogen.cpp takes member function pointers of the form:            */
/*    Error (EmitterExplicitT<Builder>::*)(const Gp&, const Mem&)      */
/*  The Builder inherits from this and provides instruction methods.   */
/* ================================================================== */

/* ================================================================== */
/*  CodeHolder — wraps PhxCodeHolder                                   */
/* ================================================================== */

class CodeHolder {
 public:
  CodeHolder() : code_(nullptr) {}
  explicit CodeHolder(PhxArch arch) : code_(phx_code_create(arch)) {}
  ~CodeHolder() {
    if (code_) phx_code_destroy(code_);
  }

  CodeHolder(const CodeHolder&) = delete;
  CodeHolder& operator=(const CodeHolder&) = delete;

  PhxCodeHolder* get() const { return code_; }
  uint64_t baseAddress() const {
    if (!code_) return 0;
    /* After relocateToBase(), return the relocated address (where the code
       will actually execute). Before relocation, fall back to buffer address. */
    return code_->base_address ? code_->base_address
                               : reinterpret_cast<uint64_t>(code_->buffer);
  }
  bool hasBaseAddress() const { return code_ && code_->buffer != nullptr; }
  size_t codeSize() const { return code_ ? code_->buffer_size : 0; }

  /* Section — provides access to the CodeHolder's code buffer */
  struct Section {
    PhxCodeHolder* holder = nullptr;
    const char* name() const { return ".text"; }
    size_t realSize() const { return holder ? holder->buffer_size : 0; }
    size_t bufferSize() const { return holder ? holder->buffer_size : 0; }
    size_t virtualSize() const { return holder ? holder->buffer_size : 0; }
    size_t offset() const { return 0; }
    void setOffset(size_t) {}
    size_t alignment() const { return 16; }
    uint32_t flags() const { return 0; }
    const uint8_t* data() const { return holder ? holder->buffer : nullptr; }
  };
  mutable Section text_section_{};  /* per-instance, not static */
  Section* sectionByName(const char*) const { text_section_.holder = code_; return &text_section_; }
  Section* textSection() const { text_section_.holder = code_; return &text_section_; }
  template <typename SectionPtr>
  Error newSection(SectionPtr*, const char*, size_t = 0, uint32_t = 0, uint32_t = 0, uint32_t = 0) {
    return kErrorOk;
  }
  uint64_t labelOffsetFromBase(const Label& l) const {
    return builder_ ? labelOffsetImpl(l) : 0;
  }
  uint64_t labelOffset(const Label& l) const {
    return builder_ ? labelOffsetImpl(l) : 0;
  }

  void setBuilder(PhxBuilder* b) { builder_ = b; }

 private:
  uint64_t labelOffsetImpl(const Label& l) const {
    if (!builder_) { fprintf(stderr, "PHX labelOffset: no builder\n"); return 0; }
    if (l.id() == UINT32_MAX) { fprintf(stderr, "PHX labelOffset: invalid label\n"); return 0; }
    if (l.id() >= builder_->next_label_id) { fprintf(stderr, "PHX labelOffset: id %u >= next %u\n", l.id(), builder_->next_label_id); return 0; }
    PhxNode* node = builder_->label_nodes[l.id()];
    if (!node) { fprintf(stderr, "PHX labelOffset: label %u unbound\n", l.id()); return 0; }
    return node->offset;
  }
  PhxBuilder* builder_ = nullptr;
 public:

  /* Sections — phoenix-asm uses a single section.
   * code_allocator.cpp iterates via code->_sections and code->sections().
   * We store a single Section* to satisfy range-for loops. */
  size_t sectionCount() const { return 1; }

  /* _sections is a public member in asmjit::CodeHolder, accessed directly
   * by code_allocator.cpp. We use a simple vector-like wrapper. */
  struct SectionVec {
    Section* s;
    Section** begin() { return &s; }
    Section** end() { return &s + (s ? 1 : 0); }
    const Section* const* begin() const { return const_cast<const Section* const*>(&s); }
    const Section* const* end() const { return begin() + (s ? 1 : 0); }
    size_t size() const { return s ? 1 : 0; }
    Section* operator[](size_t) const { return s; }
  };
  SectionVec _sections{&text_section_};
  SectionVec sections() { return _sections; }
  Error flatten() { return kErrorOk; }
  Error resolveUnresolvedLinks() { return kErrorOk; }
  Error relocateToBase(uint64_t base) {
    if (code_) {
      code_->base_address = base;
    }
    if (builder_) {
      int rc = phx_relocate_to_base(builder_, base);
      if (rc != 0) return static_cast<Error>(rc);
    }
    return kErrorOk;
  }

  /* Init — create the PhxCodeHolder if not already created.
   * In asmjit, CodeHolder::init(env) sets up the code buffer.
   * Phoenix-asm defers buffer creation to finalize. */
  template <typename... Args>
  Error init(Args&&...) {
    if (!code_) {
#if defined(CINDER_X86_64)
      code_ = phx_code_create(PHX_ARCH_X86_64);
#elif defined(CINDER_AARCH64)
      code_ = phx_code_create(PHX_ARCH_ARM64);
#endif
    }
    if (code_) text_section_.holder = code_;
    return code_ ? kErrorOk : 1;
  }

  /* Error handler */
  void setErrorHandler(void*) {}

  /* Attach builder */
  void attach(void*) {}

 private:
  PhxCodeHolder* code_;
};

/* ================================================================== */
/*  EmitterExplicitT — forward declaration (arch-specific below)       */
/* ================================================================== */

template <typename CRTP>
class EmitterExplicitT;

} /* namespace phx */

/* ================================================================== */
/*  x86_64 register constants (asmjit::x86 namespace compatibility)   */
/* ================================================================== */

#if defined(CINDER_X86_64)

#include "x86_64.h"

namespace phx { namespace x86 {

/* 64-bit GP registers */
constexpr Gp rax{0, 8};
constexpr Gp rcx{1, 8};
constexpr Gp rdx{2, 8};
constexpr Gp rbx{3, 8};
constexpr Gp rsp{4, 8};
constexpr Gp rbp{5, 8};
constexpr Gp rsi{6, 8};
constexpr Gp rdi{7, 8};
constexpr Gp r8{8, 8};
constexpr Gp r9{9, 8};
constexpr Gp r10{10, 8};
constexpr Gp r11{11, 8};
constexpr Gp r12{12, 8};
constexpr Gp r13{13, 8};
constexpr Gp r14{14, 8};
constexpr Gp r15{15, 8};

/* 32-bit GP registers */
constexpr Gp eax{0, 4};
constexpr Gp ecx{1, 4};
constexpr Gp edx{2, 4};
constexpr Gp ebx{3, 4};
constexpr Gp edi{7, 4};
constexpr Gp esi{6, 4};

/* 16-bit GP registers */
constexpr Gp ax{0, 2};

/* 8-bit GP registers */
constexpr Gp al{0, 1};
constexpr Gp cl{1, 1};

/* Segment register constants (used for TLS) */
constexpr uint8_t fs = 4;  /* FS segment */
constexpr uint8_t gs = 5;  /* GS segment */

/* XMM registers */
constexpr Xmm xmm0{0};
constexpr Xmm xmm1{1};
constexpr Xmm xmm2{2};
constexpr Xmm xmm3{3};
constexpr Xmm xmm4{4};
constexpr Xmm xmm5{5};
constexpr Xmm xmm6{6};
constexpr Xmm xmm7{7};
constexpr Xmm xmm8{8};
constexpr Xmm xmm9{9};
constexpr Xmm xmm10{10};
constexpr Xmm xmm11{11};
constexpr Xmm xmm12{12};
constexpr Xmm xmm13{13};
constexpr Xmm xmm14{14};
constexpr Xmm xmm15{15};

}} /* namespace phx::x86 */

#elif defined(CINDER_AARCH64)

#include "arm64.h"

namespace phx { namespace a64 {

/* 64-bit GP registers */
constexpr Gp x0{0, 8};
constexpr Gp x1{1, 8};
constexpr Gp x2{2, 8};
constexpr Gp x3{3, 8};
constexpr Gp x4{4, 8};
constexpr Gp x5{5, 8};
constexpr Gp x6{6, 8};
constexpr Gp x7{7, 8};
constexpr Gp x8{8, 8};
constexpr Gp x9{9, 8};
constexpr Gp x10{10, 8};
constexpr Gp x11{11, 8};
constexpr Gp x12{12, 8};
constexpr Gp x13{13, 8};
constexpr Gp x14{14, 8};
constexpr Gp x15{15, 8};
constexpr Gp x16{16, 8};
constexpr Gp x17{17, 8};
constexpr Gp x18{18, 8};
constexpr Gp x19{19, 8};
constexpr Gp x20{20, 8};
constexpr Gp x21{21, 8};
constexpr Gp x22{22, 8};
constexpr Gp x23{23, 8};
constexpr Gp x24{24, 8};
constexpr Gp x25{25, 8};
constexpr Gp x26{26, 8};
constexpr Gp x27{27, 8};
constexpr Gp x28{28, 8};
constexpr Gp x29{29, 8};
constexpr Gp x30{30, 8};
constexpr Gp sp{31, 8};

/* FP/SIMD registers — Vec = Gp on ARM64 (needs id+size), Vec = Xmm on x86 */
#ifdef CINDER_AARCH64
constexpr Vec d0{0, 8};
constexpr Vec d1{1, 8};
constexpr Vec d2{2, 8};
#else
constexpr Vec d0{0};
constexpr Vec d1{1};
constexpr Vec d2{2};
#endif

}} /* namespace phx::a64 */

#endif /* architecture */

/* ================================================================== */
/*  EmitterExplicitT — x86_64 CRTP base with instruction methods       */
/*                                                                     */
/*  autogen.cpp takes member function pointers of type:                */
/*    Error (EmitterExplicitT<Builder>::*)(const Gp&, const Gp&)       */
/*  So all instruction methods MUST be declared here, not on Builder.  */
/*  CRTP gives us access to Builder::impl() via b_().                  */
/* ================================================================== */

#if defined(CINDER_X86_64)

#include "x86_64.h"

namespace phx {

template <typename CRTP>
class EmitterExplicitT {
 protected:
  PhxBuilder* b_() { return static_cast<CRTP*>(this)->impl(); }

 public:
  /* -- MOV -- */
  Error mov(const Gp& d, const Gp& s)  { phx_x86_mov_rr(b_(), d, s); return kErrorOk; }
  Error mov(const Gp& d, const Mem& s)  { phx_x86_mov_rm(b_(), d, s); return kErrorOk; }
  Error mov(const Mem& d, const Gp& s)  { phx_x86_mov_mr(b_(), d, s); return kErrorOk; }
  Error mov(const Gp& d, uint64_t v)    { phx_x86_mov_ri(b_(), d, (int64_t)v); return kErrorOk; }
  /* mov(Gp, void*) removed — causes ambiguity with mov(Gp, uint64_t) when
   * argument is 0 or NULL. Callers should use reinterpret_cast<uint64_t>. */
  Error mov(const Mem& d, uint64_t v)   { phx_x86_mov_mi(b_(), d, (int32_t)v); return kErrorOk; }
  Error mov(const Gp& d, const Imm& i)  { phx_x86_mov_ri(b_(), d, i.value()); return kErrorOk; }
  Error mov(const Mem& d, const Imm& i) { phx_x86_mov_mi(b_(), d, (int32_t)i.value()); return kErrorOk; }
  /* -- LEA -- */
  Error lea(const Gp& d, const Mem& s)  { phx_x86_lea(b_(), d, s); return kErrorOk; }
  /* -- ADD -- */
  Error add(const Gp& d, const Gp& s)   { phx_x86_add_rr(b_(), d, s); return kErrorOk; }
  Error add(const Gp& d, const Mem& s)  { phx_x86_add_rm(b_(), d, s); return kErrorOk; }
  Error add(const Mem& d, const Gp& s)  { phx_x86_add_mr(b_(), d, s); return kErrorOk; }
  Error add(const Gp& d, int32_t v)     { phx_x86_add_ri(b_(), d, v); return kErrorOk; }
  /* add(Gp, uint64_t) removed — ambiguous with int32_t for uint32_t args */
  Error add(const Mem& d, int32_t v)    { phx_x86_add_mi(b_(), d, v); return kErrorOk; }
  Error add(const Gp& d, const Imm& i)  { phx_x86_add_ri(b_(), d, (int32_t)i.value()); return kErrorOk; }
  Error add(const Mem& d, const Imm& i) { phx_x86_add_mi(b_(), d, (int32_t)i.value()); return kErrorOk; }
  /* -- SUB -- */
  Error sub(const Gp& d, const Gp& s)   { phx_x86_sub_rr(b_(), d, s); return kErrorOk; }
  Error sub(const Gp& d, const Mem& s)  { phx_x86_sub_rm(b_(), d, s); return kErrorOk; }
  Error sub(const Mem& d, const Gp& s)  { phx_x86_sub_mr(b_(), d, s); return kErrorOk; }
  Error sub(const Gp& d, int32_t v)     { phx_x86_sub_ri(b_(), d, v); return kErrorOk; }
  /* sub(Gp, uint64_t) removed — ambiguous with int32_t for uint32_t args */
  Error sub(const Mem& d, int32_t v)    { phx_x86_sub_mi(b_(), d, v); return kErrorOk; }
  Error sub(const Gp& d, const Imm& i)  { phx_x86_sub_ri(b_(), d, (int32_t)i.value()); return kErrorOk; }
  /* -- CMP -- */
  Error cmp(const Gp& a, const Gp& c)   { phx_x86_cmp_rr(b_(), a, c); return kErrorOk; }
  Error cmp(const Gp& a, const Mem& c)  { phx_x86_cmp_rm(b_(), a, c); return kErrorOk; }
  Error cmp(const Mem& a, const Gp& c)  { phx_x86_cmp_mr(b_(), a, c); return kErrorOk; }
  Error cmp(const Gp& a, int32_t v)     { phx_x86_cmp_ri(b_(), a, v); return kErrorOk; }
  /* cmp(Gp, uint64_t) removed — ambiguous with int32_t for uint32_t args */
  Error cmp(const Mem& a, int32_t v)    { phx_x86_cmp_mi(b_(), a, v); return kErrorOk; }
  Error cmp(const Gp& a, const Imm& i)  { phx_x86_cmp_ri(b_(), a, (int32_t)i.value()); return kErrorOk; }
  Error cmp(const Mem& a, const Imm& i) { phx_x86_cmp_mi(b_(), a, (int32_t)i.value()); return kErrorOk; }
  /* -- TEST -- */
  Error test(const Gp& a, const Gp& c)  { phx_x86_test_rr(b_(), a, c); return kErrorOk; }
  Error test(const Gp& a, int32_t v)    { phx_x86_test_ri(b_(), a, v); return kErrorOk; }
  Error test(const Gp& a, const Imm& i) { phx_x86_test_ri(b_(), a, (int32_t)i.value()); return kErrorOk; }
  Error test(const Mem& a, int32_t v)   { phx_x86_test_mi(b_(), a, v); return kErrorOk; }
  /* -- AND / OR / XOR / NOT -- */
  Error and_(const Gp& d, const Gp& s)  { phx_x86_and_rr(b_(), d, s); return kErrorOk; }
  Error and_(const Gp& d, const Mem& s) { phx_x86_and_rm(b_(), d, s); return kErrorOk; }
  Error and_(const Gp& d, int32_t v)    { phx_x86_and_ri(b_(), d, v); return kErrorOk; }
  Error and_(const Gp& d, const Imm& i) { phx_x86_and_ri(b_(), d, (int32_t)i.value()); return kErrorOk; }
  Error and_(const Mem& d, int32_t v)   { phx_x86_and_mi(b_(), d, v); return kErrorOk; }
  Error or_(const Gp& d, const Gp& s)   { phx_x86_or_rr(b_(), d, s); return kErrorOk; }
  Error or_(const Gp& d, const Mem& s)  { phx_x86_or_rm(b_(), d, s); return kErrorOk; }
  Error or_(const Gp& d, int32_t v)     { phx_x86_or_ri(b_(), d, v); return kErrorOk; }
  Error or_(const Gp& d, const Imm& i)  { phx_x86_or_ri(b_(), d, (int32_t)i.value()); return kErrorOk; }
  Error or_(const Mem& d, int32_t v)    { phx_x86_or_mi(b_(), d, v); return kErrorOk; }
  Error xor_(const Gp& d, const Gp& s)  { phx_x86_xor_rr(b_(), d, s); return kErrorOk; }
  Error xor_(const Gp& d, const Mem& s) { phx_x86_xor_rm(b_(), d, s); return kErrorOk; }
  Error xor_(const Gp& d, int32_t v)    { phx_x86_xor_ri(b_(), d, v); return kErrorOk; }
  Error xor_(const Gp& d, const Imm& i) { phx_x86_xor_ri(b_(), d, (int32_t)i.value()); return kErrorOk; }
  Error not_(const Gp& d)               { phx_x86_not_r(b_(), d); return kErrorOk; }
  Error not_(const Mem& d)              { phx_x86_not_m(b_(), d); return kErrorOk; }
  /* -- NEG / INC / DEC -- */
  Error neg(const Gp& d)                { phx_x86_neg_r(b_(), d); return kErrorOk; }
  Error neg(const Mem& d)               { phx_x86_neg_m(b_(), d); return kErrorOk; }
  Error inc(const Gp& d)                { phx_x86_inc_r(b_(), d); return kErrorOk; }
  Error inc(const Mem& d)               { phx_x86_inc_m(b_(), d); return kErrorOk; }
  Error dec(const Gp& d)                { phx_x86_dec_r(b_(), d); return kErrorOk; }
  Error dec(const Mem& d)               { phx_x86_dec_m(b_(), d); return kErrorOk; }
  /* -- IMUL / IDIV / DIV -- */
  Error imul(const Gp& d, const Gp& s)  { phx_x86_imul_rr(b_(), d, s); return kErrorOk; }
  Error imul(const Gp& d, const Mem& s) { phx_x86_imul_rm(b_(), d, s); return kErrorOk; }
  Error imul(const Gp& d, const Imm& i) { phx_x86_imul_rri(b_(), d, d, (int32_t)i.value()); return kErrorOk; }
  Error imul(const Gp& d, const Gp& s, int32_t v) { phx_x86_imul_rri(b_(), d, s, v); return kErrorOk; }
  Error imul(const Gp& d, const Gp& s, const Imm& i) { phx_x86_imul_rri(b_(), d, s, (int32_t)i.value()); return kErrorOk; }
  Error idiv(const Gp& s)               { phx_x86_idiv_r(b_(), s); return kErrorOk; }
  Error idiv(const Mem& s)              { phx_x86_idiv_m(b_(), s); return kErrorOk; }
  /* Multi-arg forms for autogen.cpp: idiv(rdx, rax, src) → implicit RDX:RAX */
  Error idiv(const Gp&, const Gp&, const Gp& s)  { phx_x86_idiv_r(b_(), s); return kErrorOk; }
  Error idiv(const Gp&, const Gp&, const Mem& s) { phx_x86_idiv_m(b_(), s); return kErrorOk; }
  Error idiv(const Gp&, const Gp& s)             { phx_x86_idiv_r(b_(), s); return kErrorOk; }
  Error idiv(const Gp&, const Mem& s)            { phx_x86_idiv_m(b_(), s); return kErrorOk; }
  Error div(const Gp& s)                { phx_x86_div_r(b_(), s); return kErrorOk; }
  Error div(const Gp&, const Gp&, const Gp& s)  { phx_x86_div_r(b_(), s); return kErrorOk; }
  Error div(const Gp&, const Gp&, const Mem& s) { phx_x86_div_m(b_(), s); return kErrorOk; }
  Error div(const Gp&, const Gp& s)             { phx_x86_div_r(b_(), s); return kErrorOk; }
  Error div(const Gp&, const Mem& s)            { phx_x86_div_m(b_(), s); return kErrorOk; }
  /* -- BT -- */
  Error bt(const Gp& a, const Gp& bit)  { phx_x86_bt_rr(b_(), a, bit); return kErrorOk; }
  Error bt(const Gp& a, uint8_t bit)    { phx_x86_bt_ri(b_(), a, bit); return kErrorOk; }
  Error bt(const Gp& a, const Imm& i)   { phx_x86_bt_ri(b_(), a, (uint8_t)i.value()); return kErrorOk; }
  /* -- BTS -- */
  Error bts(const Gp& d, uint8_t bit)   { phx_x86_bts_ri(b_(), d, bit); return kErrorOk; }
  Error bts(const Gp& d, const Imm& i)  { phx_x86_bts_ri(b_(), d, (uint8_t)i.value()); return kErrorOk; }
  /* -- LOCK ADD -- */
  Error lock_add(const Mem& d, int32_t v)   { phx_x86_lock_add_mi(b_(), d, v); return kErrorOk; }
  Error lock_add(const Mem& d, const Imm& i){ phx_x86_lock_add_mi(b_(), d, (int32_t)i.value()); return kErrorOk; }
  /* -- PUSH / POP -- */
  Error push(const Gp& s)               { phx_x86_push_r(b_(), s); return kErrorOk; }
  Error push(const Mem& s)              { phx_x86_push_m(b_(), s); return kErrorOk; }
  Error push(uint64_t v)                { phx_x86_push_i(b_(), (int32_t)v); return kErrorOk; }
  Error push(const Imm& i)              { phx_x86_push_i(b_(), (int32_t)i.value()); return kErrorOk; }
  Error pop(const Gp& d)                { phx_x86_pop_r(b_(), d); return kErrorOk; }
  Error pop(const Mem& d)               { phx_x86_pop_m(b_(), d); return kErrorOk; }
  /* -- XCHG -- */
  Error xchg(const Gp& a, const Gp& c)  { phx_x86_xchg_rr(b_(), a, c); return kErrorOk; }
  /* -- JMP -- */
  Error jmp(const Label& t)             { phx_x86_jmp_label(b_(), t); return kErrorOk; }
  Error jmp(const Gp& t)                { phx_x86_jmp_r(b_(), t); return kErrorOk; }
  Error jmp(const Mem& t)               { phx_x86_jmp_m(b_(), t); return kErrorOk; }
  Error jmp(uint64_t addr)              { phx_x86_mov_ri(b_(), PHX_R11, (int64_t)addr); phx_x86_jmp_r(b_(), PHX_R11); return kErrorOk; }
  /* -- Jcc -- */
  Error je(const Label& t)  { phx_x86_je(b_(), t); return kErrorOk; }
  Error jne(const Label& t) { phx_x86_jne(b_(), t); return kErrorOk; }
  Error jz(const Label& t)  { phx_x86_jz(b_(), t); return kErrorOk; }
  Error jnz(const Label& t) { phx_x86_jnz(b_(), t); return kErrorOk; }
  Error ja(const Label& t)  { phx_x86_ja(b_(), t); return kErrorOk; }
  Error jae(const Label& t) { phx_x86_jae(b_(), t); return kErrorOk; }
  Error jb(const Label& t)  { phx_x86_jb(b_(), t); return kErrorOk; }
  Error jbe(const Label& t) { phx_x86_jbe(b_(), t); return kErrorOk; }
  Error jg(const Label& t)  { phx_x86_jg(b_(), t); return kErrorOk; }
  Error jge(const Label& t) { phx_x86_jge(b_(), t); return kErrorOk; }
  Error jl(const Label& t)  { phx_x86_jl(b_(), t); return kErrorOk; }
  Error jle(const Label& t) { phx_x86_jle(b_(), t); return kErrorOk; }
  Error jc(const Label& t)  { phx_x86_jc(b_(), t); return kErrorOk; }
  Error jnc(const Label& t) { phx_x86_jnc(b_(), t); return kErrorOk; }
  Error jo(const Label& t)  { phx_x86_jo(b_(), t); return kErrorOk; }
  Error jno(const Label& t) { phx_x86_jno(b_(), t); return kErrorOk; }
  Error js(const Label& t)  { phx_x86_js(b_(), t); return kErrorOk; }
  Error jns(const Label& t) { phx_x86_jns(b_(), t); return kErrorOk; }
  /* -- SETcc -- */
  Error sete(const Gp& d)   { phx_x86_sete(b_(), d); return kErrorOk; }
  Error setne(const Gp& d)  { phx_x86_setne(b_(), d); return kErrorOk; }
  Error setg(const Gp& d)   { phx_x86_setg(b_(), d); return kErrorOk; }
  Error setge(const Gp& d)  { phx_x86_setge(b_(), d); return kErrorOk; }
  Error setl(const Gp& d)   { phx_x86_setl(b_(), d); return kErrorOk; }
  Error setle(const Gp& d)  { phx_x86_setle(b_(), d); return kErrorOk; }
  Error seta(const Gp& d)   { phx_x86_seta(b_(), d); return kErrorOk; }
  Error setae(const Gp& d)  { phx_x86_setae(b_(), d); return kErrorOk; }
  Error setb(const Gp& d)   { phx_x86_setb(b_(), d); return kErrorOk; }
  Error setbe(const Gp& d)  { phx_x86_setbe(b_(), d); return kErrorOk; }
  /* -- CALL -- */
  Error call(const Label& t)  { phx_x86_call_label(b_(), t); return kErrorOk; }
  Error call(const Gp& t)     { phx_x86_call_r(b_(), t); return kErrorOk; }
  Error call(const Mem& t)    { phx_x86_call_m(b_(), t); return kErrorOk; }
  Error call(uint64_t addr)   { phx_x86_mov_ri(b_(), PHX_R11, (int64_t)addr); phx_x86_call_r(b_(), PHX_R11); return kErrorOk; }
  Error call(const Imm& i)    { return call((uint64_t)i.value()); }
  /* Call via function pointer — converts to uint64_t address */
  template <typename Ret, typename... Args>
  Error call(Ret (*fn)(Args...)) { return call(reinterpret_cast<uint64_t>(fn)); }
  /* -- RET / LEAVE / UD2 -- */
  Error ret()    { phx_x86_ret(b_()); return kErrorOk; }
  Error leave()  { phx_x86_leave(b_()); return kErrorOk; }
  Error ud2()    { phx_x86_ud2(b_()); return kErrorOk; }
  /* -- MOVSX / MOVSXD / MOVZX -- */
  Error movsx(const Gp& d, const Gp& s)  { phx_x86_movsx_rr(b_(), d, s); return kErrorOk; }
  Error movsx(const Gp& d, const Mem& s) { phx_x86_movsx_rm(b_(), d, s); return kErrorOk; }
  Error movsxd(const Gp& d, const Gp& s) { phx_x86_movsxd_rr(b_(), d, s); return kErrorOk; }
  Error movsxd(const Gp& d, const Mem& s){ phx_x86_movsxd_rm(b_(), d, s); return kErrorOk; }
  Error movzx(const Gp& d, const Gp& s)  { phx_x86_movzx_rr(b_(), d, s); return kErrorOk; }
  Error movzx(const Gp& d, const Mem& s) { phx_x86_movzx_rm(b_(), d, s); return kErrorOk; }
  /* -- CMOVNZ -- */
  Error cmovnz(const Gp& d, const Gp& s) { phx_x86_cmovnz_rr(b_(), d, s); return kErrorOk; }
  Error cmovnz(const Gp& d, const Mem& s){ phx_x86_cmovnz_rm(b_(), d, s); return kErrorOk; }
  /* -- CDQ / CQO / CWD -- */
  Error cdq() { phx_x86_cdq(b_()); return kErrorOk; }
  Error cdq(const Gp&, const Gp&) { phx_x86_cdq(b_()); return kErrorOk; }  /* autogen: cdq(rdx, rax) */
  Error cqo() { phx_x86_cqo(b_()); return kErrorOk; }
  Error cqo(const Gp&, const Gp&) { phx_x86_cqo(b_()); return kErrorOk; }  /* autogen: cqo(rdx, rax) */
  Error cwd(const Gp&, const Gp&) { phx_x86_cdq(b_()); return kErrorOk; }  /* autogen: cwd(rdx, rax) */
  /* -- SSE: MOVSD -- */
  Error movsd(const Xmm& d, const Xmm& s) { phx_x86_movsd_rr(b_(), d, s); return kErrorOk; }
  Error movsd(const Xmm& d, const Mem& s)  { phx_x86_movsd_rm(b_(), d, s); return kErrorOk; }
  Error movsd(const Mem& d, const Xmm& s)  { phx_x86_movsd_mr(b_(), d, s); return kErrorOk; }
  /* -- SSE: Arithmetic -- */
  Error addsd(const Xmm& d, const Xmm& s) { phx_x86_addsd_rr(b_(), d, s); return kErrorOk; }
  Error addsd(const Xmm& d, const Mem& s)  { phx_x86_addsd_rm(b_(), d, s); return kErrorOk; }
  Error subsd(const Xmm& d, const Xmm& s) { phx_x86_subsd_rr(b_(), d, s); return kErrorOk; }
  Error subsd(const Xmm& d, const Mem& s)  { phx_x86_subsd_rm(b_(), d, s); return kErrorOk; }
  Error mulsd(const Xmm& d, const Xmm& s) { phx_x86_mulsd_rr(b_(), d, s); return kErrorOk; }
  Error mulsd(const Xmm& d, const Mem& s)  { phx_x86_mulsd_rm(b_(), d, s); return kErrorOk; }
  Error divsd(const Xmm& d, const Xmm& s) { phx_x86_divsd_rr(b_(), d, s); return kErrorOk; }
  Error divsd(const Xmm& d, const Mem& s)  { phx_x86_divsd_rm(b_(), d, s); return kErrorOk; }
  /* -- SSE: Misc -- */
  Error ptest(const Xmm& a, const Xmm& c)    { phx_x86_ptest_rr(b_(), a, c); return kErrorOk; }
  Error pcmpeqw(const Xmm& d, const Xmm& s)  { phx_x86_pcmpeqw_rr(b_(), d, s); return kErrorOk; }
  Error psrlq(const Xmm& d, uint8_t v)        { phx_x86_psrlq_ri(b_(), d, v); return kErrorOk; }
  Error psrlq(const Xmm& d, const Imm& i)     { phx_x86_psrlq_ri(b_(), d, (uint8_t)i.value()); return kErrorOk; }
  Error pxor(const Xmm& d, const Xmm& s)      { phx_x86_pxor_rr(b_(), d, s); return kErrorOk; }
  Error comisd(const Xmm& a, const Xmm& c)    { phx_x86_comisd(b_(), a, c); return kErrorOk; }
  /* -- MOVQ (GP <-> XMM) -- */
  Error movq(const Xmm& d, const Gp& s)  { phx_x86_movq_rr(b_(), d, s); return kErrorOk; }
  Error movq(const Gp& d, const Xmm& s)  { phx_x86_movq_rr(b_(), d, s); return kErrorOk; }
  /* -- MOVDQU -- */
  Error movdqu(const Xmm& d, const Mem& s) { phx_x86_movdqu_rm(b_(), d, s); return kErrorOk; }
  Error movdqu(const Mem& d, const Xmm& s) { phx_x86_movdqu_mr(b_(), d, s); return kErrorOk; }
};

/* ================================================================== */
/*  Builder — lifecycle, cursor, label management                      */
/*  Inherits all instruction methods from EmitterExplicitT.            */
/* ================================================================== */

class Builder : public EmitterExplicitT<Builder> {
 public:
  explicit Builder(PhxCodeHolder* code) : impl_(phx_builder_create(code)) {}
  explicit Builder(CodeHolder* code) : impl_(nullptr), code_(code) {
    if (code) {
      if (!code->get()) code->init();
      impl_ = phx_builder_create(code->get());
      code->setBuilder(impl_);
    }
  }
  ~Builder() { if (impl_) phx_builder_destroy(impl_); }

  Builder(const Builder&) = delete;
  Builder& operator=(const Builder&) = delete;

  PhxBuilder* impl() const { return impl_; }

  /* code() accessor — returns the CodeHolder this builder was attached to */
  CodeHolder* code() const { return code_; }

  Label newLabel() { return Label(phx_builder_new_label(impl_)); }
  Error bind(const Label& label) { phx_builder_bind(impl_, label); return kErrorOk; }

  BaseNode* cursor() { return phx_builder_cursor(impl_); }
  void setCursor(BaseNode* node) { phx_builder_set_cursor(impl_, node); }

  Error align(AlignMode, uint32_t alignment) {
    phx_builder_align(impl_, (int)alignment); return kErrorOk;
  }
  Error embed(const void* data, size_t size) {
    phx_builder_embed(impl_, data, size); return kErrorOk;
  }
  Error finalize() {
    if (!impl_ || !impl_->code) return 1;
    return phx_x86_finalize(impl_);
  }

  /* Section stub — phoenix-asm uses single code section */
  Error section(void*) { return kErrorOk; }

  /* Jump hint stubs — asmjit uses these for short/long branch encoding.
   * Phoenix-asm always uses the appropriate encoding in finalize. */
  /* short_() returns a proxy that emits short-encoded branches.
   * Usage: as_->short_().jmp(label) emits EB rel8 instead of E9 rel32. */
  struct ShortProxy {
    PhxBuilder* b;
    Error jmp(const Label& t) {
      PhxNode* n = phx_builder_alloc_node(b);
      if (!n) return 1;
      n->node_type = PHX_NODE_INST;
      n->opcode = 0xEB;
      n->encoded[0] = 0xEB;
      n->encoded[1] = 0x00;
      n->encoded_size = 2;
      phx_builder_append_node(b, n);
      phx_builder_add_fixup(b, n, t.id(), 0);
      return kErrorOk;
    }
  };
  ShortProxy short_() { return ShortProxy{impl_}; }
  Builder& long_() { return *this; }

 private:
  PhxBuilder* impl_;
  CodeHolder* code_;
  bool use_short_ = false;
};

} /* namespace phx */

#endif /* CINDER_X86_64 */

/* ================================================================== */
/*  EmitterExplicitT — ARM64 CRTP base with instruction methods        */
/* ================================================================== */

#if defined(CINDER_AARCH64)

#include "arm64.h"

namespace phx {

template <typename CRTP>
class EmitterExplicitT {
 protected:
  PhxBuilder* b_() { return static_cast<CRTP*>(this)->impl(); }

 public:
  /* -- MOV -- */
  Error mov(const Gp& d, const Gp& s)   { phx_a64_mov_rr(b_(), d, s); return kErrorOk; }
  Error mov(const Gp& d, uint64_t v)    { phx_a64_mov_ri(b_(), d, v); return kErrorOk; }
  Error mov(const Gp& d, const Imm& i)  { phx_a64_mov_ri(b_(), d, (uint64_t)i.value()); return kErrorOk; }
  /* mov(Gp, any_non_integer) — catches function pointers, object pointers, etc. */
  template <typename T>
  typename std::enable_if<!std::is_integral<T>::value && !std::is_same<typename std::decay<T>::type, Imm>::value
      && !std::is_same<typename std::decay<T>::type, Gp>::value, Error>::type
  mov(const Gp& d, T val) {
    uint64_t addr;
    memcpy(&addr, &val, sizeof(addr));
    phx_a64_mov_ri(b_(), d, addr);
    return kErrorOk;
  }
  /* -- LDR / STR -- */
  Error ldr(const Gp& d, const Mem& m)  { phx_a64_ldr(b_(), d, m); return kErrorOk; }
  Error ldrb(const Gp& d, const Mem& m) { phx_a64_ldrb(b_(), d, m); return kErrorOk; }
  Error ldrh(const Gp& d, const Mem& m) { phx_a64_ldrh(b_(), d, m); return kErrorOk; }
  Error ldrsb(const Gp& d, const Mem& m){ phx_a64_ldrsb(b_(), d, m); return kErrorOk; }
  Error ldrsh(const Gp& d, const Mem& m){ phx_a64_ldrsh(b_(), d, m); return kErrorOk; }
  Error ldrsw(const Gp& d, const Mem& m){ phx_a64_ldrsw(b_(), d, m); return kErrorOk; }
  Error str(const Gp& s, const Mem& m)  { phx_a64_str(b_(), s, m); return kErrorOk; }
  Error strb(const Gp& s, const Mem& m) { phx_a64_strb(b_(), s, m); return kErrorOk; }
  Error strh(const Gp& s, const Mem& m) { phx_a64_strh(b_(), s, m); return kErrorOk; }
  /* -- LDP / STP -- */
  Error ldp(const Gp& r1, const Gp& r2, const Mem& m) { phx_a64_ldp(b_(), r1, r2, m); return kErrorOk; }
  Error stp(const Gp& r1, const Gp& r2, const Mem& m) { phx_a64_stp(b_(), r1, r2, m); return kErrorOk; }
  /* -- FP move -- */
  Error fmov(const Gp& d, const Gp& s)  { phx_a64_fmov(b_(), d, s); return kErrorOk; }
  /* -- ADR -- */
  Error adr(const Gp& d, const Label& l) { phx_a64_adr(b_(), d, l); return kErrorOk; }
  /* -- ADD / SUB -- */
  Error add(const Gp& d, const Gp& a, const Gp& c)    { phx_a64_add_rrr(b_(), d, a, c); return kErrorOk; }
  Error add(const Gp& d, const Gp& a, int64_t v)        { phx_a64_add_rri(b_(), d, a, v); return kErrorOk; }
  Error add(const Gp& d, const Gp& a, const Imm& i)    { phx_a64_add_rri(b_(), d, a, i.value()); return kErrorOk; }
  Error adds(const Gp& d, const Gp& a, const Gp& c)    { phx_a64_adds_rrr(b_(), d, a, c); return kErrorOk; }
  Error adds(const Gp& d, const Gp& a, uint64_t v)      { phx_a64_adds_rri(b_(), d, a, (int64_t)v); return kErrorOk; }
  /* Shifted register forms: add(d, a, c, lsl(N)) */
  Error add(const Gp& d, const Gp& a, const Gp& c, const Shift& s) {
    phx_a64_add_rrr_shifted(b_(), d, a, c, s.type, s.amount); return kErrorOk;
  }
  Error sub(const Gp& d, const Gp& a, const Gp& c, const Shift& s) {
    phx_a64_sub_rrr_shifted(b_(), d, a, c, s.type, s.amount); return kErrorOk;
  }
  Error sub(const Gp& d, const Gp& a, const Gp& c)     { phx_a64_sub_rrr(b_(), d, a, c); return kErrorOk; }
  Error sub(const Gp& d, const Gp& a, int64_t v)         { phx_a64_sub_rri(b_(), d, a, v); return kErrorOk; }
  Error sub(const Gp& d, const Gp& a, const Imm& i)     { phx_a64_sub_rri(b_(), d, a, i.value()); return kErrorOk; }
  Error subs(const Gp& d, const Gp& a, const Gp& c)    { phx_a64_subs_rrr(b_(), d, a, c); return kErrorOk; }
  Error subs(const Gp& d, const Gp& a, uint64_t v)      { phx_a64_subs_rri(b_(), d, a, (int64_t)v); return kErrorOk; }
  /* NEG Xd, Xn = SUB Xd, XZR, Xn */
  Error neg(const Gp& d, const Gp& src) {
    Gp zr(31, src.size());
    phx_a64_sub_rrr(b_(), d, zr, src); return kErrorOk;
  }
  /* -- LDP/STP pre/post indexed -- */
  Error ldp_pre(const Gp& r1, const Gp& r2, const Gp& base, int32_t off) {
    phx_a64_ldp_pre(b_(), r1, r2, base, off); return kErrorOk;
  }
  Error ldp_post(const Gp& r1, const Gp& r2, const Gp& base, int32_t off) {
    phx_a64_ldp_post(b_(), r1, r2, base, off); return kErrorOk;
  }
  Error stp_pre(const Gp& r1, const Gp& r2, const Gp& base, int32_t off) {
    phx_a64_stp_pre(b_(), r1, r2, base, off); return kErrorOk;
  }
  Error stp_post(const Gp& r1, const Gp& r2, const Gp& base, int32_t off) {
    phx_a64_stp_post(b_(), r1, r2, base, off); return kErrorOk;
  }
  /* -- MUL / MADD / DIV -- */
  Error mul(const Gp& d, const Gp& a, const Gp& c)     { phx_a64_mul(b_(), d, a, c); return kErrorOk; }
  Error madd(const Gp& d, const Gp& a, const Gp& c, const Gp& acc) {
    phx_a64_madd(b_(), d, a, c, acc); return kErrorOk;
  }
  Error sdiv(const Gp& d, const Gp& a, const Gp& c)    { phx_a64_sdiv(b_(), d, a, c); return kErrorOk; }
  Error udiv(const Gp& d, const Gp& a, const Gp& c)    { phx_a64_udiv(b_(), d, a, c); return kErrorOk; }
  /* -- AND / EOR / ORR / MVN -- */
  Error and_(const Gp& d, const Gp& a, const Gp& c)    { phx_a64_and_rrr(b_(), d, a, c); return kErrorOk; }
  Error and_(const Gp& d, const Gp& a, uint64_t v)      { phx_a64_and_rri(b_(), d, a, v); return kErrorOk; }
  Error eor(const Gp& d, const Gp& a, const Gp& c)     { phx_a64_eor_rrr(b_(), d, a, c); return kErrorOk; }
  Error eor(const Gp& d, const Gp& a, uint64_t v)       { phx_a64_eor_rri(b_(), d, a, v); return kErrorOk; }
  Error orr(const Gp& d, const Gp& a, const Gp& c)     { phx_a64_orr_rrr(b_(), d, a, c); return kErrorOk; }
  Error orr(const Gp& d, const Gp& a, uint64_t v)       { phx_a64_orr_rri(b_(), d, a, v); return kErrorOk; }
  Error mvn(const Gp& d, const Gp& s)                   { phx_a64_mvn(b_(), d, s); return kErrorOk; }
  /* -- CMP / TST -- */
  Error cmp(const Gp& a, const Gp& c)   { phx_a64_cmp_rr(b_(), a, c); return kErrorOk; }
  Error cmp(const Gp& a, uint64_t v)    { phx_a64_cmp_ri(b_(), a, (int64_t)v); return kErrorOk; }
  Error cmp(const Gp& a, int32_t v)     { phx_a64_cmp_ri(b_(), a, v); return kErrorOk; }
  Error cmp(const Gp& a, const Imm& i)  { phx_a64_cmp_ri(b_(), a, i.value()); return kErrorOk; }
  Error tst(const Gp& a, const Gp& c)   { phx_a64_tst_rr(b_(), a, c); return kErrorOk; }
  Error tst(const Gp& a, uint64_t v)    { phx_a64_tst_ri(b_(), a, v); return kErrorOk; }
  /* -- FP compare -- */
  Error fcmp(const Vec& a, const Vec& c) { phx_a64_fcmp(b_(), a, c); return kErrorOk; }
  /* -- CSEL / CSET -- */
  Error csel(const Gp& d, const Gp& a, const Gp& c, uint32_t cond) {
    phx_a64_csel(b_(), d, a, c, static_cast<PhxArm64Cond>(cond)); return kErrorOk;
  }
  Error cset(const Gp& d, uint32_t cond) {
    phx_a64_cset(b_(), d, static_cast<PhxArm64Cond>(cond)); return kErrorOk;
  }
  /* -- LSL (shift instruction) -- */
  Error lsl(const Gp& d, const Gp& s, uint32_t shift) {
    phx_a64_lsl(b_(), d, s, shift); return kErrorOk;
  }
  /* -- Branches -- */
  Error b(const Label& t)    { phx_a64_b(b_(), t); return kErrorOk; }
  Error bl(const Label& t)   { phx_a64_bl(b_(), t); return kErrorOk; }
  Error blr(const Gp& t)     { phx_a64_blr(b_(), t); return kErrorOk; }
  Error br(const Gp& t)      { phx_a64_br(b_(), t); return kErrorOk; }
  Error b_eq(const Label& t) { phx_a64_b_cond(b_(), PHX_COND_EQ, t); return kErrorOk; }
  Error b_ne(const Label& t) { phx_a64_b_cond(b_(), PHX_COND_NE, t); return kErrorOk; }
  Error b_mi(const Label& t) { phx_a64_b_cond(b_(), PHX_COND_MI, t); return kErrorOk; }
  Error b_ge(const Label& t) { phx_a64_b_cond(b_(), PHX_COND_GE, t); return kErrorOk; }
  Error b_lt(const Label& t) { phx_a64_b_cond(b_(), PHX_COND_LT, t); return kErrorOk; }
  Error b_gt(const Label& t) { phx_a64_b_cond(b_(), PHX_COND_GT, t); return kErrorOk; }
  Error b_le(const Label& t) { phx_a64_b_cond(b_(), PHX_COND_LE, t); return kErrorOk; }
  Error b_hi(const Label& t) { phx_a64_b_cond(b_(), PHX_COND_HI, t); return kErrorOk; }
  Error b_ls(const Label& t) { phx_a64_b_cond(b_(), PHX_COND_LS, t); return kErrorOk; }
  Error b_cs(const Label& t) { phx_a64_b_cond(b_(), PHX_COND_CS, t); return kErrorOk; }
  Error b_cc(const Label& t) { phx_a64_b_cond(b_(), PHX_COND_CC, t); return kErrorOk; }
  Error b_lo(const Label& t) { phx_a64_b_cond(b_(), PHX_COND_CC, t); return kErrorOk; }
  Error b_hs(const Label& t) { phx_a64_b_cond(b_(), PHX_COND_CS, t); return kErrorOk; }
  Error b_vs(const Label& t) { phx_a64_b_cond(b_(), PHX_COND_VS, t); return kErrorOk; }
  Error b_vc(const Label& t) { phx_a64_b_cond(b_(), PHX_COND_VC, t); return kErrorOk; }
  Error b_pl(const Label& t) { phx_a64_b_cond(b_(), PHX_COND_PL, t); return kErrorOk; }
  Error cbz(const Gp& s, const Label& t)  { phx_a64_cbz(b_(), s, t); return kErrorOk; }
  Error cbnz(const Gp& s, const Label& t) { phx_a64_cbnz(b_(), s, t); return kErrorOk; }
  /* -- Sign/Zero extend -- */
  Error sxtb(const Gp& d, const Gp& s)  { phx_a64_sxtb(b_(), d, s); return kErrorOk; }
  Error sxth(const Gp& d, const Gp& s)  { phx_a64_sxth(b_(), d, s); return kErrorOk; }
  Error sxtw(const Gp& d, const Gp& s)  { phx_a64_sxtw(b_(), d, s); return kErrorOk; }
  Error uxtb(const Gp& d, const Gp& s)  { phx_a64_uxtb(b_(), d, s); return kErrorOk; }
  Error uxth(const Gp& d, const Gp& s)  { phx_a64_uxth(b_(), d, s); return kErrorOk; }
  /* -- RET -- */
  Error ret()                { phx_a64_ret(b_()); return kErrorOk; }
  Error ret(const Gp& t)    { phx_a64_ret_reg(b_(), t); return kErrorOk; }
  /* -- TBNZ (test bit and branch if nonzero) -- */
  Error tbnz(const Gp& s, uint32_t bit, const Label& t) {
    phx_a64_tbnz(b_(), s, bit, t); return kErrorOk;
  }
  /* -- UDF -- */
  Error udf(uint16_t v)     { phx_a64_udf(b_(), v); return kErrorOk; }
  /* -- Exclusive load/store -- */
  Error ldxr(const Gp& d, const Gp& base)                { phx_a64_ldxr(b_(), d, base); return kErrorOk; }
  Error stxr(const Gp& st, const Gp& s, const Gp& base)  { phx_a64_stxr(b_(), st, s, base); return kErrorOk; }
  /* -- MRS -- */
  Error mrs(const Gp& d, uint16_t sysreg) { phx_a64_mrs(b_(), d, sysreg); return kErrorOk; }
  /* -- FP arithmetic (Gp overloads for CRTP VecD resolution) -- */
  Error fadd(const Gp& d, const Gp& a, const Gp& c) { phx_a64_fadd(b_(), d, a, c); return kErrorOk; }
  Error fsub(const Gp& d, const Gp& a, const Gp& c) { phx_a64_fsub(b_(), d, a, c); return kErrorOk; }
  Error fmul(const Gp& d, const Gp& a, const Gp& c) { phx_a64_fmul(b_(), d, a, c); return kErrorOk; }
  Error fdiv(const Gp& d, const Gp& a, const Gp& c) { phx_a64_fdiv(b_(), d, a, c); return kErrorOk; }
  /* fmov with float immediate — in Builder class only (not in EmitterExplicitT
     to avoid ambiguity with fmov(Gp,Gp) for CRTP member function pointer resolution) */
};

class Builder : public EmitterExplicitT<Builder> {
 public:
  explicit Builder(PhxCodeHolder* code) : impl_(phx_builder_create(code)), code_(nullptr) {}
  explicit Builder(CodeHolder* code) : impl_(nullptr), code_(code) {
    if (code) {
      if (!code->get()) code->init();
      impl_ = phx_builder_create(code->get());
      code->setBuilder(impl_);
    }
  }
  ~Builder() { if (impl_) phx_builder_destroy(impl_); }

  /* fmov with float immediate — in Builder (not EmitterExplicitT)
     to avoid overload ambiguity in CRTP member function pointer resolution */
  Error fmov(const Gp& d, double imm) {
    union { double dv; uint64_t u; } val; val.dv = imm;
    Gp scratch(12, 8);
    phx_a64_mov_ri(impl_, scratch, (int64_t)val.u);
    phx_a64_fmov(impl_, d, scratch);
    return kErrorOk;
  }
  using EmitterExplicitT<Builder>::fmov; /* inherit fmov(Gp,Gp) */

  Builder(const Builder&) = delete;
  Builder& operator=(const Builder&) = delete;

  PhxBuilder* impl() const { return impl_; }

  Label newLabel() { return Label(phx_builder_new_label(impl_)); }
  Error bind(const Label& label) { phx_builder_bind(impl_, label); return kErrorOk; }

  BaseNode* cursor() { return phx_builder_cursor(impl_); }
  void setCursor(BaseNode* node) { phx_builder_set_cursor(impl_, node); }

  Error align(AlignMode, uint32_t alignment) {
    phx_builder_align(impl_, (int)alignment); return kErrorOk;
  }
  Error embed(const void* data, size_t size) {
    phx_builder_embed(impl_, data, size); return kErrorOk;
  }
  Error finalize() { return (impl_ && impl_->code) ? phx_a64_finalize(impl_) : 1; }

  Error section(void*) { return kErrorOk; }
  Builder& short_() { return *this; }
  Builder& long_() { return *this; }
  CodeHolder* code() const { return code_; }

 private:
  PhxBuilder* impl_;
  CodeHolder* code_;
};

} /* namespace phx */

#endif /* CINDER_AARCH64 */

#endif /* __cplusplus */

#endif /* PHOENIX_ASM_WRAPPER_H */
