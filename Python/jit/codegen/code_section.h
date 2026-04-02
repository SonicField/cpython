// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "cinderx/Jit/codegen/arch.h"
#include "cinderx/Jit/config.h"

#if defined(PHOENIX_ASM) || defined(__aarch64__)
#include "jit/phoenix_asm/asmjit_compat.h"
#else
#include <asmjit/asmjit.h>
#endif

#include <cstring>
#include <vector>

/* ---- C API (implemented in code_section.c) ---- */
#ifdef __cplusplus
extern "C" {
#endif

const char* jit_code_section_name(int section);
int jit_code_section_from_name(const char* name);

#ifdef __cplusplus
} /* extern "C" */
#endif

namespace jit::codegen {

enum class CodeSection {
  kHot,
  kCold,
};

/* Inline C++ wrappers that forward to the C functions */
inline const char* codeSectionName(CodeSection section) {
  return jit_code_section_name(static_cast<int>(section));
}

inline CodeSection codeSectionFromName(const char* name) {
  return static_cast<CodeSection>(jit_code_section_from_name(name));
}

class CodeHolderMetadata {
 public:
  explicit CodeHolderMetadata(CodeSection section) : section_(section) {}

  void setSection(CodeSection section) {
    section_ = section;
  }

 private:
  friend class CodeSectionOverride;

  CodeSection section_;
};

// RAII device for overriding the previous code section.
class CodeSectionOverride {
 public:
  CodeSectionOverride() = delete;
  CodeSectionOverride(const CodeSectionOverride&) = delete;
  CodeSectionOverride& operator=(const CodeSectionOverride&) = delete;
  CodeSectionOverride(CodeSectionOverride&&) = delete;
  CodeSectionOverride& operator=(CodeSectionOverride&&) = delete;

  CodeSectionOverride(
      arch::Builder* as,
      const asmjit::CodeHolder* code,
      CodeHolderMetadata* metadata,
      CodeSection section)
      : as_{as}, code_{code}, metadata_{metadata} {
    if (getConfig().multiple_code_sections) {
      previous_section_ = metadata->section_;
      metadata->section_ = section;
      as->section(code->sectionByName(codeSectionName(section)));
    } else {
      previous_section_ = section;
    }
  }

  ~CodeSectionOverride() {
    // Guard against partial initialization to make GCC happy.
    if (as_ == nullptr || code_ == nullptr) {
      return;
    }
    if (getConfig().multiple_code_sections) {
      as_->section(code_->sectionByName(codeSectionName(previous_section_)));
      metadata_->section_ = previous_section_;
    }
  }

 private:
  arch::Builder* as_;
  const asmjit::CodeHolder* code_;
  CodeSection previous_section_;
  CodeHolderMetadata* metadata_;
};

// Call f with each code section.
template <typename F>
void forEachSection(F f) {
  f(CodeSection::kHot);
  f(CodeSection::kCold);
}

// Inlined here (was in code_section.cpp) — uses C++ types (std::vector, CodeHolder).
inline void populateCodeSections(
    std::vector<std::pair<void*, std::size_t>>& code_sections,
    asmjit::CodeHolder& code,
    void* code_base_ptr) {
  forEachSection([&](CodeSection section) {
    auto asmjit_section = code.sectionByName(codeSectionName(section));
    if (asmjit_section == nullptr || asmjit_section->realSize() == 0) {
      return;
    }
    auto section_start =
        static_cast<char*>(code_base_ptr) + asmjit_section->offset();
    code_sections.emplace_back(
        reinterpret_cast<void*>(section_start), asmjit_section->realSize());
  });
}

} // namespace jit::codegen
