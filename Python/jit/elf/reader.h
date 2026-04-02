// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

/* ---- C API (implemented in reader.c) ---- */
#include "cinderx/Jit/elf/reader_c.h"

#include "cinderx/Jit/elf/note.h"

#include <cstddef>
#include <iosfwd>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>

#include <fmt/format.h>

namespace jit::elf {

// Find an ELF section by name from an ELF file.
// Returns an empty span if the section cannot be found.
// Throws std::runtime_error on malformed ELF.
inline std::span<const std::byte> findSection(
    std::span<const std::byte> elf,
    std::string_view name) {
  const uint8_t *out_data = nullptr;
  size_t out_size = 0;
  std::string name_str(name);
  int rc = jit_elf_find_section(
      reinterpret_cast<const uint8_t*>(elf.data()), elf.size(),
      name_str.c_str(), &out_data, &out_size);
  if (rc != 0) {
    throw std::runtime_error{"ELF section headers are invalid"};
  }
  if (out_data == nullptr) {
    return std::span<const std::byte>{};
  }
  return std::span<const std::byte>{
      reinterpret_cast<const std::byte*>(out_data), out_size};
}

// Read the ELF notes out of an ELF section (from raw bytes).
inline NoteArray readNoteSection(std::span<const std::byte> bytes) {
  JitElfNoteArray c_arr;
  int rc = jit_elf_read_note_section(
      reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size(), &c_arr);
  if (rc != 0) {
    throw std::runtime_error{"Failed to read note section"};
  }
  NoteArray result;
  for (size_t i = 0; i < jit_elf_note_array_len(&c_arr); i++) {
    const JitElfNote *cn = jit_elf_note_array_get(&c_arr, i);
    result.insert(Note::fromC(*cn));
  }
  jit_elf_note_array_free(&c_arr);
  return result;
}

// Read the ELF notes out of an ELF section (from istream).
inline NoteArray readNoteSection(std::istream& is, size_t size) {
  std::string buf(size, '\0');
  is.read(buf.data(), size);
  JitElfNoteArray c_arr;
  int rc = jit_elf_read_note_section(
      reinterpret_cast<const uint8_t*>(buf.data()),
      static_cast<size_t>(is.gcount()), &c_arr);
  if (rc != 0) {
    throw std::runtime_error{"Failed to read note section"};
  }
  NoteArray result;
  for (size_t i = 0; i < jit_elf_note_array_len(&c_arr); i++) {
    const JitElfNote *cn = jit_elf_note_array_get(&c_arr, i);
    result.insert(Note::fromC(*cn));
  }
  jit_elf_note_array_free(&c_arr);
  return result;
}

// Parse a function's code note data out of an ELF note.
inline CodeNoteData parseCodeNote(const Note& note) {
  JitElfNote cn = note.toC();
  JitElfCodeNoteData cd;
  int rc = jit_elf_parse_code_note(&cn, &cd);
  jit_elf_note_free(&cn);
  if (rc != 0) {
    throw std::runtime_error{"Failed to parse code note"};
  }
  CodeNoteData result;
  result.file_name = cd.file_name ? cd.file_name : "";
  result.lineno = cd.lineno;
  result.hash = cd.hash;
  result.size = cd.size;
  result.normal_entry_offset = cd.normal_entry_offset;
  result.static_entry_offset = cd.has_static_entry
      ? std::make_optional(cd.static_entry_offset)
      : std::nullopt;
  jit_elf_code_note_data_free(&cd);
  return result;
}

} // namespace jit::elf
