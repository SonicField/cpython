// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <stddef.h>
#include <stdint.h>

/* ---- C API (implemented in mmap_file.c) ---- */
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const uint8_t *data;
    size_t size;
} JitMmapFile;

void jit_mmap_file_init(JitMmapFile *f);

/*
 * Open and mmap a file for reading.
 * On success, returns 0.  On error, returns -1 and writes a message
 * to errbuf (up to errbuf_size bytes).
 */
int jit_mmap_file_open(JitMmapFile *f, const char *filename,
                       char *errbuf, size_t errbuf_size);

/* Close and munmap.  Returns 0 on success, -1 on error. */
int jit_mmap_file_close(JitMmapFile *f);

#ifdef __cplusplus
} /* extern "C" */
#endif

/* ---- C++ convenience ---- */
#ifdef __cplusplus

#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>

namespace jit {

class MmapFile {
 public:
  constexpr MmapFile() = default;

  ~MmapFile() {
    jit_mmap_file_close(&f_);
  }

  void open(const char* filename) {
    char errbuf[256];
    int rc = jit_mmap_file_open(&f_, filename, errbuf, sizeof(errbuf));
    if (rc != 0) {
      throw std::runtime_error{errbuf};
    }
  }

  void close() {
    int rc = jit_mmap_file_close(&f_);
    if (rc != 0) {
      throw std::runtime_error{"Failed to munmap file"};
    }
  }

  bool isOpen() const {
    return f_.data != nullptr;
  }

  std::span<const std::byte> data() {
    return std::span<const std::byte>{
        reinterpret_cast<const std::byte*>(f_.data), f_.size};
  }

 private:
  JitMmapFile f_{nullptr, 0};
};

} // namespace jit

#endif
