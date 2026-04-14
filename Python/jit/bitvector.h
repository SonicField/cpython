// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// Phase 3D: C++ wrapper around PhxBitVector (bitvector_c.h/c).
// Implementation is in pure C; this header provides C++ operator syntax
// for existing callers (dataflow.h, analysis.h, etc.).

#pragma once

#include "cinderx/Jit/bitvector_c.h"
#include "cinderx/Common/log.h"
#include "fmt/ostream.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <iosfwd>
#include <type_traits>
#include <utility>

namespace jit::util {

class BitVector {
 public:
  BitVector() {
    phx_bv_init(&bv_, 0);
  }

  ~BitVector() {
    phx_bv_destroy(&bv_);
  }

  template <typename T>
  BitVector(size_t nb, T val) {
    static_assert(std::is_integral_v<T>, "val must be of an integral type.");
    JIT_CHECK(nb <= sizeof(void*) * 8, "Bit width is too large.")
    JIT_CHECK(
        nb == 64 || (val & ~((T{1} << nb) - 1)) == 0,
        "Val has too many bits for bit width");
    phx_bv_init_val(&bv_, nb, static_cast<uintptr_t>(val));
  }

  /* implicit */ BitVector(size_t size) {
    phx_bv_init(&bv_, size);
  }

  BitVector(const BitVector& other) {
    phx_bv_copy(&bv_, &other.bv_);
  }

  BitVector(BitVector&& other) {
    phx_bv_move(&bv_, &other.bv_);
  }

  BitVector& operator=(const BitVector& other) {
    if (this != &other) {
      phx_bv_destroy(&bv_);
      phx_bv_copy(&bv_, &other.bv_);
    }
    return *this;
  }

  BitVector& operator=(BitVector&& other) {
    if (this != &other) {
      phx_bv_destroy(&bv_);
      phx_bv_move(&bv_, &other.bv_);
    }
    return *this;
  }

  bool operator==(const BitVector& rhs) const {
    return phx_bv_equal(&bv_, &rhs.bv_) != 0;
  }
  bool operator!=(const BitVector& rhs) const {
    return !(*this == rhs);
  }

  BitVector operator&(const BitVector& rhs) const {
    BitVector r;
    r.bv_ = phx_bv_and(&bv_, &rhs.bv_);
    return r;
  }
  BitVector operator|(const BitVector& rhs) const {
    BitVector r;
    r.bv_ = phx_bv_or(&bv_, &rhs.bv_);
    return r;
  }
  BitVector operator-(const BitVector& rhs) const {
    BitVector r;
    r.bv_ = phx_bv_sub(&bv_, &rhs.bv_);
    return r;
  }

  BitVector& operator&=(const BitVector& rhs) {
    phx_bv_and_assign(&bv_, &rhs.bv_);
    return *this;
  }
  BitVector& operator|=(const BitVector& rhs) {
    phx_bv_or_assign(&bv_, &rhs.bv_);
    return *this;
  }
  BitVector& operator-=(const BitVector& rhs) {
    phx_bv_sub_assign(&bv_, &rhs.bv_);
    return *this;
  }

  void ResetAll() { phx_bv_reset_all(&bv_); }
  void fill(bool v) { phx_bv_fill(&bv_, v ? 1 : 0); }

  bool GetBit(size_t bit) const { return phx_bv_get_bit(&bv_, bit) != 0; }
  void SetBit(size_t bit, bool v = true) { phx_bv_set_bit(&bv_, bit, v ? 1 : 0); }

  uint64_t GetBitChunk(size_t chunk = 0) const { return phx_bv_get_chunk(&bv_, chunk); }
  void SetBitChunk(size_t chunk, uint64_t bits) { phx_bv_set_chunk(&bv_, chunk, bits); }

  size_t AddBits(size_t i) { return phx_bv_add_bits(&bv_, i); }
  void SetBitWidth(size_t size) { phx_bv_set_width(&bv_, size); }

  size_t GetNumBits() const { return phx_bv_num_bits(&bv_); }
  size_t GetPopCount() const { return phx_bv_popcount(&bv_); }
  bool IsEmpty() const { return phx_bv_is_empty(&bv_) != 0; }

  void forEachSetBit(std::function<void(size_t)> per_bit_func) const {
    // Wrap std::function in a C callback + context pair.
    phx_bv_for_each_set_bit(&bv_,
        [](size_t bit, void* ctx) {
          (*static_cast<std::function<void(size_t)>*>(ctx))(bit);
        },
        &per_bit_func);
  }

 private:
  PhxBitVector bv_;
};

inline std::ostream& operator<<(std::ostream& os, const BitVector& bv) {
  os << '[';
  for (std::size_t i = 0, n = bv.GetNumBits(); i < n; ++i) {
    if (i > 0 && (i % 8) == 0) {
      os << ';';
    }
    os << (bv.GetBit(i) ? '1' : '0');
  }
  os << ']';
  return os;
}

} // namespace jit::util

template <>
struct fmt::formatter<jit::util::BitVector> : fmt::ostream_formatter {};
