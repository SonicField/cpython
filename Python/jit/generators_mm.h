// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include "cinderx/python.h"

#if PY_VERSION_HEX >= 0x030C0000

#include "cinderx/Common/ref.h"
#include "cinderx/Common/log.h"
#include "cinderx/Common/util.h"
#include "cinderx/Jit/gen_data_footer.h"
#include "cinderx/module_state.h"
#include "internal/pycore_object.h"
#if PY_VERSION_HEX >= 0x030E0000
#include "internal/pycore_interpframe.h"
#endif
#include "cinderx/Jit/generators_mm_iface.h"

#include <array>

namespace jit {

struct JitGenObject;

#if defined(ENABLE_LIGHTWEIGHT_FRAMES) && defined(__aarch64__)
static_assert(sizeof(GenDataFooter) == 80, "GenDataFooter size mismatch");
#elif defined(ENABLE_LIGHTWEIGHT_FRAMES) || defined(__aarch64__)
static_assert(sizeof(GenDataFooter) == 72, "GenDataFooter size mismatch");
#else
static_assert(sizeof(GenDataFooter) == 64, "GenDataFooter size mismatch");
#endif

inline size_t computeGenSlots(PyCodeObject* code, uint64_t jit_data_size) {
  static_assert(sizeof(uint64_t) == sizeof(PyObject*));
  return _PyFrame_NumSlotsForCodeObject(code) + 1 + ceilDiv(jit_data_size, 8);
}

inline std::pair<JitGenObject*, size_t> allocateNonFreeList(
    size_t slots, bool is_coro) {
  PyTypeObject* gen_tp = cinderx::getModuleState()->genType();
  size_t size = _PyObject_VAR_SIZE(gen_tp, slots);
  JitGenObject* gen = is_coro
      ? reinterpret_cast<JitGenObject*>(PyObject_GC_NewVar(
            PyCoroObject, cinderx::getModuleState()->coroType(), slots))
      : reinterpret_cast<JitGenObject*>(
            PyObject_GC_NewVar(PyGenObject, gen_tp, slots));
  JIT_CHECK(gen != nullptr, "Failed to allocate JitGenObject");
  return {gen, size};
}


// These values were determined experimentally on IG's webservers by utilizing
// the stats above. The number of outstanding requests seems to burst up to ~60k
// on startup but then quickly settles down to around 1-2k, so 2048 entries
// should be enough. The average size seems to be ~400 bytes with the max being
// about 10x that. Performance experiments showed a size of 512 was a greater
// improvement compared to 1024. Presumably the trade off in extra fixed memory
// allocation cost on workers isn't worth it for greater sizes.
constexpr size_t kGenFreeListEntries = 2048;
constexpr size_t kGenFreeListEntrySize = 512;

// Basically a free-list but the backing memory is pre-allocated in a single
// block. This makes it possible to determine if the storage is from this pool
// even after deopt by just examining a generator's pointer value.
class JitGenFreeList : public IJitGenFreeList {
 public:
  JitGenFreeList() { // NOLINT
    Entry* next = nullptr;
    for (size_t i = 0; i < kGenFreeListEntries; ++i) { entries_[i].next = next; next = &entries_[i]; }
    head_ = next;
  }
  ~JitGenFreeList() override = default;

  std::pair<JitGenObject*, size_t> allocate(PyCodeObject* code, uint64_t jit_data_size) override;
  void free(PyObject* ptr) override;
  bool contains(void* ptr) const override { return ptr >= &entries_ && ptr < &entries_[kGenFreeListEntries - 1] + 1; }

 private:
  void* rawAllocate() { JIT_DCHECK(head_, "No free generator entries"); Entry* e = head_; head_ = e->next; Py_INCREF(cinderx::getModuleState()->module()); return e->data; }
  bool fromThisArena(void* ptr) { return ptr >= &entries_ && ptr < &entries_[kGenFreeListEntries - 1] + 1; }

  struct Entry {
    union {
      uint8_t data[kGenFreeListEntrySize];
      Entry* next;
    };
  };

  std::array<Entry, kGenFreeListEntries> entries_;
  Entry* head_;
};

class JITGenFreeThreadedFreeList : public IJitGenFreeList {
 public:
  ~JITGenFreeThreadedFreeList() override = default;

  std::pair<JitGenObject*, size_t> allocate(
      PyCodeObject* code, uint64_t jit_data_size) override {
    size_t slots = computeGenSlots(code, jit_data_size);
    return allocateNonFreeList(slots, !!(code->co_flags & CO_COROUTINE));
  }
  void free(PyObject* ptr) override { PyObject_GC_Del(ptr); }
  bool contains(void* ptr) const override { return false; }
};

// Deferred inline definitions for JitGenFreeList complex methods.
inline std::pair<JitGenObject*, size_t> JitGenFreeList::allocate(
    PyCodeObject* code, uint64_t jit_data_size) {
  PyTypeObject* gen_tp = cinderx::getModuleState()->genType();
  size_t slots = computeGenSlots(code, jit_data_size);
  size_t size = _PyObject_VAR_SIZE(gen_tp, slots);
  size_t total_size = sizeof(PyGC_Head) + size;
  bool is_coro = !!(code->co_flags & CO_COROUTINE);
  if (!head_ || total_size > kGenFreeListEntrySize) {
    return allocateNonFreeList(slots, is_coro);
  }
  void* raw = rawAllocate();
  (reinterpret_cast<PyObject**>(raw))[0] = nullptr;
  (reinterpret_cast<PyObject**>(raw))[1] = nullptr;
  auto* op = reinterpret_cast<PyVarObject*>(
      reinterpret_cast<uintptr_t>(raw) + sizeof(PyGC_Head));
  PyTypeObject* tp = is_coro ? cinderx::getModuleState()->coroType() : gen_tp;
  _PyObject_InitVar(op, tp, slots);
  return {reinterpret_cast<JitGenObject*>(op), size};
}

inline void JitGenFreeList::free(PyObject* ptr) {
  if (!fromThisArena(ptr)) { PyObject_GC_Del(ptr); return; }
  auto* entry = reinterpret_cast<Entry*>(
      reinterpret_cast<uintptr_t>(ptr) - sizeof(PyGC_Head));
  entry->next = head_;
  head_ = entry;
  Py_DECREF(cinderx::getModuleState()->module());
}

} // namespace jit

#endif // PY_VERSION_HEX >= 0x030C0000
