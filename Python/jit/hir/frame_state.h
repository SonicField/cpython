// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "cinderx/Common/log.h"
#include "cinderx/Jit/bytecode.h"
#include "cinderx/Jit/hir/phx_ptr_array.h"
#include "cinderx/Jit/hir/register.h"
#include "cinderx/Jit/stack.h"

namespace jit::hir {

// An entry in the CPython block stack
struct ExecutionBlock {
  // The CPython opcode for the block
  int opcode;

  // Offset in the bytecode of the handler for this block
  BCOffset handler_off;

  // Level to pop the operand stack when the block is exited
  int stack_level;

  bool operator==(const ExecutionBlock& other) const {
    return (opcode == other.opcode) && (handler_off == other.handler_off) &&
        (stack_level == other.stack_level);
  }

  bool operator!=(const ExecutionBlock& other) const {
    return !(*this == other);
  }

  bool isTryBlock() const {
    return opcode == SETUP_FINALLY;
  }

  bool isAsyncForHeaderBlock(const BytecodeInstructionBlock& instrs) const {
    return opcode == SETUP_FINALLY &&
        instrs.at(handler_off).opcode() == END_ASYNC_FOR;
  }
};

using BlockStack = jit::Stack<ExecutionBlock>;
using OperandStack = jit::Stack<Register*>;

// FrameState→C: C dynamic array for ExecutionBlock (replaces BlockStack).
struct PhxExecBlockArray {
  ExecutionBlock *data = nullptr;
  size_t count = 0;
  size_t capacity = 0;

  void destroy() { free(data); data = nullptr; count = 0; capacity = 0; }
  void clear() { count = 0; }
  void push(const ExecutionBlock& v) {
    if (count == capacity) {
      size_t new_cap = capacity ? capacity * 2 : 4;
      data = static_cast<ExecutionBlock*>(realloc(data, new_cap * sizeof(ExecutionBlock)));
      capacity = new_cap;
    }
    data[count++] = v;
  }
  ExecutionBlock pop() { return data[--count]; }
  const ExecutionBlock& top() const { return data[count - 1]; }
  const ExecutionBlock& at(size_t i) const { return data[i]; }
  size_t size() const { return count; }
  bool isEmpty() const { return count == 0; }

  bool operator==(const PhxExecBlockArray& other) const {
    if (count != other.count) return false;
    for (size_t i = 0; i < count; i++) {
      if (!(data[i] == other.data[i])) return false;
    }
    return true;
  }
  bool operator!=(const PhxExecBlockArray& other) const { return !(*this == other); }

  // Deep copy semantics (POD elements, malloc+memcpy).
  PhxExecBlockArray(const PhxExecBlockArray& other)
      : data(nullptr), count(other.count), capacity(other.count) {
    if (count) {
      data = static_cast<ExecutionBlock*>(malloc(count * sizeof(ExecutionBlock)));
      memcpy(data, other.data, count * sizeof(ExecutionBlock));
    }
  }
  PhxExecBlockArray& operator=(const PhxExecBlockArray& other) {
    if (this != &other) {
      free(data);
      data = nullptr;
      count = other.count;
      capacity = other.count;
      if (count) {
        data = static_cast<ExecutionBlock*>(malloc(count * sizeof(ExecutionBlock)));
        memcpy(data, other.data, count * sizeof(ExecutionBlock));
      }
    }
    return *this;
  }
  ~PhxExecBlockArray() { free(data); }

  PhxExecBlockArray() = default;
};

// The abstract state of the python frame
struct FrameState {
  FrameState() = default;
  FrameState(
      BorrowedRef<PyCodeObject> code,
      BorrowedRef<PyDictObject> globals,
      BorrowedRef<PyDictObject> builtins,
      FrameState* parent)
      : code(code), globals(globals), builtins(builtins), parent(parent) {
    JIT_DCHECK(this != parent, "FrameStates should not be self-referential");
  }

  // Used for testing only.
  explicit FrameState(BCOffset bc_off) : cur_instr_offs(bc_off) {}

  // FrameState→C step 2: explicit lifecycle with PhxPtrArray localsplus.
  // Shallow copy of Register* pointers (borrowed, not owned).
  FrameState(const FrameState& other)
      : cur_instr_offs(other.cur_instr_offs),
        localsplus{},
        nlocals(other.nlocals),
        stack{},
        block_stack(other.block_stack),
        code(other.code),
        globals(other.globals),
        builtins(other.builtins),
        parent(other.parent) {
    if (other.localsplus.count) {
      localsplus.data = static_cast<void**>(
          malloc(other.localsplus.count * sizeof(void*)));
      memcpy(localsplus.data, other.localsplus.data,
             other.localsplus.count * sizeof(void*));
      localsplus.count = other.localsplus.count;
      localsplus.capacity = other.localsplus.count;
    }
    if (other.stack.count) {
      stack.data = static_cast<void**>(
          malloc(other.stack.count * sizeof(void*)));
      memcpy(stack.data, other.stack.data,
             other.stack.count * sizeof(void*));
      stack.count = other.stack.count;
      stack.capacity = other.stack.count;
    }
    // block_stack deep-copied via PhxExecBlockArray copy ctor in initializer list.
  }

  FrameState& operator=(const FrameState& other) {
    if (this != &other) {
      cur_instr_offs = other.cur_instr_offs;
      phx_ptr_arr_destroy(&localsplus);
      if (other.localsplus.count) {
        localsplus.data = static_cast<void**>(
            malloc(other.localsplus.count * sizeof(void*)));
        memcpy(localsplus.data, other.localsplus.data,
               other.localsplus.count * sizeof(void*));
        localsplus.count = other.localsplus.count;
        localsplus.capacity = other.localsplus.count;
      }
      nlocals = other.nlocals;
      phx_ptr_arr_destroy(&stack);
      if (other.stack.count) {
        stack.data = static_cast<void**>(
            malloc(other.stack.count * sizeof(void*)));
        memcpy(stack.data, other.stack.data,
               other.stack.count * sizeof(void*));
        stack.count = other.stack.count;
        stack.capacity = other.stack.count;
      }
      block_stack = other.block_stack;  // PhxExecBlockArray operator= does deep copy
      code = other.code;
      globals = other.globals;
      builtins = other.builtins;
      parent = other.parent;
    }
    return *this;
  }

  ~FrameState() {
    phx_ptr_arr_destroy(&localsplus);
    phx_ptr_arr_destroy(&stack);
    // block_stack cleaned up by ~PhxExecBlockArray()
  }

  bool operator==(const FrameState& other) const {
    if (cur_instr_offs != other.cur_instr_offs ||
        localsplus.count != other.localsplus.count ||
        nlocals != other.nlocals ||
        stack.count != other.stack.count ||
        block_stack != other.block_stack ||
        code != other.code ||
        globals != other.globals ||
        builtins != other.builtins ||
        parent != other.parent) {
      return false;
    }
    for (size_t i = 0; i < localsplus.count; i++) {
      if (localsplus.data[i] != other.localsplus.data[i]) return false;
    }
    for (size_t i = 0; i < stack.count; i++) {
      if (stack.data[i] != other.stack.data[i]) return false;
    }
    return true;
  }

  bool operator!=(const FrameState& other) const {
    return !(*this == other);
  }

  // If the function is inlined into another function, the depth at which it
  // is inlined (nested function calls may be inlined). Starts at 1. If the
  // function is not inlined, 0.
  size_t inlineDepth() const {
    int depth = -1;
    for (auto frame = this; frame != nullptr; frame = frame->parent) {
      depth++;
    }
    return depth;
  }

  // The bytecode offset of the current instruction, or -sizeof(_Py_CODEUNIT) if
  // no instruction has executed. This corresponds to the `f_lasti` field of
  // PyFrameObject.
  BCOffset instrOffset() const {
    return cur_instr_offs;
  }

  bool visitUses(const std::function<bool(Register*&)>& func) {
    for (size_t i = 0; i < stack.count; i++) {
      Register*& reg = reinterpret_cast<Register*&>(stack.data[i]);
      if (!func(reg)) {
        return false;
      }
    }
    for (size_t i = 0; i < localsplus.count; i++) {
      Register*& reg = reinterpret_cast<Register*&>(localsplus.data[i]);
      if (reg != nullptr && !func(reg)) {
        return false;
      }
    }
    if (parent != nullptr) {
      return parent->visitUses(func);
    }
    return true;
  }

  // The currently executing instruction.
  BCOffset cur_instr_offs{-static_cast<ssize_t>(sizeof(_Py_CODEUNIT))};

  // Combination of local variables, cell variables (used by closures of inner
  // functions), and free variables (our closure). Locals are at the start and
  // free variables are at the end, but note locals can be cells so there is no
  // guarantee cells are all in the middle.
  // FrameState→C step 2: was std::vector<Register*>.
  PhxPtrArray localsplus{};

  // Number of local variables. Stored as a field directly because in tests
  // there's no code object for us to inspect.
  int nlocals{0};

  PhxPtrArray stack{};
  // FrameState→C step 2c: was BlockStack (jit::Stack<ExecutionBlock>).
  PhxExecBlockArray block_stack{};
  PyCodeObject* code{nullptr};
  PyDictObject* globals{nullptr};
  PyDictObject* builtins{nullptr};

  // Points to the FrameState, if any, into which this was inlined. Used to
  // construct the metadata needed to reify PyFrameObjects for inlined
  // functions during e.g. deopt.
  FrameState* parent{nullptr};
};

} // namespace jit::hir
