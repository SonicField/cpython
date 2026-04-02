// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "cinderx/Jit/codegen/copy_graph_c.h"

#include "cinderx/Common/log.h"
#include "cinderx/Common/util.h"

#include <unordered_map>
#include <utility>
#include <vector>

namespace jit::codegen {

// CopyGraph is used to generate a sequence of copies and/or exchanges to
// shuffle data between registers (non-negative ints) and memory locations
// (negative ints).
//
// Every location may have up to one incoming edge and arbitrarily many
// outgoing edges.
//
// CopyGraph::kTempLoc is used to indicate an arbitrary temporary location that
// is used to break cycles involving memory operands. The choice of this
// location, including ensuring that it doesn't conflict with any locations in
// the graph, is up to the caller.
class CopyGraph {
 public:
  static constexpr int kTempLoc = JIT_COPY_GRAPH_TEMP_LOC;

  struct Op {
    enum class Kind {
      kCopy,
      kExchange,
    };

    Op(Kind kind, int from, int to) : kind{kind}, from{from}, to{to} {}

    bool operator==(const Op& other) const {
      return kind == other.kind && from == other.from && to == other.to;
    }

    Kind kind;
    int from;
    int to;
  };

  CopyGraph() : impl_(jit_copy_graph_create()) {}
  ~CopyGraph() { jit_copy_graph_destroy(impl_); }

  // Non-copyable
  CopyGraph(const CopyGraph&) = delete;
  CopyGraph& operator=(const CopyGraph&) = delete;

  // Add a copy edge to the graph.
  void addEdge(int from, int to) {
    jit_copy_graph_add_edge(impl_, from, to);
  }

  // Process the graph and return the sequence of copies and/or exchanges.
  std::vector<Op> process() {
    Py_ssize_t count = 0;
    JitCopyOp* raw = jit_copy_graph_process(impl_, &count);
    std::vector<Op> result;
    result.reserve(count);
    for (Py_ssize_t i = 0; i < count; i++) {
      result.emplace_back(
          raw[i].kind == JIT_COPY_OP_COPY ? Op::Kind::kCopy
                                          : Op::Kind::kExchange,
          raw[i].from,
          raw[i].to);
    }
    jit_copy_graph_ops_free(raw);
    return result;
  }

  // Check if the copy graph is empty.
  bool isEmpty() const {
    return jit_copy_graph_is_empty(impl_);
  }

 private:
  JitCopyGraph* impl_;
};

// the same as CopyGraph, but preserves certain types of `from` nodes.
template <typename FromType>
class CopyGraphWithType : public CopyGraph {
 public:
  struct Op : CopyGraph::Op {
    Op(const CopyGraph::Op& op, FromType t) : CopyGraph::Op(op), type(t) {}

    const FromType type;
  };

  void addEdge(int from, int to, FromType type) {
    auto pair = from_types_.emplace(from, type);
    JIT_DCHECK(
        pair.second || pair.first->second == type,
        "Different type for from {}.",
        from);

    CopyGraph::addEdge(from, to);
  }

  std::vector<Op> process() {
    auto ops = CopyGraph::process();
    std::vector<Op> ret;
    ret.reserve(ops.size());

    for (auto& op : ops) {
      auto from_type = map_get(from_types_, op.from);
      ret.emplace_back(op, from_type);

      if (op.to == kTempLoc) {
        from_types_[kTempLoc] = from_type;
      }
    }

    return ret;
  }

 private:
  std::unordered_map<int, std::remove_cv_t<FromType>> from_types_;
};

} // namespace jit::codegen
