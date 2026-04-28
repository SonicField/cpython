// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "cinderx/Jit/hir/printer.h"

#include "cinderx/Common/log.h"          /* jit::repr (P-5b LoadAttrSpecial) */
#include "cinderx/Common/util.h"
#include "cinderx/Jit/code_patcher.h"     /* JumpPatcher (P-5b DeoptPatchpoint) */
#include "cinderx/Jit/hir/printer_c.h"  /* PhxHirPrinter (B2 commit-2 bridges) */
#include "cinderx/Jit/symbolizer.h"

#if PY_VERSION_HEX >= 0x030E0000
#include "pycore_ceval.h"
#include "pycore_intrinsics.h"
#endif

#include <fmt/format.h>
#include <fmt/ostream.h>

#include <algorithm>
#include <ostream>
#include <sstream>
#include <vector>

namespace jit::hir {

namespace {

const char* fvc_to_string(int conversion) {
  static const char* kFvcStrings[] = {
      "None", // FVC_NONE (0)
      "Str", // FVC_STR (1)
      "Repr", // FVC_REPR (2)
      "ASCII" // FVC_ASCII (3)
  };

  if (conversion >= 0 && conversion < 4) {
    return kFvcStrings[conversion];
  }
  return "Unknown";
}
} // namespace

// B2 commit-5b sole-path swap: HIRPrinter::Print(*) bodies route
// through the C-side phx_hir_print_* ports via open_memstream.
// Indent/Dedent/Indented are no longer used (C-side has its own
// inline equivalents in printer_c.h); their bodies and the
// indent_level_ mutators are deleted post-swap.
namespace {
// Build a C-side PhxHirPrinter from this C++ HIRPrinter state, run
// `body` on it, and pipe the open_memstream output to the C++ stream.
template <typename Body>
void run_via_c_side(
    std::ostream& os,
    const std::string& line_prefix,
    int indent_level,
    bool full_snapshots,
    const Function* func,
    Body&& body) {
  char* buf = nullptr;
  size_t size = 0;
  FILE* f = open_memstream(&buf, &size);
  if (f == nullptr) {
    return;
  }
  PhxHirPrinter p = phx_hir_printer_default();
  if (!line_prefix.empty()) {
    p.line_prefix = line_prefix.c_str();
  }
  p.indent_level = indent_level;
  p.full_snapshots = full_snapshots ? 1 : 0;
  p.func = static_cast<const void*>(func);
  body(f, &p);
  fclose(f);
  if (buf != nullptr) {
    os.write(buf, static_cast<std::streamsize>(size));
    free(buf);
  }
}
} // namespace

void HIRPrinter::Print(std::ostream& os, const Function& func) {
  run_via_c_side(os, line_prefix_, indent_level_, full_snapshots_, &func,
      [&](FILE* f, PhxHirPrinter* p) {
        phx_hir_print_function(f, p, &func);
      });
}

void HIRPrinter::Print(std::ostream& os, const CFG& cfg) {
  run_via_c_side(os, line_prefix_, indent_level_, full_snapshots_, func_,
      [&](FILE* f, PhxHirPrinter* p) {
        phx_hir_print_cfg(f, p, &cfg);
      });
}

void HIRPrinter::Print(std::ostream& os, const BasicBlock& block) {
  run_via_c_side(os, line_prefix_, indent_level_, full_snapshots_, func_,
      [&](FILE* f, PhxHirPrinter* p) {
        phx_hir_print_basic_block(f, p, &block);
      });
}

// B2 commit-5b: print_reg_states C++ static RETAINED — it remains a
// dependency of the still-C++ format_immediates RaiseStatic case
// (printer.cpp ~line 665), part of the bridged cluster covered by the
// W-PRINTER-IMMEDIATES-PORT residual workstream.  HIRPrinter::Print(Instr)
// no longer calls it (delegates to C-side phx_hir_print_instr →
// phx_hir_print_reg_states from commit-5a); kept for the in-bridge
// consumer until the full format_immediates port lands.
static void print_reg_states(
    std::ostream& os,
    const PhxRegStateArray& reg_states) {
  auto rss = reg_states;
  std::sort(rss.begin(), rss.end(), [](RegState& a, RegState& b) {
    return a.reg->id() < b.reg->id();
  });
  os << fmt::format("<{}>", rss.size());
  if (!rss.empty()) {
    os << " ";
  }
  auto sep = "";
  for (auto& reg_state : rss) {
    const char* prefix = "?";
    switch (reg_state.value_kind) {
      case ValueKind::kSigned: prefix = "s"; break;
      case ValueKind::kUnsigned: prefix = "uns"; break;
      case ValueKind::kBool: prefix = "bool"; break;
      case ValueKind::kDouble: prefix = "double"; break;
      case ValueKind::kObject: {
        switch (reg_state.ref_kind) {
          case RefKind::kUncounted: prefix = "unc"; break;
          case RefKind::kBorrowed: prefix = "b"; break;
          case RefKind::kOwned: prefix = "o"; break;
        }
        break;
      }
    }
    os << fmt::format("{}{}:{}", sep, prefix, reg_state.reg->name());
    sep = " ";
  }
}

static const int kMaxASCII = 127;

static std::string escape_unicode(const char* data, Py_ssize_t size) {
  std::string ret = "\"";
  for (Py_ssize_t i = 0; i < size; ++i) {
    char c = data[i];
    switch (c) {
      case '"':
      case '\\':
        ret += '\\';
        ret += c;
        break;
      case '\n':
        ret += "\\n";
        break;
      default:
        if (static_cast<unsigned char>(c) > kMaxASCII) {
          ret += '\\';
          ret += std::to_string(static_cast<unsigned char>(c));
        } else {
          ret += c;
        }
        break;
    }
  }
  ret += '"';
  return ret;
}

static std::string escape_unicode(PyObject* str) {
  Py_ssize_t size;
  const char* data = PyUnicode_AsUTF8AndSize(str, &size);
  if (data == nullptr) {
    PyErr_Clear();
    return "";
  }
  return escape_unicode(data, size);
}

static std::string format_name_impl(int idx, PyObject* names) {
  return fmt::format(
      "{}; {}", idx, escape_unicode(PyTuple_GET_ITEM(names, idx)));
}

static std::string
format_name(const Function* func, const Instr& instr, int idx) {
  auto code = func != nullptr ? func->codeFor(instr) : nullptr;
  if (idx < 0 || code == nullptr) {
    return fmt::format("{}", idx);
  }

  return format_name_impl(idx, code->co_names);
}

static std::string format_load_super(
    const Function* func,
    const LoadSuperBase& load) {
  auto code = func != nullptr ? func->codeFor(load) : nullptr;
  if (code == nullptr) {
    return fmt::format("{} {}", load.name_idx(), load.no_args_in_super_call());
  }
  return fmt::format(
      "{}, {}",
      format_name_impl(load.name_idx(), code->co_names),
      load.no_args_in_super_call());
}

static std::string
format_varname(const Function* func, const Instr& instr, int idx) {
  auto code = func != nullptr ? func->codeFor(instr) : nullptr;
  if (idx < 0 || code == nullptr) {
    return fmt::format("{}", idx);
  }

  auto names = getVarnameTuple(code, &idx);
  return format_name_impl(idx, names);
}

static std::string format_immediates(const Function* func, const Instr& instr) {
  switch (instr.opcode()) {
    case Opcode::kAssign:
    case Opcode::kBatchDecref:
    case Opcode::kBitCast:
    case Opcode::kBuildString:
    case Opcode::kBuildTemplate:
    case Opcode::kCheckErrOccurred:
    case Opcode::kCheckExc:
    case Opcode::kCheckNeg:
    case Opcode::kCheckSequenceBounds:
    case Opcode::kCIntToCBool:
    case Opcode::kCopyDictWithoutKeys:
    case Opcode::kDecref:
    case Opcode::kDeleteSubscr:
    case Opcode::kDeopt:
    case Opcode::kDictMerge:
    case Opcode::kDictSubscr:
    case Opcode::kDictUpdate:
    case Opcode::kEndInlinedFunction:
    case Opcode::kFormatWithSpec:
    case Opcode::kGetAIter:
    case Opcode::kGetANext:
    case Opcode::kGetIter:
    case Opcode::kGetLength:
    case Opcode::kGetTuple:
    case Opcode::kGuard:
    case Opcode::kIncref:
    case Opcode::kInitialYield:
    case Opcode::kInvokeIterNext:
    case Opcode::kIsInstance:
    case Opcode::kIsNegativeAndErrOccurred:
    case Opcode::kIsTruthy:
    case Opcode::kListAppend:
    case Opcode::kListExtend:
    case Opcode::kLoadCellItem:
    case Opcode::kLoadCurrentFunc:
    case Opcode::kLoadFrame:
    case Opcode::kLoadEvalBreaker:
    case Opcode::kAtQuiescentState:
    case Opcode::kLoadFieldAddress:
    case Opcode::kLoadVarObjectSize:
    case Opcode::kMakeCell:
    case Opcode::kMakeFunction:
    case Opcode::kMakeSet:
    case Opcode::kMakeTupleFromList:
    case Opcode::kMatchClass:
    case Opcode::kMatchKeys:
    case Opcode::kMergeSetUnpack:
    case Opcode::kPrimitiveBoxBool:
    case Opcode::kRaise:
    case Opcode::kRunPeriodicTasks:
    case Opcode::kSend:
    case Opcode::kSetCurrentAwaiter:
    case Opcode::kSetCellItem:
    case Opcode::kSetDictItem:
    case Opcode::kSetSetItem:
    case Opcode::kSetUpdate:
    case Opcode::kSnapshot:
    case Opcode::kStealCellItem:
    case Opcode::kSwapCellItem:
    case Opcode::kStoreArrayItem:
    case Opcode::kStoreSubscr:
    case Opcode::kWaitHandleLoadCoroOrResult:
    case Opcode::kWaitHandleLoadWaiter:
    case Opcode::kWaitHandleRelease:
    case Opcode::kXDecref:
    case Opcode::kXIncref:
    case Opcode::kYieldAndYieldFrom:
    case Opcode::kYieldFrom:
    case Opcode::kYieldFromHandleStopAsyncIteration:
    case Opcode::kUnicodeConcat:
    case Opcode::kUnicodeRepeat:
    case Opcode::kUnicodeSubscr:
    case Opcode::kUnreachable:
    case Opcode::kYieldValue: {
      return "";
    }
    case Opcode::kBeginInlinedFunction:
      return static_cast<const BeginInlinedFunction&>(instr).fullname();
    case Opcode::kLoadArrayItem: {
      const auto& load = static_cast<const LoadArrayItem&>(instr);
      return load.offset() == 0 ? "" : fmt::format("Offset[{}]", load.offset());
    }
    case Opcode::kLoadSplitDictItem: {
      const auto& load = static_cast<const LoadSplitDictItem&>(instr);
      return fmt::format("{}", load.itemIdx());
    }
    case Opcode::kReturn: {
      const auto& ret = static_cast<const Return&>(instr);
      return ret.type() != TObject ? ret.type().toString() : "";
    }
    case Opcode::kCallEx: {
      const auto& call = static_cast<const CallEx&>(instr);
      return fmt::format(
          "{}{}",
          (call.flags() & CallFlags::Awaited) ? ", awaited" : "",
          (call.flags() & CallFlags::KwArgs) ? ", kwargs" : "");
    }
    case Opcode::kCallInd: {
      const auto& call = static_cast<const CallInd&>(instr);
      return fmt::format("{}", call.name());
    }
    case Opcode::kBinaryOp: {
      const auto& bin_op = static_cast<const BinaryOp&>(instr);
      return std::string{GetBinaryOpName(bin_op.op())};
    }
    case Opcode::kUnaryOp: {
      const auto& unary_op = static_cast<const UnaryOp&>(instr);
      return std::string{GetUnaryOpName(unary_op.op())};
    }
    case Opcode::kBranch: {
      const auto& branch = static_cast<const Branch&>(instr);
      return fmt::format("{}", branch.target()->id);
    }
    case Opcode::kVectorCall: {
      const auto& call = static_cast<const VectorCall&>(instr);
      return fmt::format(
          "{}{}{}{}",
          call.numArgs(),
          (call.flags() & CallFlags::Awaited) ? ", awaited" : "",
          (call.flags() & CallFlags::KwArgs) ? ", kwnames" : "",
          (call.flags() & CallFlags::Static) ? ", static" : "");
    }
    case Opcode::kCallCFunc: {
      const auto& call = static_cast<const CallCFunc&>(instr);
      return std::string{call.funcName()};
    }
    case Opcode::kCallIntrinsic: {
      const auto& call = static_cast<const CallIntrinsic&>(instr);
#if PY_VERSION_HEX >= 0x030E0000
      switch (call.NumOperands()) {
        case 1:
          return _PyIntrinsics_UnaryFunctions[call.index()].name;
        case 2:
          return _PyIntrinsics_BinaryFunctions[call.index()].name;
        default:
          JIT_ABORT("Invalid number of intrinsic args: {}", call.NumOperands());
      }
#else
      return fmt::format("{}", call.index());
#endif
    }
    case Opcode::kCallMethod: {
      const auto& call = static_cast<const CallMethod&>(instr);
      return fmt::format(
          "{}{}",
          call.NumOperands(),
          (call.flags() & CallFlags::Awaited) ? ", awaited" : "");
    }
    case Opcode::kCallStatic: {
      const auto& call = static_cast<const CallStatic&>(instr);
      std::optional<std::string> func_name = symbolize(call.addr());
      if (func_name.has_value()) {
        return fmt::format(
            "{}@{}, {}",
            *func_name,
            getStablePointer(call.addr()),
            call.NumOperands());
      }
      return fmt::format(
          "{}, {}", getStablePointer(call.addr()), call.NumOperands());
    }
    case Opcode::kCallStaticRetVoid: {
      const auto& call = static_cast<const CallStaticRetVoid&>(instr);
      std::optional<std::string> func_name = symbolize(call.addr());
      if (func_name.has_value()) {
        return fmt::format(
            "{}@{}, {}",
            *func_name,
            getStablePointer(call.addr()),
            call.NumOperands());
      }
      return fmt::format(
          "{}, {}", getStablePointer(call.addr()), call.NumOperands());
    }
    case Opcode::kInvokeStaticFunction: {
      const auto& call = static_cast<const InvokeStaticFunction&>(instr);
      return fmt::format(
          "{}.{}, {}, {}",
          PyUnicode_AsUTF8(call.func()->func_module),
          PyUnicode_AsUTF8(call.func()->func_qualname),
          call.NumOperands(),
          call.ret_type());
    }
    case Opcode::kInitFrameCellVars: {
      const auto& init = static_cast<const InitFrameCellVars&>(instr);
      return fmt::format("{}", init.num_cell_vars());
    }
    case Opcode::kLoadField: {
      const auto& lf = static_cast<const LoadField&>(instr);
      std::size_t offset = lf.offset();
#ifdef Py_TRACE_REFS
      // Keep these stable from the offset of ob_refcnt, in trace refs
      // we have 2 extra next/prev pointers linking all objects together
      offset -= (sizeof(PyObject*) * 2);
#endif

      return fmt::format(
          "{}@{}, {}, {}",
          lf.name(),
          offset,
          lf.type(),
          lf.borrowed() ? "borrowed" : "owned");
    }
    case Opcode::kStoreField: {
      const auto& sf = static_cast<const StoreField&>(instr);
      return fmt::format("{}@{}", sf.name(), sf.offset());
    }
    case Opcode::kCast: {
      const auto& cast = static_cast<const Cast&>(instr);
      std::string result = cast.pytype()->tp_name;
      if (cast.exact()) {
        result = fmt::format("Exact[{}]", result);
      }
      if (cast.optional()) {
        result = fmt::format("Optional[{}]", result);
      }
      return result;
    }
    case Opcode::kTpAlloc: {
      const auto& tp_alloc = static_cast<const TpAlloc&>(instr);
      return fmt::format("{}", tp_alloc.pytype()->tp_name);
    }
    case Opcode::kCompare: {
      const auto& cmp = static_cast<const Compare&>(instr);
      return std::string{GetCompareOpName(cmp.op())};
    }
    case Opcode::kFloatCompare: {
      const auto& cmp = static_cast<const FloatCompare&>(instr);
      return std::string{GetCompareOpName(cmp.op())};
    }
    case Opcode::kLongCompare: {
      const auto& cmp = static_cast<const LongCompare&>(instr);
      return std::string{GetCompareOpName(cmp.op())};
    }
    case Opcode::kUnicodeCompare: {
      const auto& cmp = static_cast<const UnicodeCompare&>(instr);
      return std::string{GetCompareOpName(cmp.op())};
    }
    case Opcode::kLongBinaryOp: {
      const auto& bin = static_cast<const LongBinaryOp&>(instr);
      return std::string{GetBinaryOpName(bin.op())};
    }
    case Opcode::kLongInPlaceOp: {
      const auto& inplace = static_cast<const LongInPlaceOp&>(instr);
      return std::string{GetInPlaceOpName(inplace.op())};
    }
    case Opcode::kFloatBinaryOp: {
      const auto& bin = static_cast<const FloatBinaryOp&>(instr);
      return std::string{GetBinaryOpName(bin.op())};
    }
    case Opcode::kCompareBool: {
      const auto& cmp = static_cast<const Compare&>(instr);
      return std::string{GetCompareOpName(cmp.op())};
    }
    case Opcode::kIntConvert: {
      const auto& conv = static_cast<const IntConvert&>(instr);
      return conv.type().toString();
    }
    case Opcode::kPrimitiveUnaryOp: {
      const auto& unary = static_cast<const PrimitiveUnaryOp&>(instr);
      return std::string{GetPrimitiveUnaryOpName(unary.op())};
    }
    case Opcode::kCondBranch:
    case Opcode::kCondBranchIterNotDone:
    case Opcode::kCondBranchCheckType: {
      const auto& cond = static_cast<const CondBranchBase&>(instr);
      auto targets =
          fmt::format("{}, {}", cond.true_bb()->id, cond.false_bb()->id);
      if (cond.IsCondBranchCheckType()) {
        Type type = static_cast<const CondBranchCheckType&>(cond).type();
        return fmt::format("{}, {}", targets, type);
      }
      return targets;
    }
    case Opcode::kDoubleBinaryOp: {
      const auto& bin_op = static_cast<const DoubleBinaryOp&>(instr);
      return std::string{GetBinaryOpName(bin_op.op())};
    }
    case Opcode::kLoadArg: {
      const auto& load = static_cast<const LoadArg&>(instr);
      auto varname = format_varname(func, load, load.arg_idx());
      if (load.type() == TObject) {
        return varname;
      }
      return fmt::format("{}, {}", varname, load.type());
    }
    case Opcode::kLoadAttrSpecial: {
      const auto& load = static_cast<const LoadAttrSpecial&>(instr);
#if PY_VERSION_HEX < 0x030C0000
      _Py_Identifier* id = load.id();
      return fmt::format("\"{}\"", id->string);
#else
      return fmt::format("\"{}\"", repr(load.id()));
#endif
    }
    case Opcode::kLoadMethod:
    case Opcode::kLoadMethodCached:
    case Opcode::kLoadModuleMethodCached: {
      const auto& load = static_cast<const LoadMethodBase&>(instr);
      return format_name(func, load, load.name_idx());
    }
    case Opcode::kLoadMethodSuper: {
      return format_load_super(func, static_cast<const LoadSuperBase&>(instr));
    }
    case Opcode::kLoadAttrSuper: {
      return format_load_super(func, static_cast<const LoadSuperBase&>(instr));
    }
    case Opcode::kLoadConst: {
      const auto& load = static_cast<const LoadConst&>(instr);
      return fmt::format("{}", load.type());
    }
    case Opcode::kLoadFunctionIndirect: {
      const auto& load = static_cast<const LoadFunctionIndirect&>(instr);
      PyObject* py_func = *load.funcptr();
      const char* name;
      if (PyFunction_Check(py_func)) {
        name = PyUnicode_AsUTF8(((PyFunctionObject*)py_func)->func_name);
      } else {
        name = Py_TYPE(py_func)->tp_name;
      }
      return fmt::format("{}", name);
    }
    case Opcode::kIntBinaryOp: {
      const auto& bin_op = static_cast<const IntBinaryOp&>(instr);
      return std::string{GetBinaryOpName(bin_op.op())};
    }
    case Opcode::kPrimitiveCompare: {
      const auto& cmp = static_cast<const PrimitiveCompare&>(instr);
      return std::string{GetPrimitiveCompareOpName(cmp.op())};
    }
    case Opcode::kPrimitiveBox: {
      const auto& box = static_cast<const PrimitiveBox&>(instr);
      return fmt::format("{}", box.type());
    }
    case Opcode::kPrimitiveUnbox: {
      const auto& unbox = static_cast<const PrimitiveUnbox&>(instr);
      return fmt::format("{}", unbox.type());
    }
    case Opcode::kIndexUnbox: {
      const auto& unbox = static_cast<const IndexUnbox&>(instr);
      return fmt::format(
          "{}", reinterpret_cast<PyTypeObject*>(unbox.exception())->tp_name);
    }
    case Opcode::kLoadGlobalCached: {
      const auto& load = static_cast<const LoadGlobalCached&>(instr);
      return format_name(func, load, load.name_idx());
    }
    case Opcode::kLoadGlobal: {
      const auto& load = static_cast<const DeoptBaseWithNameIdx&>(instr);
      return format_name(func, instr, load.name_idx());
    }
    case Opcode::kMakeList: {
      const auto& make = static_cast<const MakeList&>(instr);
      return fmt::format("{}", make.nvalues());
    }
    case Opcode::kMakeTuple: {
      const auto& make = static_cast<const MakeTuple&>(instr);
      return fmt::format("{}", make.nvalues());
    }
    case Opcode::kGetSecondOutput: {
      return fmt::format(
          "{}", static_cast<const GetSecondOutput&>(instr).type());
    }
    case Opcode::kLoadTupleItem: {
      const auto& loaditem = static_cast<const LoadTupleItem&>(instr);
      return fmt::format("{}", loaditem.idx());
    }
    case Opcode::kMakeCheckedDict: {
      const auto& makedict = static_cast<const MakeCheckedDict&>(instr);
      return fmt::format("{} {}", makedict.type(), makedict.GetCapacity());
    }
    case Opcode::kMakeCheckedList: {
      const auto& makelist = static_cast<const MakeCheckedList&>(instr);
      return fmt::format("{} {}", makelist.type(), makelist.nvalues());
    }
    case Opcode::kMakeDict: {
      const auto& makedict = static_cast<const MakeDict&>(instr);
      return fmt::format("{}", makedict.GetCapacity());
    }
    case Opcode::kPhi: {
      const auto& phi = static_cast<const Phi&>(instr);
      std::stringstream ss;
      bool first = true;
      for (auto& bb : phi.basic_blocks()) {
        if (first) {
          first = false;
        } else {
          ss << ", ";
        }
        ss << bb->id;
      }

      return ss.str();
    }
    case Opcode::kDeleteAttr:
    case Opcode::kLoadAttr:
    case Opcode::kLoadAttrCached:
    case Opcode::kLoadModuleAttrCached:
    case Opcode::kStoreAttr:
    case Opcode::kStoreAttrCached: {
      const auto& named = static_cast<const DeoptBaseWithNameIdx&>(instr);
      return format_name(func, named, named.name_idx());
    }
    case Opcode::kInPlaceOp: {
      const auto& inplace_op = static_cast<const InPlaceOp&>(instr);
      return std::string{GetInPlaceOpName(inplace_op.op())};
    }
    case Opcode::kBuildSlice: {
      const auto& build_slice = static_cast<const BuildSlice&>(instr);
      return fmt::format("{}", build_slice.NumOperands());
    }
    case Opcode::kLoadTypeAttrCacheEntryType: {
      const auto& i = static_cast<const LoadTypeAttrCacheEntryType&>(instr);
      return fmt::format("{}", i.cache_id());
    }
    case Opcode::kLoadTypeAttrCacheEntryValue: {
      const auto& i = static_cast<const LoadTypeAttrCacheEntryValue&>(instr);
      return fmt::format("{}", i.cache_id());
    }
    case Opcode::kFillTypeAttrCache: {
      const auto& ftac = static_cast<const FillTypeAttrCache&>(instr);
      return fmt::format("{}, {}", ftac.cache_id(), ftac.name_idx());
    }
    case Opcode::kLoadTypeMethodCacheEntryValue: {
      const auto& i = static_cast<const LoadTypeMethodCacheEntryValue&>(instr);
      return fmt::format("{}", i.cache_id());
    }
    case Opcode::kLoadTypeMethodCacheEntryType: {
      const auto& i = static_cast<const LoadTypeMethodCacheEntryType&>(instr);
      return fmt::format("{}", i.cache_id());
    }
    case Opcode::kFillTypeMethodCache: {
      const auto& ftmc = static_cast<const FillTypeMethodCache&>(instr);
      return fmt::format("{}, {}", ftmc.cache_id(), ftmc.name_idx());
    }
    case Opcode::kSetFunctionAttr: {
      const auto& set_fn_attr = static_cast<const SetFunctionAttr&>(instr);
      return fmt::format("{}", functionFieldName(set_fn_attr.field()));
    }
    case Opcode::kCheckField:
    case Opcode::kCheckFreevar:
    case Opcode::kCheckVar: {
      const auto& check = static_cast<const CheckBaseWithName&>(instr);
      return escape_unicode(check.name());
    }
    case Opcode::kGuardIs: {
      const auto& gs = static_cast<const GuardIs&>(instr);
      return fmt::format("{}", getStablePointer(gs.target()));
    }
    case Opcode::kGuardType: {
      const auto& gs = static_cast<const GuardType&>(instr);
      return fmt::format("{}", gs.target().toString());
    }
    case Opcode::kHintType: {
      std::ostringstream os;
      auto profile_sep = "";
      const auto& hint = static_cast<const HintType&>(instr);
      os << fmt::format("{}, ", hint.NumOperands());
      for (auto types_seen : hint.seenTypes()) {
        os << fmt::format("{}<", profile_sep);
        auto type_sep = "";
        for (auto type : types_seen) {
          os << fmt::format("{}{}", type_sep, type.toString());
          type_sep = ", ";
        }
        os << ">";
        profile_sep = ", ";
      }
      return os.str();
    }
    case Opcode::kUseType: {
      const auto& gs = static_cast<const UseType&>(instr);
      return fmt::format("{}", gs.type().toString());
    }
    case Opcode::kRaiseAwaitableError: {
      const auto& ra = static_cast<const RaiseAwaitableError&>(instr);
      return ra.isAEnter() ? "__aenter__" : "__aexit__";
    }
    case Opcode::kRaiseStatic: {
      const auto& pyerr = static_cast<const RaiseStatic&>(instr);
      std::ostringstream os;
      print_reg_states(os, pyerr.live_regs());
      return fmt::format(
          "{}, \"{}\", <{}>",
          PyExceptionClass_Name(pyerr.excType()),
          pyerr.fmt(),
          os.str());
    }
    case Opcode::kImportFrom: {
      const auto& import_from = static_cast<const ImportFrom&>(instr);
      return format_name(func, import_from, import_from.name_idx());
    }
    case Opcode::kImportName: {
      const auto& import_name = static_cast<const ImportName&>(instr);
      return format_name(func, import_name, import_name.name_idx());
    }
    case Opcode::kEagerImportName: {
      const auto& eager_import_name =
          static_cast<const EagerImportName&>(instr);
      return format_name(func, eager_import_name, eager_import_name.name_idx());
    }
    case Opcode::kRefineType: {
      const auto& rt = static_cast<const RefineType&>(instr);
      return rt.type().toString();
    }
    case Opcode::kFormatValue: {
      int conversion = static_cast<const FormatValue&>(instr).conversion();
      return fvc_to_string(conversion);
    }
    case Opcode::kUnpackExToTuple: {
      const auto& i = static_cast<const UnpackExToTuple&>(instr);
      return fmt::format("{}, {}", i.before(), i.after());
    }
    case Opcode::kDeoptPatchpoint: {
      const auto& dp = static_cast<const DeoptPatchpoint&>(instr);
      auto patcher = dp.patcher();
      if (patcher->isLinked()) {
        return fmt::format(
            "{} -> {}",
            getStablePointer(patcher->patchpoint()),
            getStablePointer(patcher->jumpTarget()));
      }
      return fmt::format("Patcher {}", getStablePointer(dp.patcher()));
    }
    case Opcode::kUpdatePrevInstr: {
      const auto& upi = static_cast<const UpdatePrevInstr&>(instr);
      return fmt::format(
          "idx:{} line_no:{}: {}",
          upi.bytecodeOffset().asIndex(),
          upi.lineNo(),
          upi.parent() != nullptr ? "has parent" : "no parent");
    }
    case Opcode::kBuildInterpolation: {
      const auto& bi = static_cast<const BuildInterpolation&>(instr);
      return fvc_to_string(bi.conversion());
    }
    case Opcode::kLoadSpecial: {
#if PY_VERSION_HEX >= 0x030E0000
      const auto& ls = static_cast<const LoadSpecial&>(instr);
      switch (ls.specialIdx()) {
        case SPECIAL___ENTER__:
          return "__enter__";
        case SPECIAL___EXIT__:
          return "__exit__";
        case SPECIAL___AENTER__:
          return "__aenter__";
        case SPECIAL___AEXIT__:
          return "__aexit__";
        default:
          JIT_ABORT("Unknown special index: {}", ls.specialIdx());
      }
#else
      JIT_ABORT("LoadSpecial not supported before 3.14");
#endif
    }
    case Opcode::kConvertValue: {
      const auto& cv = static_cast<const ConvertValue&>(instr);
      return fvc_to_string(cv.converterIdx());
    }
  }
  JIT_ABORT("Invalid opcode {}", static_cast<int>(instr.opcode()));
}

void HIRPrinter::Print(std::ostream& os, const Instr& instr) {
  run_via_c_side(os, line_prefix_, indent_level_, full_snapshots_, func_,
      [&](FILE* f, PhxHirPrinter* p) {
        phx_hir_print_instr(f, p, &instr);
      });
}

void HIRPrinter::Print(std::ostream& os, const FrameState& state) {
  run_via_c_side(os, line_prefix_, indent_level_, full_snapshots_, func_,
      [&](FILE* f, PhxHirPrinter* p) {
        phx_hir_print_frame_state(f, p, &state);
      });
}

HIRPrinter& HIRPrinter::setFullSnapshots(bool full) {
  full_snapshots_ = full;
  return *this;
}

HIRPrinter& HIRPrinter::setLinePrefix(std::string_view prefix) {
  line_prefix_ = std::string{prefix};
  return *this;
}

std::ostream& operator<<(std::ostream& os, const Function& func) {
  HIRPrinter{}.Print(os, func);
  return os;
}

std::ostream& operator<<(std::ostream& os, const CFG& cfg) {
  HIRPrinter{}.Print(os, cfg);
  return os;
}

std::ostream& operator<<(std::ostream& os, const BasicBlock& block) {
  HIRPrinter{}.Print(os, block);
  return os;
}

std::ostream& operator<<(std::ostream& os, const Instr& instr) {
  HIRPrinter{}.Print(os, instr);
  return os;
}

std::ostream& operator<<(std::ostream& os, const FrameState& state) {
  HIRPrinter{}.Print(os, state);
  return os;
}

} // namespace jit::hir

// B2 commit-2 bridges phx_hir_print_instr_cpp + phx_hir_print_frame_state_cpp
// were a transitional shim used by the C-side phx_hir_print_basic_block
// to recurse into the not-yet-ported C++ HIRPrinter.  Removed in commit-5b
// sole-path swap: phx_hir_print_basic_block now calls the C-side
// phx_hir_print_instr port (printer_c.c), which in turn uses the
// C-side phx_hir_print_frame_state port (commit-5a).
extern "C" {

// B2 commit-3: C-callable accessors needed by the C-side format_name
// family (printer.cpp:200-237 originals).  Each wraps a non-trivial
// C++ method whose port to pure C would require many sub-accessors;
// keeping them as 1-line bridges here avoids cluttering hir_c_api with
// printer-specific surface.  Removed in commit-5 if/when the
// underlying classes get full C-side coverage.
const void* phx_hir_func_code_for(const void* func, const void* instr) {
  return reinterpret_cast<const jit::hir::Function*>(func)->codeFor(
      *reinterpret_cast<const jit::hir::Instr*>(instr));
}

int phx_hir_load_super_name_idx(const void* instr) {
  return reinterpret_cast<const jit::hir::LoadSuperBase*>(instr)->name_idx();
}

int phx_hir_load_super_no_args_in_super_call(const void* instr) {
  return reinterpret_cast<const jit::hir::LoadSuperBase*>(instr)
      ->no_args_in_super_call();
}

// B2 commit-5a: structural HIR accessors for print_reg_states +
// Print(FrameState) C-side bodies.  Per theologian 11:35:47Z these
// are reusable C-API surface, not the single-use bridge pattern.
size_t phx_hir_reg_state_array_size(const void* array) {
  return reinterpret_cast<const jit::hir::PhxRegStateArray*>(array)->size();
}

const void* phx_hir_reg_state_array_at(const void* array, size_t i) {
  return &(*reinterpret_cast<const jit::hir::PhxRegStateArray*>(array))[i];
}

const void* phx_hir_reg_state_reg(const void* rs) {
  return reinterpret_cast<const jit::hir::RegState*>(rs)->reg;
}

int phx_hir_reg_state_ref_kind(const void* rs) {
  return static_cast<int>(reinterpret_cast<const jit::hir::RegState*>(rs)->ref_kind);
}

int phx_hir_reg_state_value_kind(const void* rs) {
  return static_cast<int>(reinterpret_cast<const jit::hir::RegState*>(rs)->value_kind);
}

int phx_hir_register_id(const void* reg) {
  return reinterpret_cast<const jit::hir::Register*>(reg)->id();
}

const char* phx_hir_register_name(const void* reg) {
  // Register::name() returns std::string by value; cache via thread_local
  // to keep the returned const char* live for the duration of the print
  // (printer.cpp doesn't outlive a single Print() call).  Acceptable for
  // tooling; a later cleanup could plumb name() through std::string_view.
  thread_local std::string s_name_buf;
  s_name_buf = reinterpret_cast<const jit::hir::Register*>(reg)->name();
  return s_name_buf.c_str();
}

int phx_hir_frame_state_cur_instr_offs(const void* state) {
  return reinterpret_cast<const jit::hir::FrameState*>(state)
      ->cur_instr_offs.value();
}

int phx_hir_frame_state_nlocals(const void* state) {
  return reinterpret_cast<const jit::hir::FrameState*>(state)->nlocals;
}

size_t phx_hir_frame_state_localsplus_count(const void* state) {
  return reinterpret_cast<const jit::hir::FrameState*>(state)->localsplus.count;
}

const void* phx_hir_frame_state_localsplus_at(const void* state, size_t i) {
  return reinterpret_cast<const jit::hir::FrameState*>(state)
      ->localsplus.data[i];
}

size_t phx_hir_frame_state_stack_count(const void* state) {
  return reinterpret_cast<const jit::hir::FrameState*>(state)->stack.count;
}

const void* phx_hir_frame_state_stack_at(const void* state, size_t i) {
  return reinterpret_cast<const jit::hir::FrameState*>(state)->stack.data[i];
}

size_t phx_hir_frame_state_block_stack_size(const void* state) {
  return reinterpret_cast<const jit::hir::FrameState*>(state)->block_stack.size();
}

int phx_hir_frame_state_block_stack_opcode(const void* state, size_t i) {
  return reinterpret_cast<const jit::hir::FrameState*>(state)
      ->block_stack.at(i).opcode;
}

int phx_hir_frame_state_block_stack_handler_off(const void* state, size_t i) {
  return reinterpret_cast<const jit::hir::FrameState*>(state)
      ->block_stack.at(i).handler_off.value();
}

int phx_hir_frame_state_block_stack_stack_level(const void* state, size_t i) {
  return reinterpret_cast<const jit::hir::FrameState*>(state)
      ->block_stack.at(i).stack_level;
}

// B2 commit-5b: Print(Instr) port accessors.  Per theologian 11:51:45Z
// PORT confirmation — Type stringification, Instr opname, DeoptBase
// fields are structural HIR-level surface (reusable for future C-side
// debug tooling), distinct from format_immediates' instruction-
// specific case-explosion.
const char* phx_hir_instr_opname(const void* instr) {
  // Instr::opname() returns std::string_view by value; cache via
  // thread_local std::string to keep returned const char* live for the
  // print-call duration (same pattern as phx_hir_register_name).
  thread_local std::string s_opname_buf;
  s_opname_buf = reinterpret_cast<const jit::hir::Instr*>(instr)->opname();
  return s_opname_buf.c_str();
}

int phx_hir_register_type_is_top(const void* reg) {
  return reinterpret_cast<const jit::hir::Register*>(reg)->type() ==
                 jit::hir::TTop
             ? 1
             : 0;
}

const char* phx_hir_register_type_to_string(const void* reg) {
  thread_local std::string s_type_buf;
  std::ostringstream ss;
  ss << reinterpret_cast<const jit::hir::Register*>(reg)->type();
  s_type_buf = ss.str();
  return s_type_buf.c_str();
}

const char* phx_hir_deopt_descr(const void* deopt) {
  return reinterpret_cast<const jit::hir::DeoptBase*>(deopt)->descr();
}

const void* phx_hir_deopt_guilty_reg(const void* deopt) {
  return reinterpret_cast<const jit::hir::DeoptBase*>(deopt)->guiltyReg();
}

const void* phx_hir_deopt_live_regs(const void* deopt) {
  return &reinterpret_cast<const jit::hir::DeoptBase*>(deopt)->live_regs();
}

// Type-aware FrameState dispatch — mirror of jit::hir::get_frame_state
// (hir.cpp:1250-1261).  hir_c_get_frame_state in hir_instr_c.h is the
// raw deopt-layout field accessor used by simplify pass on instrs
// already known to be deopt-class; the printer needs the type-aware
// variant since Print(Instr) runs on every opcode.
const void* phx_hir_instr_get_frame_state(const void* instr) {
  return jit::hir::get_frame_state(
      *reinterpret_cast<const jit::hir::Instr*>(instr));
}

// W-PRINTER-IMMEDIATES-PORT P-2: opname accessors for the
// BinaryOp/UnaryOp/InPlaceOp/Compare instr classes.  Wraps the
// existing C++ GetXxxOpName(enum_class) helpers (hir.h:654/705/755/1772);
// returns a thread_local-buffered C string for the duration of the
// printer call.  Each ported case in phx_format_immediates calls
// these directly via the int op field on the C-side instr struct.
const char* phx_hir_binary_op_name(int op) {
  thread_local std::string s_buf;
  s_buf = jit::hir::GetBinaryOpName(static_cast<jit::hir::BinaryOpKind>(op));
  return s_buf.c_str();
}

const char* phx_hir_unary_op_name(int op) {
  thread_local std::string s_buf;
  s_buf = jit::hir::GetUnaryOpName(static_cast<jit::hir::UnaryOpKind>(op));
  return s_buf.c_str();
}

const char* phx_hir_in_place_op_name(int op) {
  thread_local std::string s_buf;
  s_buf = jit::hir::GetInPlaceOpName(static_cast<jit::hir::InPlaceOpKind>(op));
  return s_buf.c_str();
}

const char* phx_hir_compare_op_name(int op) {
  thread_local std::string s_buf;
  s_buf = jit::hir::GetCompareOpName(static_cast<jit::hir::CompareOp>(op));
  return s_buf.c_str();
}

// BeginInlinedFunction::fullname() returns const char* directly; this
// is a thin wrapper to keep the C-side switch case body uniform.
const char* phx_hir_begin_inlined_function_fullname(const void* instr) {
  return reinterpret_cast<const jit::hir::BeginInlinedFunction*>(instr)->fullname();
}

// LoadArrayItem::offset() — int field accessor.
intptr_t phx_hir_load_array_item_offset(const void* instr) {
  return reinterpret_cast<const jit::hir::LoadArrayItem*>(instr)->offset();
}

// LoadSplitDictItem::itemIdx() — int field accessor.
int phx_hir_load_split_dict_item_idx(const void* instr) {
  return reinterpret_cast<const jit::hir::LoadSplitDictItem*>(instr)->itemIdx();
}

// Return::type() — Type stringification; "" if TObject (C++ side
// suppresses default type per printer.cpp:308).
const char* phx_hir_return_type_or_empty(const void* instr) {
  thread_local std::string s_buf;
  const auto* ret = reinterpret_cast<const jit::hir::Return*>(instr);
  if (ret->type() == jit::hir::TObject) {
    s_buf.clear();
  } else {
    s_buf = ret->type().toString();
  }
  return s_buf.c_str();
}

// Branch::target()->id — destination block id for unconditional branches.
int phx_hir_branch_target_id(const void* instr) {
  return reinterpret_cast<const jit::hir::Branch*>(instr)->target()->id;
}

// W-PRINTER-IMMEDIATES-PORT P-4b: complex-call bridges.
// CallStatic / CallStaticRetVoid emit "<name@stable_ptr, num_ops>"
// when symbolize succeeds, "<stable_ptr, num_ops>" otherwise.  Single
// bridge that writes the formatted text directly to FILE* — caller
// adds the "<...>" wrapping.
void phx_format_call_addr(FILE* out, const void* addr, size_t num_ops) {
  auto sym = jit::symbolize(addr);
  const void* sp = jit::getStablePointer(addr);
  if (sym.has_value()) {
    fprintf(out, "%s@%p, %zu", sym->c_str(), sp, num_ops);
  } else {
    fprintf(out, "%p, %zu", sp, num_ops);
  }
}

// CallCFunc::funcName() returns std::string_view — enum-to-string
// lookup over CallCFunc_FUNCS X-macro (hir.cpp:CallCFunc::funcName).
// Take instr pointer and dispatch via the existing member function.
const char* phx_hir_call_cfunc_name(const void* instr) {
  thread_local std::string s_buf;
  const auto* call = reinterpret_cast<const jit::hir::CallCFunc*>(instr);
  s_buf = std::string{call->funcName()};
  return s_buf.c_str();
}

// CallIntrinsic name lookup.  3.14+ uses _PyIntrinsics_*Functions
// tables indexed by the intrinsic's index field; older versions
// just emit the index as a number.  Returns NULL when caller should
// fall back to printing the index integer directly.
#if PY_VERSION_HEX >= 0x030E0000
const char* phx_hir_call_intrinsic_name(size_t index, size_t num_operands) {
  switch (num_operands) {
    case 1:
      return _PyIntrinsics_UnaryFunctions[index].name;
    case 2:
      return _PyIntrinsics_BinaryFunctions[index].name;
    default:
      return nullptr;
  }
}
#else
const char* phx_hir_call_intrinsic_name(size_t /*index*/, size_t /*num_operands*/) {
  return nullptr;  // caller emits index integer
}
#endif

// InvokeStaticFunction.func() returns PyFunctionObject*.  Two
// PyUnicode_AsUTF8 lookups for module + qualname; returned strings are
// thread_local-buffered.  ret_type is read directly from the
// HirInvokeStaticFunction struct's HirType field by the caller.
const char* phx_hir_pyfunc_module_name(const void* func_obj) {
  thread_local std::string s_buf;
  const auto* func = static_cast<const PyFunctionObject*>(func_obj);
  const char* m = PyUnicode_AsUTF8(func->func_module);
  s_buf = m != nullptr ? std::string{m} : std::string{};
  return s_buf.c_str();
}

const char* phx_hir_pyfunc_qualname(const void* func_obj) {
  thread_local std::string s_buf;
  const auto* func = static_cast<const PyFunctionObject*>(func_obj);
  const char* m = PyUnicode_AsUTF8(func->func_qualname);
  s_buf = m != nullptr ? std::string{m} : std::string{};
  return s_buf.c_str();
}

// W-PRINTER-IMMEDIATES-PORT P-5a: simple-bridge accessors for the next
// batch of dispatcher cases — primitive op-name lookups, FunctionAttr
// stringification, and getStablePointer.  Same thread_local-buffer
// pattern as the P-2 op-name bridges (above).
const char* phx_hir_primitive_compare_op_name(int op) {
  thread_local std::string s_buf;
  s_buf = jit::hir::GetPrimitiveCompareOpName(
      static_cast<jit::hir::PrimitiveCompareOp>(op));
  return s_buf.c_str();
}

const char* phx_hir_primitive_unary_op_name(int op) {
  thread_local std::string s_buf;
  s_buf = jit::hir::GetPrimitiveUnaryOpName(
      static_cast<jit::hir::PrimitiveUnaryOpKind>(op));
  return s_buf.c_str();
}

const char* phx_hir_function_field_name(int field) {
  return jit::hir::functionFieldName(
      static_cast<jit::hir::FunctionAttr>(field));
}

const void* phx_hir_get_stable_pointer(const void* ptr) {
  return jit::getStablePointer(ptr);
}

// W-PRINTER-IMMEDIATES-PORT P-5b: bridges for the complex residual cases
// (HintType, RaiseStatic, DeoptPatchpoint, LoadAttrSpecial).

// JumpPatcher state — used by DeoptPatchpoint case to format
// "{patchpoint} -> {jumpTarget}" or fallback "Patcher {ptr}".
int phx_hir_patcher_is_linked(const void* patcher) {
  return reinterpret_cast<const jit::JumpPatcher*>(patcher)->isLinked() ? 1 : 0;
}

const void* phx_hir_patcher_patchpoint(const void* patcher) {
  return reinterpret_cast<const jit::JumpPatcher*>(patcher)->patchpoint();
}

const void* phx_hir_patcher_jump_target(const void* patcher) {
  return reinterpret_cast<const jit::JumpPatcher*>(patcher)->jumpTarget();
}

// PyObject_Repr-with-error-protection — wraps the repr() helper from
// jit_common/log.cpp (PyErr_Fetch/Restore around PyObject_Repr,
// PyUnicode_AsUTF8AndSize). thread_local-buffered.
const char* phx_hir_pyobject_repr(const void* obj) {
  thread_local std::string s_buf;
  s_buf = jit::repr(static_cast<PyObject*>(const_cast<void*>(obj)));
  return s_buf.c_str();
}

// HintType iteration — types_ is std::vector<std::vector<Type>>.  Two
// nested-vector layers are not worth bridging individually (would need
// 4-5 accessors); single bridge writes the whole "<N, <T,T>, <T>>"
// formatted text directly to FILE*.  Caller adds the outer "<...>"
// wrapping (matches printer.cpp:673-689 NumOperands+for-loop pattern).
void phx_format_hint_type(FILE* out, const void* instr) {
  const auto* hint = reinterpret_cast<const jit::hir::HintType*>(instr);
  fmt::print(out, "{}, ", hint->NumOperands());
  const char* profile_sep = "";
  for (auto& types_seen : hint->seenTypes()) {
    fmt::print(out, "{}<", profile_sep);
    const char* type_sep = "";
    for (auto& type : types_seen) {
      fmt::print(out, "{}{}", type_sep, type.toString());
      type_sep = ", ";
    }
    fputc('>', out);
    profile_sep = ", ";
  }
}

// B2 commit-4: bridge into the static format_immediates above.  Per
// theologian 11:23:20Z option A, format_immediates (185 case-branches)
// stays C++ — porting it requires hundreds of HIR-instruction-specific
// C accessors which is months of work for tooling code.  W-PRINTER-
// IMMEDIATES-PORT residual workstream tracks the eventual port; until
// then commit-5 Print(Instr) C wrapper calls this bridge.
//
// Wrapping `<` ... `>` happens HERE (not in the C caller) so the
// no-immediates case writes nothing — matches printer.cpp:815-818
// `if (!immed.empty()) os << "<" << immed << ">"` semantics.
void phx_format_immediates_cpp(
    FILE* out,
    const struct PhxHirPrinter* p,
    const void* instr_ptr) {
  const auto* instr = static_cast<const jit::hir::Instr*>(instr_ptr);
  const auto* func = static_cast<const jit::hir::Function*>(p->func);
  std::string s = jit::hir::format_immediates(func, *instr);
  if (!s.empty()) {
    std::fputc('<', out);
    std::fwrite(s.data(), 1, s.size(), out);
    std::fputc('>', out);
  }
}

} // extern "C"
