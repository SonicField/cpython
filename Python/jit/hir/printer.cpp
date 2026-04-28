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

#include <ostream>
#include <sstream>
#include <vector>

namespace jit::hir {

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

// W-PRINTER-IMMEDIATES-PORT P-5c: print_reg_states / escape_unicode pair /
// format_name family / format_immediates static deleted.  All call-sites
// were format_immediates' own switch branches (theologian 15:00:30Z G1.5
// pre-analysis: 10 deletion targets ALL-CLEAR via call-site grep).  The
// dispatcher now lives entirely in printer_c.c (162/168 explicit cases +
// abort default for the 6 LoadSpecial-class 3.14+ residuals that never
// fire on the Phoenix 3.12.13 target).

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

// W-PRINTER-IMMEDIATES-PORT P-5c: phx_format_immediates_cpp bridge deleted.
// Its only caller was the C-side dispatcher's default branch; with all
// 162/168 cases ported (P-1..P-5b), the default branch now aborts on the
// 6 LoadSpecial-class 3.14+ residuals (which never fire on Phoenix's
// 3.12.13 target).  Bridge declaration removed from printer_c.h.

} // extern "C"
