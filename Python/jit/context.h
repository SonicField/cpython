// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "cinderx/python.h"

#if PY_VERSION_HEX < 0x030C0000
#include "cinder/genobject_jit.h"
#endif

#include "cinderx/Common/ref.h"
#include "cinderx/Common/slab_arena.h"
#include "cinderx/Common/util.h"
#include "cinderx/Jit/code_runtime.h"
#include "cinderx/Jit/codegen/arch.h"
#include "cinderx/Jit/compiled_function.h"
#include "cinderx/Jit/containers.h"
#include "cinderx/Jit/context_iface.h"
#include "cinderx/Jit/deopt.h"
#include "cinderx/Jit/elf/note.h"
#include "cinderx/Jit/fixed_type_profiler.h"
#include "cinderx/Jit/gen_data_footer.h"
#include "cinderx/Jit/hir/preload.h"
#include "cinderx/Jit/inline_cache.h"
#include "cinderx/Jit/pyjit_result.h"
#include "cinderx/Jit/type_deopt_patchers.h"
// Forward declaration — full definition in global_deopt_patcher.h.
// context.h only uses GlobalDeoptPatcher* (pointer), never dereferences it.
namespace jit { class GlobalDeoptPatcher; }

#include <functional>
#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace jit {

#if PY_VERSION_HEX < 0x030C0000
// Memory management functions for JIT generator data.
// In 3.12+ there is no gen->gi_jit_data and this functionality is part of
// JitGenObject.

jit::GenDataFooter* jitgen_data_allocate(size_t spill_words);
void jitgen_data_free(PyGenObject* gen);

inline GenDataFooter* genDataFooter(PyGenObject* gen) {
  return reinterpret_cast<GenDataFooter*>(gen->gi_jit_data);
}

// The number of words for pre-allocated blocks in the generator suspend data
// free-list. I chose this based on it covering 99% of the JIT generator
// spill-sizes needed when running 'make testcinder_jit' at the time I collected
// this data. For reference:
//   99.9% coverage came at 256 spill size
//   99.99% was at 1552
//   max was 4999
// There were about ~15k JIT generators in total during the run.
constexpr size_t kMinGenSpillWords = 89;

// Pre 3.12 these fields needed to be at a fixed offset so they can be quickly
// accessed from C code in genobject.c.
static_assert(
    offsetof(GenDataFooter, state) == Ci_GEN_JIT_DATA_OFFSET_STATE,
    "Byte offset for state shifted");
static_assert(
    offsetof(GenDataFooter, yieldPoint) == Ci_GEN_JIT_DATA_OFFSET_YIELD_POINT,
    "Byte offset for yieldPoint shifted");

#endif

PyObject* yieldFromValue(
    GenDataFooter* gen_footer,
    const GenYieldPoint* yield_point);

// Information about the runtime behavior of a single deopt point: how often
// it's been hit, and the frequency of guilty types, if applicable.
struct DeoptStat {
  std::size_t count;
  FixedTypeProfiler<4> types;
};

// Map from CodeRuntime to stats about each deopt point.
//
// Uses an unordered map to store the deopt stats for each code object as it's
// meant to be sparse.  We expect most deopt points to be unused.
using DeoptStats = jit::
    UnorderedMap<const CodeRuntime*, jit::UnorderedMap<std::size_t, DeoptStat>>;

using InlineCacheStats = std::vector<CacheStats>;

class Builtins {
 public:
  void init();
  bool isInitialized() const;
  std::optional<std::string> find(PyMethodDef* meth) const;
  std::optional<PyMethodDef*> find(const std::string& name) const;

 private:
  std::atomic<bool> is_initialized_{false};
  UnorderedMap<PyMethodDef*, std::string> cfunc_to_name_;
  UnorderedMap<std::string, PyMethodDef*> name_to_cfunc_;
};

// Lookup key for compiled functions in Context: a code object and the globals
// and builtins dicts it was JIT-compiled with.
struct CompilationKey {
  // These three are borrowed references; the values are kept alive by strong
  // references in the corresponding jit::CodeRuntime.
  PyObject* code;
  PyObject* builtins;
  PyObject* globals;

  explicit CompilationKey(PyFunctionObject* func)
      : code{func->func_code},
        builtins{func->func_builtins},
        globals{func->func_globals} {}

  CompilationKey(PyObject* code, PyObject* builtins, PyObject* globals)
      : code(code), builtins(builtins), globals(globals) {}

  constexpr bool operator==(const CompilationKey& other) const = default;
};

} // namespace jit

template <>
struct std::hash<jit::CompilationKey> {
  std::size_t operator()(const jit::CompilationKey& key) const {
    std::hash<PyObject*> hasher;
    return jit::combineHash(
        hasher(key.code), hasher(key.globals), hasher(key.builtins));
  }
};

namespace jit {

/*
 * A jit::Context encapsulates all the state managed by an instance of the JIT.
 */
class Context : public IJitContext {
 public:
  Context();

  /*
   * Adds a function to the list of deopted functions - this means the function
   * was once compiled but has now been turned back into a normal Python
   * function. If the JIT is re-enabled the function can be re-initialized to
   * the JITed form.
   */
  void addDeoptedFunc(PyFunctionObject* func);

  /*
   * Removes a function from the deopted functions set.
   */
  void removeDeoptedFunc(PyFunctionObject* func);

  /*
   * Fully remove all effects of compilation from a function.
   */
  void uncompile(PyFunctionObject* func);

  /*
   * Adds a record indicating that the specified function is currently being
   * compiled. This is used to prevent multiple threads from compiling the same
   * function at the same time.
   */
  bool addActiveCompile(CompilationKey& key);

  /*
   * Indicates that the specified function is no longer being compiled.
   */
  void removeActiveCompile(CompilationKey& key);

  /*
   * Creates the CompiledFunction object for a given compilation key.
   * The compiled code can then be shared amongst compatible functions.
   */
  CompiledFunction* makeCompiledFunction(
      PyFunctionObject* func,
      const CompilationKey& key,
      CompiledFunctionData&& compiled_func);

  /*
   * Record per-function metadata for a newly compiled function and set the
   * function's entrypoint.
   */
  void finalizeFunc(
      PyFunctionObject* func,
      const CompiledFunction& compiled);

  /*
   * Adds a compiled function to the Context. Returns false if the function was
   * previously added.
   */
  bool addCompiledFunc(PyFunctionObject* func);

  /*
   * Removes a function from the set of functions that are known to be compiled.
   * This happens if a function is deopted.
   *
   * Returns true if the function was removed.
   */
  bool removeCompiledFunc(PyFunctionObject* func);

  /*
   * Return whether or not this context compiled the supplied function.
   */
  bool didCompile(PyFunctionObject* func) override;
  bool isDeoptimized(PyFunctionObject* func) override;

  /*
   * Remove the specified code object from the known compiled codes.
   */
  void forgetCode(PyFunctionObject* func);
  /*
   * Look up the compiled function object for a given Python function object.
   */
  CompiledFunction* lookupFunc(PyFunctionObject* func);

  /*
   * Gets the CompiledFunction for a given code/builtins/globals triplet.
   */
  CompiledFunction* lookupCode(
      BorrowedRef<PyCodeObject> code,
      BorrowedRef<PyDictObject> builtins,
      BorrowedRef<PyDictObject> globals);

  /*
   * Looks up the CodeRuntime for a given function.
   */
  CodeRuntime* lookupCodeRuntime(PyFunctionObject* func);

  /*
   * Get the map of all compiled code objects, keyed by their address and also
   * their builtins and globals objects.
   */
  const UnorderedMap<CompilationKey, std::unique_ptr<CompiledFunction>>&
  compiledCodes() const;

  /*
   * Get a range over all function objects that have been compiled.
   */
  const UnorderedSet<PyFunctionObject*>& compiledFuncs();

  /*
   * Get a range over all function objects that have been compiled and since
   * deopted.
   */
  const UnorderedSet<PyFunctionObject*>& deoptedFuncs();

  /*
   * Get the total time spent compiling functions thus far.
   */
  std::chrono::milliseconds totalCompileTime() const;

  /*
   * Adds time to the record of how much time has been spent compiling
   * functions.
   */
  void addCompileTime(std::chrono::nanoseconds time);
  /*
   * Set and hold a reference to the cinderjit Python module.
   */
  void setCinderJitModule(Ref<> mod);

  /*
   * Clear cache of compiled code such that subsequent compilations are always
   * full rather than just re-binding pre-compiled code. Only intended to be
   * used during multithreaded_compile_test.
   */
  void clearCache();

  /*
   * Callbacks invoked by the runtime when a PyFunctionObject is destroyed.
   */
  void funcDestroyed(PyFunctionObject* func);

  // Methods moved from Runtime class

  template <typename... Args>
  CodeRuntime* allocateCodeRuntime(Args&&... args) {
    return code_runtimes_.allocate(std::forward<Args>(args)...);
  }

  void mlockProfilerDependencies();

  // Find a cache for the indirect static entry point for a function.
  void** findFunctionEntryCache(PyFunctionObject* function);

  void clearFunctionEntryCache(PyFunctionObject* function);

  // Checks to see if we already have an entry for indirect static entry point
  bool hasFunctionEntryCache(PyFunctionObject* function) const;

  // Gets information about the primitive arguments that a function
  // is typed to.  Typed object references are explicitly excluded.
  _PyTypedArgsInfo* findFunctionPrimitiveArgInfo(PyFunctionObject* function);

  // Record that a deopt of the given index happened at runtime, with an
  // optional guilty value.
  void recordDeopt(
      CodeRuntime* code_runtime,
      std::size_t idx,
      BorrowedRef<> guilty_value);

  // Get the stat object for a given deopt.  It will not exist if the deopt has
  // never been hit.
  const DeoptStat* deoptStat(
      const CodeRuntime* code_runtime,
      std::size_t deopt_idx) const;

  // Clear all deopt stats.
  void clearDeoptStats();

  // Clear type deopt patchers before final GC to prevent dangling pointers.
  void clearTypeDeoptPatchers();

  // Get and clear inline cache stats.
  InlineCacheStats getAndClearLoadMethodCacheStats();
  InlineCacheStats getAndClearLoadTypeMethodCacheStats();

  using GuardFailureCallback = std::function<void(const DeoptMetadata&)>;

  // Add a function to be called when deoptimization occurs due to guard
  // failure. Intended to be used for testing/debugging only.
  void setGuardFailureCallback(GuardFailureCallback cb);
  void guardFailed(const DeoptMetadata& deopt_meta);
  void clearGuardFailureCallback();

  // Ensure that this Context owns a reference to the given borrowed object,
  // keeping it alive for use by the compiled code. Make CodeRuntime a new
  // owner of the object.
  void addReference(BorrowedRef<> obj);

  // Release any references this Context holds to Python objects.
  void releaseReferences();

  // Allocate a new attribute cache.
  LoadAttrCache* allocateLoadAttrCache();
  LoadTypeAttrCache* allocateLoadTypeAttrCache();
  LoadMethodCache* allocateLoadMethodCache();
  // Lookup-or-create: returns existing warm IC for (code, bc_offset) if available
  LoadMethodCache* allocateLoadMethodCache(BorrowedRef<PyCodeObject> code, int bc_offset);
  LoadModuleAttrCache* allocateLoadModuleAttrCache();
  LoadModuleMethodCache* allocateLoadModuleMethodCache();
  LoadTypeMethodCache* allocateLoadTypeMethodCache();
  StoreAttrCache* allocateStoreAttrCache();

  const Builtins& builtins();

  // Some profilers need to walk the code_rt->code->qualname chain for jitted
  // functions on the call stack. The JIT rarely touches this memory and, as a
  // result, the OS may page it out. Out of process profilers (i.e. those that
  // use eBPF) that attempt to read the memory after it has been paged out will
  // fail; the read would cause a page fault which is currently unsupported
  // inside of an eBPF probe. Periodically calling this function will ensure
  // that the OS doesn't page out the memory too aggressively.
  //
  // Returns a PyListObject containing the qualnames of the units for which
  // memory was paged in.
  Ref<> pageInProfilerDependencies();

  // When type is modified or an instance of type has __class__ assigned to,
  // call patcher->maybePatch(new_ty).
  void watchType(BorrowedRef<PyTypeObject> type, TypeDeoptPatcher* patcher);
  void unwatchType(BorrowedRef<PyTypeObject> type, TypeDeoptPatcher* patcher);

  // Watch a module global for changes.  When the global (identified by
  // dict + key) changes from its expected value, the patcher fires.
  void watchGlobal(
      BorrowedRef<PyDictObject> globals,
      BorrowedRef<PyUnicodeObject> key,
      GlobalDeoptPatcher* patcher);

  // Unregister a GlobalDeoptPatcher.  Called from the patcher destructor
  // when a compiled function is destroyed (e.g., during recompilation)
  // to prevent dangling pointers in global_deopt_patchers_.
  void unwatchGlobal(
      BorrowedRef<PyDictObject> globals,
      BorrowedRef<PyUnicodeObject> key,
      GlobalDeoptPatcher* patcher);

  // Callback from GlobalCacheManager::notifyDictUpdate when a watched
  // global changes.
  void notifyGlobalModified(
      BorrowedRef<PyDictObject> dict,
      BorrowedRef<PyUnicodeObject> key,
      BorrowedRef<> new_value);

  // Callback for when a type is modified or destroyed. lookup_type should be
  // the type that triggered the call (the type that's being
  // modified/deleted/otherwise messed with), and new_type should be the "new"
  // type that is taking its place.
  //
  // In the case of a modification, this new type will be the same as
  // lookup_type, and for type destruction it will be nullptr. For __class__
  // assignment, it will be the new type assigned to the object, in case the
  // deopt patcher determines that the new type is still suitable for the
  // specialized code.
  void notifyTypeModified(
      BorrowedRef<PyTypeObject> lookup_type,
      BorrowedRef<PyTypeObject> new_type);

  // Checks to see if we've compiled a code but not yet created a
  // CompiledFunction object.
  bool hasCompletedCompile(CompilationKey& key);

  void finalizeMultiThreadedCompile();

  // Notifies that a compilation is complete. If we're not in multi-threaded
  // compile the CompiledFunction will immediately be created, otherwise the
  // CompiledFunctionData will be preserved until the multi-threaded compile can
  // finalize things.
  void codeCompiled(
      PyFunctionObject* func,
      CompilationKey& key,
      CompiledFunctionData&& compiled_func);

#if PY_VERSION_HEX < 0x030C0000
  // In 3.12+ the equivalent of this is in generators_rt.cpp.
  template <typename F>
    requires std::is_invocable_r_v<int, F, PyObject*>
  int forEachOwnedRef(PyGenObject* gen, const DeoptMetadata& meta, F func) {
    auto base = reinterpret_cast<char*>(genDataFooter(gen));
    for (const LiveValue& value : meta.live_values) {
      if (value.ref_kind != hir::RefKind::kOwned) {
        continue;
      }
      codegen::PhyLocation loc = value.location;
      JIT_CHECK(
          !loc.is_register(),
          "DeoptMetadata for Yields should not reference registers");
      int ret = func(*reinterpret_cast<PyObject**>(base + loc.loc));
      if (ret != 0) {
        return ret;
      }
    }
    return 0;
  }
#endif

  BorrowedRef<> zero() override;
  BorrowedRef<> strBuildClass();

  void watchPendingTypes();
  void fixupFunctionEntryCachePostMultiThreadedCompile();

  const hir::Type& typeForCommonConstant(int i) const;

  // Map of all code objects to the functions that they were found in.
  // Needed for printing the name of the code object and for preloading.
  UnorderedMap<BorrowedRef<PyCodeObject>, PyFunctionObject*>&
  codeOuterFunctions() {
    return code_outer_funcs_;
  }

  // Allocate all CodeRuntimes together so they can be mlocked() without
  // including any other data that happened to be on the same page.
  SlabArena<CodeRuntime> code_runtimes_;

  // These SlabAreas hold data that is allocated at compile-time and likely to
  // change at runtime, and should be isolated from other data to avoid COW
  // casualties.
  SlabArena<LoadAttrCache, AttributeCacheSizeTrait> load_attr_caches_;
  SlabArena<LoadTypeAttrCache> load_type_attr_caches_;
  SlabArena<LoadMethodCache> load_method_caches_;
  // Map from (code, bc_offset) -> IC for tier 2 reuse
  std::unordered_map<std::pair<PyCodeObject*, int>, LoadMethodCache*,
    decltype([](const std::pair<PyCodeObject*, int>& p) {
      return std::hash<void*>{}(p.first) ^ std::hash<int>{}(p.second);
    })> load_method_cache_map_;
  SlabArena<LoadModuleAttrCache> load_module_attr_caches_;
  SlabArena<LoadModuleMethodCache> load_module_method_caches_;
  SlabArena<LoadTypeMethodCache> load_type_method_caches_;
  SlabArena<StoreAttrCache, AttributeCacheSizeTrait> store_attr_caches_;
  SlabArena<void*> pointer_caches_;

  FunctionEntryCacheMap function_entry_caches_;

  std::vector<DeoptMetadata> deopt_metadata_;
  DeoptStats deopt_stats_;
  GuardFailureCallback guard_failure_callback_;

  // Deopt backoff: suppress JIT for code objects that deopt repeatedly.
  //
  // After kDeoptBackoffThreshold runtime guard failures for a single
  // CodeRuntime, set CI_CO_SUPPRESS_JIT on its code object. This prevents
  // reoptFunc() (pyjit.cpp:798) and scheduleJitCompile() (pyjit.cpp:3892)
  // from re-attaching JIT code that will just deopt again.
  //
  // CODE PATTERNS THAT TRIGGER DEOPT BACKOFF:
  //   - Inner-class methods with polymorphic self (e.g. nn_module_forward:
  //     multiple classes define forward(), GuardType on self fails when
  //     different subclasses alternate). ~40K deopts per benchmark run.
  //   - Deep inheritance __init__ chains (e.g. deep_class_super: 5-level
  //     hierarchy where super().__init__() triggers GuardType failures on
  //     each level). ~1.1M deopts per benchmark run.
  //   - Decorator-wrapped functions (e.g. decorator_chain: GuardType on
  //     class_op fails across decorated variants). ~50K deopts.
  //
  // THRESHOLD RATIONALE (1000):
  //   Import-time deopts (e.g. Tokenizer.__next during json import) peak
  //   at ~100 for simple imports, but complex import graphs (dozens of
  //   modules) can exceed 1000. Runtime deopt loops are much larger:
  //   40K+ (nn_module_forward), 50K+ (pytorch_cm, decorator_chain), 1.1M
  //   (deep_class_super). A threshold of 100 crashed importlib when deopt
  //   backoff detached JIT from Tokenizer.__next mid-import (spec_from_loader
  //   interaction). Threshold 1000 prevents this for simple imports but
  //   does NOT eliminate the need for -S (skip site.py) when the full
  //   import graph exceeds 1000 deopts. The real fix is preventing JIT
  //   activation during imports (via -S or deferred JIT init).
  //
  // MECHANISM: Vectorcall reset (pointer swap on PyFunctionObject), NOT
  //   co_flags mutation on PyCodeObject. The v1 design used co_flags but
  //   crashed because co_flags is shared across all function objects using
  //   the same code object, and mutation during active execution is unsafe.
  //
  // Separate from deopt_stats_ because deopt_stats_ can be cleared by
  // cinderjit.get_and_clear_runtime_stats().
  //
  // NOTE: Thread safety relies on GIL (Py_GIL_DISABLED is NOT defined in
  // our build). If moving to nogil, this map needs a mutex.
  static constexpr uint32_t kDeoptBackoffThreshold = 1000;
  UnorderedMap<const CodeRuntime*, uint32_t> deopt_backoff_counts_;

  // Check if a CodeRuntime has exceeded the deopt backoff threshold.
  // Called from reoptFunc() to decide whether to set CI_CO_SUPPRESS_JIT
  // and refuse re-attachment for new function objects (inner-class methods).
  bool isDeoptBackoffTriggered(const CodeRuntime* runtime) const {
    auto it = deopt_backoff_counts_.find(runtime);
    return it != deopt_backoff_counts_.end() &&
           it->second >= kDeoptBackoffThreshold;
  }

  // Deopt all compiled functions using a given CodeRuntime by resetting
  // their vectorcall to the interpreter entry. Called from recordDeopt()
  // when the backoff threshold is reached. This handles the case where
  // module-level functions (persistent PyFunctionObjects) loop through
  // JIT entry -> guard fail -> deopt -> JIT entry without ever re-entering
  // reoptFunc(). Resetting vectorcall (a pointer swap on PyFunctionObject)
  // is safe mid-deopt, unlike co_flags mutation on PyCodeObject which
  // crashed in v1.
  void deoptBackoffSuppressFunctions(CodeRuntime* code_runtime);

  // References to Python objects held by this Context
  std::unordered_set<ThreadedRef<PyObject>> references_;
  Builtins builtins_;

  std::unordered_map<BorrowedRef<PyTypeObject>, std::vector<TypeDeoptPatcher*>>
      type_deopt_patchers_;

  // Key: (dict, key_name) pair.  Value: list of patchers watching that global.
  using GlobalWatchKey = std::pair<BorrowedRef<PyDictObject>, BorrowedRef<PyUnicodeObject>>;
  struct GlobalWatchKeyHash {
    size_t operator()(const GlobalWatchKey& k) const {
      return std::hash<PyObject*>{}(reinterpret_cast<PyObject*>(k.first.get()))
           ^ (std::hash<PyObject*>{}(reinterpret_cast<PyObject*>(k.second.get())) << 1);
    }
  };
  std::unordered_map<GlobalWatchKey, std::vector<GlobalDeoptPatcher*>, GlobalWatchKeyHash>
      global_deopt_patchers_;

  Ref<> zero_;
  Ref<> str_build_class_;
  std::unordered_set<BorrowedRef<PyTypeObject>> pending_watches_;

  std::vector<hir::Type> common_constant_types_;

 private:
  /* Deopts a function but doesn't touch deopted_funcs_. */
  bool deoptFuncImpl(PyFunctionObject* func);

  /*
   * Map of all compiled code objects, keyed by their address and also their
   * builtins and globals objects.
   */
  UnorderedMap<CompilationKey, std::unique_ptr<CompiledFunction>>
      compiled_codes_;

  /* Set of which functions have JIT-compiled entrypoints. */
  UnorderedSet<PyFunctionObject*> compiled_funcs_;

  /* Set of which functions were JIT-compiled but have since been deopted. */
  UnorderedSet<PyFunctionObject*> deopted_funcs_;

  /*
   * Set of compilations that are currently active, across all threads.
   */
  UnorderedSet<CompilationKey> active_compiles_;

  /*
   * Compilations which have been finished but we haven't created the
   * CompiledFunction objects yet. These are used in the multi-threaded compile
   * case to avoid creating CompiledFunction objects until after all the
   * compiles have completed.
   */
  UnorderedMap<
      CompilationKey,
      std::pair<CompiledFunctionData, ThreadedRef<PyFunctionObject>>>
      completed_compiles_;

  /*
   * Code which is being kept alive in case it was in use when
   * clearCache was called. Only intended to be used during
   * multithreaded_compile_test.
   */
  std::vector<std::unique_ptr<CompiledFunction>> orphaned_compiled_codes_;

  Ref<> cinderjit_module_;

  std::atomic_size_t total_compile_time_ms_;

  // Map of all code objects to the functions that they were found in.
  UnorderedMap<BorrowedRef<PyCodeObject>, PyFunctionObject*>
      code_outer_funcs_;
};

// A CompilerContext is like a Context but it also holds a compiler object
// of the consumers choosing.
template <typename T>
class CompilerContext : public Context {
 public:
  T& compiler() {
    return compiler_;
  }

 private:
  T compiler_;
};

/*
 * An AotContext is like the JIT context, but it holds onto state for
 * ahead-of-time compiled functions.
 */
class AotContext {
 public:
  struct FuncState {
    elf::CodeNoteData note;
    PyFunctionObject* func;
    std::span<const std::byte> compiled_code;

    vectorcallfunc normalEntry() const {
      return reinterpret_cast<vectorcallfunc>(const_cast<std::byte*>(
          compiled_code.data() + note.normal_entry_offset));
    }
  };

  /*
   * Initialize the context with the handle to the AOT bundle created by
   * dlopen().
   */
  void init(void* bundle_handle);

  /* Clean up the context object. */
  void destroy();

  /*
   * Register a new function whose metadata has been parsed out of the AOT
   * bundle.
   */
  void registerFunc(const elf::Note& note);

  /* Look up the state associated with a given Python function. */
  const FuncState* lookupFuncState(PyFunctionObject* func);

 private:
  // The handle to the AOT bundle created by dlopen().
  void* bundle_handle_{nullptr};

  jit::UnorderedMap<std::string, FuncState> funcs_;
};

extern AotContext g_aot_ctx;

// Get the global JIT context. Returns nullptr if the JIT is not initialized.
// This is equivalent to jitCtx() but can be used without depending on pyjit.
Context* getContext();

} // namespace jit
