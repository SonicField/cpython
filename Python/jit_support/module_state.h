// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "cinderx/python.h"

#include "cinderx/Common/watchers_c.h"
#include "cinderx/Jit/code_allocator_iface.h"
#include "cinderx/Jit/containers.h"
#include "cinderx/Jit/context_iface.h"
#include "cinderx/Jit/generators_mm_iface.h"
#include "cinderx/Jit/global_cache_iface.h"
#include "cinderx/Jit/jit_list_iface.h"
#include "cinderx/Jit/symbolizer_iface.h"
#include "cinderx/async_lazy_value_iface.h"

#include <memory>
#include <unordered_map>

namespace cinderx {

class ModuleState {
 public:
  // Implements CPython's traverse functionality for tracing through to GC
  // references
  int traverse(visitproc visit, void* arg);

  // Implements CPython's clear functionality for dropping GC references
  int clear();

  jit::IGlobalCacheManager* cacheManager() const {
    return cache_manager_.get();
  }

  void setCacheManager(jit::IGlobalCacheManager* cache_manager) {
    cache_manager_ = std::unique_ptr<jit::IGlobalCacheManager>(cache_manager);
  }

  jit::ICodeAllocator* codeAllocator() const {
    return code_allocator_.get();
  }

  void setCodeAllocator(jit::ICodeAllocator* code_allocator) {
    code_allocator_.reset(code_allocator);
  }

  jit::IJitContext* jitContext() const {
    return jit_context_.get();
  }

  void setJitContext(jit::IJitContext* context) {
    jit_context_ = std::unique_ptr<jit::IJitContext>(context);
  }

  jit::IJITList* jitList() const {
    return jit_list_.get();
  }

  void setJitList(std::unique_ptr<jit::IJITList> jit_list) {
    jit_list_ = std::move(jit_list);
  }

  jit::ISymbolizer* symbolizer() const {
    return symbolizer_.get();
  }

  void setSymbolizer(jit::ISymbolizer* symbolizer) {
    symbolizer_ = std::unique_ptr<jit::ISymbolizer>(symbolizer);
  }

  IAsyncLazyValueState* asyncLazyValueState() {
    return async_lazy_value_.get();
  }

  void setAsyncLazyValueState(IAsyncLazyValueState* state) {
    async_lazy_value_ = std::unique_ptr<IAsyncLazyValueState>(state);
  }

  void setCoroType(PyTypeObject* coro_type) {
    coro_type_ = Ref<PyTypeObject>::create(coro_type);
  }

  PyTypeObject* coroType() const {
    return coro_type_;
  }

  void setGenType(PyTypeObject* gen_type) {
    gen_type_ = Ref<PyTypeObject>::create(gen_type);
  }

  PyTypeObject* genType() const {
    return gen_type_;
  }

  // Sets the value of sys._clear_type_caches when CinderX was initialized.
  // We then replace it with a function which forwards to the original.
  void setSysClearCaches(PyObject* clear_caches) {
    sys_clear_caches_ = Ref<>::create(clear_caches);
  }

#if PY_VERSION_HEX < 0x030E0000
  void setFrameReifier(PyObject* frame_reifier) {
    frame_reifier_ = Ref<>::create(frame_reifier);
  }

  PyObject* frameReifier() const {
    return frame_reifier_;
  }
#endif

  // Gets the value of sys._clear_type_caches when CinderX was initialized.
  PyObject* sysClearCaches() const {
    return sys_clear_caches_;
  }

  PyObject* getOriginalSysMonitoringRegisterCallback() const {
    return orig_sys_monitoring_register_callback_;
  }

  void setOriginalSysMonitoringRegisterCallback(PyObject* func) {
    orig_sys_monitoring_register_callback_ = Ref<>::create(func);
  }

  PyObject* getOriginalSysSetProfile() const {
    return orig_sys_setprofile_;
  }

  void setOriginalSysSetProfile(PyObject* func) {
    orig_sys_setprofile_ = Ref<>::create(func);
  }

  PyObject* getOriginalSysSetTrace() const {
    return orig_sys_settrace_;
  }

  void setOriginalSysSetTrace(PyObject* func) {
    orig_sys_settrace_ = Ref<>::create(func);
  }

  void setAnextAwaitableType(PyTypeObject* type) {
    anext_awaitable_type_ = Ref<PyTypeObject>::create(type);
  }

  PyTypeObject* anextAwaitableType() const {
    return anext_awaitable_type_;
  }

  void setBuiltinNext(PyObject* builtin_next) {
    builtin_next_ = Ref<>::create(builtin_next);
  }

  PyObject* builtinNext() const {
    return builtin_next_;
  }

  jit::UnorderedSet<PyFunctionObject*>& perfTrampolineWorklist() {
    return perf_trampoline_worklist_;
  }

  void setModule(PyObject* module) {
    cinderx_module_ = module;
  }

  // Returns the PyModule instance for the CinderX module. This can be useful if
  // we have live data backed by the module, in which case we can increase the
  // refcount of the module to prevent it from being freed prematurely.
  PyObject* module() const {
    return cinderx_module_;
  }

  jit::IJitGenFreeList* jitGenFreeList() const {
    return jit_gen_free_list_.get();
  }

  void setJitGenFreeList(jit::IJitGenFreeList* jit_gen_free_list) {
    jit_gen_free_list_ =
        std::unique_ptr<jit::IJitGenFreeList>(jit_gen_free_list);
  }

  // Returns a dictionary of type->dict[name, members] for standard builtin
  // types.
  std::unordered_map<PyTypeObject*, Ref<>>& builtinMembers() {
    return builtin_members_;
  }

  bool initBuiltinMembers();

  CiWatcherState& watcherState();

  jit::UnorderedSet<PyObject*>& registeredCompilationUnits();

 private:
  CiWatcherState watcher_state_;

  std::unique_ptr<jit::IGlobalCacheManager> cache_manager_;
  std::unique_ptr<jit::ICodeAllocator> code_allocator_;
  std::unique_ptr<jit::ISymbolizer> symbolizer_;
  std::unique_ptr<jit::IJitContext> jit_context_;
  std::unique_ptr<jit::IJITList> jit_list_;
  std::unique_ptr<IAsyncLazyValueState> async_lazy_value_;
  std::unique_ptr<jit::IJitGenFreeList> jit_gen_free_list_;
  Ref<PyTypeObject> coro_type_, gen_type_, anext_awaitable_type_;
  std::unordered_map<PyTypeObject*, Ref<>> builtin_members_;
#if PY_VERSION_HEX < 0x030E0000
  Ref<> frame_reifier_;
#endif
  Ref<> sys_clear_caches_, builtin_next_;
  Ref<> orig_sys_monitoring_register_callback_;
  Ref<> orig_sys_setprofile_;
  Ref<> orig_sys_settrace_;

  // Function and code objects ("units") registered for compilation.
  jit::UnorderedSet<PyObject*> registered_compilation_units;

  // Function objects registered for pre-fork perf-trampoline compilation.
  jit::UnorderedSet<PyFunctionObject*> perf_trampoline_worklist_;

  PyObject* cinderx_module_;
};

namespace detail {
// Backing storage for the global ModuleState singleton.  Defined in
// module_state.cpp; declared here so getModuleState() can be inlined and
// each call site lowers to a single load (eliminates ~17.4M PLT-mediated
// function calls per gen_simple measurement, W-PHASE2-CODEGEN-SLOW Phase 1).
extern ModuleState* s_cinderx_state;
} // namespace detail

// Get the global ModuleState singleton.  Inline so callers in hot paths
// (JitGen_CheckAny per-yield, JITRT_InvokeIterNext per-iter) avoid the
// cross-TU function-call overhead.
inline ModuleState* getModuleState() {
  return detail::s_cinderx_state;
}

// Get the ModuleState from the CinderX module object.
//
// Prefer this to using the global singleton when possible.
ModuleState* getModuleState(PyObject* mod);

// Set the global ModuleState singleton, using the CinderX module object.
void setModuleState(PyObject* mod);

// Unset the global ModuleState singleton, but don't destroy it.
//
// Destroying the state object is done manually.
void removeModuleState();

} // namespace cinderx

// C bridge for accessing the ModuleState's builtin_members_ cache.
//
// Returns a borrowed reference to the cached member, or NULL if the type
// is not in the cache OR the type's cached dict has no entry for `name`.
//
// Used by the pure-C blme_helpers_c.c port of
// builtin_load_method_elimination.cpp's getMethodObjectFromType +
// immutableMultithreadedTypeLookup helpers (W42-class Cat-A extraction
// per theologian 20:03Z + supervisor 20:04Z + W27e PARTIAL precedent).
//
// Single-purpose, single-direction (read-only lookup) bridge. Internal
// access to the C++ unordered_map is encapsulated; C callers see a
// PyObject*-only surface.
#ifdef __cplusplus
extern "C" {
#endif
PyObject* phx_builtin_members_lookup(PyTypeObject* type, PyObject* name);
#ifdef __cplusplus
}
#endif
