/*
 * Phoenix JIT initialization — minimal init path that bypasses the full
 * CinderX module infrastructure and only sets up what the JIT needs.
 *
 * Called from PyInit__cinderx instead of the full _cinderx_exec_impl.
 */

#include "cinderx/python.h"
#include "cinderx/module_state.h"
#include "cinderx/Jit/pyjit.h"
#include "cinderx/Jit/config.h"
#include "cinderx/Jit/global_cache.h"
#include "cinderx/Jit/generators_rt.h"
#include "cinderx/Common/code.h"
#include "cinderx/Common/log.h"
#include "cinderx/Common/watchers.h"
#include "cinderx/module_c_state.h"

/* Watcher callbacks — notify the JIT when types/dicts/funcs/code change */
static int phoenix_code_watcher(PyCodeEvent event, PyCodeObject* co) {
    if (event == PY_CODE_EVENT_DESTROY) {
        jit::codeDestroyed(co);
    }
    return 0;
}

static int phoenix_dict_watcher(
    PyDict_WatchEvent event, PyObject* dict_obj,
    PyObject* key_obj, PyObject* new_value) {
    auto state = cinderx::getModuleState();
    jit::IGlobalCacheManager* globalCaches =
        state != nullptr ? state->cacheManager() : nullptr;
    if (globalCaches == nullptr) return 0;

    BorrowedRef<PyDictObject> dict{dict_obj};
    switch (event) {
        case PyDict_EVENT_ADDED:
        case PyDict_EVENT_MODIFIED:
        case PyDict_EVENT_DELETED:
            if (key_obj == nullptr || !PyUnicode_CheckExact(key_obj)) {
                globalCaches->notifyDictUnwatch(dict);
            } else {
                /* Only process already-interned keys. Do NOT call
                   PyUnicode_InternInPlace — it allocates memory which
                   can trigger GC re-entrancy, crashing during automatic
                   garbage collection. Non-interned keys are rare for
                   module globals (Python interns most identifiers). */
                if (PyUnicode_CHECK_INTERNED(key_obj)) {
                    BorrowedRef<PyUnicodeObject> key{key_obj};
                    globalCaches->notifyDictUpdate(dict, key, new_value);
                }
            }
            break;
        case PyDict_EVENT_CLEARED:
            globalCaches->notifyDictClear(dict);
            break;
        case PyDict_EVENT_CLONED:
        case PyDict_EVENT_DEALLOCATED:
            globalCaches->notifyDictUnwatch(dict);
            break;
    }
    return 0;
}

static int phoenix_func_watcher(
    PyFunction_WatchEvent event, PyFunctionObject* func, PyObject* new_value) {
    switch (event) {
        case PyFunction_EVENT_CREATE:
            jit::scheduleJitCompile(func);
            break;
        case PyFunction_EVENT_MODIFY_CODE:
            jit::funcModified(func);
            break;
        case PyFunction_EVENT_DESTROY:
            jit::funcDestroyed(func);
            break;
        default:
            break;
    }
    return 0;
}

static int phoenix_type_watcher(PyTypeObject* type) {
    jit::typeModified(type);
    return 0;
}

namespace {

/* Minimal module definition for the _cinderx module */
static int phoenix_exec(PyObject* m);
static void phoenix_free(void* m);

static struct PyModuleDef_Slot phoenix_slots[] = {
    {Py_mod_exec, (void*)phoenix_exec},
    {0, nullptr},
};

static struct PyModuleDef phoenix_module_def = {
    PyModuleDef_HEAD_INIT,
    "_cinderx",
    "Phoenix JIT for CPython",
    sizeof(cinderx::ModuleState),
    nullptr,  /* methods */
    phoenix_slots,
    nullptr,  /* traverse */
    nullptr,  /* clear */
    phoenix_free,
};

static bool phoenix_initialized = false;

static int phoenix_exec(PyObject* m) {
    /* Guard against double-init — the test runner can trigger
       site.main() re-execution which reloads _cinderx */
    if (phoenix_initialized) {
        return 0;
    }
    phoenix_initialized = true;

    /* Save CPython's original function vectorcall — needed by
       getInterpretedVectorcall() to delegate to the interpreter */
    Ci_PyFunction_Vectorcall = _PyFunction_Vectorcall;

    /* Initialize module state in-place */
    void* state_mem = PyModule_GetState(m);
    if (state_mem == nullptr) {
        PyErr_SetString(PyExc_RuntimeError, "Phoenix: no module state");
        return -1;
    }
    auto state = new (state_mem) cinderx::ModuleState();
    cinderx::setModuleState(m);

    /* Create JIT-specific generator/coroutine types from specs.
       These are DISTINCT from PyGen_Type/PyCoro_Type — JitGen_CheckAny
       uses them to distinguish JIT generators from regular ones.
       Setting genType/coroType to PyGen_Type would cause jitgen_dealloc
       to treat ALL generators as JIT generators, crashing on dealloc. */
    PyTypeObject* gen_type = (PyTypeObject*)PyType_FromSpec(&jit::JitGen_Spec);
    if (gen_type == nullptr) {
        return -1;
    }
    state->setGenType(gen_type);
    Py_DECREF(gen_type);

    PyTypeObject* coro_type = (PyTypeObject*)PyType_FromSpec(&jit::JitCoro_Spec);
    if (coro_type == nullptr) {
        return -1;
    }
    state->setCoroType(coro_type);
    Py_DECREF(coro_type);

    /* Create global cache manager — needed by the preloader for
       LOAD_GLOBAL resolution and dict watching */
    auto cache_manager = new (std::nothrow) jit::GlobalCacheManager();
    if (cache_manager == nullptr) {
        PyErr_SetString(PyExc_MemoryError, "Phoenix: failed to allocate GlobalCacheManager");
        return -1;
    }
    state->setCacheManager(cache_manager);

    /* Initialize watchers — the JIT's inline caches need these to
       track type/dict/func/code changes */
    auto& ws = state->watcherState();
    ws.setCodeWatcher(phoenix_code_watcher);
    ws.setDictWatcher(phoenix_dict_watcher);
    ws.setFuncWatcher(phoenix_func_watcher);
    ws.setTypeWatcher(phoenix_type_watcher);
    if (ws.init() < 0) {
        PyErr_SetString(PyExc_RuntimeError, "Phoenix: failed to init watchers");
        return -1;
    }

    /* Initialize builtin type member caches */
    if (!state->initBuiltinMembers()) {
        PyErr_SetString(PyExc_RuntimeError, "Phoenix: failed to init builtin members");
        return -1;
    }

    /* Initialize CodeExtra index — needed by the counting trampoline
       and inline caches to store per-code-object JIT metadata */
    initCodeExtraIndex();

    /* Initialize the JIT */
    int ret = jit::initialize();
    if (ret == -2) {
        return 0;
    }
    if (ret != 0) {
        if (!PyErr_Occurred()) {
            return 0;
        }
        return -1;
    }

    /* Enable auto-compilation: newly created functions will be scheduled
       for JIT compilation via the func watcher -> scheduleJitCompile.
       The counting trampoline compiles after this many calls.
       Does NOT stamp existing functions (safe for CPython internals). */
    jit::getMutableConfig().compile_after_n_calls = 1000;

    return 0;
}

static void phoenix_free(void* m) {
    auto state = cinderx::getModuleState();
    if (state == nullptr) {
        return;
    }

    /* Shutdown in reverse init order:
       1. Clear watchers — prevents callbacks during cleanup
       2. Finalize JIT — deopts generators, releases references
       3. Clean up CodeExtra index
       4. Destruct ModuleState (placement-new'd, needs explicit dtor)
       5. Clear global pointer */
    state->watcherState().fini();
    jit::finalize();
    finiCodeExtraIndex();
    state->~ModuleState();
    cinderx::removeModuleState();
}

}  /* anonymous namespace */

extern "C" PyObject* PyInit__cinderx(void) {
    return PyModuleDef_Init(&phoenix_module_def);
}
