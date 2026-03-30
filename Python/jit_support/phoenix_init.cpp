/*
 * Phoenix JIT initialization — minimal init path that bypasses the full
 * CinderX module infrastructure and only sets up what the JIT needs.
 *
 * Called from PyInit__cinderx instead of the full _cinderx_exec_impl.
 */

#include "cinderx/python.h"
#include "cinderx/module_state.h"
#include "cinderx/Jit/pyjit.h"
#include "cinderx/Common/log.h"
#include "cinderx/Common/watchers.h"

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
                if (!PyUnicode_CHECK_INTERNED(key_obj)) {
                    Py_INCREF(key_obj);
                    PyUnicode_InternInPlace(&key_obj);
                    Py_DECREF(key_obj);
                }
                BorrowedRef<PyUnicodeObject> key{key_obj};
                globalCaches->notifyDictUpdate(dict, key, new_value);
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

static int phoenix_exec(PyObject* m) {
    fprintf(stderr, "Phoenix: initializing JIT\n");
    fflush(stderr);

    /* Initialize module state in-place */
    void* state_mem = PyModule_GetState(m);
    if (state_mem == nullptr) {
        PyErr_SetString(PyExc_RuntimeError, "Phoenix: no module state");
        return -1;
    }
    auto state = new (state_mem) cinderx::ModuleState();
    cinderx::setModuleState(m);

    /* Set gen/coro types to stock CPython types */
    state->setGenType(&PyGen_Type);
    state->setCoroType(&PyCoro_Type);

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
    fprintf(stderr, "Phoenix: watchers initialized\n");

    /* Initialize builtin type member caches */
    if (!state->initBuiltinMembers()) {
        PyErr_SetString(PyExc_RuntimeError, "Phoenix: failed to init builtin members");
        return -1;
    }
    fprintf(stderr, "Phoenix: builtin members initialized\n");

    fprintf(stderr, "Phoenix: calling jit::initialize()\n");
    fflush(stderr);

    /* Initialize the JIT */
    int ret = jit::initialize();
    fprintf(stderr, "Phoenix: jit::initialize() returned %d\n", ret);
    fflush(stderr);
    if (ret == -2) {
        return 0;
    }
    if (ret != 0) {
        if (!PyErr_Occurred()) {
            fprintf(stderr, "Phoenix: JIT not initialized (ret=%d)\n", ret);
            return 0;
        }
        return -1;
    }

    fprintf(stderr, "Phoenix: JIT initialized successfully, returning 0\n");
    fflush(stderr);
    return 0;
}

/* Prevent the existing phoenix_free from calling jit::finalize which
   accesses uninitialized code_extra_index */

static void phoenix_free(void* m) {
    fprintf(stderr, "Phoenix: module free called\n");
    fflush(stderr);
    /* Don't call jit::finalize() yet — it accesses code_extra_index
       which may not have been properly initialized */
    cinderx::removeModuleState();
}

}  /* anonymous namespace */

extern "C" PyObject* PyInit__cinderx(void) {
    fprintf(stderr, "Phoenix: PyInit__cinderx called\n");
    fflush(stderr);
    return PyModuleDef_Init(&phoenix_module_def);
}
