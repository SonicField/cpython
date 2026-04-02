/* watchers.c -- Pure C implementation of watcher state management.
 *
 * Phase 3D: Replaces watchers.cpp (cinderx::WatcherState class).
 * All functions use CPython C API directly — no C++ dependencies. */

#include "cinderx/Common/watchers_c.h"

void ci_watcher_state_init_struct(CiWatcherState *ws) {
    ws->code_watcher = NULL;
    ws->dict_watcher = NULL;
    ws->func_watcher = NULL;
    ws->type_watcher = NULL;
    ws->code_watcher_id = -1;
    ws->dict_watcher_id = -1;
    ws->func_watcher_id = -1;
    ws->type_watcher_id = -1;
}

int ci_watcher_state_init(CiWatcherState *ws) {
    if (ws->code_watcher != NULL) {
        ws->code_watcher_id = PyCode_AddWatcher(ws->code_watcher);
        if (ws->code_watcher_id < 0) {
            return -1;
        }
    }
    if (ws->dict_watcher != NULL) {
        ws->dict_watcher_id = PyDict_AddWatcher(ws->dict_watcher);
        if (ws->dict_watcher_id < 0) {
            return -1;
        }
    }
    if (ws->func_watcher != NULL) {
        ws->func_watcher_id = PyFunction_AddWatcher(ws->func_watcher);
        if (ws->func_watcher_id < 0) {
            return -1;
        }
    }
    if (ws->type_watcher != NULL) {
        ws->type_watcher_id = PyType_AddWatcher(ws->type_watcher);
        if (ws->type_watcher_id < 0) {
            return -1;
        }
    }
    return 0;
}

int ci_watcher_state_fini(CiWatcherState *ws) {
    if (ws->type_watcher_id != -1 && PyType_ClearWatcher(ws->type_watcher_id) < 0) {
        return -1;
    }
    ws->type_watcher_id = -1;

    if (ws->func_watcher_id != -1 && PyFunction_ClearWatcher(ws->func_watcher_id) < 0) {
        return -1;
    }
    ws->func_watcher_id = -1;

    if (ws->dict_watcher_id != -1 && PyDict_ClearWatcher(ws->dict_watcher_id) < 0) {
        return -1;
    }
    ws->dict_watcher_id = -1;

    if (ws->code_watcher_id != -1 && PyCode_ClearWatcher(ws->code_watcher_id) < 0) {
        return -1;
    }
    ws->code_watcher_id = -1;

    return 0;
}

void ci_watcher_state_set_code_watcher(CiWatcherState *ws, CiCodeWatcher watcher) {
    ws->code_watcher = watcher;
}

void ci_watcher_state_set_dict_watcher(CiWatcherState *ws, CiDictWatcher watcher) {
    ws->dict_watcher = watcher;
}

void ci_watcher_state_set_func_watcher(CiWatcherState *ws, CiFuncWatcher watcher) {
    ws->func_watcher = watcher;
}

void ci_watcher_state_set_type_watcher(CiWatcherState *ws, CiTypeWatcher watcher) {
    ws->type_watcher = watcher;
}

int ci_watcher_state_watch_dict(CiWatcherState *ws, PyObject *dict) {
    return PyDict_Watch(ws->dict_watcher_id, dict);
}

int ci_watcher_state_unwatch_dict(CiWatcherState *ws, PyObject *dict) {
    return PyDict_Unwatch(ws->dict_watcher_id, dict);
}

int ci_watcher_state_watch_type(CiWatcherState *ws, PyTypeObject *type) {
    return PyType_Watch(ws->type_watcher_id, (PyObject *)type);
}

int ci_watcher_state_unwatch_type(CiWatcherState *ws, PyTypeObject *type) {
    return PyType_Unwatch(ws->type_watcher_id, (PyObject *)type);
}
