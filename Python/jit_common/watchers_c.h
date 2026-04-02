/* watchers_c.h -- C struct and API for watcher state management.
 *
 * Phase 3D: Pure C replacement for cinderx::WatcherState class.
 * Manages Python code/dict/func/type watchers via CPython C API. */

#ifndef CINDERX_WATCHERS_C_H
#define CINDERX_WATCHERS_C_H

#include "Python.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Watcher callback types (match CPython watcher APIs). */
typedef int (*CiCodeWatcher)(PyCodeEvent, PyCodeObject*);
typedef int (*CiDictWatcher)(PyDict_WatchEvent, PyObject*, PyObject*, PyObject*);
typedef int (*CiFuncWatcher)(PyFunction_WatchEvent, PyFunctionObject*, PyObject*);
typedef int (*CiTypeWatcher)(PyTypeObject*);

/* Watcher state: holds registered watcher callbacks and their IDs. */
typedef struct {
    CiCodeWatcher code_watcher;
    CiDictWatcher dict_watcher;
    CiFuncWatcher func_watcher;
    CiTypeWatcher type_watcher;
    int code_watcher_id;
    int dict_watcher_id;
    int func_watcher_id;
    int type_watcher_id;
} CiWatcherState;

/* Initialize struct fields to defaults (NULL callbacks, -1 IDs). */
void ci_watcher_state_init_struct(CiWatcherState *ws);

/* Register all non-NULL watchers with CPython. Returns -1 on error. */
int ci_watcher_state_init(CiWatcherState *ws);

/* Unregister all watchers from CPython. Returns -1 on error. */
int ci_watcher_state_fini(CiWatcherState *ws);

/* Set individual watcher callbacks (call before init). */
void ci_watcher_state_set_code_watcher(CiWatcherState *ws, CiCodeWatcher watcher);
void ci_watcher_state_set_dict_watcher(CiWatcherState *ws, CiDictWatcher watcher);
void ci_watcher_state_set_func_watcher(CiWatcherState *ws, CiFuncWatcher watcher);
void ci_watcher_state_set_type_watcher(CiWatcherState *ws, CiTypeWatcher watcher);

/* Watch/unwatch individual dicts and types. */
int ci_watcher_state_watch_dict(CiWatcherState *ws, PyObject *dict);
int ci_watcher_state_unwatch_dict(CiWatcherState *ws, PyObject *dict);
int ci_watcher_state_watch_type(CiWatcherState *ws, PyTypeObject *type);
int ci_watcher_state_unwatch_type(CiWatcherState *ws, PyTypeObject *type);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CINDERX_WATCHERS_C_H */
