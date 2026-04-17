// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "cinderx/python.h"

#include "cinderx/Common/ref.h"
#include "cinderx/async_lazy_value_iface.h"

#include <memory>

#if PY_VERSION_HEX >= 0x030C0000

/*
  Unboxed result of calling _asyncio_future_blocking
  Normally we'd use int with commit convention to indicate:
    true (>0), false (0), error(<0)
  However it is permitted for this property to return None
  we need to represent 4 states so enum fits quite nicely
*/
typedef enum {
  BLOCKING_TRUE,
  BLOCKING_FALSE,
  BLOCKING_ERROR,
  BLOCKING_NONE,
} fut_blocking_state;

typedef PyObject* (*get_source_traceback)(PyObject* fut);

typedef struct {
  PyWeakReference weakref;
  // actual method table
  get_source_traceback source_traceback;
} PyMethodTableRef;

#ifdef __cplusplus

namespace cinderx {
class AsyncLazyValueState : public IAsyncLazyValueState {
 public:
  AsyncLazyValueState() {}
  ~AsyncLazyValueState() override {}

  bool init() override;

  PyTypeObject* asyncLazyValueType() override {
    return async_lazy_value_type_;
  }

  PyTypeObject* asyncLazyValueComputeType() {
    return async_lazy_value_compute_type_;
  }

  PyTypeObject* awaitableValueType() {
    return awaitable_value_type_;
  }

  PyTypeObject* futureType();

  PyTypeObject* methodTableRefType() {
    return _method_table_ref_type_;
  }

  PyObject* getEventLoop();

  PyMethodTableRef* lastUsedTaskTypeTableRef() {
    return last_used_task_type_table_ref_;
  }

  void setLastUsedTaskTypeTableRef(PyMethodTableRef* tableref) {
    last_used_task_type_table_ref_ = tableref;
  }

  PyTypeObject* lastUsedTaskType() {
    return last_used_task_type_;
  }

  void setLastUsedTaskType(PyTypeObject* type) {
    last_used_task_type_ = type;
  }

  PyObject* methodRefCallback() {
    return methodref_callback_;
  }

  PyObject* asyncioFutureBlocking();
  PyTypeObject* cancalledError();

  PyGetSetDescrObject* futureSourceTraceback();

  Ref<PyGetSetDescrObject> lookupFutureGetSet(const char* name);

  PyMethodTableRef* futureTableRef();

 private:
  Ref<PyTypeObject> async_lazy_value_type_;
  Ref<PyTypeObject> async_lazy_value_compute_type_;
  Ref<PyTypeObject> awaitable_value_type_;
  Ref<PyTypeObject> future_type_;
  Ref<PyTypeObject> _method_table_ref_type_;
  Ref<> asyncio_future_blocking_;
  Ref<> get_event_loop_;
  Ref<PyTypeObject> cancelled_error_;
  Ref<> methodref_callback_;
  Ref<PyGetSetDescrObject> future_source_traceback_;
  Ref<PyMethodTableRef> future_table_ref_;

  // borrowed reference to the last task type that was created in create_task
  PyTypeObject* last_used_task_type_;
  // method table for the last used task type
  PyMethodTableRef* last_used_task_type_table_ref_;
};
} // namespace cinderx

#endif

#endif // #if PY_VERSION_HEX >= 0x030C0000
