/*
 * gen_data_footer.c -- JIT generator data footer access (pure C)
 *
 * Phase 3D conversion: gen_data_footer.cpp -> gen_data_footer.c
 * Provides access to the GenDataFooter pointer stored after
 * the Python frame data in a generator object.
 */

#include "cinderx/Jit/gen_data_footer.h"
#include "cinderx/module_c_state.h"

#if PY_VERSION_HEX >= 0x030C0000

#include "internal/pycore_frame.h"
#include "jit_common/py-portability.h"

void **
jit_gen_data_footer_ptr_code(PyGenObject *gen, PyCodeObject *gen_code)
{
    PyTypeObject *gen_type = Ci_JitGenType();
    size_t python_frame_data_bytes =
        _PyFrame_NumSlotsForCodeObject(gen_code) * gen_type->tp_itemsize;
    return (void **)(
        (uintptr_t)gen + gen_type->tp_basicsize +
        python_frame_data_bytes);
}

void **
jit_gen_data_footer_ptr(PyGenObject *gen)
{
#if PY_VERSION_HEX >= 0x030E0000
    _PyInterpreterFrame *gen_frame = &gen->gi_iframe;
#else
    _PyInterpreterFrame *gen_frame = (_PyInterpreterFrame *)(gen->gi_iframe);
#endif
    return jit_gen_data_footer_ptr_code(gen, _PyFrame_GetCode(gen_frame));
}

void *
jit_gen_data_footer(PyGenObject *gen)
{
    return *jit_gen_data_footer_ptr(gen);
}

#endif /* PY_VERSION_HEX >= 0x030C0000 */
