/*
 * C Extension module to test Python internal C APIs (Include/internal).
 */

#ifndef Py_BUILD_CORE_BUILTIN
#  define Py_BUILD_CORE_MODULE 1
#endif

/* Always enable assertions */
#undef NDEBUG

#define PY_SSIZE_T_CLEAN

#include "Python.h"
#include "frameobject.h"
#include "interpreteridobject.h" // _PyInterpreterID_LookUp()
#include "pycore_atomic_funcs.h" // _Py_atomic_int_get()
#include "pycore_bitutils.h"     // _Py_bswap32()
#include "pycore_bytesobject.h"  // _PyBytes_Find()
#include "pycore_compile.h"      // _PyCompile_CodeGen, _PyCompile_OptimizeCfg, _PyCompile_Assemble
#include "pycore_ceval.h"        // _PyEval_AddPendingCall
#include "pycore_fileutils.h"    // _Py_normpath
#include "pycore_frame.h"        // _PyInterpreterFrame
#include "pycore_gc.h"           // PyGC_Head
#include "pycore_hashtable.h"    // _Py_hashtable_new()
#include "pycore_initconfig.h"   // _Py_GetConfigsAsDict()
#include "pycore_interp.h"       // _PyInterpreterState_GetConfigCopy()
#include "pycore_pathconfig.h"   // _PyPathConfig_ClearGlobal()
#include "pycore_pyerrors.h"     // _Py_UTF8_Edit_Cost()
#include "pycore_pystate.h"      // _PyThreadState_GET()
#include "osdefs.h"              // MAXPATHLEN

#include "pycore_ws_deque.h"       // _PyWSDeque, _PyGCLocalBuffer
#include "pycore_gc_barrier.h"     // _PyGCBarrier
#ifdef Py_PARALLEL_GC
#include "pycore_gc_parallel.h"    // _PyGCSplitVector, _PyGCWorkQueue, _PyGCSemaphore
#endif

#include <pthread.h>                // pthread_create

#include "clinic/_testinternalcapi.c.h"


#define MODULE_NAME "_testinternalcapi"


static PyObject *
_get_current_module(void)
{
    // We ensured it was imported in _run_script().
    PyObject *name = PyUnicode_FromString(MODULE_NAME);
    if (name == NULL) {
        return NULL;
    }
    PyObject *mod = PyImport_GetModule(name);
    Py_DECREF(name);
    if (mod == NULL) {
        return NULL;
    }
    assert(mod != Py_None);
    return mod;
}


/* module state *************************************************************/

typedef struct {
    PyObject *record_list;
} module_state;

static inline module_state *
get_module_state(PyObject *mod)
{
    assert(mod != NULL);
    module_state *state = PyModule_GetState(mod);
    assert(state != NULL);
    return state;
}

static int
traverse_module_state(module_state *state, visitproc visit, void *arg)
{
    Py_VISIT(state->record_list);
    return 0;
}

static int
clear_module_state(module_state *state)
{
    Py_CLEAR(state->record_list);
    return 0;
}


/* module functions *********************************************************/

/*[clinic input]
module _testinternalcapi
[clinic start generated code]*/
/*[clinic end generated code: output=da39a3ee5e6b4b0d input=7bb583d8c9eb9a78]*/
static PyObject *
get_configs(PyObject *self, PyObject *Py_UNUSED(args))
{
    return _Py_GetConfigsAsDict();
}


static PyObject*
get_recursion_depth(PyObject *self, PyObject *Py_UNUSED(args))
{
    PyThreadState *tstate = _PyThreadState_GET();

    return PyLong_FromLong(tstate->py_recursion_limit - tstate->py_recursion_remaining);
}


static PyObject*
test_bswap(PyObject *self, PyObject *Py_UNUSED(args))
{
    uint16_t u16 = _Py_bswap16(UINT16_C(0x3412));
    if (u16 != UINT16_C(0x1234)) {
        PyErr_Format(PyExc_AssertionError,
                     "_Py_bswap16(0x3412) returns %u", u16);
        return NULL;
    }

    uint32_t u32 = _Py_bswap32(UINT32_C(0x78563412));
    if (u32 != UINT32_C(0x12345678)) {
        PyErr_Format(PyExc_AssertionError,
                     "_Py_bswap32(0x78563412) returns %lu", u32);
        return NULL;
    }

    uint64_t u64 = _Py_bswap64(UINT64_C(0xEFCDAB9078563412));
    if (u64 != UINT64_C(0x1234567890ABCDEF)) {
        PyErr_Format(PyExc_AssertionError,
                     "_Py_bswap64(0xEFCDAB9078563412) returns %llu", u64);
        return NULL;
    }

    Py_RETURN_NONE;
}


static int
check_popcount(uint32_t x, int expected)
{
    // Use volatile to prevent the compiler to optimize out the whole test
    volatile uint32_t u = x;
    int bits = _Py_popcount32(u);
    if (bits != expected) {
        PyErr_Format(PyExc_AssertionError,
                     "_Py_popcount32(%lu) returns %i, expected %i",
                     (unsigned long)x, bits, expected);
        return -1;
    }
    return 0;
}


static PyObject*
test_popcount(PyObject *self, PyObject *Py_UNUSED(args))
{
#define CHECK(X, RESULT) \
    do { \
        if (check_popcount(X, RESULT) < 0) { \
            return NULL; \
        } \
    } while (0)

    CHECK(0, 0);
    CHECK(1, 1);
    CHECK(0x08080808, 4);
    CHECK(0x10000001, 2);
    CHECK(0x10101010, 4);
    CHECK(0x10204080, 4);
    CHECK(0xDEADCAFE, 22);
    CHECK(0xFFFFFFFF, 32);
    Py_RETURN_NONE;

#undef CHECK
}


static int
check_bit_length(unsigned long x, int expected)
{
    // Use volatile to prevent the compiler to optimize out the whole test
    volatile unsigned long u = x;
    int len = _Py_bit_length(u);
    if (len != expected) {
        PyErr_Format(PyExc_AssertionError,
                     "_Py_bit_length(%lu) returns %i, expected %i",
                     x, len, expected);
        return -1;
    }
    return 0;
}


static PyObject*
test_bit_length(PyObject *self, PyObject *Py_UNUSED(args))
{
#define CHECK(X, RESULT) \
    do { \
        if (check_bit_length(X, RESULT) < 0) { \
            return NULL; \
        } \
    } while (0)

    CHECK(0, 0);
    CHECK(1, 1);
    CHECK(0x1000, 13);
    CHECK(0x1234, 13);
    CHECK(0x54321, 19);
    CHECK(0x7FFFFFFF, 31);
    CHECK(0xFFFFFFFF, 32);
    Py_RETURN_NONE;

#undef CHECK
}


#define TO_PTR(ch) ((void*)(uintptr_t)ch)
#define FROM_PTR(ptr) ((uintptr_t)ptr)
#define VALUE(key) (1 + ((int)(key) - 'a'))

static Py_uhash_t
hash_char(const void *key)
{
    char ch = (char)FROM_PTR(key);
    return ch;
}


static int
hashtable_cb(_Py_hashtable_t *table,
             const void *key_ptr, const void *value_ptr,
             void *user_data)
{
    int *count = (int *)user_data;
    char key = (char)FROM_PTR(key_ptr);
    int value = (int)FROM_PTR(value_ptr);
    assert(value == VALUE(key));
    *count += 1;
    return 0;
}


static PyObject*
test_hashtable(PyObject *self, PyObject *Py_UNUSED(args))
{
    _Py_hashtable_t *table = _Py_hashtable_new(hash_char,
                                               _Py_hashtable_compare_direct);
    if (table == NULL) {
        return PyErr_NoMemory();
    }

    // Using an newly allocated table must not crash
    assert(table->nentries == 0);
    assert(table->nbuckets > 0);
    assert(_Py_hashtable_get(table, TO_PTR('x')) == NULL);

    // Test _Py_hashtable_set()
    char key;
    for (key='a'; key <= 'z'; key++) {
        int value = VALUE(key);
        if (_Py_hashtable_set(table, TO_PTR(key), TO_PTR(value)) < 0) {
            _Py_hashtable_destroy(table);
            return PyErr_NoMemory();
        }
    }
    assert(table->nentries == 26);
    assert(table->nbuckets > table->nentries);

    // Test _Py_hashtable_get_entry()
    for (key='a'; key <= 'z'; key++) {
        _Py_hashtable_entry_t *entry = _Py_hashtable_get_entry(table, TO_PTR(key));
        assert(entry != NULL);
        assert(entry->key == TO_PTR(key));
        assert(entry->value == TO_PTR(VALUE(key)));
    }

    // Test _Py_hashtable_get()
    for (key='a'; key <= 'z'; key++) {
        void *value_ptr = _Py_hashtable_get(table, TO_PTR(key));
        assert((int)FROM_PTR(value_ptr) == VALUE(key));
    }

    // Test _Py_hashtable_steal()
    key = 'p';
    void *value_ptr = _Py_hashtable_steal(table, TO_PTR(key));
    assert((int)FROM_PTR(value_ptr) == VALUE(key));
    assert(table->nentries == 25);
    assert(_Py_hashtable_get_entry(table, TO_PTR(key)) == NULL);

    // Test _Py_hashtable_foreach()
    int count = 0;
    int res = _Py_hashtable_foreach(table, hashtable_cb, &count);
    assert(res == 0);
    assert(count == 25);

    // Test _Py_hashtable_clear()
    _Py_hashtable_clear(table);
    assert(table->nentries == 0);
    assert(table->nbuckets > 0);
    assert(_Py_hashtable_get(table, TO_PTR('x')) == NULL);

    _Py_hashtable_destroy(table);
    Py_RETURN_NONE;
}


static PyObject *
test_get_config(PyObject *Py_UNUSED(self), PyObject *Py_UNUSED(args))
{
    PyConfig config;
    PyConfig_InitIsolatedConfig(&config);
    if (_PyInterpreterState_GetConfigCopy(&config) < 0) {
        PyConfig_Clear(&config);
        return NULL;
    }
    PyObject *dict = _PyConfig_AsDict(&config);
    PyConfig_Clear(&config);
    return dict;
}


static PyObject *
test_set_config(PyObject *Py_UNUSED(self), PyObject *dict)
{
    PyConfig config;
    PyConfig_InitIsolatedConfig(&config);
    if (_PyConfig_FromDict(&config, dict) < 0) {
        goto error;
    }
    if (_PyInterpreterState_SetConfig(&config) < 0) {
        goto error;
    }
    PyConfig_Clear(&config);
    Py_RETURN_NONE;

error:
    PyConfig_Clear(&config);
    return NULL;
}


static PyObject *
test_reset_path_config(PyObject *Py_UNUSED(self), PyObject *Py_UNUSED(arg))
{
    _PyPathConfig_ClearGlobal();
    Py_RETURN_NONE;
}


static PyObject*
test_atomic_funcs(PyObject *self, PyObject *Py_UNUSED(args))
{
    // Test _Py_atomic_size_get() and _Py_atomic_size_set()
    Py_ssize_t var = 1;
    _Py_atomic_size_set(&var, 2);
    assert(_Py_atomic_size_get(&var) == 2);
    Py_RETURN_NONE;
}


static int
check_edit_cost(const char *a, const char *b, Py_ssize_t expected)
{
    int ret = -1;
    PyObject *a_obj = NULL;
    PyObject *b_obj = NULL;

    a_obj = PyUnicode_FromString(a);
    if (a_obj == NULL) {
        goto exit;
    }
    b_obj = PyUnicode_FromString(b);
    if (b_obj == NULL) {
        goto exit;
    }
    Py_ssize_t result = _Py_UTF8_Edit_Cost(a_obj, b_obj, -1);
    if (result != expected) {
        PyErr_Format(PyExc_AssertionError,
                     "Edit cost from '%s' to '%s' returns %zd, expected %zd",
                     a, b, result, expected);
        goto exit;
    }
    // Check that smaller max_edits thresholds are exceeded.
    Py_ssize_t max_edits = result;
    while (max_edits > 0) {
        max_edits /= 2;
        Py_ssize_t result2 = _Py_UTF8_Edit_Cost(a_obj, b_obj, max_edits);
        if (result2 <= max_edits) {
            PyErr_Format(PyExc_AssertionError,
                         "Edit cost from '%s' to '%s' (threshold %zd) "
                         "returns %zd, expected greater than %zd",
                         a, b, max_edits, result2, max_edits);
            goto exit;
        }
    }
    // Check that bigger max_edits thresholds don't change anything
    Py_ssize_t result3 = _Py_UTF8_Edit_Cost(a_obj, b_obj, result * 2 + 1);
    if (result3 != result) {
        PyErr_Format(PyExc_AssertionError,
                     "Edit cost from '%s' to '%s' (threshold %zd) "
                     "returns %zd, expected %zd",
                     a, b, result * 2, result3, result);
        goto exit;
    }
    ret = 0;
exit:
    Py_XDECREF(a_obj);
    Py_XDECREF(b_obj);
    return ret;
}

static PyObject *
test_edit_cost(PyObject *self, PyObject *Py_UNUSED(args))
{
    #define CHECK(a, b, n) do {              \
        if (check_edit_cost(a, b, n) < 0) {  \
            return NULL;                     \
        }                                    \
    } while (0)                              \

    CHECK("", "", 0);
    CHECK("", "a", 2);
    CHECK("a", "A", 1);
    CHECK("Apple", "Aple", 2);
    CHECK("Banana", "B@n@n@", 6);
    CHECK("Cherry", "Cherry!", 2);
    CHECK("---0---", "------", 2);
    CHECK("abc", "y", 6);
    CHECK("aa", "bb", 4);
    CHECK("aaaaa", "AAAAA", 5);
    CHECK("wxyz", "wXyZ", 2);
    CHECK("wxyz", "wXyZ123", 8);
    CHECK("Python", "Java", 12);
    CHECK("Java", "C#", 8);
    CHECK("AbstractFoobarManager", "abstract_foobar_manager", 3+2*2);
    CHECK("CPython", "PyPy", 10);
    CHECK("CPython", "pypy", 11);
    CHECK("AttributeError", "AttributeErrop", 2);
    CHECK("AttributeError", "AttributeErrorTests", 10);

    #undef CHECK
    Py_RETURN_NONE;
}


static int
check_bytes_find(const char *haystack0, const char *needle0,
                 int offset, Py_ssize_t expected)
{
    Py_ssize_t len_haystack = strlen(haystack0);
    Py_ssize_t len_needle = strlen(needle0);
    Py_ssize_t result_1 = _PyBytes_Find(haystack0, len_haystack,
                                        needle0, len_needle, offset);
    if (result_1 != expected) {
        PyErr_Format(PyExc_AssertionError,
                    "Incorrect result_1: '%s' in '%s' (offset=%zd)",
                    needle0, haystack0, offset);
        return -1;
    }
    // Allocate new buffer with no NULL terminator.
    char *haystack = PyMem_Malloc(len_haystack);
    if (haystack == NULL) {
        PyErr_NoMemory();
        return -1;
    }
    char *needle = PyMem_Malloc(len_needle);
    if (needle == NULL) {
        PyMem_Free(haystack);
        PyErr_NoMemory();
        return -1;
    }
    memcpy(haystack, haystack0, len_haystack);
    memcpy(needle, needle0, len_needle);
    Py_ssize_t result_2 = _PyBytes_Find(haystack, len_haystack,
                                        needle, len_needle, offset);
    PyMem_Free(haystack);
    PyMem_Free(needle);
    if (result_2 != expected) {
        PyErr_Format(PyExc_AssertionError,
                    "Incorrect result_2: '%s' in '%s' (offset=%zd)",
                    needle0, haystack0, offset);
        return -1;
    }
    return 0;
}

static int
check_bytes_find_large(Py_ssize_t len_haystack, Py_ssize_t len_needle,
                       const char *needle)
{
    char *zeros = PyMem_RawCalloc(len_haystack, 1);
    if (zeros == NULL) {
        PyErr_NoMemory();
        return -1;
    }
    Py_ssize_t res = _PyBytes_Find(zeros, len_haystack, needle, len_needle, 0);
    PyMem_RawFree(zeros);
    if (res != -1) {
        PyErr_Format(PyExc_AssertionError,
                    "check_bytes_find_large(%zd, %zd) found %zd",
                    len_haystack, len_needle, res);
        return -1;
    }
    return 0;
}

static PyObject *
test_bytes_find(PyObject *self, PyObject *Py_UNUSED(args))
{
    #define CHECK(H, N, O, E) do {               \
        if (check_bytes_find(H, N, O, E) < 0) {  \
            return NULL;                         \
        }                                        \
    } while (0)

    CHECK("", "", 0, 0);
    CHECK("Python", "", 0, 0);
    CHECK("Python", "", 3, 3);
    CHECK("Python", "", 6, 6);
    CHECK("Python", "yth", 0, 1);
    CHECK("ython", "yth", 1, 1);
    CHECK("thon", "yth", 2, -1);
    CHECK("Python", "thon", 0, 2);
    CHECK("ython", "thon", 1, 2);
    CHECK("thon", "thon", 2, 2);
    CHECK("hon", "thon", 3, -1);
    CHECK("Pytho", "zz", 0, -1);
    CHECK("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "ab", 0, -1);
    CHECK("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "ba", 0, -1);
    CHECK("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "bb", 0, -1);
    CHECK("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab", "ab", 0, 30);
    CHECK("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaba", "ba", 0, 30);
    CHECK("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaabb", "bb", 0, 30);
    #undef CHECK

    // Hunt for segfaults
    // n, m chosen here so that (n - m) % (m + 1) == 0
    // This would make default_find in fastsearch.h access haystack[n].
    if (check_bytes_find_large(2048, 2, "ab") < 0) {
        return NULL;
    }
    if (check_bytes_find_large(4096, 16, "0123456789abcdef") < 0) {
        return NULL;
    }
    if (check_bytes_find_large(8192, 2, "ab") < 0) {
        return NULL;
    }
    if (check_bytes_find_large(16384, 4, "abcd") < 0) {
        return NULL;
    }
    if (check_bytes_find_large(32768, 2, "ab") < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}


static PyObject *
normalize_path(PyObject *self, PyObject *filename)
{
    Py_ssize_t size = -1;
    wchar_t *encoded = PyUnicode_AsWideCharString(filename, &size);
    if (encoded == NULL) {
        return NULL;
    }

    PyObject *result = PyUnicode_FromWideChar(_Py_normpath(encoded, size), -1);
    PyMem_Free(encoded);

    return result;
}

static PyObject *
get_getpath_codeobject(PyObject *self, PyObject *Py_UNUSED(args)) {
    return _Py_Get_Getpath_CodeObject();
}


static PyObject *
encode_locale_ex(PyObject *self, PyObject *args)
{
    PyObject *unicode;
    int current_locale = 0;
    wchar_t *wstr;
    PyObject *res = NULL;
    const char *errors = NULL;

    if (!PyArg_ParseTuple(args, "U|is", &unicode, &current_locale, &errors)) {
        return NULL;
    }
    wstr = PyUnicode_AsWideCharString(unicode, NULL);
    if (wstr == NULL) {
        return NULL;
    }
    _Py_error_handler error_handler = _Py_GetErrorHandler(errors);

    char *str = NULL;
    size_t error_pos;
    const char *reason = NULL;
    int ret = _Py_EncodeLocaleEx(wstr,
                                 &str, &error_pos, &reason,
                                 current_locale, error_handler);
    PyMem_Free(wstr);

    switch(ret) {
    case 0:
        res = PyBytes_FromString(str);
        PyMem_RawFree(str);
        break;
    case -1:
        PyErr_NoMemory();
        break;
    case -2:
        PyErr_Format(PyExc_RuntimeError, "encode error: pos=%zu, reason=%s",
                     error_pos, reason);
        break;
    case -3:
        PyErr_SetString(PyExc_ValueError, "unsupported error handler");
        break;
    default:
        PyErr_SetString(PyExc_ValueError, "unknown error code");
        break;
    }
    return res;
}


static PyObject *
decode_locale_ex(PyObject *self, PyObject *args)
{
    char *str;
    int current_locale = 0;
    PyObject *res = NULL;
    const char *errors = NULL;

    if (!PyArg_ParseTuple(args, "y|is", &str, &current_locale, &errors)) {
        return NULL;
    }
    _Py_error_handler error_handler = _Py_GetErrorHandler(errors);

    wchar_t *wstr = NULL;
    size_t wlen = 0;
    const char *reason = NULL;
    int ret = _Py_DecodeLocaleEx(str,
                                 &wstr, &wlen, &reason,
                                 current_locale, error_handler);

    switch(ret) {
    case 0:
        res = PyUnicode_FromWideChar(wstr, wlen);
        PyMem_RawFree(wstr);
        break;
    case -1:
        PyErr_NoMemory();
        break;
    case -2:
        PyErr_Format(PyExc_RuntimeError, "decode error: pos=%zu, reason=%s",
                     wlen, reason);
        break;
    case -3:
        PyErr_SetString(PyExc_ValueError, "unsupported error handler");
        break;
    default:
        PyErr_SetString(PyExc_ValueError, "unknown error code");
        break;
    }
    return res;
}

static PyObject *
set_eval_frame_default(PyObject *self, PyObject *Py_UNUSED(args))
{
    module_state *state = get_module_state(self);
    _PyInterpreterState_SetEvalFrameFunc(PyInterpreterState_Get(), _PyEval_EvalFrameDefault);
    Py_CLEAR(state->record_list);
    Py_RETURN_NONE;
}

static PyObject *
record_eval(PyThreadState *tstate, struct _PyInterpreterFrame *f, int exc)
{
    if (PyFunction_Check(f->f_funcobj)) {
        PyObject *module = _get_current_module();
        assert(module != NULL);
        module_state *state = get_module_state(module);
        Py_DECREF(module);
        int res = PyList_Append(state->record_list,
                                ((PyFunctionObject *)f->f_funcobj)->func_name);
        if (res < 0) {
            return NULL;
        }
    }
    return _PyEval_EvalFrameDefault(tstate, f, exc);
}


static PyObject *
set_eval_frame_record(PyObject *self, PyObject *list)
{
    module_state *state = get_module_state(self);
    if (!PyList_Check(list)) {
        PyErr_SetString(PyExc_TypeError, "argument must be a list");
        return NULL;
    }
    Py_XSETREF(state->record_list, Py_NewRef(list));
    _PyInterpreterState_SetEvalFrameFunc(PyInterpreterState_Get(), record_eval);
    Py_RETURN_NONE;
}

/*[clinic input]

_testinternalcapi.compiler_codegen -> object

  ast: object
  filename: object
  optimize: int
  compile_mode: int = 0

Apply compiler code generation to an AST.
[clinic start generated code]*/

static PyObject *
_testinternalcapi_compiler_codegen_impl(PyObject *module, PyObject *ast,
                                        PyObject *filename, int optimize,
                                        int compile_mode)
/*[clinic end generated code: output=40a68f6e13951cc8 input=a0e00784f1517cd7]*/
{
    PyCompilerFlags *flags = NULL;
    return _PyCompile_CodeGen(ast, filename, flags, optimize, compile_mode);
}


/*[clinic input]

_testinternalcapi.optimize_cfg -> object

  instructions: object
  consts: object
  nlocals: int

Apply compiler optimizations to an instruction list.
[clinic start generated code]*/

static PyObject *
_testinternalcapi_optimize_cfg_impl(PyObject *module, PyObject *instructions,
                                    PyObject *consts, int nlocals)
/*[clinic end generated code: output=57c53c3a3dfd1df0 input=6a96d1926d58d7e5]*/
{
    return _PyCompile_OptimizeCfg(instructions, consts, nlocals);
}

static int
get_nonnegative_int_from_dict(PyObject *dict, const char *key) {
    PyObject *obj = PyDict_GetItemString(dict, key);
    if (obj == NULL) {
        return -1;
    }
    return PyLong_AsLong(obj);
}

/*[clinic input]

_testinternalcapi.assemble_code_object -> object

  filename: object
  instructions: object
  metadata: object

Create a code object for the given instructions.
[clinic start generated code]*/

static PyObject *
_testinternalcapi_assemble_code_object_impl(PyObject *module,
                                            PyObject *filename,
                                            PyObject *instructions,
                                            PyObject *metadata)
/*[clinic end generated code: output=38003dc16a930f48 input=e713ad77f08fb3a8]*/

{
    assert(PyDict_Check(metadata));
    _PyCompile_CodeUnitMetadata umd;

    umd.u_name = PyDict_GetItemString(metadata, "name");
    umd.u_qualname = PyDict_GetItemString(metadata, "qualname");

    assert(PyUnicode_Check(umd.u_name));
    assert(PyUnicode_Check(umd.u_qualname));

    umd.u_consts = PyDict_GetItemString(metadata, "consts");
    umd.u_names = PyDict_GetItemString(metadata, "names");
    umd.u_varnames = PyDict_GetItemString(metadata, "varnames");
    umd.u_cellvars = PyDict_GetItemString(metadata, "cellvars");
    umd.u_freevars = PyDict_GetItemString(metadata, "freevars");
    umd.u_fasthidden = PyDict_GetItemString(metadata, "fasthidden");

    assert(PyDict_Check(umd.u_consts));
    assert(PyDict_Check(umd.u_names));
    assert(PyDict_Check(umd.u_varnames));
    assert(PyDict_Check(umd.u_cellvars));
    assert(PyDict_Check(umd.u_freevars));
    assert(PyDict_Check(umd.u_fasthidden));

    umd.u_argcount = get_nonnegative_int_from_dict(metadata, "argcount");
    umd.u_posonlyargcount = get_nonnegative_int_from_dict(metadata, "posonlyargcount");
    umd.u_kwonlyargcount = get_nonnegative_int_from_dict(metadata, "kwonlyargcount");
    umd.u_firstlineno = get_nonnegative_int_from_dict(metadata, "firstlineno");

    assert(umd.u_argcount >= 0);
    assert(umd.u_posonlyargcount >= 0);
    assert(umd.u_kwonlyargcount >= 0);
    assert(umd.u_firstlineno >= 0);

    return (PyObject*)_PyCompile_Assemble(&umd, filename, instructions);
}


static PyObject *
get_interp_settings(PyObject *self, PyObject *args)
{
    int interpid = -1;
    if (!PyArg_ParseTuple(args, "|i:get_interp_settings", &interpid)) {
        return NULL;
    }

    PyInterpreterState *interp = NULL;
    if (interpid < 0) {
        PyThreadState *tstate = _PyThreadState_GET();
        interp = tstate ? tstate->interp : _PyInterpreterState_Main();
    }
    else if (interpid == 0) {
        interp = _PyInterpreterState_Main();
    }
    else {
        PyErr_Format(PyExc_NotImplementedError,
                     "%zd", interpid);
        return NULL;
    }
    assert(interp != NULL);

    PyObject *settings = PyDict_New();
    if (settings == NULL) {
        return NULL;
    }

    /* Add the feature flags. */
    PyObject *flags = PyLong_FromUnsignedLong(interp->feature_flags);
    if (flags == NULL) {
        Py_DECREF(settings);
        return NULL;
    }
    int res = PyDict_SetItemString(settings, "feature_flags", flags);
    Py_DECREF(flags);
    if (res != 0) {
        Py_DECREF(settings);
        return NULL;
    }

    /* "own GIL" */
    PyObject *own_gil = interp->ceval.own_gil ? Py_True : Py_False;
    if (PyDict_SetItemString(settings, "own_gil", own_gil) != 0) {
        Py_DECREF(settings);
        return NULL;
    }

    return settings;
}


static PyObject *
clear_extension(PyObject *self, PyObject *args)
{
    PyObject *name = NULL, *filename = NULL;
    if (!PyArg_ParseTuple(args, "OO:clear_extension", &name, &filename)) {
        return NULL;
    }
    if (_PyImport_ClearExtension(name, filename) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *
write_perf_map_entry(PyObject *self, PyObject *args)
{
    PyObject *code_addr_v;
    const void *code_addr;
    unsigned int code_size;
    const char *entry_name;

    if (!PyArg_ParseTuple(args, "OIs", &code_addr_v, &code_size, &entry_name))
        return NULL;
    code_addr = PyLong_AsVoidPtr(code_addr_v);
    if (code_addr == NULL) {
        return NULL;
    }

    int ret = PyUnstable_WritePerfMapEntry(code_addr, code_size, entry_name);
    if (ret < 0) {
        PyErr_SetFromErrno(PyExc_OSError);
        return NULL;
    }
    return PyLong_FromLong(ret);
}

static PyObject *
perf_map_state_teardown(PyObject *Py_UNUSED(self), PyObject *Py_UNUSED(ignored))
{
    PyUnstable_PerfMapState_Fini();
    Py_RETURN_NONE;
}

static PyObject *
iframe_getcode(PyObject *self, PyObject *frame)
{
    if (!PyFrame_Check(frame)) {
        PyErr_SetString(PyExc_TypeError, "argument must be a frame");
        return NULL;
    }
    struct _PyInterpreterFrame *f = ((PyFrameObject *)frame)->f_frame;
    return PyUnstable_InterpreterFrame_GetCode(f);
}

static PyObject *
iframe_getline(PyObject *self, PyObject *frame)
{
    if (!PyFrame_Check(frame)) {
        PyErr_SetString(PyExc_TypeError, "argument must be a frame");
        return NULL;
    }
    struct _PyInterpreterFrame *f = ((PyFrameObject *)frame)->f_frame;
    return PyLong_FromLong(PyUnstable_InterpreterFrame_GetLine(f));
}

static PyObject *
iframe_getlasti(PyObject *self, PyObject *frame)
{
    if (!PyFrame_Check(frame)) {
        PyErr_SetString(PyExc_TypeError, "argument must be a frame");
        return NULL;
    }
    struct _PyInterpreterFrame *f = ((PyFrameObject *)frame)->f_frame;
    return PyLong_FromLong(PyUnstable_InterpreterFrame_GetLasti(f));
}


static int _pending_callback(void *arg)
{
    /* we assume the argument is callable object to which we own a reference */
    PyObject *callable = (PyObject *)arg;
    PyObject *r = PyObject_CallNoArgs(callable);
    Py_DECREF(callable);
    Py_XDECREF(r);
    return r != NULL ? 0 : -1;
}

/* The following requests n callbacks to _pending_callback.  It can be
 * run from any python thread.
 */
static PyObject *
pending_threadfunc(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyObject *callable;
    int ensure_added = 0;
    static char *kwlist[] = {"", "ensure_added", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs,
                                     "O|$p:pending_threadfunc", kwlist,
                                     &callable, &ensure_added))
    {
        return NULL;
    }
    PyInterpreterState *interp = PyInterpreterState_Get();

    /* create the reference for the callbackwhile we hold the lock */
    Py_INCREF(callable);

    int r;
    Py_BEGIN_ALLOW_THREADS
    r = _PyEval_AddPendingCall(interp, &_pending_callback, callable, 0);
    Py_END_ALLOW_THREADS
    if (r < 0) {
        /* unsuccessful add */
        if (!ensure_added) {
            Py_DECREF(callable);
            Py_RETURN_FALSE;
        }
        do {
            Py_BEGIN_ALLOW_THREADS
            r = _PyEval_AddPendingCall(interp, &_pending_callback, callable, 0);
            Py_END_ALLOW_THREADS
        } while (r < 0);
    }

    Py_RETURN_TRUE;
}


static struct {
    int64_t interpid;
} pending_identify_result;

static int
_pending_identify_callback(void *arg)
{
    PyThread_type_lock mutex = (PyThread_type_lock)arg;
    assert(pending_identify_result.interpid == -1);
    PyThreadState *tstate = PyThreadState_Get();
    pending_identify_result.interpid = PyInterpreterState_GetID(tstate->interp);
    PyThread_release_lock(mutex);
    return 0;
}

static PyObject *
pending_identify(PyObject *self, PyObject *args)
{
    PyObject *interpid;
    if (!PyArg_ParseTuple(args, "O:pending_identify", &interpid)) {
        return NULL;
    }
    PyInterpreterState *interp = _PyInterpreterID_LookUp(interpid);
    if (interp == NULL) {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_ValueError, "interpreter not found");
        }
        return NULL;
    }

    pending_identify_result.interpid = -1;

    PyThread_type_lock mutex = PyThread_allocate_lock();
    if (mutex == NULL) {
        return NULL;
    }
    PyThread_acquire_lock(mutex, WAIT_LOCK);
    /* It gets released in _pending_identify_callback(). */

    int r;
    do {
        Py_BEGIN_ALLOW_THREADS
        r = _PyEval_AddPendingCall(interp,
                                   &_pending_identify_callback, (void *)mutex,
                                   0);
        Py_END_ALLOW_THREADS
    } while (r < 0);

    /* Wait for the pending call to complete. */
    PyThread_acquire_lock(mutex, WAIT_LOCK);
    PyThread_release_lock(mutex);
    PyThread_free_lock(mutex);

    PyObject *res = PyLong_FromLongLong(pending_identify_result.interpid);
    pending_identify_result.interpid = -1;
    if (res == NULL) {
        return NULL;
    }
    return res;
}


/*[clinic input]
gh_119213_getargs

    spam: object = None

Test _PyArg_Parser.kwtuple
[clinic start generated code]*/

static PyObject *
gh_119213_getargs_impl(PyObject *module, PyObject *spam)
/*[clinic end generated code: output=d8d9c95d5b446802 input=65ef47511da80fc2]*/
{
    // It must never have been called in the main interprer
    assert(!_Py_IsMainInterpreter(PyInterpreterState_Get()));
    return Py_NewRef(spam);
}


// ============================================================================
// Parallel GC Infrastructure Tests
// ============================================================================

// --- Work-Stealing Deque: Basic Operations ---

static PyObject *
test_ws_deque_init_fini(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    _PyWSDeque deque;
    _PyWSDeque_Init(&deque);

    if (_PyWSDeque_Size(&deque) != 0) {
        _PyWSDeque_Fini(&deque);
        PyErr_SetString(PyExc_AssertionError, "New deque should be empty");
        return NULL;
    }

    if (_PyWSDeque_GetNumResizes(&deque) != 0) {
        _PyWSDeque_Fini(&deque);
        PyErr_SetString(PyExc_AssertionError, "New deque should have 0 resizes");
        return NULL;
    }

    _PyWSDeque_Fini(&deque);
    Py_RETURN_NONE;
}

static PyObject *
test_ws_deque_push_take_single(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    _PyWSDeque deque;
    _PyWSDeque_Init(&deque);

    PyObject *obj = PyLong_FromLong(42);
    if (obj == NULL) {
        _PyWSDeque_Fini(&deque);
        return NULL;
    }
    Py_INCREF(obj);

    _PyWSDeque_Push(&deque, obj);

    if (_PyWSDeque_Size(&deque) != 1) {
        Py_DECREF(obj);
        _PyWSDeque_Fini(&deque);
        PyErr_SetString(PyExc_AssertionError, "Deque size should be 1 after push");
        return NULL;
    }

    PyObject *result = _PyWSDeque_Take(&deque);

    if (result != obj) {
        Py_XDECREF(result);
        Py_DECREF(obj);
        _PyWSDeque_Fini(&deque);
        PyErr_SetString(PyExc_AssertionError, "Take should return the pushed object");
        return NULL;
    }

    if (_PyWSDeque_Size(&deque) != 0) {
        Py_DECREF(obj);
        _PyWSDeque_Fini(&deque);
        PyErr_SetString(PyExc_AssertionError, "Deque should be empty after take");
        return NULL;
    }

    Py_DECREF(obj);
    _PyWSDeque_Fini(&deque);
    Py_RETURN_NONE;
}

static PyObject *
test_ws_deque_push_steal_single(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    _PyWSDeque deque;
    _PyWSDeque_Init(&deque);

    PyObject *obj = PyLong_FromLong(123);
    if (obj == NULL) {
        _PyWSDeque_Fini(&deque);
        return NULL;
    }
    Py_INCREF(obj);

    _PyWSDeque_Push(&deque, obj);

    PyObject *result = (PyObject *)_PyWSDeque_Steal(&deque);

    if (result != obj) {
        Py_XDECREF(result);
        Py_DECREF(obj);
        _PyWSDeque_Fini(&deque);
        PyErr_SetString(PyExc_AssertionError, "Steal should return the pushed object");
        return NULL;
    }

    if (_PyWSDeque_Size(&deque) != 0) {
        Py_DECREF(obj);
        _PyWSDeque_Fini(&deque);
        PyErr_SetString(PyExc_AssertionError, "Deque should be empty after steal");
        return NULL;
    }

    Py_DECREF(obj);
    _PyWSDeque_Fini(&deque);
    Py_RETURN_NONE;
}

static PyObject *
test_ws_deque_lifo_order(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    _PyWSDeque deque;
    _PyWSDeque_Init(&deque);

    const int count = 10;
    PyObject *objects[10];

    for (int i = 0; i < count; i++) {
        objects[i] = PyLong_FromLong(i);
        if (objects[i] == NULL) {
            for (int j = 0; j < i; j++) {
                Py_DECREF(objects[j]);
            }
            _PyWSDeque_Fini(&deque);
            return NULL;
        }
        Py_INCREF(objects[i]);
        _PyWSDeque_Push(&deque, objects[i]);
    }

    for (int i = count - 1; i >= 0; i--) {
        PyObject *result = _PyWSDeque_Take(&deque);
        if (result != objects[i]) {
            for (int j = 0; j < count; j++) {
                Py_DECREF(objects[j]);
            }
            _PyWSDeque_Fini(&deque);
            PyErr_Format(PyExc_AssertionError,
                        "Expected object %d, got different object", i);
            return NULL;
        }
    }

    for (int i = 0; i < count; i++) {
        Py_DECREF(objects[i]);
    }

    _PyWSDeque_Fini(&deque);
    Py_RETURN_NONE;
}

static PyObject *
test_ws_deque_fifo_order(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    _PyWSDeque deque;
    _PyWSDeque_Init(&deque);

    const int count = 10;
    PyObject *objects[10];

    for (int i = 0; i < count; i++) {
        objects[i] = PyLong_FromLong(i);
        if (objects[i] == NULL) {
            for (int j = 0; j < i; j++) {
                Py_DECREF(objects[j]);
            }
            _PyWSDeque_Fini(&deque);
            return NULL;
        }
        Py_INCREF(objects[i]);
        _PyWSDeque_Push(&deque, objects[i]);
    }

    for (int i = 0; i < count; i++) {
        PyObject *result = (PyObject *)_PyWSDeque_Steal(&deque);
        if (result != objects[i]) {
            for (int j = 0; j < count; j++) {
                Py_DECREF(objects[j]);
            }
            _PyWSDeque_Fini(&deque);
            PyErr_Format(PyExc_AssertionError,
                        "Expected object %d, got different object", i);
            return NULL;
        }
    }

    for (int i = 0; i < count; i++) {
        Py_DECREF(objects[i]);
    }

    _PyWSDeque_Fini(&deque);
    Py_RETURN_NONE;
}

// --- Work-Stealing Deque: Edge Cases ---

static PyObject *
test_ws_deque_take_empty(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    _PyWSDeque deque;
    _PyWSDeque_Init(&deque);

    PyObject *result = _PyWSDeque_Take(&deque);

    if (result != NULL) {
        Py_DECREF(result);
        _PyWSDeque_Fini(&deque);
        PyErr_SetString(PyExc_AssertionError,
                       "Take from empty deque should return NULL");
        return NULL;
    }

    _PyWSDeque_Fini(&deque);
    Py_RETURN_NONE;
}

static PyObject *
test_ws_deque_steal_empty(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    _PyWSDeque deque;
    _PyWSDeque_Init(&deque);

    PyObject *result = (PyObject *)_PyWSDeque_Steal(&deque);

    if (result != NULL) {
        Py_DECREF(result);
        _PyWSDeque_Fini(&deque);
        PyErr_SetString(PyExc_AssertionError,
                       "Steal from empty deque should return NULL");
        return NULL;
    }

    _PyWSDeque_Fini(&deque);
    Py_RETURN_NONE;
}

static PyObject *
test_ws_deque_resize(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    _PyWSDeque deque;
    _PyWSDeque_Init(&deque);

    const int count = 5000;
    PyObject *obj = PyLong_FromLong(42);
    if (obj == NULL) {
        _PyWSDeque_Fini(&deque);
        return NULL;
    }

    for (int i = 0; i < count; i++) {
        Py_INCREF(obj);
        _PyWSDeque_Push(&deque, obj);
    }

    int num_resizes = _PyWSDeque_GetNumResizes(&deque);
    if (num_resizes < 1) {
        Py_DECREF(obj);
        _PyWSDeque_Fini(&deque);
        PyErr_Format(PyExc_AssertionError,
                    "Expected at least 1 resize, got %d", num_resizes);
        return NULL;
    }

    size_t size = _PyWSDeque_Size(&deque);
    if (size != (size_t)count) {
        Py_DECREF(obj);
        _PyWSDeque_Fini(&deque);
        PyErr_Format(PyExc_AssertionError,
                    "Expected size %d, got %zu", count, size);
        return NULL;
    }

    for (int i = 0; i < count; i++) {
        PyObject *result = _PyWSDeque_Take(&deque);
        if (result == NULL) {
            Py_DECREF(obj);
            _PyWSDeque_Fini(&deque);
            PyErr_Format(PyExc_AssertionError,
                        "Failed to take element %d", i);
            return NULL;
        }
        Py_DECREF(result);
    }

    Py_DECREF(obj);
    _PyWSDeque_Fini(&deque);
    Py_RETURN_NONE;
}

static PyObject *
test_ws_deque_init_with_undersized_buffer(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    _PyWSDeque deque;

    char tiny_buffer[16];
    size_t requested_size = 64;
    int result = _PyWSDeque_InitWithBuffer(
        &deque, tiny_buffer, sizeof(tiny_buffer), requested_size);

    if (result != 0) {
        _PyWSDeque_Fini(&deque);
        PyErr_SetString(PyExc_AssertionError,
                        "Expected fallback (return 0) for undersized buffer");
        return NULL;
    }

    PyObject *obj = PyLong_FromLong(99);
    if (obj == NULL) {
        _PyWSDeque_Fini(&deque);
        return NULL;
    }

    Py_INCREF(obj);
    _PyWSDeque_Push(&deque, obj);

    PyObject *taken = _PyWSDeque_Take(&deque);
    if (taken != obj) {
        Py_DECREF(obj);
        _PyWSDeque_Fini(&deque);
        PyErr_SetString(PyExc_AssertionError,
                        "Deque push/take failed after buffer fallback");
        return NULL;
    }
    Py_DECREF(taken);
    Py_DECREF(obj);
    _PyWSDeque_Fini(&deque);
    Py_RETURN_NONE;
}

static PyObject *
test_ws_deque_init_with_exact_buffer(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    _PyWSDeque deque;

    size_t requested_size = 64;
    size_t required = sizeof(_PyWSArray) + sizeof(uintptr_t) * requested_size;
    char *buffer = PyMem_RawCalloc(1, required);
    if (buffer == NULL) {
        return PyErr_NoMemory();
    }

    int result = _PyWSDeque_InitWithBuffer(
        &deque, buffer, required, requested_size);

    if (result != 1) {
        _PyWSDeque_Fini(&deque);
        PyMem_RawFree(buffer);
        PyErr_SetString(PyExc_AssertionError,
                        "Expected success (return 1) for exact-size buffer");
        return NULL;
    }

    PyObject *obj = PyLong_FromLong(42);
    if (obj == NULL) {
        _PyWSDeque_Fini(&deque);
        PyMem_RawFree(buffer);
        return NULL;
    }

    Py_INCREF(obj);
    _PyWSDeque_Push(&deque, obj);

    PyObject *taken = _PyWSDeque_Take(&deque);
    if (taken != obj) {
        Py_DECREF(obj);
        _PyWSDeque_Fini(&deque);
        PyMem_RawFree(buffer);
        PyErr_SetString(PyExc_AssertionError,
                        "Deque push/take failed with exact buffer");
        return NULL;
    }
    Py_DECREF(taken);
    Py_DECREF(obj);

    _PyWSDeque_FiniExternal(&deque, buffer);
    PyMem_RawFree(buffer);
    Py_RETURN_NONE;
}

// --- Work-Stealing Deque: Concurrent Tests ---

typedef struct {
    _PyWSDeque *deque;
    int num_steals;
    int num_successful;
} steal_worker_args;

static void *
steal_worker(void *arg)
{
    steal_worker_args *args = (steal_worker_args *)arg;
    args->num_successful = 0;

    for (int i = 0; i < args->num_steals; i++) {
        void *obj = _PyWSDeque_Steal(args->deque);
        if (obj != NULL) {
            args->num_successful++;
        }
    }

    return NULL;
}

static PyObject *
test_ws_deque_concurrent_push_steal(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    _PyWSDeque deque;
    _PyWSDeque_Init(&deque);

    const int num_items = 1000;
    const int num_workers = 4;
    const int steals_per_worker = 300;

    PyObject *obj = PyLong_FromLong(42);
    if (obj == NULL) {
        _PyWSDeque_Fini(&deque);
        return NULL;
    }

    pthread_t workers[4];
    steal_worker_args args[4];

    for (int i = 0; i < num_workers; i++) {
        args[i].deque = &deque;
        args[i].num_steals = steals_per_worker;
        args[i].num_successful = 0;
        pthread_create(&workers[i], NULL, steal_worker, &args[i]);
    }

    for (int i = 0; i < num_items; i++) {
        Py_INCREF(obj);
        _PyWSDeque_Push(&deque, obj);
    }

    for (int i = 0; i < num_workers; i++) {
        pthread_join(workers[i], NULL);
    }

    int total_stolen = 0;
    for (int i = 0; i < num_workers; i++) {
        total_stolen += args[i].num_successful;
    }

    int drained = 0;
    PyObject *result;
    while ((result = _PyWSDeque_Take(&deque)) != NULL) {
        Py_DECREF(result);
        drained++;
    }

    if (total_stolen + drained != num_items) {
        Py_DECREF(obj);
        _PyWSDeque_Fini(&deque);
        PyErr_Format(PyExc_AssertionError,
                    "Expected %d items total, got %d stolen + %d drained = %d",
                    num_items, total_stolen, drained, total_stolen + drained);
        return NULL;
    }

    Py_DECREF(obj);
    _PyWSDeque_Fini(&deque);
    Py_RETURN_NONE;
}

// --- Barrier Tests ---

static PyObject *
test_barrier_capacity_zero(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    _PyGCBarrier barrier;
    _PyGCBarrier_Init(&barrier, 0);
    _PyGCBarrier_Fini(&barrier);
    PyErr_SetString(PyExc_AssertionError, "Init(capacity=0) should have aborted");
    return NULL;
}

typedef struct {
    _PyGCBarrier *barrier;
    int arrived;
} barrier_worker_args;

static void *
barrier_worker(void *arg)
{
    barrier_worker_args *args = (barrier_worker_args *)arg;
    args->arrived = 1;
    _PyGCBarrier_Wait(args->barrier);
    return NULL;
}

static PyObject *
test_barrier_basic(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    const int num_threads = 4;
    _PyGCBarrier barrier;
    _PyGCBarrier_Init(&barrier, num_threads);

    pthread_t threads[3];
    barrier_worker_args args[3];

    for (int i = 0; i < num_threads - 1; i++) {
        args[i].barrier = &barrier;
        args[i].arrived = 0;
        pthread_create(&threads[i], NULL, barrier_worker, &args[i]);
    }

    _PyGCBarrier_Wait(&barrier);

    for (int i = 0; i < num_threads - 1; i++) {
        pthread_join(threads[i], NULL);
        if (!args[i].arrived) {
            _PyGCBarrier_Fini(&barrier);
            PyErr_SetString(PyExc_AssertionError, "Worker did not arrive at barrier");
            return NULL;
        }
    }

    _PyGCBarrier_Fini(&barrier);
    Py_RETURN_NONE;
}

static PyObject *
test_barrier_multiple_rounds(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    _PyGCBarrier barrier;
    _PyGCBarrier_Init(&barrier, 1);

    unsigned int epoch_before = barrier.epoch;
    _PyGCBarrier_Wait(&barrier);
    if (barrier.epoch != epoch_before + 1) {
        _PyGCBarrier_Fini(&barrier);
        PyErr_SetString(PyExc_AssertionError, "Epoch should increment by 1 per cycle");
        return NULL;
    }

    _PyGCBarrier_Wait(&barrier);
    if (barrier.epoch != epoch_before + 2) {
        _PyGCBarrier_Fini(&barrier);
        PyErr_SetString(PyExc_AssertionError, "Epoch should increment by 1 per cycle (round 2)");
        return NULL;
    }

    _PyGCBarrier_Fini(&barrier);
    Py_RETURN_NONE;
}

typedef struct {
    _PyGCBarrier *barrier;
    unsigned int epoch_after_round1;
    unsigned int epoch_after_round2;
} epoch_worker_args;

static void *
epoch_worker(void *arg)
{
    epoch_worker_args *args = (epoch_worker_args *)arg;
    _PyGCBarrier_Wait(args->barrier);
    args->epoch_after_round1 = args->barrier->epoch;
    _PyGCBarrier_Wait(args->barrier);
    args->epoch_after_round2 = args->barrier->epoch;
    return NULL;
}

static PyObject *
test_barrier_epoch_distinguishes(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    _PyGCBarrier barrier;
    _PyGCBarrier_Init(&barrier, 2);

    epoch_worker_args args = { .barrier = &barrier };
    pthread_t thread;
    pthread_create(&thread, NULL, epoch_worker, &args);

    unsigned int epoch_before = barrier.epoch;
    _PyGCBarrier_Wait(&barrier);
    _PyGCBarrier_Wait(&barrier);

    pthread_join(thread, NULL);

    if (args.epoch_after_round1 == epoch_before) {
        _PyGCBarrier_Fini(&barrier);
        PyErr_SetString(PyExc_AssertionError, "Epoch should differ after round 1");
        return NULL;
    }
    if (args.epoch_after_round2 == args.epoch_after_round1) {
        _PyGCBarrier_Fini(&barrier);
        PyErr_SetString(PyExc_AssertionError, "Epoch should differ between rounds");
        return NULL;
    }

    _PyGCBarrier_Fini(&barrier);
    Py_RETURN_NONE;
}

static PyObject *
test_barrier_postcondition(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    _PyGCBarrier barrier;
    _PyGCBarrier_Init(&barrier, 1);

    unsigned int epoch_before = barrier.epoch;
    _PyGCBarrier_Wait(&barrier);

    if (barrier.epoch == epoch_before) {
        _PyGCBarrier_Fini(&barrier);
        PyErr_SetString(PyExc_AssertionError, "Epoch did not advance after Wait");
        return NULL;
    }
    if (barrier.num_left != barrier.capacity) {
        _PyGCBarrier_Fini(&barrier);
        PyErr_SetString(PyExc_AssertionError, "num_left not reset after barrier lift");
        return NULL;
    }

    _PyGCBarrier_Fini(&barrier);
    Py_RETURN_NONE;
}

// --- LocalBuffer Tests ---

static PyObject *
test_localbuffer_push_pop(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    _PyGCLocalBuffer buf;
    _PyGCLocalBuffer_Init(&buf);

    if (!_PyGCLocalBuffer_IsEmpty(&buf)) {
        PyErr_SetString(PyExc_AssertionError, "New buffer should be empty");
        return NULL;
    }

    PyObject *obj = PyLong_FromLong(42);
    if (obj == NULL) return NULL;

    _PyGCLocalBuffer_Push(&buf, obj);
    if (_PyGCLocalBuffer_IsEmpty(&buf)) {
        Py_DECREF(obj);
        PyErr_SetString(PyExc_AssertionError, "Buffer should not be empty after push");
        return NULL;
    }

    PyObject *result = _PyGCLocalBuffer_Pop(&buf);
    if (result != obj) {
        Py_DECREF(obj);
        PyErr_SetString(PyExc_AssertionError, "Pop should return pushed object");
        return NULL;
    }
    if (!_PyGCLocalBuffer_IsEmpty(&buf)) {
        Py_DECREF(obj);
        PyErr_SetString(PyExc_AssertionError, "Buffer should be empty after pop");
        return NULL;
    }

    Py_DECREF(obj);
    Py_RETURN_NONE;
}

static PyObject *
test_localbuffer_push_full(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    _PyGCLocalBuffer buf;
    _PyGCLocalBuffer_Init(&buf);

    PyObject *obj = PyLong_FromLong(99);
    if (obj == NULL) return NULL;

    for (int i = 0; i < _PyGC_LOCAL_BUFFER_SIZE; i++) {
        Py_INCREF(obj);
        _PyGCLocalBuffer_Push(&buf, obj);
    }

    if (!_PyGCLocalBuffer_IsFull(&buf)) {
        Py_DECREF(obj);
        PyErr_SetString(PyExc_AssertionError, "Buffer should be full after 1024 pushes");
        return NULL;
    }

    for (int i = 0; i < _PyGC_LOCAL_BUFFER_SIZE; i++) {
        PyObject *r = _PyGCLocalBuffer_Pop(&buf);
        Py_DECREF(r);
    }

    Py_DECREF(obj);
    Py_RETURN_NONE;
}

static PyObject *
test_localbuffer_pop_empty(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    _PyGCLocalBuffer buf;
    _PyGCLocalBuffer_Init(&buf);

    if (!_PyGCLocalBuffer_IsEmpty(&buf)) {
        PyErr_SetString(PyExc_AssertionError, "New buffer should be empty");
        return NULL;
    }

    PyObject *obj = PyLong_FromLong(1);
    if (obj == NULL) return NULL;

    _PyGCLocalBuffer_Push(&buf, obj);
    PyObject *r = _PyGCLocalBuffer_Pop(&buf);
    if (r != obj) {
        Py_DECREF(obj);
        PyErr_SetString(PyExc_AssertionError, "Pop should return pushed object");
        return NULL;
    }

    if (!_PyGCLocalBuffer_IsEmpty(&buf)) {
        Py_DECREF(obj);
        PyErr_SetString(PyExc_AssertionError, "Buffer should be empty after popping all");
        return NULL;
    }

    Py_DECREF(obj);
    Py_RETURN_NONE;
}

static PyObject *
test_overflow_flush_precondition(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    _PyGCLocalBuffer buf;
    _PyGCLocalBuffer_Init(&buf);
    _PyWSDeque deque;
    _PyWSDeque_Init(&deque);

    PyObject *obj = PyLong_FromLong(7);
    if (obj == NULL) {
        _PyWSDeque_Fini(&deque);
        return NULL;
    }

    for (int i = 0; i < _PyGC_LOCAL_BUFFER_SIZE; i++) {
        Py_INCREF(obj);
        _PyGCLocalBuffer_Push(&buf, obj);
    }

    _PyGC_OverflowFlush(&buf, &deque);

    if (buf.count != _PyGC_LOCAL_BUFFER_SIZE / 2) {
        Py_DECREF(obj);
        _PyWSDeque_Fini(&deque);
        PyErr_Format(PyExc_AssertionError,
                    "Expected %d items after flush, got %zu",
                    _PyGC_LOCAL_BUFFER_SIZE / 2, buf.count);
        return NULL;
    }

    size_t deque_size = _PyWSDeque_Size(&deque);
    if (deque_size != _PyGC_LOCAL_BUFFER_SIZE / 2) {
        Py_DECREF(obj);
        _PyWSDeque_Fini(&deque);
        PyErr_Format(PyExc_AssertionError,
                    "Expected %d items in deque after flush, got %zu",
                    _PyGC_LOCAL_BUFFER_SIZE / 2, deque_size);
        return NULL;
    }

    while (!_PyGCLocalBuffer_IsEmpty(&buf)) {
        Py_DECREF(_PyGCLocalBuffer_Pop(&buf));
    }
    PyObject *r;
    while ((r = _PyWSDeque_Take(&deque)) != NULL) {
        Py_DECREF(r);
    }

    Py_DECREF(obj);
    _PyWSDeque_Fini(&deque);
    Py_RETURN_NONE;
}

static PyObject *
test_overflow_flush_normal(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    _PyGCLocalBuffer buf;
    _PyGCLocalBuffer_Init(&buf);
    _PyWSDeque deque;
    _PyWSDeque_Init(&deque);

    PyObject *obj = PyLong_FromLong(42);
    if (obj == NULL) {
        _PyWSDeque_Fini(&deque);
        return NULL;
    }

    int total_pushed = 0;
    for (int round = 0; round < 3; round++) {
        while (!_PyGCLocalBuffer_IsFull(&buf)) {
            Py_INCREF(obj);
            _PyGCLocalBuffer_Push(&buf, obj);
            total_pushed++;
        }
        _PyGC_OverflowFlush(&buf, &deque);
    }

    int counted = (int)buf.count;
    PyObject *r;
    while ((r = _PyWSDeque_Take(&deque)) != NULL) {
        Py_DECREF(r);
        counted++;
    }
    while (!_PyGCLocalBuffer_IsEmpty(&buf)) {
        Py_DECREF(_PyGCLocalBuffer_Pop(&buf));
    }

    if (counted != total_pushed) {
        Py_DECREF(obj);
        _PyWSDeque_Fini(&deque);
        PyErr_Format(PyExc_AssertionError,
                    "Conservation violated: pushed %d, found %d",
                    total_pushed, counted);
        return NULL;
    }

    Py_DECREF(obj);
    _PyWSDeque_Fini(&deque);
    Py_RETURN_NONE;
}

// --- Deque Invariant Tests ---

static PyObject *
test_deque_init_values(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    _PyWSDeque deque;
    _PyWSDeque_Init(&deque);

    size_t top = _Py_atomic_load_ssize_relaxed((Py_ssize_t *)&deque.top);
    size_t bot = _Py_atomic_load_ssize_relaxed((Py_ssize_t *)&deque.bot);

    if (top != 1 || bot != 1) {
        _PyWSDeque_Fini(&deque);
        PyErr_Format(PyExc_AssertionError,
                    "Expected top=1, bot=1, got top=%zu, bot=%zu", top, bot);
        return NULL;
    }

    _PyWSDeque_Fini(&deque);
    Py_RETURN_NONE;
}

static PyObject *
test_deque_top_leq_bot(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    _PyWSDeque deque;
    _PyWSDeque_Init(&deque);

    PyObject *obj = PyLong_FromLong(42);
    if (obj == NULL) {
        _PyWSDeque_Fini(&deque);
        return NULL;
    }

    for (int i = 0; i < 100; i++) {
        Py_INCREF(obj);
        _PyWSDeque_Push(&deque, obj);

        size_t top = _Py_atomic_load_ssize_relaxed((Py_ssize_t *)&deque.top);
        size_t bot = _Py_atomic_load_ssize_relaxed((Py_ssize_t *)&deque.bot);
        if (top > bot) {
            Py_DECREF(obj);
            _PyWSDeque_Fini(&deque);
            PyErr_Format(PyExc_AssertionError,
                        "top > bot after push %d: top=%zu, bot=%zu", i, top, bot);
            return NULL;
        }
    }

    for (int i = 0; i < 100; i++) {
        PyObject *r = _PyWSDeque_Take(&deque);
        Py_DECREF(r);

        size_t top = _Py_atomic_load_ssize_relaxed((Py_ssize_t *)&deque.top);
        size_t bot = _Py_atomic_load_ssize_relaxed((Py_ssize_t *)&deque.bot);
        if (top > bot) {
            Py_DECREF(obj);
            _PyWSDeque_Fini(&deque);
            PyErr_Format(PyExc_AssertionError,
                        "top > bot after take %d: top=%zu, bot=%zu", i, top, bot);
            return NULL;
        }
    }

    PyObject *r = _PyWSDeque_Take(&deque);
    if (r != NULL) {
        Py_DECREF(r);
    }
    size_t top = _Py_atomic_load_ssize_relaxed((Py_ssize_t *)&deque.top);
    size_t bot = _Py_atomic_load_ssize_relaxed((Py_ssize_t *)&deque.bot);
    if (top > bot) {
        Py_DECREF(obj);
        _PyWSDeque_Fini(&deque);
        PyErr_Format(PyExc_AssertionError,
                    "top > bot after empty take: top=%zu, bot=%zu", top, bot);
        return NULL;
    }

    Py_DECREF(obj);
    _PyWSDeque_Fini(&deque);
    Py_RETURN_NONE;
}

static PyObject *
test_deque_grow_chain_fini(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    _PyWSDeque deque;
    _PyWSDeque_Init(&deque);

    PyObject *obj = PyLong_FromLong(42);
    if (obj == NULL) {
        _PyWSDeque_Fini(&deque);
        return NULL;
    }

    const int count = 10000;
    for (int i = 0; i < count; i++) {
        Py_INCREF(obj);
        _PyWSDeque_Push(&deque, obj);
    }

    int resizes = _PyWSDeque_GetNumResizes(&deque);
    if (resizes < 1) {
        Py_DECREF(obj);
        _PyWSDeque_Fini(&deque);
        PyErr_SetString(PyExc_AssertionError, "Expected at least 1 resize");
        return NULL;
    }

    for (int i = 0; i < count; i++) {
        Py_DECREF(_PyWSDeque_Take(&deque));
    }

    Py_DECREF(obj);
    _PyWSDeque_Fini(&deque);
    Py_RETURN_NONE;
}

// --- T1 Infrastructure Tests (SplitVector, WorkQueue, Semaphore) ---

#ifdef Py_PARALLEL_GC

static PyObject *
test_splitvector_init_push(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    _PyGCSplitVector vec;
    if (_PyGCSplitVector_Init(&vec) < 0) {
        return PyErr_NoMemory();
    }

    if (vec.entries == NULL || vec.count != 0) {
        _PyGCSplitVector_Fini(&vec);
        PyErr_SetString(PyExc_AssertionError, "Init should set entries!=NULL, count=0");
        return NULL;
    }

    size_t initial_cap = vec.capacity;
    for (size_t i = 0; i < initial_cap + 10; i++) {
        if (_PyGCSplitVector_Push(&vec, (PyGC_Head *)(uintptr_t)(i + 1)) < 0) {
            _PyGCSplitVector_Fini(&vec);
            return PyErr_NoMemory();
        }
    }

    if (vec.count != initial_cap + 10) {
        _PyGCSplitVector_Fini(&vec);
        PyErr_Format(PyExc_AssertionError,
                    "Expected count=%zu, got %zu", initial_cap + 10, vec.count);
        return NULL;
    }
    if (vec.capacity <= initial_cap) {
        _PyGCSplitVector_Fini(&vec);
        PyErr_SetString(PyExc_AssertionError, "Capacity should have grown");
        return NULL;
    }

    _PyGCSplitVector_Clear(&vec);
    if (vec.count != 0) {
        _PyGCSplitVector_Fini(&vec);
        PyErr_SetString(PyExc_AssertionError, "Clear should set count=0");
        return NULL;
    }

    _PyGCSplitVector_Fini(&vec);
    Py_RETURN_NONE;
}

static PyObject *
test_workqueue_init_push(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    _PyGCWorkQueue queue;
    if (_PyGCWorkQueue_Init(&queue) < 0) {
        return PyErr_NoMemory();
    }

    const int count = 100;
    PyObject *obj = PyLong_FromLong(42);
    if (obj == NULL) {
        _PyGCWorkQueue_Fini(&queue);
        return NULL;
    }

    for (int i = 0; i < count; i++) {
        Py_INCREF(obj);
        if (_PyGCWorkQueue_Push(&queue, obj) < 0) {
            Py_DECREF(obj);
            _PyGCWorkQueue_Fini(&queue);
            return PyErr_NoMemory();
        }
    }

    Py_ssize_t write_idx = _Py_atomic_load_ssize_relaxed(&queue.write_index);
    if (write_idx != count) {
        Py_DECREF(obj);
        _PyGCWorkQueue_Fini(&queue);
        PyErr_Format(PyExc_AssertionError,
                    "Expected write_index=%d, got %zd", count, write_idx);
        return NULL;
    }

    Py_ssize_t read_idx = _Py_atomic_load_ssize_relaxed(&queue.read_index);
    if (read_idx > write_idx) {
        Py_DECREF(obj);
        _PyGCWorkQueue_Fini(&queue);
        PyErr_SetString(PyExc_AssertionError, "read_index > write_index");
        return NULL;
    }

    for (int i = 0; i < count; i++) {
        Py_DECREF(obj);
    }
    Py_DECREF(obj);
    _PyGCWorkQueue_Fini(&queue);
    Py_RETURN_NONE;
}

static PyObject *
test_semaphore_post_wait(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    _PyGCSemaphore sema;
    if (_PyGCSemaphore_Init(&sema) < 0) {
        PyErr_SetString(PyExc_RuntimeError, "Semaphore init failed");
        return NULL;
    }

    _PyGCSemaphore_Post(&sema, 3);

    for (int i = 0; i < 3; i++) {
        _PyGCSemaphore_Wait(&sema);
    }

    if (sema.tokens != 0) {
        _PyGCSemaphore_Fini(&sema);
        PyErr_Format(PyExc_AssertionError,
                    "Expected 0 tokens, got %zd", sema.tokens);
        return NULL;
    }

    _PyGCSemaphore_Fini(&sema);
    Py_RETURN_NONE;
}

static PyObject *
test_semaphore_post_multiple(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    _PyGCSemaphore sema;
    if (_PyGCSemaphore_Init(&sema) < 0) {
        PyErr_SetString(PyExc_RuntimeError, "Semaphore init failed");
        return NULL;
    }

    _PyGCSemaphore_Post(&sema, 5);
    _PyGCSemaphore_Post(&sema, 3);

    if (sema.tokens != 8) {
        _PyGCSemaphore_Fini(&sema);
        PyErr_Format(PyExc_AssertionError,
                    "Expected 8 tokens, got %zd", sema.tokens);
        return NULL;
    }

    for (int i = 0; i < 8; i++) {
        _PyGCSemaphore_Wait(&sema);
    }

    _PyGCSemaphore_Fini(&sema);
    Py_RETURN_NONE;
}

typedef struct {
    _PyGCSemaphore *sema;
    int count;
    int received;
} sema_worker_args;

static void *
sema_consumer(void *arg)
{
    sema_worker_args *args = (sema_worker_args *)arg;
    args->received = 0;
    for (int i = 0; i < args->count; i++) {
        _PyGCSemaphore_Wait(args->sema);
        args->received++;
    }
    return NULL;
}

static PyObject *
test_semaphore_concurrent(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    _PyGCSemaphore sema;
    if (_PyGCSemaphore_Init(&sema) < 0) {
        PyErr_SetString(PyExc_RuntimeError, "Semaphore init failed");
        return NULL;
    }

    const int count = 100;
    sema_worker_args args = { .sema = &sema, .count = count, .received = 0 };

    pthread_t consumer_thread;
    pthread_create(&consumer_thread, NULL, sema_consumer, &args);

    for (int i = 0; i < count; i++) {
        _PyGCSemaphore_Post(&sema, 1);
    }

    pthread_join(consumer_thread, NULL);

    if (args.received != count) {
        _PyGCSemaphore_Fini(&sema);
        PyErr_Format(PyExc_AssertionError,
                    "Expected %d received, got %d", count, args.received);
        return NULL;
    }

    _PyGCSemaphore_Fini(&sema);
    Py_RETURN_NONE;
}

static PyObject *
test_workqueue_reset(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    _PyGCWorkQueue queue;
    if (_PyGCWorkQueue_Init(&queue) < 0) {
        return PyErr_NoMemory();
    }

    PyObject *obj = PyLong_FromLong(1);
    if (obj == NULL) {
        _PyGCWorkQueue_Fini(&queue);
        return NULL;
    }

    for (int i = 0; i < 10; i++) {
        Py_INCREF(obj);
        _PyGCWorkQueue_Push(&queue, obj);
    }

    _PyGCWorkQueue_Reset(&queue);

    Py_ssize_t write_idx = _Py_atomic_load_ssize_relaxed(&queue.write_index);
    Py_ssize_t read_idx = _Py_atomic_load_ssize_relaxed(&queue.read_index);
    int done = _Py_atomic_load_int_relaxed(&queue.producer_done);

    if (write_idx != 0 || read_idx != 0 || done != 0) {
        Py_DECREF(obj);
        _PyGCWorkQueue_Fini(&queue);
        PyErr_SetString(PyExc_AssertionError,
                       "Reset should zero indices and producer_done");
        return NULL;
    }

    for (int i = 0; i < 10; i++) {
        Py_DECREF(obj);
    }
    Py_DECREF(obj);
    _PyGCWorkQueue_Fini(&queue);
    Py_RETURN_NONE;
}

#endif  /* Py_PARALLEL_GC */

// ============================================================================
// End of Parallel GC Infrastructure Tests
// ============================================================================


static PyMethodDef module_functions[] = {
    {"get_configs", get_configs, METH_NOARGS},
    {"get_recursion_depth", get_recursion_depth, METH_NOARGS},
    {"test_bswap", test_bswap, METH_NOARGS},
    {"test_popcount", test_popcount, METH_NOARGS},
    {"test_bit_length", test_bit_length, METH_NOARGS},
    {"test_hashtable", test_hashtable, METH_NOARGS},
    {"get_config", test_get_config, METH_NOARGS},
    {"set_config", test_set_config, METH_O},
    {"reset_path_config", test_reset_path_config, METH_NOARGS},
    {"test_atomic_funcs", test_atomic_funcs, METH_NOARGS},
    {"test_edit_cost", test_edit_cost, METH_NOARGS},
    {"test_bytes_find", test_bytes_find, METH_NOARGS},
    {"normalize_path", normalize_path, METH_O, NULL},
    {"get_getpath_codeobject", get_getpath_codeobject, METH_NOARGS, NULL},
    {"EncodeLocaleEx", encode_locale_ex, METH_VARARGS},
    {"DecodeLocaleEx", decode_locale_ex, METH_VARARGS},
    {"set_eval_frame_default", set_eval_frame_default, METH_NOARGS, NULL},
    {"set_eval_frame_record", set_eval_frame_record, METH_O, NULL},
    _TESTINTERNALCAPI_COMPILER_CODEGEN_METHODDEF
    _TESTINTERNALCAPI_OPTIMIZE_CFG_METHODDEF
    _TESTINTERNALCAPI_ASSEMBLE_CODE_OBJECT_METHODDEF
    {"get_interp_settings", get_interp_settings, METH_VARARGS, NULL},
    {"clear_extension", clear_extension, METH_VARARGS, NULL},
    {"write_perf_map_entry", write_perf_map_entry, METH_VARARGS},
    {"perf_map_state_teardown", perf_map_state_teardown, METH_NOARGS},
    {"iframe_getcode", iframe_getcode, METH_O, NULL},
    {"iframe_getline", iframe_getline, METH_O, NULL},
    {"iframe_getlasti", iframe_getlasti, METH_O, NULL},
    {"pending_threadfunc", _PyCFunction_CAST(pending_threadfunc),
     METH_VARARGS | METH_KEYWORDS},
    {"pending_identify", pending_identify, METH_VARARGS, NULL},
    GH_119213_GETARGS_METHODDEF
    // Parallel GC infrastructure tests
    {"test_ws_deque_init_fini", test_ws_deque_init_fini, METH_NOARGS, NULL},
    {"test_ws_deque_push_take_single", test_ws_deque_push_take_single, METH_NOARGS, NULL},
    {"test_ws_deque_push_steal_single", test_ws_deque_push_steal_single, METH_NOARGS, NULL},
    {"test_ws_deque_lifo_order", test_ws_deque_lifo_order, METH_NOARGS, NULL},
    {"test_ws_deque_fifo_order", test_ws_deque_fifo_order, METH_NOARGS, NULL},
    {"test_ws_deque_take_empty", test_ws_deque_take_empty, METH_NOARGS, NULL},
    {"test_ws_deque_steal_empty", test_ws_deque_steal_empty, METH_NOARGS, NULL},
    {"test_ws_deque_resize", test_ws_deque_resize, METH_NOARGS, NULL},
    {"test_ws_deque_init_with_undersized_buffer", test_ws_deque_init_with_undersized_buffer, METH_NOARGS, NULL},
    {"test_ws_deque_init_with_exact_buffer", test_ws_deque_init_with_exact_buffer, METH_NOARGS, NULL},
    {"test_ws_deque_concurrent_push_steal", test_ws_deque_concurrent_push_steal, METH_NOARGS, NULL},
    {"test_barrier_capacity_zero", test_barrier_capacity_zero, METH_NOARGS, NULL},
    {"test_barrier_basic", test_barrier_basic, METH_NOARGS, NULL},
    {"test_barrier_multiple_rounds", test_barrier_multiple_rounds, METH_NOARGS, NULL},
    {"test_barrier_epoch_distinguishes", test_barrier_epoch_distinguishes, METH_NOARGS, NULL},
    {"test_barrier_postcondition", test_barrier_postcondition, METH_NOARGS, NULL},
    {"test_localbuffer_push_pop", test_localbuffer_push_pop, METH_NOARGS, NULL},
    {"test_localbuffer_push_full", test_localbuffer_push_full, METH_NOARGS, NULL},
    {"test_localbuffer_pop_empty", test_localbuffer_pop_empty, METH_NOARGS, NULL},
    {"test_overflow_flush_precondition", test_overflow_flush_precondition, METH_NOARGS, NULL},
    {"test_overflow_flush_normal", test_overflow_flush_normal, METH_NOARGS, NULL},
    {"test_deque_init_values", test_deque_init_values, METH_NOARGS, NULL},
    {"test_deque_top_leq_bot", test_deque_top_leq_bot, METH_NOARGS, NULL},
    {"test_deque_grow_chain_fini", test_deque_grow_chain_fini, METH_NOARGS, NULL},
#ifdef Py_PARALLEL_GC
    {"test_splitvector_init_push", test_splitvector_init_push, METH_NOARGS, NULL},
    {"test_workqueue_init_push", test_workqueue_init_push, METH_NOARGS, NULL},
    {"test_workqueue_reset", test_workqueue_reset, METH_NOARGS, NULL},
    {"test_semaphore_post_wait", test_semaphore_post_wait, METH_NOARGS, NULL},
    {"test_semaphore_post_multiple", test_semaphore_post_multiple, METH_NOARGS, NULL},
    {"test_semaphore_concurrent", test_semaphore_concurrent, METH_NOARGS, NULL},
#endif
    {NULL, NULL} /* sentinel */
};


/* initialization function */

static int
module_exec(PyObject *module)
{
    if (_PyModule_Add(module, "SIZEOF_PYGC_HEAD",
                        PyLong_FromSsize_t(sizeof(PyGC_Head))) < 0) {
        return 1;
    }

    return 0;
}

static struct PyModuleDef_Slot module_slots[] = {
    {Py_mod_exec, module_exec},
    {Py_mod_multiple_interpreters, Py_MOD_PER_INTERPRETER_GIL_SUPPORTED},
    {0, NULL},
};

static int
module_traverse(PyObject *module, visitproc visit, void *arg)
{
    module_state *state = get_module_state(module);
    assert(state != NULL);
    traverse_module_state(state, visit, arg);
    return 0;
}

static int
module_clear(PyObject *module)
{
    module_state *state = get_module_state(module);
    assert(state != NULL);
    (void)clear_module_state(state);
    return 0;
}

static void
module_free(void *module)
{
    module_state *state = get_module_state(module);
    assert(state != NULL);
    (void)clear_module_state(state);
}

static struct PyModuleDef _testcapimodule = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = MODULE_NAME,
    .m_doc = NULL,
    .m_size = sizeof(module_state),
    .m_methods = module_functions,
    .m_slots = module_slots,
    .m_traverse = module_traverse,
    .m_clear = module_clear,
    .m_free = (freefunc)module_free,
};


PyMODINIT_FUNC
PyInit__testinternalcapi(void)
{
    return PyModuleDef_Init(&_testcapimodule);
}
