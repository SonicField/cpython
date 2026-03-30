/* Phoenix: CPython with JIT */

#include "Python.h"

/* Forward declaration of _cinderx module init */
extern PyObject* PyInit__cinderx(void);

#ifdef MS_WINDOWS
int
wmain(int argc, wchar_t **argv)
{
    /* Register _cinderx as a builtin module before Py_Initialize */
    if (PyImport_AppendInittab("_cinderx", PyInit__cinderx) == -1) {
        fprintf(stderr, "Phoenix: failed to register _cinderx module\n");
        return 1;
    }
    return Py_Main(argc, argv);
}
#else
int
main(int argc, char **argv)
{
    /* Register _cinderx as a builtin module before Py_Initialize */
    if (PyImport_AppendInittab("_cinderx", PyInit__cinderx) == -1) {
        fprintf(stderr, "Phoenix: failed to register _cinderx module\n");
        return 1;
    }
    return Py_BytesMain(argc, argv);
}
#endif
