# Phoenix JIT pre-loader for CPython test suite.
# Import _cinderx early so the JIT is active during all tests.
try:
    import _cinderx
except ImportError:
    pass
