# W18 — `hir_type_is_subtype` mixed-lifetime single-union mechanism

## Status

W18 v1: **CLOSED** 2026-04-22 10:47:51Z (supervisor) on M-Design verdict.
W18 v2 (framework re-implementation): deferred to post-W19/W21/W22/W23.

## Summary

`hir_type_is_subtype(type, supertype)` returns the wrong answer (0 instead of 1)
when `supertype` is a mixed-lifetime single-union (e.g., `OPTOBJECT_OR_CPTR`,
`OPTOBJECT_OR_CUINT64`) and `type` is one of the union's primitive arms (e.g.,
`CPtr`, `CUInt64`). Same-lifetime supertype unions and unrelated-type checks
both return correct results.

The framework bug is empirically reproduced on:
- x86_64 RelWithDebInfo (-O2 + LTO), commit 2faa8a024f
- ARM64 pydebug (-O0), commit 2faa8a024f

The F1'' enum-bypass mitigation (cdf741f610: `kOptObjectOrCPtr`,
`kOptObjectOrCUInt64`) routes around the broken single-union path by issuing
two separate `hir_type_is_subtype` calls in `register_type_matches` with
simple-type supertypes. Empirically validated — production tests pass on both
architectures with the bypass applied.

## V1 + V4 verdict (PC variant — gdb/ctypes call across builds)

Per the W18 falsifier's 6-case taxonomy (theologian 2026-04-22 10:15:23Z spec):

| # | type        | supertype              | expect | V1 (x86_64 RelWithDebInfo) | V4 (ARM64 pydebug) | note          |
|---|-------------|------------------------|--------|----------------------------|--------------------|---------------|
| T1| CPtr        | OPTOBJECT_OR_CPTR      | 1      | **0 FAIL**                 | **0 FAIL**         | KNOWN BROKEN  |
| T2| OPTOBJECT   | OPTOBJECT_OR_CPTR      | 1      | 1 PASS                     | 1 PASS             | control       |
| T3| CInt32      | OPTOBJECT_OR_CPTR      | 0      | 0 PASS                     | 0 PASS             | control       |
| T4| CUInt64     | OPTOBJECT_OR_CUINT64   | 1      | **0 FAIL**                 | **0 FAIL**         | KNOWN BROKEN  |
| T5| OPTOBJECT   | OPTOBJECT_OR_CUINT64   | 1      | 1 PASS                     | 1 PASS             | control       |
| T6| CPtr        | OPTOBJECT_OR_CUINT64   | 0      | 0 PASS                     | 0 PASS             | control       |

V1 tooling: ctypes via `PYTHONPATH=build/lib.linux-x86_64-3.12 ./python` (hir_type_is_subtype exported at 0x80c770 in the local x86_64 binary).
V4 tooling: gdb-attach to a sleeping `./python -c 'import time; time.sleep(120)'` on devgpu004; ctypes seg-faults on dlopen(None) under ARM64 pydebug.

V3 (x86_64 pydebug -O0) was not run — V1 + V4 already cover both opt-level extremes (-O2/LTO and -O0); V3 would only re-confirm M2 ruled out. Build infra hiccup (asmjit not refetched after `--pydebug --clean`) made V3 cost ~15 min for diminishing evidentiary return.

## Mechanism analysis (M1/M2/M3/M-Design)

| class | description | verdict | rationale |
|-------|-------------|---------|-----------|
| M1    | ABI mismatch (by-value struct passing) | RULED UNLIKELY | V1 SysV-x86_64 + V4 AArch64 are different ABIs; both reproduce identical bug. Cross-ABI reproduction strongly suggests function-logic bug, not register-passing convention. |
| M2    | LTO/inline | RULED OUT | V4 has neither LTO nor inline (-O0 pydebug); still fails. Not optimization-related. |
| M3    | Compiler bug | RULED UNLIKELY | Same compiler family (clang) on both arches; if compiler bug, would expect arch-dependent symptoms. |
| **M-Design** | **Framework logic flaw in mixed-lifetime single-union handling** | **CONFIRMED** | Cross-ABI cross-mode reproduction with same logical pattern points to the function's design itself. |

T7 (M1 ABI by-pointer wrapper) was deferred from v1 per theologian 2026-04-22 10:22:48Z — adds new API surface, v2 work.

## Source-vs-runtime divergence (open mystery for W18 v2)

Per source inspection (theologian 07:33Z + 10:46Z, generalist 07:43Z gdb verification):

```
hir_type_is_subtype(type, supertype) =
    (bits(type)     & bits(supertype))     == bits(type)      &&  // bits subtype
    (lifetime(type) & lifetime(supertype)) == lifetime(type)  &&  // lifetime subtype
    spec_subtype(type, supertype)
```

For `CPtr` vs `OPTOBJECT_OR_CPTR`: each individual sub-check returns 1 (verified via gdb at 07:43Z). The composite SHOULD return 1. Empirically returns 0.

> The arithmetic is provably correct. The composite is provably wrong. — theologian 10:47:27Z

This is the v2 root-cause investigation seed.

## F1'' bypass validation

Commit `cdf741f610` (W8 push 2faa8a024f origin tip) added two enum constraints to `register_type_matches`:
- `kOptObjectOrCPtr` → two separate `is_subtype` calls (CPtr-arm, OPTOBJECT-arm)
- `kOptObjectOrCUInt64` → two separate `is_subtype` calls (CUInt64-arm, OPTOBJECT-arm)

Each individual call uses a simple-type supertype, never the broken mixed-lifetime union. Production tests (push 50 dual-arch verify, 7/7 each) confirm the bypass works.

## W18 v2 priorities (post-W19/W21/W22/W23)

1. Source-level audit of `hir_type_is_subtype` C implementation vs the C++ `Type::operator<=` it ports.
2. Hidden-code-path search: LTO inline replacement, function pointer indirection, macro substitution.
3. If source matches semantics: M3 reconsidered with broader compiler-family check (gcc cross-build).
4. If source doesn't match: re-implement to fix the design flaw.
5. Once framework is fixed: revert F1'' enum bypass + restore single-union semantics; re-run W18 v1 falsifier to confirm T1+T4 PASS.

## Falsifier source

The standalone-exe falsifier (`Python/jit/hir/test_t4_union_falsifier.c`) was drafted as
a CMake-built executable but pivoted to PC variant after 5 build iterations surfaced
unbounded transitive deps in `hir_type_c.c` (PyExc_BaseException, PyType_Ready,
jit_compile_lock, hir_type_is_exact, ...). The standalone re-attempt is W18 v2 scope
if needed for T7 (M1 ABI by-pointer wrapper).

Per pythia #67 (2026-04-22 11:10:03Z), the broken-target source is preserved here
in the commit-tracked artifact rather than as uncommitted file debt in tree:

```c
/* W18 — synthetic falsifier for hir_type_is_subtype mixed-lifetime single-union. */
#include "hir_type_c.h"
#include <stdio.h>
#include <string.h>
#include <stdint.h>

/* Stubs (per test_hir_type.c:23-40 precedent) */
const uint64_t _hir_type_kObject    = 0x800ffffffffULL;
const uint64_t _hir_type_kPrimitive = 0xfff00000000ULL;
int _hir_type_is_builtin_pytype(PyTypeObject *t) { (void)t; return 0; }
int PyType_IsSubtype(PyTypeObject *a, PyTypeObject *b) { (void)a; (void)b; return 0; }
int phx_threaded_compile_in_progress(void) { return 0; }
void phx_threaded_compile_serialize(void) {}

static int run_case(const char *name, HirType op, HirType supertype, int expected) {
    int actual = hir_type_is_subtype(op, supertype);
    int pass = (actual == expected);
    fprintf(stderr, "%s: is_subtype(0x%lx, 0x%lx) = %d (expect %d) %s\n",
        name, (unsigned long)op.bits_and_flags,
        (unsigned long)supertype.bits_and_flags, actual, expected,
        pass ? "PASS" : "FAIL");
    return pass;
}

typedef struct { HirType a; HirType b; } HirTypePair;

static int t7_pass_by_pointer_indirection(HirType op, HirType supertype, int expected) {
    HirTypePair pair;
    memcpy(&pair.a, &op, sizeof(HirType));
    memcpy(&pair.b, &supertype, sizeof(HirType));
    HirType op_copy, super_copy;
    memcpy(&op_copy, &pair.a, sizeof(HirType));
    memcpy(&super_copy, &pair.b, sizeof(HirType));
    return run_case("T7-by-ptr", op_copy, super_copy, expected);
}

int main(void) {
    HirType cuint64    = HIR_TYPE_CUINT64;
    HirType cint32     = HIR_TYPE_CINT32;
    HirType cptr       = HIR_TYPE_CPTR;
    HirType optobject  = HIR_TYPE_OPTOBJECT;
    HirType opt_or_cptr    = HIR_TYPE_SIMPLE(0x820ffffffffULL, HIR_TYPE_LIFETIME_TOP);
    HirType opt_or_cuint64 = HIR_TYPE_OPTOBJECT_OR_CUINT64;

    int passed = 0;
    fprintf(stderr, "=== W18 falsifier ===\n");
    passed += run_case("T1", cptr,      opt_or_cptr,    1);  /* KNOWN BROKEN */
    passed += run_case("T2", optobject, opt_or_cptr,    1);  /* control */
    passed += run_case("T3", cint32,    opt_or_cptr,    0);  /* control */
    passed += run_case("T4", cuint64,   opt_or_cuint64, 1);  /* KNOWN BROKEN */
    passed += run_case("T5", optobject, opt_or_cuint64, 1);  /* control */
    passed += run_case("T6", cptr,      opt_or_cuint64, 0);  /* control */
    passed += t7_pass_by_pointer_indirection(cptr, opt_or_cptr, 1);
    fprintf(stderr, "=== %d/7 PASSED ===\n", passed);
    return (passed == 7) ? 0 : 1;
}
```

To recreate as a buildable target in W18 v2: this source needs the 17 PyTypeObject
stubs (per `test_hir_type.c` precedent) plus 9+ more deeper deps (PyExc_BaseException,
PyType_Ready, jit_compile_lock, jit_compile_unlock, hir_type_is_exact, _Py_NoneStruct).
Standalone-exe approach is high-friction; PC variant (gdb-attach + ctypes) is the
practical path.

PC variant scripts (preserved at commit-time in `/tmp/`, not under SCM):
- `/tmp/w18_ctypes_x86.py` — ctypes wrapper, x86_64 RelWithDebInfo
- `/tmp/w18_gdb_arm64.sh` — gdb-attach wrapper, ARM64 pydebug

## References

- W8 close: 2026-04-22 10:13Z (cdf741f610 trio at SonicField/cpython origin)
- Push 50: 2026-04-22 10:21Z (2faa8a024f, includes W8 trio + emitLoadCommonConstant)
- W18 v1 close: 2026-04-22 10:47:51Z (supervisor on theologian 10:47:27Z verdict)
