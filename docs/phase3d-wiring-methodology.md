# Phase 3D: C++ → C Wiring Methodology

## Overview

Proven methodology for replacing C++ implementations with C API calls.
Validated on hir::Type operators (2026-04-14). Two bugs found and fixed
via this process before any incorrect code was pushed.

## Pattern: Write → Assert → Wire → Delete

### Step 1: Write C Implementation (Additive)
- Create pure C functions in `_c.h`/`_c.c` files
- Gate with triple gate (x86_64 + ARM64 + benchmark)
- C and C++ coexist — zero behavioral change

### Step 2: Add Conversion Helpers
For types that cross the C/C++ boundary, add field-by-field conversion
helpers as member functions of the C++ class:

```cpp
HirType toHirType() const {
    HirType h;
    h.bits_and_flags = (uint64_t(bits_)
        | (uint64_t(lifetime_) << HIR_TYPE_LIFETIME_SHIFT)
        | (uint64_t(spec_kind_) << HIR_TYPE_SPEC_SHIFT));
    h.pytype = spec_.pytype;
    return h;
}
static Type fromHirType(HirType h) {
    Type r(Type::bits_t{0}, Type::bits_t{0});
    r.bits_ = h.bits_and_flags & HIR_TYPE_BITS_MASK;
    r.lifetime_ = (h.bits_and_flags >> HIR_TYPE_LIFETIME_SHIFT) & 0x3;
    r.spec_kind_ = (h.bits_and_flags >> HIR_TYPE_SPEC_SHIFT) & 0x7;
    r.spec_.pytype = h.pytype;
    return r;
}
```

**Critical rules:**
- Use field-by-field extraction, NOT memcpy (C++ bitfield layout is
  implementation-defined and may differ from C shift/mask constants)
- Use `Type(Type::bits_t{0}, Type::bits_t{0})`, NOT `Type{0,0}` (ambiguous)

### Step 3: Assertion Wrappers (ONE function at a time)
Before deleting C++ body, add runtime comparison:

```cpp
Type Type::operator&(Type other) const {
    Type cpp_result = /* original C++ body */;
    Type c_result = fromHirType(
        hir_type_intersect(toHirType(), other.toHirType()));
    JIT_CHECK(cpp_result == c_result, "intersect mismatch");
    return cpp_result;
}
```

Run 972/972+ Phoenix tests. First JIT_CHECK failure reveals exact
diverging inputs. Fix the C implementation before proceeding.

### Step 4: Wire and Delete
Once assertions pass with zero mismatches:

```cpp
Type Type::operator&(Type other) const {
    return fromHirType(
        hir_type_intersect(toHirType(), other.toHirType()));
}
```

Gate with full triple gate (x86_64 + ARM64 + benchmark).

## Ceremony Classification

Not all files need the same level of rigor.

### HIGH Ceremony (assertion wrappers required)
Semantic-heavy files where C/C++ divergence causes wrong codegen:
- type.cpp — type system operators (PROVEN need: 2 bugs found)
- hir.cpp — instruction type checking
- simplify.cpp — algebraic optimizations
- refcount_insertion.cpp — ownership semantics
- builder.cpp — bytecode→HIR, type inference
- gen_asm.cpp — codegen, instruction encoding
- lir/generator.cpp — HIR→LIR lowering

### LOW Ceremony (triple gate sufficient, no assertion wrappers)
Mechanical conversions with no semantic risk:
- code_patcher.cpp — byte-level ops
- perf_jitdump.cpp — perf profiling
- symbolizer.cpp — symbol lookup
- debug_info.cpp — debug metadata
- jit_time_log.cpp — timing diagnostics
- frame_shadow.cpp — frame tracking
- Bridge files (lir_c_api.cpp, hir_c_api.cpp) — die when callers convert

### MEDIUM Ceremony (limited assertion coverage)
Some semantic content but not type-system critical:
- HIR passes (analysis, ssa, cfg, etc.)
- LIR passes (regalloc, parser, etc.)

## Lessons Learned

### kSpecObject Bugs (2026-04-14)
Two bugs found during hir::Type wiring, both in the same area:

1. **Predicate completeness:** `hir_type_has_type_spec()` must include
   `kSpecObject` (object-specialized types ARE type-specialized). Missing
   this caused `specSubtype` to return false for object specs → wrong type
   analysis → wrong codegen → SIGILL (ud2 deopt guard).

2. **Union member confusion:** HirType spec union stores different types
   per spec_kind:
   - SPEC_TYPE / SPEC_TYPE_EXACT → pytype (PyTypeObject*)
   - SPEC_OBJECT → pyobject (PyObject*) — need Py_TYPE(pyobject)
   - SPEC_INT → int_val
   - SPEC_DOUBLE → double_val
   
   Passing pyobject to PyType_IsSubtype (which expects PyTypeObject*)
   caused SEGFAULT.

3. **Unit test gap:** 17 behavioral tests all passed but didn't cover
   object-specialized cross-kind comparisons. Added 3 kSpecObject
   regression tests to prevent recurrence.

4. **Registration map values:** Hand-computed bit constants diverge from
   C++ HIR_TYPES macro. Always extract exact bit patterns at runtime from
   C++ and paste into C tables. Never hand-compute.

### Process Lessons
- Batch wiring (multiple functions at once) crashed twice. Incremental
  wiring (one function at a time with assertion wrapper) succeeded.
- The assertion wrapper approach found bugs in under 5 minutes of runtime
  that would have taken hours to find with GDB alone.
- GDB showed WHERE the crash happened (ud2 in JIT code), assertions
  showed WHY (which Type values diverged).
