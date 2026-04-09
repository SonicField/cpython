/* test_hir_type.c — C unit tests for hir::Type operator implementations
 *
 * Tests hir_type_intersect, hir_type_subtract, hir_type_union against
 * theologian's 17 behavioral invariants. Same pattern as test_arm64_corpus.c.
 *
 * Build:
 *   Requires linking with hir_type_c.c and the bridge constants.
 *   Built as part of the JIT test suite via CMake.
 *
 * Note: This file inlines the type bit constants from type_generated.h
 * because it's compiled as C (not C++).
 */

#include <stdio.h>
#include <string.h>
#include <stdint.h>

/* Include the C type header */
#include "hir_type_c.h"

static int g_pass = 0, g_fail = 0;

/* ---- Type bit constants (from type_generated.h) ---- */
/* These MUST match the generated values. Verified by static_asserts in bridge. */

#define K_BOTTOM    0x0UL
#define K_OBJECT    0x800ffffffffUL
#define K_PRIMITIVE 0xfff00000000UL
#define K_LONG      0x00000200402UL
#define K_LIST      0x00008010000UL
#define K_BOOL      0x00000000001UL

/* Provide the extern constants that hir_type_c.c needs */
const uint64_t _hir_type_kObject = K_OBJECT;
const uint64_t _hir_type_kPrimitive = K_PRIMITIVE;

/* Stub for builtin type check — for unit tests, treat all types as non-builtin */
int _hir_type_is_builtin_pytype(PyTypeObject *type) {
    (void)type;
    return 0;  /* conservative: no builtin types in unit test context */
}

/* Stub PyType_IsSubtype — not reached in unit tests (no real PyTypeObject*) */
int PyType_IsSubtype(PyTypeObject *a, PyTypeObject *b) {
    (void)a; (void)b;
    return 0;
}

/* ---- Test helpers ---- */

static HirType make_type(uint64_t bits, uint64_t lifetime, uint64_t spec_kind,
                          intptr_t spec) {
    HirType t;
    t.bits_and_flags = (bits & HIR_TYPE_BITS_MASK)
                     | ((lifetime & 0x3) << HIR_TYPE_LIFETIME_SHIFT)
                     | ((spec_kind & 0x7) << HIR_TYPE_SPEC_SHIFT);
    t.int_val = spec;
    return t;
}

static HirType make_simple(uint64_t bits, uint64_t lifetime) {
    return make_type(bits, lifetime, HIR_SPEC_TOP, 0);
}

static HirType make_bottom(void) {
    return make_type(0, 0, HIR_SPEC_BOTTOM, 0);
}

static int is_bottom(HirType t) {
    return hir_type_bits(&t) == 0;
}

static void check(const char *name, int condition) {
    if (condition) {
        printf("  PASS  %s\n", name);
        g_pass++;
    } else {
        printf("  FAIL  %s\n", name);
        g_fail++;
    }
}

/* ---- INTERSECT tests (6 cases) ---- */

static void test_intersect(void) {
    printf("\n=== Intersect (operator&) ===\n");

    /* 1. bits = a.bits & b.bits, lifetime = a.lifetime & b.lifetime */
    {
        HirType a = make_simple(K_LONG | K_LIST, HIR_LIFETIME_TOP);
        HirType b = make_simple(K_LONG, HIR_LIFETIME_MORTAL);
        HirType r = hir_type_intersect(a, b);
        check("bits AND + lifetime AND",
              hir_type_bits(&r) == K_LONG &&
              hir_type_lifetime(&r) == HIR_LIFETIME_MORTAL);
    }

    /* 2. kObject==0 → lifetime cleared */
    {
        HirType a = make_simple(K_LONG, HIR_LIFETIME_MORTAL);
        HirType b = make_simple(K_PRIMITIVE, HIR_LIFETIME_TOP);
        HirType r = hir_type_intersect(a, b);
        /* K_LONG & K_PRIMITIVE = 0 (no overlap) → bottom */
        check("kObject zero → bottom", is_bottom(r));
    }

    /* 3. lifetime==0 → kObject bits cleared */
    {
        HirType a = make_simple(K_LONG, HIR_LIFETIME_MORTAL);
        HirType b = make_simple(K_LONG, HIR_LIFETIME_IMMORTAL);
        HirType r = hir_type_intersect(a, b);
        /* lifetime & = mortal & immortal = 0 → clear kObject bits → bottom */
        check("lifetime zero → kObject cleared → bottom", is_bottom(r));
    }

    /* 4. Bottom result when bits == 0 */
    {
        HirType a = make_simple(K_LONG, HIR_LIFETIME_MORTAL);
        HirType b = make_simple(K_LIST, HIR_LIFETIME_MORTAL);
        HirType r = hir_type_intersect(a, b);
        check("disjoint bits → bottom", is_bottom(r));
    }

    /* 5. Spec subtype: more-specific wins */
    {
        HirType a = make_simple(K_LONG, HIR_LIFETIME_TOP);
        HirType b = make_simple(K_LONG, HIR_LIFETIME_TOP);
        HirType r = hir_type_intersect(a, b);
        check("same type → preserves bits",
              hir_type_bits(&r) == K_LONG &&
              hir_type_lifetime(&r) == HIR_LIFETIME_TOP);
    }

    /* 6. Self-intersection is identity */
    {
        HirType a = make_simple(K_LONG | K_LIST, HIR_LIFETIME_MORTAL);
        HirType r = hir_type_intersect(a, a);
        check("self-intersect → identity",
              hir_type_bits(&r) == (K_LONG | K_LIST) &&
              hir_type_lifetime(&r) == HIR_LIFETIME_MORTAL);
    }
}

/* ---- SUBTRACT tests (5 cases) ---- */

static void test_subtract(void) {
    printf("\n=== Subtract (operator-) ===\n");

    /* 1. If A <= B → Bottom */
    {
        HirType a = make_simple(K_LONG, HIR_LIFETIME_MORTAL);
        HirType b = make_simple(K_LONG | K_LIST, HIR_LIFETIME_TOP);
        HirType r = hir_type_subtract(a, b);
        check("subset → bottom", is_bottom(r));
    }

    /* 2. No specSubtype → return self */
    {
        HirType a = make_type(K_LONG, HIR_LIFETIME_MORTAL, HIR_SPEC_TYPE, 0x1234);
        HirType b = make_type(K_LONG, HIR_LIFETIME_MORTAL, HIR_SPEC_TYPE, 0x5678);
        HirType r = hir_type_subtract(a, b);
        /* Different type specs, neither is subtype of other → return self */
        check("no specSubtype → self",
              hir_type_bits(&r) == K_LONG &&
              r.int_val == 0x1234);
    }

    /* 3. Primitive bits subtracted directly */
    {
        HirType a = make_simple(K_LONG | K_LIST | K_PRIMITIVE, HIR_LIFETIME_TOP);
        HirType b = make_simple(K_PRIMITIVE, 0);
        HirType r = hir_type_subtract(a, b);
        check("primitive bits removed",
              (hir_type_bits(&r) & K_PRIMITIVE) == 0 &&
              (hir_type_bits(&r) & K_LONG) == K_LONG);
    }

    /* 4. Self-subtract → bottom */
    {
        HirType a = make_simple(K_LONG, HIR_LIFETIME_MORTAL);
        HirType r = hir_type_subtract(a, a);
        check("self-subtract → bottom", is_bottom(r));
    }

    /* 5. Subtract bottom → self */
    {
        HirType a = make_simple(K_LONG, HIR_LIFETIME_MORTAL);
        HirType b = make_bottom();
        HirType r = hir_type_subtract(a, b);
        check("subtract bottom → self",
              hir_type_bits(&r) == K_LONG &&
              hir_type_lifetime(&r) == HIR_LIFETIME_MORTAL);
    }
}

/* ---- UNION tests (6 cases) ---- */

static void test_union(void) {
    printf("\n=== Union (operator|) ===\n");

    /* 1. Trivial subset: if A <= B, result is B */
    {
        HirType a = make_simple(K_LONG, HIR_LIFETIME_MORTAL);
        HirType b = make_simple(K_LONG | K_LIST, HIR_LIFETIME_TOP);
        HirType r = hir_type_union(a, b);
        check("subset → supertype",
              hir_type_bits(&r) == (K_LONG | K_LIST) &&
              hir_type_lifetime(&r) == HIR_LIFETIME_TOP);
    }

    /* 2. bits = a.bits | b.bits, lifetime = a.lifetime | b.lifetime */
    {
        HirType a = make_simple(K_LONG, HIR_LIFETIME_MORTAL);
        HirType b = make_simple(K_LIST, HIR_LIFETIME_IMMORTAL);
        HirType r = hir_type_union(a, b);
        check("bits OR + lifetime OR",
              hir_type_bits(&r) == (K_LONG | K_LIST) &&
              hir_type_lifetime(&r) == HIR_LIFETIME_TOP);
    }

    /* 3. No type spec → unspecialized result */
    {
        HirType a = make_simple(K_LONG, HIR_LIFETIME_MORTAL);
        HirType b = make_simple(K_LIST, HIR_LIFETIME_MORTAL);
        HirType r = hir_type_union(a, b);
        check("no type spec → unspecialized",
              hir_type_spec_kind(&r) == HIR_SPEC_TOP);
    }

    /* 4. Self-union is identity */
    {
        HirType a = make_simple(K_LONG, HIR_LIFETIME_MORTAL);
        HirType r = hir_type_union(a, a);
        check("self-union → identity",
              hir_type_bits(&r) == K_LONG &&
              hir_type_lifetime(&r) == HIR_LIFETIME_MORTAL);
    }

    /* 5. Union with bottom → self */
    {
        HirType a = make_simple(K_LONG, HIR_LIFETIME_MORTAL);
        HirType b = make_bottom();
        HirType r = hir_type_union(a, b);
        check("union with bottom → self",
              hir_type_bits(&r) == K_LONG &&
              hir_type_lifetime(&r) == HIR_LIFETIME_MORTAL);
    }

    /* 6. Commutativity */
    {
        HirType a = make_simple(K_LONG, HIR_LIFETIME_MORTAL);
        HirType b = make_simple(K_LIST, HIR_LIFETIME_IMMORTAL);
        HirType r1 = hir_type_union(a, b);
        HirType r2 = hir_type_union(b, a);
        check("union commutativity",
              hir_type_bits(&r1) == hir_type_bits(&r2) &&
              hir_type_lifetime(&r1) == hir_type_lifetime(&r2));
    }
}

/* ---- Main ---- */

int main(void) {
    printf("HIR Type Operator Tests — C Unit Tests\n");
    printf("=======================================\n");

    test_intersect();
    test_subtract();
    test_union();

    printf("\n=======================================\n");
    printf("Results: %d PASS, %d FAIL (out of %d)\n",
           g_pass, g_fail, g_pass + g_fail);
    printf("=======================================\n");

    return g_fail > 0 ? 1 : 0;
}
