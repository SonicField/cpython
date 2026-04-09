/* Compatibility shim: CPython 3.15-style atomic operations for raw pointers.
 *
 * CPython 3.12's pycore_atomic.h provides atomics on wrapper types
 * (_Py_atomic_address, _Py_atomic_int).  The parallel GC backport from 3.15
 * needs atomic operations on raw fields (uintptr_t, int, Py_ssize_t) such as
 * PyGC_Head._gc_prev.  This header bridges the gap.
 *
 * Two backends:
 *   1. HAVE_STD_ATOMIC  -- C11 <stdatomic.h>
 *   2. HAVE_BUILTIN_ATOMIC -- GCC/Clang __atomic builtins
 *
 * No volatile-only fallback is provided; parallel GC requires real atomics.
 */

#ifndef Py_GC_ATOMIC_COMPAT_H
#define Py_GC_ATOMIC_COMPAT_H

#ifndef Py_BUILD_CORE
#  error "this header requires Py_BUILD_CORE define"
#endif

#include "pyconfig.h"

#include <stdint.h>     /* uintptr_t */
#include <stddef.h>     /* Py_ssize_t via Python.h, but be safe */

#if defined(HAVE_STD_ATOMIC)
#  include <stdatomic.h>
#elif !defined(HAVE_BUILTIN_ATOMIC)
#  error "Parallel GC requires C11 stdatomic.h or GCC __atomic builtins"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ===================================================================
 * uintptr_t atomics
 * =================================================================== */

#if defined(HAVE_STD_ATOMIC)

static inline uintptr_t
_Py_atomic_load_uintptr_relaxed(const uintptr_t *addr)
{
    return atomic_load_explicit((const _Atomic(uintptr_t) *)addr,
                                memory_order_relaxed);
}

static inline void
_Py_atomic_store_uintptr_relaxed(uintptr_t *addr, uintptr_t val)
{
    atomic_store_explicit((_Atomic(uintptr_t) *)addr, val,
                          memory_order_relaxed);
}

static inline uintptr_t
_Py_atomic_load_uintptr_acquire(const uintptr_t *addr)
{
    return atomic_load_explicit((const _Atomic(uintptr_t) *)addr,
                                memory_order_acquire);
}

static inline void
_Py_atomic_store_uintptr_release(uintptr_t *addr, uintptr_t val)
{
    atomic_store_explicit((_Atomic(uintptr_t) *)addr, val,
                          memory_order_release);
}

static inline uintptr_t
_Py_atomic_add_uintptr(uintptr_t *addr, uintptr_t val)
{
    return atomic_fetch_add_explicit((_Atomic(uintptr_t) *)addr, val,
                                    memory_order_seq_cst);
}

static inline uintptr_t
_Py_atomic_and_uintptr(uintptr_t *addr, uintptr_t val)
{
    return atomic_fetch_and_explicit((_Atomic(uintptr_t) *)addr, val,
                                    memory_order_seq_cst);
}

static inline int
_Py_atomic_compare_exchange_uintptr(uintptr_t *addr,
                                    uintptr_t *expected,
                                    uintptr_t desired)
{
    return atomic_compare_exchange_strong_explicit(
        (_Atomic(uintptr_t) *)addr, expected, desired,
        memory_order_seq_cst, memory_order_seq_cst);
}

/* ===================================================================
 * pointer atomics
 * =================================================================== */

static inline void *
_Py_atomic_load_ptr(const void *addr)
{
    return (void *)atomic_load_explicit(
        (const _Atomic(uintptr_t) *)addr, memory_order_seq_cst);
}

static inline void *
_Py_atomic_load_ptr_acquire(const void *addr)
{
    return (void *)atomic_load_explicit(
        (const _Atomic(uintptr_t) *)addr, memory_order_acquire);
}

static inline void
_Py_atomic_store_ptr_relaxed(void *addr, void *val)
{
    atomic_store_explicit((_Atomic(uintptr_t) *)addr, (uintptr_t)val,
                          memory_order_relaxed);
}

static inline void
_Py_atomic_store_ptr_release(void *addr, void *val)
{
    atomic_store_explicit((_Atomic(uintptr_t) *)addr, (uintptr_t)val,
                          memory_order_release);
}

/* ===================================================================
 * int atomics
 * =================================================================== */

static inline int
_Py_atomic_load_int(const int *addr)
{
    return atomic_load_explicit((const _Atomic(int) *)addr,
                                memory_order_seq_cst);
}

static inline void
_Py_atomic_store_int(int *addr, int val)
{
    atomic_store_explicit((_Atomic(int) *)addr, val,
                          memory_order_seq_cst);
}

static inline int
_Py_atomic_load_int_relaxed(const int *addr)
{
    return atomic_load_explicit((const _Atomic(int) *)addr,
                                memory_order_relaxed);
}

static inline void
_Py_atomic_store_int_relaxed(int *addr, int val)
{
    atomic_store_explicit((_Atomic(int) *)addr, val,
                          memory_order_relaxed);
}

static inline int
_Py_atomic_load_int_acquire(const int *addr)
{
    return atomic_load_explicit((const _Atomic(int) *)addr,
                                memory_order_acquire);
}

static inline void
_Py_atomic_store_int_release(int *addr, int val)
{
    atomic_store_explicit((_Atomic(int) *)addr, val,
                          memory_order_release);
}

static inline int
_Py_atomic_add_int(int *addr, int val)
{
    return atomic_fetch_add_explicit((_Atomic(int) *)addr, val,
                                    memory_order_seq_cst);
}

/* ===================================================================
 * Py_ssize_t atomics
 * =================================================================== */

static inline Py_ssize_t
_Py_atomic_load_ssize_relaxed(const Py_ssize_t *addr)
{
    return atomic_load_explicit((const _Atomic(Py_ssize_t) *)addr,
                                memory_order_relaxed);
}

static inline void
_Py_atomic_store_ssize_relaxed(Py_ssize_t *addr, Py_ssize_t val)
{
    atomic_store_explicit((_Atomic(Py_ssize_t) *)addr, val,
                          memory_order_relaxed);
}

static inline Py_ssize_t
_Py_atomic_load_ssize_acquire(const Py_ssize_t *addr)
{
    return atomic_load_explicit((const _Atomic(Py_ssize_t) *)addr,
                                memory_order_acquire);
}

static inline void
_Py_atomic_store_ssize_release(Py_ssize_t *addr, Py_ssize_t val)
{
    atomic_store_explicit((_Atomic(Py_ssize_t) *)addr, val,
                          memory_order_release);
}

static inline int
_Py_atomic_compare_exchange_ssize(Py_ssize_t *addr,
                                  Py_ssize_t *expected,
                                  Py_ssize_t desired)
{
    return atomic_compare_exchange_strong_explicit(
        (_Atomic(Py_ssize_t) *)addr, expected, desired,
        memory_order_seq_cst, memory_order_seq_cst);
}

/* ===================================================================
 * Fences
 * =================================================================== */

static inline void
_Py_atomic_fence_seq_cst(void)
{
    atomic_thread_fence(memory_order_seq_cst);
}

static inline void
_Py_atomic_fence_release(void)
{
    atomic_thread_fence(memory_order_release);
}

static inline void
_Py_atomic_fence_acquire(void)
{
    atomic_thread_fence(memory_order_acquire);
}

#elif defined(HAVE_BUILTIN_ATOMIC)

/* ===================================================================
 * uintptr_t atomics  (GCC __atomic builtins)
 * =================================================================== */

static inline uintptr_t
_Py_atomic_load_uintptr_relaxed(const uintptr_t *addr)
{
    return __atomic_load_n(addr, __ATOMIC_RELAXED);
}

static inline void
_Py_atomic_store_uintptr_relaxed(uintptr_t *addr, uintptr_t val)
{
    __atomic_store_n(addr, val, __ATOMIC_RELAXED);
}

static inline uintptr_t
_Py_atomic_load_uintptr_acquire(const uintptr_t *addr)
{
    return __atomic_load_n(addr, __ATOMIC_ACQUIRE);
}

static inline void
_Py_atomic_store_uintptr_release(uintptr_t *addr, uintptr_t val)
{
    __atomic_store_n(addr, val, __ATOMIC_RELEASE);
}

static inline uintptr_t
_Py_atomic_add_uintptr(uintptr_t *addr, uintptr_t val)
{
    return __atomic_fetch_add(addr, val, __ATOMIC_SEQ_CST);
}

static inline uintptr_t
_Py_atomic_and_uintptr(uintptr_t *addr, uintptr_t val)
{
    return __atomic_fetch_and(addr, val, __ATOMIC_SEQ_CST);
}

static inline int
_Py_atomic_compare_exchange_uintptr(uintptr_t *addr,
                                    uintptr_t *expected,
                                    uintptr_t desired)
{
    return __atomic_compare_exchange_n(addr, expected, desired,
                                      /*weak=*/0,
                                      __ATOMIC_SEQ_CST,
                                      __ATOMIC_SEQ_CST);
}

/* ===================================================================
 * pointer atomics  (GCC __atomic builtins)
 * =================================================================== */

static inline void *
_Py_atomic_load_ptr(const void *addr)
{
    uintptr_t val;
    __atomic_load((const uintptr_t *)addr, &val, __ATOMIC_SEQ_CST);
    return (void *)val;
}

static inline void *
_Py_atomic_load_ptr_acquire(const void *addr)
{
    uintptr_t val;
    __atomic_load((const uintptr_t *)addr, &val, __ATOMIC_ACQUIRE);
    return (void *)val;
}

static inline void
_Py_atomic_store_ptr_relaxed(void *addr, void *val)
{
    uintptr_t v = (uintptr_t)val;
    __atomic_store((uintptr_t *)addr, &v, __ATOMIC_RELAXED);
}

static inline void
_Py_atomic_store_ptr_release(void *addr, void *val)
{
    uintptr_t v = (uintptr_t)val;
    __atomic_store((uintptr_t *)addr, &v, __ATOMIC_RELEASE);
}

/* ===================================================================
 * int atomics  (GCC __atomic builtins)
 * =================================================================== */

static inline int
_Py_atomic_load_int(const int *addr)
{
    return __atomic_load_n(addr, __ATOMIC_SEQ_CST);
}

static inline void
_Py_atomic_store_int(int *addr, int val)
{
    __atomic_store_n(addr, val, __ATOMIC_SEQ_CST);
}

static inline int
_Py_atomic_load_int_relaxed(const int *addr)
{
    return __atomic_load_n(addr, __ATOMIC_RELAXED);
}

static inline void
_Py_atomic_store_int_relaxed(int *addr, int val)
{
    __atomic_store_n(addr, val, __ATOMIC_RELAXED);
}

static inline int
_Py_atomic_load_int_acquire(const int *addr)
{
    return __atomic_load_n(addr, __ATOMIC_ACQUIRE);
}

static inline void
_Py_atomic_store_int_release(int *addr, int val)
{
    __atomic_store_n(addr, val, __ATOMIC_RELEASE);
}

static inline int
_Py_atomic_add_int(int *addr, int val)
{
    return __atomic_fetch_add(addr, val, __ATOMIC_SEQ_CST);
}

/* ===================================================================
 * Py_ssize_t atomics  (GCC __atomic builtins)
 * =================================================================== */

static inline Py_ssize_t
_Py_atomic_load_ssize_relaxed(const Py_ssize_t *addr)
{
    return __atomic_load_n(addr, __ATOMIC_RELAXED);
}

static inline void
_Py_atomic_store_ssize_relaxed(Py_ssize_t *addr, Py_ssize_t val)
{
    __atomic_store_n(addr, val, __ATOMIC_RELAXED);
}

static inline Py_ssize_t
_Py_atomic_load_ssize_acquire(const Py_ssize_t *addr)
{
    return __atomic_load_n(addr, __ATOMIC_ACQUIRE);
}

static inline void
_Py_atomic_store_ssize_release(Py_ssize_t *addr, Py_ssize_t val)
{
    __atomic_store_n(addr, val, __ATOMIC_RELEASE);
}

static inline int
_Py_atomic_compare_exchange_ssize(Py_ssize_t *addr,
                                  Py_ssize_t *expected,
                                  Py_ssize_t desired)
{
    return __atomic_compare_exchange_n(addr, expected, desired,
                                      /*weak=*/0,
                                      __ATOMIC_SEQ_CST,
                                      __ATOMIC_SEQ_CST);
}

/* ===================================================================
 * Fences  (GCC __atomic builtins)
 * =================================================================== */

static inline void
_Py_atomic_fence_seq_cst(void)
{
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
}

static inline void
_Py_atomic_fence_release(void)
{
    __atomic_thread_fence(__ATOMIC_RELEASE);
}

static inline void
_Py_atomic_fence_acquire(void)
{
    __atomic_thread_fence(__ATOMIC_ACQUIRE);
}

#endif  /* HAVE_STD_ATOMIC / HAVE_BUILTIN_ATOMIC */

#ifdef __cplusplus
}
#endif

#endif  /* Py_GC_ATOMIC_COMPAT_H */
