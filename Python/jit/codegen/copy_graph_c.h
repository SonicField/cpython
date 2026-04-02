/*
 * copy_graph_c.h -- Pure C interface for CopyGraph
 *
 * Phase 3D conversion: copy_graph.cpp -> copy_graph.c
 *
 * CopyGraph generates optimal sequences of copy and exchange operations
 * to shuffle data between registers (non-negative ints) and memory
 * locations (negative ints).
 */

#ifndef COPY_GRAPH_C_H
#define COPY_GRAPH_C_H

#include "Python.h"

#include <limits.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle */
typedef struct JitCopyGraph JitCopyGraph;

/* Operation kinds */
typedef enum {
    JIT_COPY_OP_COPY = 0,
    JIT_COPY_OP_EXCHANGE = 1,
} JitCopyOpKind;

/* A single copy or exchange operation */
typedef struct {
    JitCopyOpKind kind;
    int from;
    int to;
} JitCopyOp;

/* Sentinel location used to break memory cycles.
 * The caller is responsible for mapping this to a real temp location
 * that doesn't conflict with any locations in the graph. */
#define JIT_COPY_GRAPH_TEMP_LOC INT_MAX

/* Create a new empty copy graph. */
JitCopyGraph *jit_copy_graph_create(void);

/* Destroy a copy graph and free all resources. */
void jit_copy_graph_destroy(JitCopyGraph *g);

/* Add a copy edge: data flows from location `from` to location `to`.
 * Each `to` location may have at most one incoming edge. */
void jit_copy_graph_add_edge(JitCopyGraph *g, int from, int to);

/* Return nonzero if the graph contains no nodes. */
int jit_copy_graph_is_empty(const JitCopyGraph *g);

/* Process the graph and return an array of copy/exchange operations.
 * Sets *out_count to the number of operations.
 * The returned array must be freed with jit_copy_graph_ops_free().
 * After processing, the graph is empty. */
JitCopyOp *jit_copy_graph_process(JitCopyGraph *g, Py_ssize_t *out_count);

/* Free an operations array returned by jit_copy_graph_process(). */
void jit_copy_graph_ops_free(JitCopyOp *ops);

#ifdef __cplusplus
}
#endif

#endif /* COPY_GRAPH_C_H */
