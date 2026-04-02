/*
 * copy_graph.c -- Graph-based copy/exchange sequence generator (pure C)
 *
 * Phase 3D conversion: copy_graph.cpp -> copy_graph.c
 *
 * Generates optimal sequences of copy and exchange operations to
 * shuffle data between registers and memory locations.  Handles
 * chains (via sequential copies) and cycles (via exchanges for
 * register-only cycles, temp location for memory cycles).
 *
 * Uses _Py_hashtable for the node map (int location -> CopyNode*).
 */

#include "cinderx/Jit/codegen/copy_graph_c.h"

#include "Python.h"
#include "pycore_hashtable.h"

#include <assert.h>
#include <stddef.h>

/* ---- Intrusive doubly-linked list helpers ---- */

/*
 * Circular doubly-linked list node, embeddable in structs.
 * Self-referencing (prev == next == self) means "not in any list".
 */
typedef struct CopyListNode {
    struct CopyListNode *prev;
    struct CopyListNode *next;
} CopyListNode;

static inline void
clist_node_init(CopyListNode *n)
{
    n->prev = n;
    n->next = n;
}

static inline int
clist_node_is_linked(const CopyListNode *n)
{
    return n->prev != n;
}

static inline void
clist_node_unlink(CopyListNode *n)
{
    n->prev->next = n->next;
    n->next->prev = n->prev;
    n->prev = n;
    n->next = n;
}

/* Sentinel-based circular doubly-linked list head */
typedef struct {
    CopyListNode sentinel;
} CopyListHead;

static inline void
clist_init(CopyListHead *list)
{
    list->sentinel.prev = &list->sentinel;
    list->sentinel.next = &list->sentinel;
}

static inline int
clist_is_empty(const CopyListHead *list)
{
    return list->sentinel.next == &list->sentinel;
}

/* Insert item after the given node */
static inline void
clist_insert_after(CopyListNode *after, CopyListNode *item)
{
    assert(!clist_node_is_linked(item));
    item->next = after->next;
    item->prev = after;
    after->next->prev = item;
    after->next = item;
}

static inline void
clist_push_back(CopyListHead *list, CopyListNode *item)
{
    clist_insert_after(list->sentinel.prev, item);
}

static inline void
clist_push_front(CopyListHead *list, CopyListNode *item)
{
    clist_insert_after(&list->sentinel, item);
}

static inline CopyListNode *
clist_front(CopyListHead *list)
{
    assert(!clist_is_empty(list));
    return list->sentinel.next;
}

/* ---- CopyNode ---- */

typedef struct CopyNode {
    int loc;
    struct CopyNode *parent;
    CopyListNode child_link;    /* membership in parent's children list */
    CopyListNode leaf_link;     /* membership in global leaf_nodes list */
    CopyListHead children;      /* this node's children */
} CopyNode;

/* Container-of macros */
#define NODE_FROM_CHILD_LINK(ptr) \
    ((CopyNode *)((char *)(ptr) - offsetof(CopyNode, child_link)))
#define NODE_FROM_LEAF_LINK(ptr) \
    ((CopyNode *)((char *)(ptr) - offsetof(CopyNode, leaf_link)))

static CopyNode *
copy_node_new(int loc)
{
    CopyNode *n = (CopyNode *)PyMem_RawCalloc(1, sizeof(CopyNode));
    if (n == NULL) {
        return NULL;
    }
    n->loc = loc;
    n->parent = NULL;
    clist_node_init(&n->child_link);
    clist_node_init(&n->leaf_link);
    clist_init(&n->children);
    return n;
}

/* Unlink from all lists and free */
static void
copy_node_destroy(CopyNode *n)
{
    if (clist_node_is_linked(&n->child_link)) {
        clist_node_unlink(&n->child_link);
    }
    if (clist_node_is_linked(&n->leaf_link)) {
        clist_node_unlink(&n->leaf_link);
    }
    PyMem_RawFree(n);
}

/* ---- Op vector (dynamic array of JitCopyOp) ---- */

typedef struct {
    JitCopyOp *items;
    Py_ssize_t len;
    Py_ssize_t cap;
} OpVec;

static void
opvec_init(OpVec *v)
{
    v->items = NULL;
    v->len = 0;
    v->cap = 0;
}

static void
opvec_push(OpVec *v, JitCopyOpKind kind, int from, int to)
{
    if (v->len >= v->cap) {
        Py_ssize_t new_cap = v->cap ? v->cap * 2 : 8;
        v->items = (JitCopyOp *)PyMem_RawRealloc(
            v->items, (size_t)new_cap * sizeof(JitCopyOp));
        v->cap = new_cap;
    }
    v->items[v->len].kind = kind;
    v->items[v->len].from = from;
    v->items[v->len].to = to;
    v->len++;
}

/* ---- Hashtable key helpers ---- */

static inline const void *
int_to_key(int loc)
{
    return (const void *)(intptr_t)loc;
}

/* ---- JitCopyGraph ---- */

struct JitCopyGraph {
    _Py_hashtable_t *nodes;     /* int (as void*) -> CopyNode* */
    CopyListHead leaf_nodes;    /* global list of leaf nodes */
};

/* Hash table value destroy: just free the node.
 * List pointers become dangling but that's fine since all nodes are
 * being freed simultaneously during table destruction. */
static void
node_value_destroy(void *value)
{
    PyMem_RawFree(value);
}

JitCopyGraph *
jit_copy_graph_create(void)
{
    JitCopyGraph *g = (JitCopyGraph *)PyMem_RawCalloc(
        1, sizeof(JitCopyGraph));
    if (g == NULL) {
        return NULL;
    }
    g->nodes = _Py_hashtable_new_full(
        _Py_hashtable_hash_ptr,
        _Py_hashtable_compare_direct,
        NULL,                /* key_destroy: keys are ints cast to void* */
        node_value_destroy,  /* value_destroy: free CopyNode* */
        NULL);               /* allocator: use default */
    clist_init(&g->leaf_nodes);
    return g;
}

void
jit_copy_graph_destroy(JitCopyGraph *g)
{
    if (g == NULL) {
        return;
    }
    _Py_hashtable_destroy(g->nodes);
    PyMem_RawFree(g);
}

int
jit_copy_graph_is_empty(const JitCopyGraph *g)
{
    return g->nodes->nentries == 0;
}

/* Get or create a node for the given location.
 * Newly created nodes are added to leaf_nodes. */
static CopyNode *
get_node(JitCopyGraph *g, int loc)
{
    CopyNode *n = (CopyNode *)_Py_hashtable_get(g->nodes, int_to_key(loc));
    if (n != NULL) {
        return n;
    }
    n = copy_node_new(loc);
    _Py_hashtable_set(g->nodes, int_to_key(loc), n);
    /* Every new node starts as a leaf */
    clist_push_back(&g->leaf_nodes, &n->leaf_link);
    return n;
}

/* Set child's parent, updating list memberships */
static void
set_parent(JitCopyGraph *g, CopyNode *child, CopyNode *parent)
{
    assert(child != parent);
    (void)g;  /* g unused but kept for API consistency */

    /* Remove child from its current parent's children list */
    if (clist_node_is_linked(&child->child_link)) {
        clist_node_unlink(&child->child_link);
    }

    child->parent = parent;

    if (parent != NULL) {
        /* Add child to parent's children list */
        clist_push_back(&parent->children, &child->child_link);
        /* Parent is no longer a leaf */
        if (clist_node_is_linked(&parent->leaf_link)) {
            clist_node_unlink(&parent->leaf_link);
        }
    }
}

/* Remove a node from the hash table, unlink from all lists, and free */
static void
erase_node(JitCopyGraph *g, CopyNode *n)
{
    _Py_hashtable_steal(g->nodes, int_to_key(n->loc));
    copy_node_destroy(n);
}

void
jit_copy_graph_add_edge(JitCopyGraph *g, int from, int to)
{
    CopyNode *parent = get_node(g, from);
    CopyNode *child = get_node(g, to);
    assert(child->parent == NULL);
    set_parent(g, child, parent);
}

/* Check if a cycle (following parent pointers) contains only
 * register (non-negative) locations */
static int
in_register_cycle(CopyNode *node)
{
    CopyNode *cursor = node;
    do {
        if (cursor->loc < 0) {
            return 0;
        }
        cursor = cursor->parent;
    } while (cursor != node);
    return 1;
}

/* Process all leaf nodes, generating copy operations */
static void
process_leaf_nodes(JitCopyGraph *g, OpVec *ops)
{
    while (!clist_is_empty(&g->leaf_nodes)) {
        CopyNode *node = NODE_FROM_LEAF_LINK(clist_front(&g->leaf_nodes));
        clist_node_unlink(&node->leaf_link);

        CopyNode *parent = node->parent;
        opvec_push(ops, JIT_COPY_OP_COPY, parent->loc, node->loc);
        erase_node(g, node);

        if (clist_is_empty(&parent->children)) {
            if (parent->parent == NULL) {
                /* Parent has no parent — last copy in this chain */
                erase_node(g, parent);
            } else {
                /* Process the parent next */
                clist_push_front(&g->leaf_nodes, &parent->leaf_link);
            }
        }
    }
}

/* Get any node from the hash table.  Returns NULL if empty. */
static CopyNode *
get_any_node(JitCopyGraph *g)
{
    for (size_t i = 0; i < g->nodes->nbuckets; i++) {
        _Py_slist_item_t *item = g->nodes->buckets[i].head;
        if (item != NULL) {
            _Py_hashtable_entry_t *entry = (_Py_hashtable_entry_t *)item;
            return (CopyNode *)entry->value;
        }
    }
    return NULL;
}

JitCopyOp *
jit_copy_graph_process(JitCopyGraph *g, Py_ssize_t *out_count)
{
    OpVec ops;
    opvec_init(&ops);

    process_leaf_nodes(g, &ops);

    while (g->nodes->nentries > 0) {
        CopyNode *node = get_any_node(g);
        assert(node != NULL);

        if (in_register_cycle(node)) {
            /* All-register cycle: use exchange operations */
            CopyNode *first_child = NODE_FROM_CHILD_LINK(
                clist_front(&node->children));
            set_parent(g, first_child, NULL);

            while (node->parent != NULL) {
                opvec_push(&ops, JIT_COPY_OP_EXCHANGE,
                           node->loc, node->parent->loc);
                CopyNode *parent = node->parent;
                erase_node(g, node);
                node = parent;
            }
            erase_node(g, node);
            continue;
        }

        /* Memory cycle: save to temp, break cycle, process leaves */
        opvec_push(&ops, JIT_COPY_OP_COPY, node->loc,
                   JIT_COPY_GRAPH_TEMP_LOC);
        CopyNode *temp_node = get_node(g, JIT_COPY_GRAPH_TEMP_LOC);
        CopyNode *child = NODE_FROM_CHILD_LINK(
            clist_front(&node->children));
        set_parent(g, child, temp_node);
        clist_push_back(&g->leaf_nodes, &node->leaf_link);
        process_leaf_nodes(g, &ops);
    }

    *out_count = ops.len;
    return ops.items;
}

void
jit_copy_graph_ops_free(JitCopyOp *ops)
{
    PyMem_RawFree(ops);
}
