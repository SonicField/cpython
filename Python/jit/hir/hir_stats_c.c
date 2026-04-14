/*
 * hir_stats_c.c — Pure C implementation of HIR statistics collection.
 *
 * Phase 3D: Replaces hir_stats.cpp (C++ HIRStats pass class).
 * Diagnostic-only — gated by dump_hir_stats config flag.
 */

#include "cinderx/Jit/hir/hir_stats_c.h"
#include "cinderx/Jit/hir/hir_c_api.h"
#include "cinderx/Jit/hir/hir_instr_c.h"
#include "cinderx/Jit/jit_config_c.h"
#include "cinderx/Jit/threaded_compile_c.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- Simple string->int counter map ---- */

typedef struct {
    char *key;
    int count;
} StatsEntry;

typedef struct {
    StatsEntry *entries;
    size_t size;
    size_t capacity;
} StatsMap;

static void stats_map_init(StatsMap *m) {
    m->entries = NULL;
    m->size = 0;
    m->capacity = 0;
}

static void stats_map_destroy(StatsMap *m) {
    for (size_t i = 0; i < m->size; i++) {
        free(m->entries[i].key);
    }
    free(m->entries);
}

static void stats_map_increment(StatsMap *m, const char *key) {
    /* Linear search — fine for diagnostic use (small N). */
    for (size_t i = 0; i < m->size; i++) {
        if (strcmp(m->entries[i].key, key) == 0) {
            m->entries[i].count++;
            return;
        }
    }
    /* New key. */
    if (m->size == m->capacity) {
        size_t new_cap = m->capacity ? m->capacity * 2 : 32;
        StatsEntry *new_entries = (StatsEntry *)realloc(
            m->entries, new_cap * sizeof(StatsEntry));
        if (!new_entries) return;  /* OOM — skip silently */
        m->entries = new_entries;
        m->capacity = new_cap;
    }
    m->entries[m->size].key = strdup(key);
    m->entries[m->size].count = 1;
    m->size++;
}

/* ---- JSON string escaping ---- */

static void escape_json(const char *s, char *out, size_t outsz) {
    size_t j = 0;
    for (size_t i = 0; s[i] && j + 6 < outsz; i++) {
        unsigned char c = (unsigned char)s[i];
        switch (c) {
            case '"':  out[j++] = '\\'; out[j++] = '"'; break;
            case '\\': out[j++] = '\\'; out[j++] = '\\'; break;
            case '\n': out[j++] = '\\'; out[j++] = 'n'; break;
            case '\r': out[j++] = '\\'; out[j++] = 'r'; break;
            case '\t': out[j++] = '\\'; out[j++] = 't'; break;
            case '\b': out[j++] = '\\'; out[j++] = 'b'; break;
            case '\f': out[j++] = '\\'; out[j++] = 'f'; break;
            default:
                if (c < 0x20 || c >= 0x80) {
                    j += (size_t)snprintf(out + j, outsz - j, "\\u%04x", c);
                } else {
                    out[j++] = (char)c;
                }
                break;
        }
    }
    out[j] = '\0';
}

/* ---- Stats dump as JSON ---- */

static void stats_dump(const StatsMap *instrs, const StatsMap *types,
                       const char *func_name, FILE *output) {
    char escaped_name[512];
    escape_json(func_name, escaped_name, sizeof(escaped_name));

    fprintf(output, "JIT: %s:%d -- Stats for %s: {\"function\": \"%s\", "
            "\"instructions\": {", __FILE__, __LINE__, func_name, escaped_name);

    for (size_t i = 0; i < instrs->size; i++) {
        char escaped_key[256];
        escape_json(instrs->entries[i].key, escaped_key, sizeof(escaped_key));
        fprintf(output, "%s\"%s\": %d",
                i > 0 ? ", " : "", escaped_key, instrs->entries[i].count);
    }

    fprintf(output, "}, \"types\": {");

    for (size_t i = 0; i < types->size; i++) {
        char escaped_key[256];
        escape_json(types->entries[i].key, escaped_key, sizeof(escaped_key));
        fprintf(output, "%s\"%s\": %d",
                i > 0 ? ", " : "", escaped_key, types->entries[i].count);
    }

    fprintf(output, "}}\n");
    fflush(output);
}

/* ---- Public API ---- */

void hir_stats_run(HirFunction func, const char *func_name) {
    StatsMap instrs, types;
    stats_map_init(&instrs);
    stats_map_init(&types);

    int safe = jit_compile_running();

    HirCFG cfg = hir_func_cfg(func);
    HirBasicBlock block = hir_cfg_blocks_first(cfg);
    while (block) {
        HirInstr instr = hir_block_first(block);
        while (instr) {
            /* Count opcodes — pure C via opcode table. */
            const char *opname = hir_instr_info_name(hir_c_opcode(instr));
            stats_map_increment(&instrs, opname);

            /* Count output types. */
            HirRegister output = hir_c_output(instr);
            if (output) {
                HirType t = hir_register_type(output);
                char type_str[256];
                hir_type_to_string(&t, type_str, sizeof(type_str), safe);
                stats_map_increment(&types, type_str);
            }

            instr = hir_block_next(block, instr);
        }
        block = hir_cfg_blocks_next(cfg, block);
    }

    /* Dump to JIT log output. */
    const JitConfig *config = jit_get_config();
    FILE *output = config->log.output_file;
    if (!output) output = stderr;
    stats_dump(&instrs, &types, func_name, output);

    stats_map_destroy(&instrs);
    stats_map_destroy(&types);
}
