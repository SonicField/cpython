/*
 * writer.c -- ELF writer (pure C)
 *
 * Phase 3D conversion: writer.cpp -> writer.c
 * Constructs an ELF shared library containing JIT-compiled functions.
 * Uses C table types (JitElfSymTab, JitElfStrTab, etc.) and a
 * BufWriter for output instead of std::ostream.
 */

#include "cinderx/Jit/elf/writer_c.h"
#include "cinderx/Jit/elf/note_c.h"

#include "cinderx/Jit/elf/symbol.h"   /* JitElfSymTab, JitElfSymbol */
#include "cinderx/Jit/elf/string.h"   /* JitElfStrTab */
#include "cinderx/Jit/elf/hash.h"     /* JitElfHashTab */
#include "cinderx/Jit/elf/dynamic.h"  /* JitElfDynTab */

#include "Python.h"

#include <assert.h>
#include <stddef.h>
#include <string.h>
#include <zlib.h>

#ifndef WIN32

/* ---- ELF header structures (binary-compatible with header.h) ---- */

/* Section header types */
#define EW_PROGRAM      0x01
#define EW_SYMBOL_TABLE 0x02
#define EW_STRING_TABLE 0x03
#define EW_HASH         0x05
#define EW_DYNAMIC      0x06
#define EW_NOTE         0x07

/* Section header flags */
#define EW_SEC_WRITABLE   0x01
#define EW_SEC_ALLOC      0x02
#define EW_SEC_EXECUTABLE 0x04
#define EW_SEC_INFO_LINK  0x40

/* Segment header types */
#define EW_SEG_LOADABLE 0x1
#define EW_SEG_DYNAMIC  0x2
#define EW_SEG_NOTE     0x4

/* Segment header flags */
#define EW_SEG_EXECUTABLE 0x1
#define EW_SEG_WRITABLE   0x2
#define EW_SEG_READABLE   0x4

#pragma pack(push, 1)

typedef struct {
    uint32_t name_offset;
    uint32_t type;
    uint64_t flags;
    uint64_t address;
    uint64_t offset;
    uint64_t size;
    uint32_t link;
    uint32_t info;
    uint64_t align;
    uint64_t entry_size;
} EwSectionHeader;

typedef struct {
    uint32_t type;
    uint32_t flags;
    uint64_t offset;
    uint64_t address;
    uint64_t physical_address;
    uint64_t file_size;
    uint64_t mem_size;
    uint64_t align;
} EwSegmentHeader;

typedef struct {
    uint8_t magic[4];
    uint8_t elf_class;
    uint8_t endian;
    uint8_t elf_version;
    uint8_t osabi;
    uint8_t abi_version;
    uint8_t padding[7];
    uint16_t type;
    uint16_t machine;
    uint32_t version;
    uint64_t entry_address;
    uint64_t segment_header_offset;
    uint64_t section_header_offset;
    uint32_t flags;
    uint16_t header_size;
    uint16_t segment_header_size;
    uint16_t segment_header_count;
    uint16_t section_header_size;
    uint16_t section_header_count;
    uint16_t section_name_index;
} EwFileHeader;

#pragma pack(pop)

/* Section indices */
enum {
    SEC_NULL = 0,
    SEC_TEXT,
    SEC_DYNSYM,
    SEC_DYNSTR,
    SEC_HASH,
    SEC_FUNC_NOTE,
    SEC_DYNAMIC,
    SEC_SHSTRTAB,
    SEC_TOTAL
};

/* Segment indices */
enum {
    SEG_TEXT = 0,
    SEG_READONLY,
    SEG_READWRITE,
    SEG_FUNC_NOTE,
    SEG_DYNAMIC,
    SEG_TOTAL
};

static const uint64_t PAGE_SIZE = 0x1000;
static const uint64_t TEXT_START = 0x1000;

/* ---- Internal Object ---- */

typedef struct {
    EwFileHeader file_header;
    EwSectionHeader section_headers[SEC_TOTAL];
    EwSegmentHeader segment_headers[SEG_TOTAL];
    uint32_t header_padding;
    uint32_t text_padding;

    JitElfSymTab dynsym;
    JitElfStrTab dynstr;
    uint32_t dynsym_padding;

    JitElfHashTab hash;
    uint32_t hash_padding;

    JitElfNoteArray func_notes;
    uint32_t func_notes_padding;

    JitElfDynTab dynamic;
    uint32_t dynamic_padding;

    JitElfStrTab shstrtab;

    uint64_t section_offset;
    uint32_t libpython_name;
} EwObject;

/* ---- BufWriter ---- */

typedef struct {
    uint8_t *data;
    size_t len;
    size_t cap;
    int error;
} BufWriter;

static void
bw_init(BufWriter *bw)
{
    bw->data = NULL;
    bw->len = 0;
    bw->cap = 0;
    bw->error = 0;
}

static void
bw_free(BufWriter *bw)
{
    PyMem_RawFree(bw->data);
    bw->data = NULL;
    bw->len = 0;
    bw->cap = 0;
}

static void
bw_grow(BufWriter *bw, size_t needed)
{
    if (bw->error) return;
    size_t new_cap = bw->cap ? bw->cap : 4096;
    while (new_cap < bw->len + needed) {
        new_cap *= 2;
    }
    if (new_cap > bw->cap) {
        uint8_t *p = (uint8_t *)PyMem_RawRealloc(bw->data, new_cap);
        if (p == NULL) {
            bw->error = 1;
            return;
        }
        bw->data = p;
        bw->cap = new_cap;
    }
}

static void
bw_write(BufWriter *bw, const void *src, size_t n)
{
    bw_grow(bw, n);
    if (bw->error) return;
    memcpy(bw->data + bw->len, src, n);
    bw->len += n;
}

static void
bw_write_u32(BufWriter *bw, uint32_t v)
{
    bw_write(bw, &v, sizeof(v));
}

static void
bw_pad(BufWriter *bw, size_t n)
{
    bw_grow(bw, n);
    if (bw->error) return;
    memset(bw->data + bw->len, 0, n);
    bw->len += n;
}

/* ---- Utilities ---- */

static uint64_t
round_up(uint64_t n, uint64_t align)
{
    return (n + align - 1) & ~(align - 1);
}

static uint64_t
align_offset(EwObject *elf, uint64_t align)
{
    uint64_t new_off = round_up(elf->section_offset, align);
    uint64_t delta = new_off - elf->section_offset;
    elf->section_offset = new_off;
    return delta;
}

static uint32_t
hash_bytecode(PyCodeObject *code)
{
    uint32_t crc = crc32(0, NULL, 0);
    PyObject *bc = PyCode_GetCode(code);
    if (!PyBytes_Check(bc)) {
        Py_DECREF(bc);
        return crc;
    }
    char *buffer;
    Py_ssize_t len;
    if (PyBytes_AsStringAndSize(bc, &buffer, &len) < 0) {
        Py_DECREF(bc);
        return crc;
    }
    crc = crc32(crc, (unsigned char *)buffer, len);
    Py_DECREF(bc);
    return crc;
}

/* ---- Init helpers ---- */

static void
init_file_header(EwObject *elf)
{
    EwFileHeader *h = &elf->file_header;
    memset(h, 0, sizeof(*h));
    h->magic[0] = 0x7f; h->magic[1] = 'E'; h->magic[2] = 'L'; h->magic[3] = 'F';
    h->elf_class = 2;         /* 64-bit */
    h->endian = 1;            /* little-endian */
    h->elf_version = 1;
    h->osabi = 3;             /* Linux */
    h->type = 3;              /* shared library */
    h->machine = 0x3e;        /* AMD x86-64 */
    h->version = 1;
    h->header_size = 64;
    h->segment_header_size = sizeof(EwSegmentHeader);
    h->segment_header_offset = offsetof(EwObject, segment_headers);
    h->segment_header_count = SEG_TOTAL;
    h->section_header_offset = offsetof(EwObject, section_headers);
    h->section_header_count = SEC_TOTAL;
    h->section_name_index = SEC_SHSTRTAB;
}

static void
init_text_section(EwObject *elf, uint64_t text_size)
{
    EwSectionHeader *h = &elf->section_headers[SEC_TEXT];
    h->name_offset = jit_elf_strtab_insert(&elf->shstrtab, ".text", 5);
    h->type = EW_PROGRAM;
    h->flags = EW_SEC_ALLOC | EW_SEC_EXECUTABLE;
    h->address = elf->section_offset;
    h->offset = elf->section_offset;
    h->size = text_size;
    h->align = 0x10;
    elf->section_offset += h->size;
}

static void
init_dynsym_section(EwObject *elf)
{
    EwSectionHeader *h = &elf->section_headers[SEC_DYNSYM];
    h->name_offset = jit_elf_strtab_insert(&elf->shstrtab, ".dynsym", 7);
    h->type = EW_SYMBOL_TABLE;
    h->flags = EW_SEC_ALLOC | EW_SEC_INFO_LINK;
    h->address = elf->section_offset;
    h->offset = elf->section_offset;
    h->size = jit_elf_symtab_data_size(&elf->dynsym);
    h->link = SEC_DYNSTR;
    h->info = 1;
    h->align = 0x8;
    h->entry_size = sizeof(JitElfSymbol);
    elf->section_offset += h->size;
}

static void
init_dynstr_section(EwObject *elf)
{
    EwSectionHeader *h = &elf->section_headers[SEC_DYNSTR];
    h->name_offset = jit_elf_strtab_insert(&elf->shstrtab, ".dynstr", 7);
    h->type = EW_STRING_TABLE;
    h->flags = EW_SEC_ALLOC;
    h->address = elf->section_offset;
    h->offset = elf->section_offset;
    h->size = jit_elf_strtab_size(&elf->dynstr);
    h->align = 0x1;
    elf->section_offset += h->size;
}

static void
init_hash_section(EwObject *elf)
{
    EwSectionHeader *h = &elf->section_headers[SEC_HASH];
    h->name_offset = jit_elf_strtab_insert(&elf->shstrtab, ".hash", 5);
    h->type = EW_HASH;
    h->flags = EW_SEC_ALLOC;
    h->address = elf->section_offset;
    h->offset = elf->section_offset;
    h->size = jit_elf_hashtab_size_bytes(&elf->hash);
    h->link = SEC_DYNSYM;
    h->align = 0x8;
    elf->section_offset += h->size;
}

static void
init_func_note_section(EwObject *elf)
{
    EwSectionHeader *h = &elf->section_headers[SEC_FUNC_NOTE];
    h->name_offset = jit_elf_strtab_insert(
        &elf->shstrtab, JIT_ELF_FUNC_NOTE_SECTION_NAME,
        strlen(JIT_ELF_FUNC_NOTE_SECTION_NAME));
    h->type = EW_NOTE;
    h->flags = EW_SEC_ALLOC;
    h->address = elf->section_offset;
    h->offset = elf->section_offset;
    h->size = jit_elf_note_array_size_bytes(&elf->func_notes);
    h->align = 0x4;
    elf->section_offset += h->size;
}

static void
init_dynamic_section(EwObject *elf)
{
    EwSectionHeader *h = &elf->section_headers[SEC_DYNAMIC];
    h->name_offset = jit_elf_strtab_insert(&elf->shstrtab, ".dynamic", 8);
    h->type = EW_DYNAMIC;
    h->flags = EW_SEC_ALLOC | EW_SEC_WRITABLE;
    h->address = elf->section_offset;
    h->offset = elf->section_offset;
    h->size = jit_elf_dyntab_data_size(&elf->dynamic);
    h->link = SEC_DYNSTR;
    h->entry_size = sizeof(JitElfDyn);
    h->align = 0x8;
    elf->section_offset += h->size;
}

static void
init_shstrtab_section(EwObject *elf)
{
    EwSectionHeader *h = &elf->section_headers[SEC_SHSTRTAB];
    h->name_offset = jit_elf_strtab_insert(&elf->shstrtab, ".shstrtab", 9);
    h->type = EW_STRING_TABLE;
    h->offset = elf->section_offset;
    h->size = jit_elf_strtab_size(&elf->shstrtab);
    h->align = 0x1;
    elf->section_offset += h->size;
}

static void
init_text_segment(EwObject *elf)
{
    EwSectionHeader *sec = &elf->section_headers[SEC_TEXT];
    EwSegmentHeader *h = &elf->segment_headers[SEG_TEXT];
    h->type = EW_SEG_LOADABLE;
    h->flags = EW_SEG_EXECUTABLE | EW_SEG_READABLE;
    h->offset = sec->offset;
    h->address = sec->address;
    h->file_size = sec->size;
    h->mem_size = h->file_size;
    h->align = 0x1000;
}

static void
init_readonly_segment(EwObject *elf)
{
    EwSectionHeader *dynsym = &elf->section_headers[SEC_DYNSYM];
    EwSectionHeader *dynamic = &elf->section_headers[SEC_DYNAMIC];
    EwSegmentHeader *h = &elf->segment_headers[SEG_READONLY];
    h->type = EW_SEG_LOADABLE;
    h->flags = EW_SEG_READABLE;
    h->offset = dynsym->offset;
    h->address = dynsym->address;
    h->file_size = dynamic->offset - dynsym->offset;
    h->mem_size = h->file_size;
    h->align = 0x1000;
}

static void
init_readwrite_segment(EwObject *elf)
{
    EwSectionHeader *dynamic = &elf->section_headers[SEC_DYNAMIC];
    EwSegmentHeader *h = &elf->segment_headers[SEG_READWRITE];
    h->type = EW_SEG_LOADABLE;
    h->flags = EW_SEG_READABLE | EW_SEG_WRITABLE;
    h->offset = dynamic->offset;
    h->address = dynamic->address;
    h->file_size = dynamic->size;
    h->mem_size = h->file_size;
    h->align = 0x1000;
}

static void
init_func_note_segment(EwObject *elf)
{
    EwSectionHeader *note = &elf->section_headers[SEC_FUNC_NOTE];
    EwSegmentHeader *h = &elf->segment_headers[SEG_FUNC_NOTE];
    h->type = EW_SEG_NOTE;
    h->flags = EW_SEG_READABLE;
    h->offset = note->offset;
    h->address = note->address;
    h->file_size = note->size;
    h->mem_size = h->file_size;
    h->align = note->align;
}

static void
init_dynamic_segment(EwObject *elf)
{
    EwSectionHeader *dynamic = &elf->section_headers[SEC_DYNAMIC];
    EwSegmentHeader *h = &elf->segment_headers[SEG_DYNAMIC];
    h->type = EW_SEG_DYNAMIC;
    h->flags = EW_SEG_READABLE | EW_SEG_WRITABLE;
    h->offset = dynamic->offset;
    h->address = dynamic->address;
    h->file_size = dynamic->size;
    h->mem_size = h->file_size;
    h->align = 0x1000;
}

static void
init_dynamics(EwObject *elf)
{
    EwSectionHeader *dynsym = &elf->section_headers[SEC_DYNSYM];
    EwSectionHeader *dynstr = &elf->section_headers[SEC_DYNSTR];
    EwSectionHeader *hash = &elf->section_headers[SEC_HASH];

    jit_elf_dyntab_insert(&elf->dynamic, JIT_ELF_DYN_NEEDED,
                          elf->libpython_name);
    jit_elf_dyntab_insert(&elf->dynamic, JIT_ELF_DYN_HASH, hash->address);
    jit_elf_dyntab_insert(&elf->dynamic, JIT_ELF_DYN_STRTAB, dynstr->address);
    jit_elf_dyntab_insert(&elf->dynamic, JIT_ELF_DYN_STRSZ, dynstr->size);
    jit_elf_dyntab_insert(&elf->dynamic, JIT_ELF_DYN_SYMTAB, dynsym->address);
    jit_elf_dyntab_insert(&elf->dynamic, JIT_ELF_DYN_SYMENT,
                          sizeof(JitElfSymbol));
}

/* ---- Note construction ---- */

static void
write_func_note_desc(BufWriter *bw, const JitElfCodeEntry *entry)
{
    uintptr_t code_start = (uintptr_t)entry->compiled_code;
    bw_write_u32(bw, (uint32_t)strlen(entry->file_name));
    bw_write(bw, entry->file_name, strlen(entry->file_name));
    bw_write_u32(bw, (uint32_t)entry->lineno);
    bw_write_u32(bw, hash_bytecode(entry->code));
    bw_write_u32(bw, (uint32_t)entry->compiled_code_size);

    uintptr_t normal_entry = (uintptr_t)entry->normal_entry;
    uintptr_t static_entry = (uintptr_t)entry->static_entry;
    uint32_t normal_off = (uint32_t)(normal_entry - code_start);
    uint32_t static_off = static_entry != 0
        ? (uint32_t)(static_entry - code_start)
        : JIT_ELF_INVALID_STATIC_OFFSET;

    bw_write_u32(bw, normal_off);
    bw_write_u32(bw, static_off);
}

static void
init_func_notes(EwObject *elf, const JitElfCodeEntry *entries, size_t count)
{
    for (size_t i = 0; i < count; i++) {
        BufWriter desc_bw;
        bw_init(&desc_bw);
        write_func_note_desc(&desc_bw, &entries[i]);
        if (desc_bw.error) {
            bw_free(&desc_bw);
            continue;
        }
        jit_elf_note_array_insert_bin(
            &elf->func_notes, entries[i].func_name,
            (const char *)desc_bw.data, desc_bw.len, 0x30a05f0);
        bw_free(&desc_bw);
    }
}

/* ---- Write helpers ---- */

static void
write_hash(BufWriter *bw, const JitElfHashTab *ht)
{
    bw_write_u32(bw, ht->nbuckets);
    bw_write_u32(bw, ht->nchains);
    bw_write(bw, ht->buckets, ht->nbuckets * sizeof(uint32_t));
    bw_write(bw, ht->chains, ht->nchains * sizeof(uint32_t));
}

static void
write_note(BufWriter *bw, const JitElfNote *note)
{
    uint32_t name_size = (uint32_t)(strlen(note->name) + 1);
    uint32_t desc_size = (uint32_t)(note->desc_len + 1);

    bw_write(bw, &name_size, 4);
    bw_write(bw, &desc_size, 4);
    bw_write(bw, &note->type, 4);

    bw_write(bw, note->name, name_size);
    size_t name_pad = round_up(name_size, 4) - name_size;
    bw_pad(bw, name_pad);

    if (note->desc_len > 0) {
        bw_write(bw, note->desc, note->desc_len);
        /* Write the NUL terminator */
        uint8_t nul = 0;
        bw_write(bw, &nul, 1);
        size_t desc_pad = round_up(desc_size, 4) - desc_size;
        bw_pad(bw, desc_pad);
    }
}

static void
write_notes(BufWriter *bw, const JitElfNoteArray *notes)
{
    for (size_t i = 0; i < jit_elf_note_array_len(notes); i++) {
        write_note(bw, jit_elf_note_array_get(notes, i));
    }
}

static void
write_elf(BufWriter *bw, const EwObject *elf,
          const JitElfCodeEntry *entries, size_t count)
{
    /* Headers */
    bw_write(bw, &elf->file_header, sizeof(elf->file_header));
    bw_write(bw, &elf->section_headers, sizeof(elf->section_headers));
    bw_write(bw, &elf->segment_headers, sizeof(elf->segment_headers));
    bw_pad(bw, elf->header_padding);

    /* .text: compiled code */
    for (size_t i = 0; i < count; i++) {
        bw_write(bw, entries[i].compiled_code, entries[i].compiled_code_size);
    }
    bw_pad(bw, elf->text_padding);

    /* .dynsym + .dynstr */
    bw_write(bw, jit_elf_symtab_data(&elf->dynsym),
             jit_elf_symtab_data_size(&elf->dynsym));
    bw_write(bw, jit_elf_strtab_data(&elf->dynstr),
             jit_elf_strtab_size(&elf->dynstr));
    bw_pad(bw, elf->dynsym_padding);

    /* .hash */
    write_hash(bw, &elf->hash);
    bw_pad(bw, elf->hash_padding);

    /* .note.pyfunc */
    write_notes(bw, &elf->func_notes);
    bw_pad(bw, elf->func_notes_padding);

    /* .dynamic */
    bw_write(bw, jit_elf_dyntab_data(&elf->dynamic),
             jit_elf_dyntab_data_size(&elf->dynamic));
    bw_pad(bw, elf->dynamic_padding);

    /* .shstrtab */
    bw_write(bw, jit_elf_strtab_data(&elf->shstrtab),
             jit_elf_strtab_size(&elf->shstrtab));
}

/* ---- Public API ---- */

int
jit_elf_write_entries(const JitElfCodeEntry *entries, size_t count,
                      uint8_t **out_data, size_t *out_size)
{
    EwObject elf;
    memset(&elf, 0, sizeof(elf));

    jit_elf_symtab_init(&elf.dynsym);
    jit_elf_strtab_init(&elf.dynstr);
    jit_elf_hashtab_init(&elf.hash);
    jit_elf_note_array_init(&elf.func_notes);
    jit_elf_dyntab_init(&elf.dynamic);
    jit_elf_strtab_init(&elf.shstrtab);

    init_file_header(&elf);

    /* Initialize symbols */
    uint64_t text_end = TEXT_START;
    for (size_t i = 0; i < count; i++) {
        JitElfSymbol sym;
        memset(&sym, 0, sizeof(sym));
        sym.name_offset = jit_elf_strtab_insert(
            &elf.dynstr, entries[i].func_name, strlen(entries[i].func_name));
        sym.info = JIT_ELF_SYM_GLOBAL | JIT_ELF_SYM_FUNC;
        sym.section_index = SEC_TEXT;
        sym.address = text_end;
        sym.size = entries[i].compiled_code_size;
        jit_elf_symtab_insert(&elf.dynsym, &sym);
        text_end += entries[i].compiled_code_size;
    }
    uint64_t text_size = text_end - TEXT_START;

    elf.libpython_name = jit_elf_strtab_insert(
        &elf.dynstr, "libpython3.10.so", 16);

    /* Layout sections */
    elf.section_offset = offsetof(EwObject, header_padding);
    elf.header_padding = (uint32_t)align_offset(&elf, PAGE_SIZE);

    init_text_section(&elf, text_size);
    elf.text_padding = (uint32_t)align_offset(&elf, PAGE_SIZE);

    init_dynsym_section(&elf);
    init_dynstr_section(&elf);
    elf.dynsym_padding = (uint32_t)align_offset(&elf, 0x8);

    jit_elf_hashtab_build(&elf.hash, &elf.dynsym, &elf.dynstr);
    init_hash_section(&elf);
    elf.hash_padding = (uint32_t)align_offset(&elf, 4);

    init_func_notes(&elf, entries, count);
    init_func_note_section(&elf);
    elf.func_notes_padding = (uint32_t)align_offset(&elf, PAGE_SIZE);

    init_dynamics(&elf);
    init_dynamic_section(&elf);
    elf.dynamic_padding = (uint32_t)align_offset(&elf, 0x8);

    init_shstrtab_section(&elf);

    init_text_segment(&elf);
    init_readonly_segment(&elf);
    init_readwrite_segment(&elf);
    init_func_note_segment(&elf);
    init_dynamic_segment(&elf);

    /* Write to buffer */
    BufWriter bw;
    bw_init(&bw);
    write_elf(&bw, &elf, entries, count);

    /* Cleanup tables */
    jit_elf_symtab_free(&elf.dynsym);
    jit_elf_strtab_free(&elf.dynstr);
    jit_elf_hashtab_free(&elf.hash);
    jit_elf_note_array_free(&elf.func_notes);
    jit_elf_dyntab_free(&elf.dynamic);
    jit_elf_strtab_free(&elf.shstrtab);

    if (bw.error) {
        bw_free(&bw);
        return -1;
    }

    *out_data = bw.data;
    *out_size = bw.len;
    return 0;
}

#endif /* !WIN32 */
