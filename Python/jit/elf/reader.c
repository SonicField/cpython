/*
 * reader.c -- ELF note section reader (pure C)
 *
 * Phase 3D conversion: reader.cpp -> reader.c
 * Reads ELF note sections and parses code note data.
 * Uses a simple buffer reader instead of std::istream.
 */

#include "cinderx/Jit/elf/reader_c.h"
#include "cinderx/Jit/elf/note_c.h"

#include "Python.h"

#ifdef ENABLE_ELF_READER
#include <elf.h>
#include <link.h>
#endif

#include <assert.h>
#include <string.h>

/* ---- Buffer reader ---- */

typedef struct {
    const uint8_t *data;
    size_t size;
    size_t pos;
    int error;   /* nonzero if a read failed */
} BufReader;

static void
bufreader_init(BufReader *br, const uint8_t *data, size_t size)
{
    br->data = data;
    br->size = size;
    br->pos = 0;
    br->error = 0;
}

static int
bufreader_good(const BufReader *br)
{
    return !br->error && br->pos < br->size;
}

static uint32_t
bufreader_read_u32(BufReader *br)
{
    if (br->pos + sizeof(uint32_t) > br->size) {
        br->error = 1;
        return 0;
    }
    uint32_t result;
    memcpy(&result, br->data + br->pos, sizeof(uint32_t));
    br->pos += sizeof(uint32_t);
    return result;
}

/*
 * Read a string of `size` characters.
 * If has_nul_terminator is nonzero, reads one extra byte (the NUL)
 * past the string content.
 * Returns a newly allocated copy (PyMem_RawMalloc) or NULL on error.
 */
static char *
bufreader_read_str(BufReader *br, size_t size, int has_nul_terminator)
{
    if (size > 100000) {
        br->error = 1;
        return NULL;
    }
    size_t read_size = has_nul_terminator ? size + 1 : size;
    if (br->pos + read_size > br->size) {
        br->error = 1;
        return NULL;
    }
    char *result = (char *)PyMem_RawMalloc(size + 1);
    if (result == NULL) {
        br->error = 1;
        return NULL;
    }
    memcpy(result, br->data + br->pos, size);
    result[size] = '\0';
    br->pos += read_size;
    return result;
}

/*
 * Skip padding bytes.  Given that the previous item was read with
 * `previous_size` bytes, skip ahead to the next `alignment` boundary.
 */
static void
bufreader_unpad(BufReader *br, size_t previous_size, size_t alignment)
{
    size_t ignore = alignment - (previous_size % alignment);
    if (ignore == alignment) {
        ignore = 0;
    }
    if (br->pos + ignore > br->size) {
        br->error = 1;
        return;
    }
    br->pos += ignore;
}

/* ---- Note reading ---- */

static int
read_note(BufReader *br, JitElfNote *out)
{
    /* name_size and desc_size include the NUL terminator in the encoding */
    uint32_t name_size_raw = bufreader_read_u32(br);
    uint32_t desc_size_raw = bufreader_read_u32(br);
    uint32_t note_type = bufreader_read_u32(br);
    if (br->error) {
        return -1;
    }

    uint32_t name_size = name_size_raw - 1;
    uint32_t desc_size = desc_size_raw - 1;

    char *name = bufreader_read_str(br, name_size, 1 /* has NUL */);
    if (name == NULL) {
        return -1;
    }
    /* name read_size = name_size + 1 (the NUL), for unpad */
    bufreader_unpad(br, name_size + 1, 4);

    char *desc = bufreader_read_str(br, desc_size, 1 /* has NUL */);
    if (desc == NULL) {
        PyMem_RawFree(name);
        return -1;
    }
    bufreader_unpad(br, desc_size + 1, 4);

    jit_elf_note_init(out);
    jit_elf_note_set_bin(out, name, desc, desc_size, note_type);
    PyMem_RawFree(name);
    PyMem_RawFree(desc);
    return 0;
}

/* ---- Public API ---- */

int
jit_elf_find_section(const uint8_t *elf_data, size_t elf_size,
                     const char *name,
                     const uint8_t **out_data, size_t *out_size)
{
#ifdef ENABLE_ELF_READER
    if (elf_size < sizeof(ElfW(Ehdr))) {
        return -1;
    }
    const ElfW(Ehdr) *ehdr = (const ElfW(Ehdr) *)elf_data;
    if (ehdr->e_shoff + (size_t)ehdr->e_shnum * sizeof(ElfW(Shdr)) > elf_size) {
        return -1;
    }
    const ElfW(Shdr) *shdrs = (const ElfW(Shdr) *)(elf_data + ehdr->e_shoff);

    /* Find .shstrtab so we can read section names */
    if (ehdr->e_shstrndx >= ehdr->e_shnum) {
        return -1;
    }
    const ElfW(Shdr) *shstrtab = &shdrs[ehdr->e_shstrndx];
    if (shstrtab->sh_offset + shstrtab->sh_size > elf_size) {
        return -1;
    }
    const char *strtab = (const char *)(elf_data + shstrtab->sh_offset);

    for (uint16_t i = 0; i < ehdr->e_shnum; i++) {
        if (shdrs[i].sh_name >= shstrtab->sh_size) {
            continue;
        }
        const char *section_name = strtab + shdrs[i].sh_name;
        if (strcmp(section_name, name) == 0) {
            if (shdrs[i].sh_offset + shdrs[i].sh_size > elf_size) {
                return -1;
            }
            *out_data = elf_data + shdrs[i].sh_offset;
            *out_size = shdrs[i].sh_size;
            return 0;
        }
    }

    /* Section not found — not an error, just empty result */
    *out_data = NULL;
    *out_size = 0;
    return 0;
#else
    (void)elf_data;
    (void)elf_size;
    (void)name;
    (void)out_data;
    (void)out_size;
    return -1;
#endif
}

int
jit_elf_read_note_section(const uint8_t *data, size_t size,
                          JitElfNoteArray *out)
{
    jit_elf_note_array_init(out);
    BufReader br;
    bufreader_init(&br, data, size);

    while (bufreader_good(&br)) {
        JitElfNote note;
        if (read_note(&br, &note) != 0) {
            jit_elf_note_array_free(out);
            return -1;
        }
        jit_elf_note_array_insert_bin(out, note.name, note.desc, note.desc_len,
                                     note.type);
        jit_elf_note_free(&note);
    }
    return 0;
}

int
jit_elf_parse_code_note(const JitElfNote *note, JitElfCodeNoteData *out)
{
    jit_elf_code_note_data_init(out);

    BufReader br;
    bufreader_init(&br, (const uint8_t *)note->desc, note->desc_len);

    uint32_t file_name_size = bufreader_read_u32(&br);
    if (br.error) {
        return -1;
    }

    char *file_name = bufreader_read_str(&br, file_name_size,
                                         0 /* no NUL terminator */);
    if (file_name == NULL) {
        return -1;
    }
    out->file_name = file_name;

    out->lineno = bufreader_read_u32(&br);
    out->hash = bufreader_read_u32(&br);
    out->size = bufreader_read_u32(&br);
    out->normal_entry_offset = bufreader_read_u32(&br);
    uint32_t static_offset = bufreader_read_u32(&br);
    if (br.error) {
        jit_elf_code_note_data_free(out);
        return -1;
    }

    if (static_offset != JIT_ELF_INVALID_STATIC_OFFSET) {
        out->static_entry_offset = static_offset;
        out->has_static_entry = 1;
    } else {
        out->static_entry_offset = JIT_ELF_INVALID_STATIC_OFFSET;
        out->has_static_entry = 0;
    }
    return 0;
}
