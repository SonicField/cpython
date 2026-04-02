/*
 * mmap_file.c -- Memory-mapped file reader (pure C)
 *
 * Phase 3D conversion: mmap_file.cpp -> mmap_file.c
 * Provides mmap-based read-only file access for ELF AOT loading.
 */

#include "cinderx/Jit/mmap_file.h"

#include <errno.h>
#include <stdio.h>
#include <fcntl.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

void
jit_mmap_file_init(JitMmapFile *f)
{
    f->data = NULL;
    f->size = 0;
}

int
jit_mmap_file_open(JitMmapFile *f, const char *filename,
                   char *errbuf, size_t errbuf_size)
{
    if (f->data != NULL) {
        snprintf(errbuf, errbuf_size,
                 "Trying to mmap %s on top of an existing file object",
                 filename);
        return -1;
    }

    int fd = open(filename, O_RDONLY);
    if (fd == -1) {
        snprintf(errbuf, errbuf_size, "Could not open %s: %s",
                 filename, strerror(errno));
        return -1;
    }

    struct stat statbuf;
    int stat_result = fstat(fd, &statbuf);
    if (stat_result == -1) {
        snprintf(errbuf, errbuf_size, "Could not stat %s: %s",
                 filename, strerror(errno));
        close(fd);
        return -1;
    }

    off_t signed_size = statbuf.st_size;
    if (signed_size < 0) {
        snprintf(errbuf, errbuf_size,
                 "Stat'd a size of %lld for file %s",
                 (long long)signed_size, filename);
        close(fd);
        return -1;
    }

    size_t size = (size_t)signed_size;
    void *data = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);  /* fd not needed after mmap */

    if (data == MAP_FAILED) {
        snprintf(errbuf, errbuf_size, "Failed to mmap file %s: %s",
                 filename, strerror(errno));
        return -1;
    }

    f->data = (const uint8_t *)data;
    f->size = size;
    return 0;
}

int
jit_mmap_file_close(JitMmapFile *f)
{
    if (f->data == NULL) {
        return 0;
    }
    int result = munmap((void *)f->data, f->size);
    f->data = NULL;
    f->size = 0;
    return result == 0 ? 0 : -1;
}
