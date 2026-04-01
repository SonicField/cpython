/* code_section.c -- Code section name mapping (pure C)
 *
 * Phase 3D conversion: code_section.cpp -> code_section.c
 * Provides codeSectionName and codeSectionFromName as C functions.
 * populateCodeSections stays in code_section.h (C++ inline, uses asmjit types).
 */

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

const char*
jit_code_section_name(int section) {
    switch (section) {
    case 0: /* kHot */
        return ".text";
    case 1: /* kCold */
        return ".coldtext";
    }
    fprintf(stderr, "JIT: %s:%d -- Bad code section %d\n",
            __FILE__, __LINE__, section);
    abort();
}

int
jit_code_section_from_name(const char *name) {
    if (strcmp(name, ".text") == 0 || strcmp(name, ".addrtab") == 0) {
        return 0; /* kHot */
    }
    if (strcmp(name, ".coldtext") == 0) {
        return 1; /* kCold */
    }
    fprintf(stderr, "JIT: %s:%d -- Bad code section name %s\n",
            __FILE__, __LINE__, name);
    abort();
}
