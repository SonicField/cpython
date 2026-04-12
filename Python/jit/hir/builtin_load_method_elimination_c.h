#pragma once
#include "cinderx/Jit/hir/hir_c_api.h"
#ifdef __cplusplus
extern "C" {
#endif
void hir_builtin_load_method_elimination_run(HirFunction func);
#ifdef __cplusplus
}
#endif
