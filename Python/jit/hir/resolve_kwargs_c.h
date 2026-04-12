#pragma once
#include "cinderx/Jit/hir/hir_c_api.h"
#ifdef __cplusplus
extern "C" {
#endif
void hir_resolve_kwargs_run(HirFunction func);
#ifdef __cplusplus
}
#endif
