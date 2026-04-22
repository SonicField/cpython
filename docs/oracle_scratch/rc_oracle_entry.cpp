// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// W3 R4 ORACLE — C-linkage entry point for libphoenix_rc_oracle.a.
//
// compiler.cpp's #ifdef RC_ORACLE dispatcher (Step 3.5) calls this symbol
// when env RC_ORACLE_USE_CXX is set. Wraps the cc4a18e7e5 RefcountInsertion
// C++ pass class so the dispatcher only needs an extern "C" forward decl
// — no C++ header dependency in compiler.cpp.
//
// Per supervisor 2026-04-22 02:37:10Z hybrid design + theologian 02:39:08Z
// dispatcher spec.

#include "rc_oracle.h"  // jit::hir::RefcountInsertion from cc4a18e7e5

extern "C" int rc_oracle_run(void *func) {
  auto *irfunc = reinterpret_cast<jit::hir::Function *>(func);
  jit::hir::RefcountInsertion pass;
  pass.Run(*irfunc);
  return 0;
}
