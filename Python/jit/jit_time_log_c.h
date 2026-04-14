// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Parse a comma-separated list of function name patterns for compilation
// time capture. Replaces any previously set patterns.
void jit_time_parse_func_list(const char* flag_value);

// Check if function_name matches any captured pattern set via
// jit_time_parse_func_list.
int jit_time_capture_for(const char* function_name);

// Glob-style pattern matching: '?' matches any single character,
// '*' matches any sequence (including empty).
int jit_time_is_match(const char* word, const char* pattern);

#ifdef __cplusplus
}
#endif
