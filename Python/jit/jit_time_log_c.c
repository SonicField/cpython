// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// Pure C implementation of JIT compilation time log pattern matching.

#include "cinderx/Jit/jit_time_log_c.h"

#include <string.h>

#define JIT_TIME_MAX_PATTERNS 128
#define JIT_TIME_MAX_PATTERN_LEN 512

static struct {
  char patterns[JIT_TIME_MAX_PATTERNS][JIT_TIME_MAX_PATTERN_LEN];
  int count;
} s_capture_patterns;

static int
is_match_impl(const char* word, int n, int wlen, const char* pat, int m,
              int plen) {
  if (m == plen) {
    return n == wlen;
  }
  if (n == wlen) {
    for (int i = m; i < plen; i++) {
      if (pat[i] != '*') {
        return 0;
      }
    }
    return 1;
  }
  if (pat[m] == '?' || pat[m] == word[n]) {
    return is_match_impl(word, n + 1, wlen, pat, m + 1, plen);
  }
  if (pat[m] == '*') {
    return is_match_impl(word, n + 1, wlen, pat, m, plen) ||
        is_match_impl(word, n, wlen, pat, m + 1, plen);
  }
  return 0;
}

int jit_time_is_match(const char* word, const char* pattern) {
  return is_match_impl(
      word, 0, (int)strlen(word), pattern, 0, (int)strlen(pattern));
}

void jit_time_parse_func_list(const char* flag_value) {
  s_capture_patterns.count = 0;
  if (!flag_value || !*flag_value) {
    return;
  }
  const char* p = flag_value;
  while (*p && s_capture_patterns.count < JIT_TIME_MAX_PATTERNS) {
    const char* start = p;
    while (*p && *p != ',') {
      p++;
    }
    int len = (int)(p - start);
    if (len > 0 && len < JIT_TIME_MAX_PATTERN_LEN) {
      memcpy(
          s_capture_patterns.patterns[s_capture_patterns.count], start, len);
      s_capture_patterns.patterns[s_capture_patterns.count][len] = '\0';
      s_capture_patterns.count++;
    }
    if (*p == ',') {
      p++;
    }
  }
}

int jit_time_capture_for(const char* function_name) {
  for (int i = 0; i < s_capture_patterns.count; i++) {
    if (jit_time_is_match(function_name, s_capture_patterns.patterns[i])) {
      return 1;
    }
  }
  return 0;
}
