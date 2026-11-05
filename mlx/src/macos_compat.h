// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0
// getrandom(2) for macOS, where blocks/ asks for it: the system's getentropy().
#pragma once
#include <sys/random.h>
#include <sys/types.h>
#include <unistd.h>
#ifdef __cplusplus
static inline ssize_t getrandom(void *p, size_t n, unsigned) {
  size_t done = 0;
  while (done < n) {
    const size_t k = n - done > 256 ? 256 : n - done;
    if (getentropy((char *)p + done, k)) return done ? (ssize_t)done : -1;
    done += k;
  }
  return (ssize_t)n;
}
#endif
