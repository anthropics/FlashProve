// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0
// The memory that fp_prove shares with its companion, a build of the verifier that walks a proof while
// it is being written (mlx/DESIGN_PROVER.md). One file, mapped by both.
//   header: written (bytes of proof available), closed (no more will come), eq_bytes, eq_count
//   proof bytes at PROOF_AT; equations, serialized, at EQ_AT.
#pragma once
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
namespace fpstream {
constexpr uint64_t PROOF_AT = 4096, EQ_AT = 1ull << 30, SIZE = 1ull << 33;
// A proof file of this length is the companion's sign that the proof comes through the stream.
constexpr uint64_t MARK_BYTES = 1048583;
struct Head { std::atomic<uint64_t> written, closed, eq_bytes, eq_count; };
inline uint8_t *map(const char *path, bool create) {
  const int fd = open(path, create ? O_RDWR | O_CREAT | O_TRUNC : O_RDWR, 0600);
  if (fd < 0) return nullptr;
  if (create && ftruncate(fd, (off_t)SIZE)) return nullptr;
  void *p = mmap(nullptr, SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  close(fd);
  return p == MAP_FAILED ? nullptr : (uint8_t *)p;
}
inline Head *head(uint8_t *m) { return reinterpret_cast<Head *>(m); }
inline uint8_t *companion_map() {
  static uint8_t *m = [] { const char *e = getenv("FP_STREAM"); return e ? map(e, false) : nullptr; }();
  return m;
}
}
