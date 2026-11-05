#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Writes the two headers that make a companion of the verifier: copies of blocks/iop/message.h and
blocks/iop/req_rows.h with small insertions. The sources in the repository stay as they are, and
the stock cp_verify is built without these. usage: make_shadow.py <repository> <out directory>"""
import os, sys
root, out = sys.argv[1], sys.argv[2]
os.makedirs(out + "/iop", exist_ok=True)
s = open(root + "/blocks/iop/message.h").read()
old = """  Reader(const u8 *p, size_t n) : p_(p), end_(p + n) {}
"""
new = """  Reader(const u8 *p, size_t n) : p_(p), end_(p + n) {
    if (n == fpstream::MARK_BYTES && fpstream::companion_map()) { // the proof comes through the stream
      stream_ = fpstream::companion_map();
      p_ = stream_ + fpstream::PROOF_AT, end_ = p_;
    }
  }
"""
assert old in s; s = s.replace(old, new)
old = """  bool at_end() const { return ok_ && p_ == end_; }
"""
new = """  bool at_end() const {
    if (stream_) {
      fpstream::Head *h = fpstream::head(stream_);
      while (!h->closed.load() && (u64)(p_ - stream_ - fpstream::PROOF_AT) == h->written.load()) usleep(200);
      return ok_ && h->closed.load() && (u64)(p_ - stream_ - fpstream::PROOF_AT) == h->written.load();
    }
    return ok_ && p_ == end_;
  }
"""
assert old in s; s = s.replace(old, new)
old = """  const u8 *take(size_t n) {
    if (!ok_ || (size_t)(end_ - p_) < n) {"""
new = """  const u8 *take(size_t n) {
    if (stream_) {
      fpstream::Head *h = fpstream::head(stream_);
      const u64 want = (u64)(p_ - stream_ - fpstream::PROOF_AT) + n;
      while (h->written.load() < want && !h->closed.load()) usleep(200);
      end_ = stream_ + fpstream::PROOF_AT + h->written.load();
    }
    if (!ok_ || (size_t)(end_ - p_) < n) {"""
assert old in s; s = s.replace(old, new)
old = """  const u8 *p_, *end_;
  bool ok_ = true;
};"""
new = """  const u8 *p_, *end_;
  bool ok_ = true;
  u8 *stream_ = nullptr;
};"""
assert old in s; s = s.replace(old, new)
s = s.replace("namespace blk {", '#include <unistd.h>\n#include "stream.h"\nnamespace blk {', 1)
open(out + "/iop/message.h", "w").write(s)

s = open(root + "/blocks/iop/req_rows.h").read()
old = "  Row row(const ReqEquation &eq) const {\n"
new = """  Row row(const ReqEquation &eq) const {
    fp_export_equation(eq);
"""
assert old in s; s = s.replace(old, new)
old = "// Builds a segment's linear claim."
new = '''// The companion's export of a segment's equation to fp_prove.
inline void fp_put(u8 *&w, const void *p, size_t n) { memcpy(w, p, n), w += n; }
inline void fp_put_u64(u8 *&w, u64 x) { fp_put(w, &x, 8); }
inline void fp_put_es(u8 *&w, const std::vector<E> &v) { fp_put_u64(w, v.size()); if (!v.empty()) fp_put(w, v.data(), v.size() * sizeof(E)); }
inline void fp_put_iw(u8 *&w, const IndexWeights &x) {
  fp_put_es(w, x.lo), fp_put_es(w, x.hi);
  fp_put_u64(w, x.values ? 1 : 0);
  if (x.values) fp_put_es(w, *x.values);
}
inline void fp_export_equation(const ReqEquation &eq) {
  u8 *m = fpstream::companion_map();
  if (!m) return;
  fpstream::Head *h = fpstream::head(m);
  u8 *start = m + fpstream::EQ_AT + h->eq_bytes.load(), *w = start;
  fp_put_u64(w, eq.terms.size());
  for (const ReqTerm &t : eq.terms) {
    fp_put_u64(w, t.born), fp_put_u64(w, t.array.size()), fp_put(w, t.array.data(), t.array.size());
    fp_put(w, &t.scale, sizeof(E));
    fp_put_iw(w, t.on_rows), fp_put_iw(w, t.on_columns);
  }
  fp_put_es(w, eq.on_masks);
  fp_put(w, &eq.target, sizeof(E));
  h->eq_bytes.fetch_add((u64)(w - start));
  h->eq_count.fetch_add(1);
}
// Builds a segment's linear claim.'''
assert old in s; s = s.replace(old, new, 1)
s = s.replace('#pragma once\n', '#pragma once\n#include <cstring>\n#include "stream.h"\n', 1)
open(out + "/iop/req_rows.h", "w").write(s)
