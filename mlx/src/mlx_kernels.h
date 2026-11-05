// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0
// Field arithmetic on MLX arrays of uint64. Elements of F are canonical words; an array of E has a
// last axis of 3 coordinates.
#pragma once
#include <mlx/mlx.h>
#include <memory>
#include <string>
#include <vector>
#include "field/field.h"
#include "hash/hash.h"
#include "tree/opening.h"
namespace fpmlx {
namespace mx = mlx::core;
std::string read_kernel_header(const char *name);
mx::array f_binary(const char *op, const mx::array &a, const mx::array &b); // "add" "sub" "mul"
mx::array e_mul(const mx::array &a, const mx::array &b);
mx::array e_upload(const std::vector<blk::E> &v);
std::vector<blk::E> e_download(const mx::array &a);
// rows x 2^log_dim coefficients to rows x 2^log_n symbols: Encoder::encode() on every row.
mx::array encode_rows(const blk::u64 *coefficients, size_t rows, unsigned log_dim, unsigned log_n, const blk::F *tw);
blk::E dot(const mx::array &a, const mx::array &b);
void fold_round(const mx::array &c, const mx::array &m, size_t rows, size_t len, blk::E q[3]);
mx::array halve(const mx::array &t, size_t rows, size_t len, const blk::E &g);
blk::u64 search_nonce(const blk::ChainPrefix &prefix, unsigned bits);
// blk::OracleTree with its leaves and vertices hashed on the GPU. The same draws from `coins`.
class OracleTreeMlx {
public:
  OracleTreeMlx(const blk::OracleTreeShape &s, blk::Coins &coins);
  blk::Digest build(const std::vector<mx::array> &main, const std::vector<mx::array> &mask);
  template <class Fetch> void open(const blk::PositionList &list, blk::Opening &o, Fetch fetch) const {
    (void)blk::opening_given(s_, list);
    o = blk::Opening();
    o.main_width = s_.main_width(), o.mask_width = s_.mask_width();
    std::vector<blk::F> column;
    for (size_t i = 0; i < list.size(); i++) {
      fetch(list[i], column);
      BLK_REQUIRE(column.size() == s_.width(), "oracle tree: a fetched column has not the leaf's size");
      o.columns.insert(o.columns.end(), column.begin(), column.end());
      o.salts.insert(o.salts.end(), salt_.begin() + list[i] * s_.salt_bytes, salt_.begin() + (list[i] + 1) * s_.salt_bytes);
    }
    struct Side {
      struct Value {};
      const OracleTreeMlx &t;
      std::vector<blk::Digest> out;
      Value leaf(size_t) { return {}; }
      bool given(unsigned d, blk::u64 place, Value &) { out.push_back(t.digest(d, place)); return true; }
      Value node(unsigned, blk::u64, const Value &, const Value &) { return {}; }
    } side{*this, {}};
    typename Side::Value root;
    blk::tree_walk(s_.tree(), list, side, root);
    o.given = std::move(side.out);
  }
private:
  blk::Digest digest(unsigned depth, blk::u64 place) const;
  blk::OracleTreeShape s_;
  std::vector<blk::u8> salt_;
  std::vector<mx::array> level_; // level_[d]: 2^d digests of 4 words
};
}
