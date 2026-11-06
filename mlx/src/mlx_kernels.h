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
// Words that an engine keeps: an evaluated MLX array, whose memory the CPU reads too. Metal does not
// survive fork(): in a forked child (the library's tests that check that a misuse ends the process)
// a Buf holds host words and each operation below takes a plain host path. An array that a child
// inherits is read, not released.
bool in_forked_child();
class Buf {
public:
  Buf() = default;
  explicit Buf(const mx::array &a);
  explicit Buf(std::vector<blk::u64> host);
  const blk::u64 *data() const { return p_; }
  size_t size() const { return n_; }
  bool on_gpu() const { return (bool)a_; }
  const mx::array &array() const { return *a_; }
private:
  std::shared_ptr<mx::array> a_;
  std::shared_ptr<std::vector<blk::u64>> h_;
  const blk::u64 *p_ = nullptr;
  size_t n_ = 0;
};
Buf e_upload(const std::vector<blk::E> &v);
std::vector<blk::E> e_download(const Buf &a);
// rows x 2^log_dim coefficients to rows x 2^log_n symbols: Encoder::encode() on every row.
Buf encode_rows(const blk::u64 *coefficients, size_t rows, unsigned log_dim, unsigned log_n, const blk::F *tw);
blk::E dot(const Buf &a, const Buf &b);
void fold_round(const Buf &c, const Buf &m, size_t rows, size_t len, blk::E q[3]);
Buf halve(const Buf &t, size_t rows, size_t len, const blk::E &g);
blk::u64 search_nonce(const blk::ChainPrefix &prefix, unsigned bits);
// The rounds of a sumcheck, its maps on the GPU. The summand is compiled into a kernel by its shape.
struct SumShape {
  std::vector<blk::E> coeff;              // by monomial
  std::vector<std::vector<unsigned>> maps; // by monomial: indices of the maps multiplied
};
// The weight of a later round: (lo, hi) at the place `at` of the index that the round before it sees.
struct SumWeight { blk::E lo, hi; unsigned at = 0; };
class SumRounds {
public:
  // Takes the maps' values: `maps` are empty afterwards, until fetch(). later[l]: the weight that round
  // l + 1 adds to round l's table; rounds = later.size() + 1.
  SumRounds(std::vector<std::vector<blk::E>> &maps, SumShape shape, const std::vector<SumWeight> &later);
  // g at the nodes 0 .. d - 1 of round l: the sum over the pairs along `step`, with the round's weights.
  std::vector<blk::E> nodes(size_t l, unsigned d, size_t step);
  void bind(const blk::E &r, size_t step);
  void fetch(std::vector<std::vector<blk::E>> &maps) const;
private:
  SumShape shape_;
  std::vector<unsigned> used_;   // the maps that the summand reads
  mx::array m_;                  // map k's values at k * size_
  std::vector<mx::array> w_;     // w_[l]: round l's table of weights
  mx::array coeff_, program_;    // program_: for a summand too long to compile, its monomials as data
  size_t count_, size_;
};
// blk::OracleTree with its leaves and vertices hashed on the GPU. The same draws from `coins`.
class OracleTreeMlx {
public:
  OracleTreeMlx(const blk::OracleTreeShape &s, blk::Coins &coins);
  blk::Digest build(const std::vector<Buf> &main, const std::vector<Buf> &mask);
  // A table of main rows only, in chunks of equal numbers of rows: no copy joins them.
  blk::Digest build_chunks(const std::vector<Buf> &chunks, size_t rows_each);
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
  blk::Digest levels(const mx::array &leaves);
  blk::Digest digest(unsigned depth, blk::u64 place) const;
  blk::OracleTreeShape s_;
  std::vector<blk::u8> salt_;
  std::vector<Buf> level_; // level_[d]: 2^d digests of 4 words
};
}
