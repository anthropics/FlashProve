// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0
// blk::LkpCpu (blocks/iop/lkp.h) with its trees and sumchecks on Metal, and blk::WfCpu with that lookup.
// The summands of a lookup are known by their step, so each is given to the kernels as monomials:
// KEYS, TABLE y0 y1; BITS y0^2 - y0; LEAVES 1 + b R, R being the table's factor, a map of the same index.
#pragma once
#include "iop/lkp.h"
#include "iop/wf.h"
#include "mlx_kernels.h"
namespace fpmlx {
using namespace blk;
class SumcheckMlx : public SumcheckProver<E> {
public:
  template <class Maps> SumcheckMlx(Maps &&maps, unsigned count, SumShape shape, const std::vector<Weight<E>> &vars, const std::vector<unsigned> &order, size_t ends)
      : rounds_(make(std::forward<Maps>(maps), count, std::move(shape), vars, order)), vars_(vars), bits_(order), ends_(ends) {}
  std::vector<E> round() override {
    BLK_REQUIRE(j_ < vars_.size(), "sum-check: no variable is left");
    return poly_from_values_at_0_1_2<E>(rounds_->nodes(j_, vars_[j_].d, size_t(1) << bits_[j_]));
  }
  void bind(const E &g) override {
    BLK_REQUIRE(j_ < vars_.size(), "sum-check: no variable is left");
    const unsigned b = bits_[j_];
    rounds_->bind(g, size_t(1) << b);
    for (size_t t = ++j_; t < bits_.size(); t++) if (bits_[t] > b) bits_[t]--;
  }
  std::vector<E> end_values() override {
    BLK_REQUIRE(j_ == vars_.size(), "sum-check: variables are left");
    std::vector<std::vector<E>> m;
    rounds_->fetch(m);
    std::vector<E> v;
    for (size_t i = 0; i < ends_; i++) v.push_back(m[i][0]);
    return v;
  }
private:
  static std::vector<SumWeight> later_of(const std::vector<Weight<E>> &vars, std::vector<unsigned> at) {
    std::vector<SumWeight> later;
    for (size_t l = 0; l + 1 < vars.size(); l++) {
      for (size_t t = l + 1; t < at.size(); t++) if (at[t] > at[l]) at[t]--;
      later.push_back({vars[l + 1].moment[0] - vars[l + 1].moment[1], vars[l + 1].moment[1], at[l + 1]});
    }
    return later;
  }
  template <class Maps> static std::unique_ptr<SumRounds> make(Maps &&maps, unsigned count, SumShape shape, const std::vector<Weight<E>> &vars, const std::vector<unsigned> &order) {
    if constexpr (std::is_same_v<std::decay_t<Maps>, mx::array>) return std::make_unique<SumRounds>(maps, count, std::move(shape), later_of(vars, order));
    else return std::make_unique<SumRounds>(maps, std::move(shape), later_of(vars, order));
  }
  std::unique_ptr<SumRounds> rounds_;
  std::vector<Weight<E>> vars_;
  std::vector<unsigned> bits_;
  size_t ends_, j_ = 0;
};
class LkpMlx : public LkpProver {
public:
  LkpMlx(const LkpPublic &fixed, std::vector<E> keys, std::vector<E> bits) : fixed_(fixed), keys_(std::move(keys)), bits_(std::move(bits)) {
    BLK_REQUIRE(keys_.size() == fixed.all_keys() && bits_.size() == size_t(1) << (fixed.mu + fixed.d), "LKP: keys or bits have not the announced number");
  }
  void begin(const E &theta, const E &alpha) override {
    alpha_ = alpha;
    key_levels_.clear();
    size_t first = 0;
    for (unsigned part = 0; part < fixed_.part_bits().size(); part++) {
      const unsigned v = fixed_.part_bits()[part];
      std::vector<E> leaves(size_t(1) << v);
      for (size_t i = 0; i < leaves.size(); i++) {
        const LkpKeyBlock &b = lkp_block_of(fixed_, i);
        leaves[i] = alpha - (b.function < 0 ? lkp_row(fixed_, theta, b.padding_row) : keys_[first + i] + E::from(from_u64((u64)b.function)) * theta);
      }
      for (const LkpPadding &p : fixed_.padding)
        if (p.part == part)
          for (u64 i = 0; i < p.length; i++) leaves[p.first + (i << p.log_step)] = alpha - lkp_row(fixed_, theta, p.row);
      key_levels_.push_back(tree(leaves, 1));
      first += size_t(1) << v;
    }
    table_ = lkp_rows(fixed_, theta);
    const size_t w = size_t(1) << fixed_.mu;
    std::vector<E> table_leaves(bits_.size());
    for (size_t i = 0; i < bits_.size(); i++) table_leaves[i] = E::one() + bits_[i] * (alpha - E::one() - table_[i / w]);
    table_levels_ = tree(table_leaves, w);
  }
  std::vector<E> key_roots() override {
    std::vector<E> roots;
    for (const auto &levels : key_levels_) roots.push_back(e_fetch(levels[0])[0]);
    return roots;
  }
  std::vector<E> plane_roots() override { return e_fetch(table_levels_[0]); }
  std::unique_ptr<SumcheckProver<E>> sumcheck(LkpStep step, unsigned part, unsigned level, const Summand<E> &, const std::vector<Weight<E>> &vars, const std::vector<unsigned> &order) override {
    SumShape shape;
    if (step == LkpStep::KEYS || step == LkpStep::TABLE) {
      shape.coeff = {E::one()}, shape.maps = {{0, 1}};
      const mx::array &below = (step == LkpStep::KEYS ? key_levels_[part] : table_levels_)[level + 1];
      return std::make_unique<SumcheckMlx>(below, 2, std::move(shape), vars, order, 2);
    }
    std::vector<std::vector<E>> maps = {bits_};
    if (step == LkpStep::BITS) shape.coeff = {E::one(), E::zero() - E::one()}, shape.maps = {{0, 0}, {0}};
    else {
      const size_t w = size_t(1) << fixed_.mu;
      std::vector<E> R(bits_.size());
      for (size_t i = 0; i < R.size(); i++) R[i] = alpha_ - E::one() - table_[i / w];
      maps.push_back(std::move(R));
      shape.coeff = {E::one(), E::one()}, shape.maps = {{}, {0, 1}};
    }
    return std::make_unique<SumcheckMlx>(maps, 0, std::move(shape), vars, order, 1);
  }
private:
  // levels[l] from the top down; levels[0] has `top` entries.
  static std::vector<mx::array> tree(const std::vector<E> &leaves, size_t top) {
    std::vector<mx::array> up(1, e_on_gpu(leaves));
    for (size_t n = leaves.size(); n > top; n /= 2) up.push_back(product_level(up.back()));
    std::reverse(up.begin(), up.end());
    return up;
  }
  LkpPublic fixed_;
  std::vector<E> keys_, bits_, table_;
  E alpha_ = E::zero();
  std::vector<std::vector<mx::array>> key_levels_;
  std::vector<mx::array> table_levels_;
};
class WfMlx : public WfCpu {
public:
  WfMlx(const WfPublic &x, const WfMatrix &m, std::function<F(u64 row, u64 column)> at) : WfCpu(x, m, std::move(at)), fixed_(wf_lookups(x, m)) {}
  LkpProver &prover(size_t lookup) override {
    std::vector<u64> count;
    BLK_REQUIRE(multiplicities(lookup, count), "WF: a key in no row of its table");
    const LkpPublic &p = fixed_[lookup];
    std::vector<E> bits(size_t(1) << (p.mu + p.d), E::zero());
    for (size_t t = 0; t < count.size(); t++)
      for (size_t j = 0; j < p.planes(); j++)
        if ((count[t] >> j) & 1) bits[j + (t << p.mu)] = E::one();
    now_.reset(new LkpMlx(p, keys(lookup), std::move(bits)));
    return *now_;
  }
private:
  const std::vector<LkpPublic> fixed_;
  std::unique_ptr<LkpMlx> now_;
};
}
