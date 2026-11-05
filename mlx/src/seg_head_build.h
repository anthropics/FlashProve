// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0
// The witness of the output head (docs/format/11-4-output-head.md, second half): the matrices of cells,
// each position's row with its 21 further digits, and the keys of the lookup of step 1. The honest
// rule for a position's cost is the one of prover/head/forward.cuh (k_head_price, k_head_budget), which
// the format leaves open. Five limbs only (NR = 99), the step table only.
#pragma once
#include "seg_ff_witness.h"
namespace fpmlx {
// The request of 11.4: T tokens, then r and e, as T + 2 words; y: the next token.
struct HeadRequest {
  std::vector<u32> tokens;
  u32 r = 0, e = 0, y = 0;
  static HeadRequest read(const std::string &bytes, unsigned T) {
    if (bytes.size() != 4ull * (T + 2)) throw std::runtime_error("request: not T + 2 words");
    HeadRequest q;
    q.tokens.resize(T);
    memcpy(q.tokens.data(), bytes.data(), 4ull * T);
    memcpy(&q.r, bytes.data() + 4ull * T, 4);
    memcpy(&q.e, bytes.data() + 4ull * T + 4, 4);
    return q;
  }
};
struct HeadShape2 {
  int lt, T, d, V, la, ls, S2, lPd, JV, JVa, XL, NR, NRH, NS, NPr, NO;
  int lmhf, NHf, lmhd, lmhr, lmbH, lmx, nmin, MF, KF, sh, sT, F, lkW, tau, lidb;
  long long C, theta, cap;
  long long offE() const { return 8421505; }
};
inline HeadShape2 head_shape(const Policy &p, const ModelShape &m) {
  HeadShape2 s{};
  s.lt = (int)p.get("lt"), s.T = 1 << s.lt, s.d = (int)p.get("d"), s.V = (int)p.get("V"), s.la = (int)p.get("la"), s.ls = s.la - s.lt, s.S2 = 1 << s.ls;
  s.lPd = (int)clog2u(s.d), s.JV = (s.V + s.S2 - 1) / s.S2 * s.S2, s.JVa = 1 << (clog2u(s.JV) - 1);
  s.XL = (int)p.get("xl"), s.NR = 99, s.NRH = s.NR + 21, s.NS = 2 * s.XL - 1, s.NPr = 7, s.NO = 4;
  if (s.XL != 5) throw std::runtime_error("head: five limbs only");
  s.lmhf = (int)p.get("lmhf"), s.NHf = (int)p.get("NHf"), s.lmhd = (int)p.get("lmhd"), s.lmhr = (int)p.get("lmhr"), s.lmbH = (int)p.get("lmbH"), s.lmx = (int)p.get("lmx");
  s.nmin = p.v.count("nminH") ? (int)p.v.at("nminH") : (int)p.get("nmin");
  s.MF = m.MF, s.KF = m.KF, s.sh = (int)p.get("sh"), s.sT = (int)p.get("sT"), s.F = (int)p.get("F");
  if (s.sT + 4 != s.sh) throw std::runtime_error("head: the step table only");
  s.lkW = (int)p.get("table_w"), s.tau = (int)p.get("tau"), s.lidb = (int)p.get("lidb");
  s.C = p.get("C"), s.theta = p.get("theta"), s.cap = p.v.at("cap");
  return s;
}
// The table of exponentials with N = 484.
inline std::vector<u32> head_ex_table(int F) {
  std::vector<u32> EX(65536);
  unsigned __int128 G = (unsigned __int128)1 << (F + 17);
  for (u32 &x : EX) {
    x = (u32)((G + 131071) >> 17);
    G = (G * 484 + 0xFFFFFFFFull) >> 32;
  }
  return EX;
}
struct HeadWitness {
  HeadShape2 s;
  HeadRequest q;
  FeatMat hf, hd, hr, xbL; // xbL: the cells of the last group's xbO
  CellMatrix2 bH;
  std::vector<LookupCall> calls;
  std::vector<std::map<std::string, SegPlace>> place; // by call
  std::vector<int> lJ;
  LookupSizes sizes;
  std::vector<std::vector<u32>> mult;
  std::vector<NormRow> rows;                  // by position
  std::vector<I64> act, str;                  // [t * d + k]
  std::vector<I64> lam;                       // [t * V + v]
  std::vector<u32> win;                       // by position; 0 where on_t = 0
  std::vector<std::array<I64, 6>> hidden;     // gc, gb, gu, gk, gz, gs by position
  std::vector<u32> EX;
  std::vector<char> on, co;
  // False: the witness is no honest one, and the proof would be refused.
  bool norm_ok = true, margins_ok = true, bounds_ok = true, keys_ok = false;
  std::vector<I64> price;
  int call_of(const std::string &seg) const { for (size_t c = 0; c < place.size(); c++) if (place[c].count(seg)) return (int)c; throw std::runtime_error("no call holds " + seg); }
};
inline u64 head_seg_len(const HeadShape2 &s, const std::string &n) {
  if (n == "NF" || n == "NFH" || n == "CNF") return s.d;
  if (n == "HD0" || n == "HD1" || n == "HD2") return s.JV;
  if (n == "ROW") return s.NRH;
  if (n == "HE0") return s.JVa;
  if (n == "HE1") return s.JV - s.JVa;
  throw std::runtime_error("no such segment " + n);
}
// y = the lowest index of the largest logit at T - 1 if e = T, else 0.
inline u32 head_next_token(const Values &V, const HeadRequest &q, unsigned T) { return q.e == T ? (u32)V.get("first_choice", T - 1, 0) : 0; }
inline HeadWitness head_build(const Policy &pol, const ModelShape &m, const Values &V, const CallTables &tabs, const HeadRequest &q) {
  HeadWitness R;
  R.s = head_shape(pol, m), R.q = q;
  const HeadShape2 &s = R.s;
  const int T = s.T, d = s.d, Vn = s.V, JV = s.JV, JVa = s.JVa;
  R.EX = head_ex_table(s.F);
  R.hf.init(s.NHf, s.lmhf, 4ll * d + 6ll * JV, s.lt);
  R.hd.init(1, s.lmhd, 2ll * d, s.lt);
  R.hr.init(1, s.lmhr, s.NRH, s.lt);
  R.xbL.init(1, s.lmx, (long long)s.XL * d, s.lt);
  // The calls: pieces, columns, tables.
  const size_t NC = pol.callH.size();
  R.calls.resize(NC), R.place.resize(NC), R.lJ.resize(NC);
  std::vector<u64> FE(65536);
  for (u64 u = 0; u < 65536; u++) FE[u] = u + ((u64)R.EX[u] << 16);
  for (size_t c = 0; c < NC; c++) {
    struct Piece { std::string seg; u64 first, size; };
    std::vector<Piece> pieces;
    for (const std::string &n : pol.callH[c].segs) {
      const u64 len = head_seg_len(s, n);
      u64 first = 0;
      for (int b = 62; b >= 0; b--)
        if ((len >> b) & 1) pieces.push_back({n, first, 1ull << b}), first += 1ull << b;
      R.place[c][n].n = len;
    }
    std::stable_sort(pieces.begin(), pieces.end(), [](const Piece &a, const Piece &b) { return a.size > b.size; });
    u64 col = 0;
    for (const Piece &p : pieces) {
      R.place[c][p.seg].pieces.push_back({p.first, p.size}), R.place[c][p.seg].col.push_back(col);
      col += p.size;
    }
    R.lJ[c] = std::max<int>((int)clog2u(col), s.nmin - s.lt);
    LookupCall &K = R.calls[c];
    K.n = R.lJ[c] + s.lt;
    const std::string &kind = pol.callH[c].kind;
    K.k = kind == "R8" ? 8 : 16;
    if (kind == "FC") K.table = tabs.FC, K.t0 = 32768;
    else if (kind == "FE") K.table = FE, K.t0 = 0;
    K.keys.assign(1ull << K.n, K.row(K.t0)), K.padding.assign(1ull << K.n, 1);
  }
  std::map<std::string, int> callOf;
  for (size_t c = 0; c < NC; c++) for (auto &x : R.place[c]) callOf[x.first] = (int)c;
  auto key_put = [&](const std::string &seg, u64 x, int t, u64 key, bool padding = false) {
    const int c = callOf.at(seg);
    const u64 idx = R.place[c].at(seg).pos(x) * T + t;
    R.calls[c].keys[idx] = key, R.calls[c].padding[idx] = padding;
  };
  auto put = [&](FeatMat &M, int f0, int stride, int nl, int t, I64 v) { int32_t c[8]; limbs_of(v, nl, c); for (int lb = 0; lb < nl; lb++) M.cell(f0 + (long long)lb * stride, t) = c[lb]; };
  // Positions.
  R.on.assign(T, 0), R.co.assign(T, 0), R.win.assign(T, 0);
  for (int t = 0; t < T; t++) {
    R.on[t] = (u32)t >= q.r && (u32)t < q.e, R.co[t] = R.on[t] && t + 1 < T;
    if (R.on[t]) R.win[t] = t + 1 < T ? q.tokens[t + 1] : q.y;
    if (R.win[t] >= (u32)Vn) throw std::runtime_error("head: a token is not below V");
  }
  R.str.resize((size_t)T * d), R.act.resize((size_t)T * d), R.lam.resize((size_t)T * Vn), R.rows.resize(T);
  // The final norm: stream, row, wide output, remainder, activations.
  std::vector<int32_t> lim((size_t)d * s.XL);
  for (int t = 0; t < T; t++) {
    for (int k = 0; k < d; k++) {
      const I64 x = V.get("x_last", t, k);
      R.str[(size_t)t * d + k] = x;
      limbs_of(x, s.XL, &lim[(size_t)k * s.XL]);
      for (int lb = 0; lb < s.XL; lb++) R.xbL.cell((long long)lb * d + k, t) = lim[(size_t)k * s.XL + lb];
    }
    NormRow r = norm_row(s.XL, lim, d, m.epsF, s.MF);
    if (r.R != V.get("aF_Rc", t, 0) || r.sig != V.get("aF_sigma", t, 0)) R.norm_ok = false;
    for (int k = 0; k < d; k++) {
      const I64 aw = V.get("aF_w", t, k), act = V.get("aF", t, k);
      const __int128 rem = (__int128)R.str[(size_t)t * d + k] * r.R + ((__int128)1 << (s.KF - 1)) - ((__int128)aw << s.KF);
      if (rem < 0 || rem >= ((__int128)1 << s.KF)) R.norm_ok = false;
      const I64 rh = (I64)(rem >> 16), rl = (I64)(rem & 65535);
      R.act[(size_t)t * d + k] = act;
      put(R.hf, k, d, 2, t, aw), put(R.hf, 2 * d + k, d, 2, t, rh - 32896), put(R.hd, k, d, 2, t, act);
      key_put("NF", k, t, (u64)rl), key_put("NFH", k, t, (u64)rh << (32 - s.KF));
      key_put("CNF", k, t, (u64)(aw + 32768) + ((u64)(act + 2048) << 28));
    }
    R.rows[t] = std::move(r);
    for (int v = 0; v < Vn; v++) R.lam[(size_t)t * Vn + v] = V.get("lam", t, v);
  }
  // Each position's cost: k_head_price.
  const I64 BIG = 1ll << 37, most = (1ll << 40) - 1;
  R.price.assign(T, 0);
  std::vector<I64> spare(T, 0);
  for (int t = 0; t < T; t++) {
    if (!R.co[t]) continue;
    const I64 *lam = &R.lam[(size_t)t * Vn];
    const u32 w = R.win[t];
    I64 gap = 0;
    for (int v = 0; v < Vn; v++) gap = std::max<I64>(gap, lam[v] - lam[w] + (v < (int)w));
    std::vector<I64> at(Vn);
    for (int v = 0; v < Vn; v++) at[v] = (lam[w] + gap + (1ll << s.sh) - lam[v] - (v < (int)w)) >> s.sh;
    auto sum = [&](I64 j) { u64 a = 0; for (int v = 0; v < Vn; v++) a += R.EX[std::min<I64>(at[v] + j, 65535)]; return a; };
    I64 low = 0, high = 32768;
    while (low < high) {
      const I64 mid = (low + high) / 2;
      if (sum(mid) <= 1ull << s.F) high = mid; else low = mid + 1;
    }
    R.price[t] = gap + ((1 + low) << s.sh);
    spare[t] = (I64)((1ull << s.F) - sum(low));
  }
  // The six hidden numbers: k_head_budget.
  const int r0 = (int)q.r, end = std::min<int>((int)q.e, T - 1);
  I64 spent = 0, clipped = 0;
  for (int t = r0; t < end; t++) spent += std::min<I64>(R.price[t], s.C), clipped += R.price[t] > s.C;
  const I64 RC = std::max(end - r0, 0);
  I64 left = s.theta * RC - spent, rows_left = (RC ? s.cap : 0) - clipped;
  if (left < 0 || rows_left < 0) R.bounds_ok = false;
  left = std::max<I64>(left, 0), rows_left = std::max<I64>(rows_left, 0);
  R.hidden.assign(T, {0, 0, 0, 0, 0, 0});
  std::vector<I64> lift(T, 0);
  static const int dg0[6] = {0, 5, 6, 11, 16, 18}, dgn[6] = {5, 1, 5, 5, 2, 3};
  for (int t = 0; t < T; t++) {
    const bool counted = R.co[t], over = counted && R.price[t] > s.C;
    const I64 c = counted && !over ? R.price[t] : 0;
    auto &h = R.hidden[t];
    h[0] = c, h[1] = over, h[2] = counted ? std::min<I64>(std::max<I64>(left - (I64)(t - r0) * most, 0), most) : 0;
    h[3] = counted ? s.C - c : 0, h[4] = t == r0 && counted ? rows_left : 0;
    h[5] = !counted ? 0 : over ? (1ll << s.F) - Vn : spare[t];
    lift[t] = c + (over ? BIG : 0);
    std::vector<int> &D = R.rows[t].D;
    D.resize(s.NRH, 0);
    for (int i = 0; i < 6; i++) put_digits(D, s.NR + dg0[i], dgn[i], (unsigned __int128)h[i]);
    for (int j = 0; j < s.NRH; j++) {
      R.hr.cell(j, t) = D[j] - 128;
      key_put("ROW", j, t, row_mul(j) * (u64)D[j]);
    }
  }
  // Margins, digits, table values.
  const int oD1 = 4 * d, oD2 = 4 * d + 2 * JV, oE = 4 * d + 3 * JV;
  const I64 reach = 65537ll << (s.sh + 15);
  for (int t = 0; t < T; t++)
    for (int v = 0; v < JV; v++) {
      I64 K0 = 0, K1 = 0, K2 = 0;
      const bool real = v < Vn && R.on[t], padding = v >= Vn;
      if (real) {
        const I64 *lam = &R.lam[(size_t)t * Vn];
        const u32 w = R.win[t];
        I64 D = lam[w] + lift[t] - lam[v] - (v < (int)w);
        if (D < 0 || D >= reach) R.margins_ok = false, D = 0;
        const I64 mm = D >> s.sh;
        K2 = D & ((1ll << s.sh) - 1);
        K1 = mm < 32768 ? mm : 32768 + (mm & 32767);
        K0 = mm < 32768 ? 0 : (mm >> 15) - 1;
      }
      const I64 ex = R.EX[K1];
      put(R.hf, oD1 + v, JV, 2, t, real ? K1 - 32896 : 0);
      put(R.hf, oD2 + v, JV, 1, t, real ? K2 - 128 : 0);
      put(R.hf, oE + v, JV, 3, t, ex - s.offE());
      key_put("HD0", v, t, (u64)K0, padding), key_put("HD1", v, t, (u64)K1, padding), key_put("HD2", v, t, (u64)K2, padding);
      const u64 he = (u64)K1 + ((u64)ex << 16);
      if (v < JVa) key_put("HE0", v, t, he, padding); else key_put("HE1", v - JVa, t, he, padding);
    }
  // The lookup's sizes, multiplicities and planes.
  R.sizes = lookup_sizes(R.calls, s.lidb);
  R.keys_ok = lookup_multiplicities(R.calls, R.sizes, R.mult);
  const LookupSizes &z = R.sizes;
  const int wd = (int)z.kT + 5 - s.lmbH, lowT = s.lmbH - 5;
  R.bH = CellMatrix2(1ull << s.lmbH, (u64)z.NC << wd);
  for (unsigned c = 0; c < z.NC; c++)
    for (u64 t = 0; t < (1ull << z.kT); t++)
      for (int j = 0; j < 32; j++)
        if ((R.mult[c][t] >> j) & 1) R.bH.at((t & ((1ull << lowT) - 1)) * 32 + j, ((u64)c << wd) + (t >> lowT)) = 1;
  return R;
}
}
