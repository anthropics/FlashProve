// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0
// Fills the matrices of the feed-forward segment of group g, and forms the keys of its lookup.
#pragma once
#include "seg_ff_witness.h"
namespace fpmlx {
struct FFGroup {
  FFShape s;
  int g = 0;
  FeatMat af, ad, ar, qa, xbI, xbO;
  std::map<std::string, Tensor2> tf; // tensors by name
  std::vector<LookupCall> calls;
  std::vector<std::map<std::string, SegPlace>> place; // by call
  std::vector<int> lJ;
  std::vector<std::vector<std::array<std::vector<E>, 1>>> dummy;
  // by lam, n (first or second norm), token
  std::vector<std::vector<NormRow>> rows; // rows[i][t]
};
inline void lay_tensors(FFGroup &G, const std::vector<std::tuple<std::string, int, int, int>> &list) {
  int off = 0;
  for (auto &[name, nl, cnt, J] : list) {
    Tensor2 t;
    t.off = off, t.nl = nl, t.cnt = cnt, t.J = J;
    G.tf[name] = t;
    off += nl * cnt * J;
  }
}
inline long long features(const std::vector<std::tuple<std::string, int, int, int>> &list) {
  long long n = 0;
  for (auto &[name, nl, cnt, J] : list) n += (long long)nl * cnt * J;
  return n;
}
struct PutTo {
  FeatMat &m;
  const Tensor2 &t;
  void put(int ix, int j, int tok, I64 v) { int32_t c[8]; limbs_of(v, t.nl, c); for (int lb = 0; lb < t.nl; lb++) m.cell(t.feat(lb, ix, j), tok) = c[lb]; }
};
inline FFGroup ff_build(const Policy &pol, const ModelShape &m, const Values &V, const CallTables &tabs, int g) {
  FFGroup R;
  R.s = ff_shape(pol, m);
  R.g = g;
  const FFShape &s = R.s;
  const int Lam = s.Lam, LamM = s.LamM, T = s.T, d = s.d, f = s.f, fe = s.fe, NE = s.NE;
  // The tensors of af, in the order of the table.
  std::vector<std::tuple<std::string, int, int, int>> A, D, Rr, Q;
  if (s.gcl) A.push_back({"g0", 1, Lam, f}), A.push_back({"gc", 1, Lam, f});
  else A.push_back({"gi", 2, Lam, f});
  A.insert(A.end(), {{"sl", 2, Lam, f}, {"up", 2, Lam, f}, {"rg", s.hgD, Lam, f}, {"ru", s.huD, Lam, f}, {"xs", s.XL, 2 * Lam - 1, d}, {"ro", s.ho, Lam, d}, {"rd", s.hdD, Lam, d},
                     {"aw", 2, Lam, d}, {"ra", 2, Lam, d}, {"mw", 2, Lam, d}, {"rm", 2, Lam, d}, {"qkw", 2, Lam, s.nqk}, {"rqk", s.hq, Lam, s.nqk}, {"vw", 2, Lam, s.dkv}, {"rv", s.hv, Lam, s.dkv}});
  if (s.gcl) A.push_back({"gh", 2, Lam, f});
  D = {{"an", 2, Lam, d}, {"mn", 2, Lam, d}};
  if (NE) D.push_back({"xr", 2, LamM, d});
  if (NE) Rr = {{"lr", 2, LamM, NE}, {"rl", s.hr, LamM, NE}, {"sel", 1, LamM, NE}, {"sc", 3, LamM, 1}};
  Rr.push_back({"row", 1, 2 * Lam, s.NR});
  if (s.rcl) Rr.push_back({"ly", 2, LamM, NE}), Rr.push_back({"l3", 1, LamM, NE});
  Q = {{"q", 2, Lam, s.dq}, {"kx", 2, Lam, s.dkv}, {"v", 2, Lam, s.dkv}, {"o", 2, Lam, s.dq}};
  R.af.init(s.NAf, s.lmaf, features(A), s.lt);
  R.ad.init(s.NAd, s.lmad, features(D), s.lt);
  R.ar.init(1, s.lmar, features(Rr), s.lt);
  R.qa.init(1, s.lmq, features(Q), s.lt);
  R.xbI.init(1, s.lmx, (long long)s.XL * d, s.lt);
  R.xbO.init(1, s.lmx, (long long)s.XL * d, s.lt);
  lay_tensors(R, A);
  { FFGroup t; lay_tensors(t, D); for (auto &x : t.tf) R.tf[x.first] = x.second; }
  { FFGroup t; lay_tensors(t, Rr); for (auto &x : t.tf) R.tf[x.first] = x.second; }
  { FFGroup t; lay_tensors(t, Q); for (auto &x : t.tf) R.tf[x.first] = x.second; }
  auto AF = [&](const char *n) { return PutTo{R.af, R.tf.at(n)}; };
  auto AD = [&](const char *n) { return PutTo{R.ad, R.tf.at(n)}; };
  auto AR = [&](const char *n) { return PutTo{R.ar, R.tf.at(n)}; };
  auto QA = [&](const char *n) { return PutTo{R.qa, R.tf.at(n)}; };
  auto L = [&](const char *n, int lam) { return std::string(n) + "." + std::to_string(g * Lam + lam); };
  auto str = [&](int sg, int t, int j) -> I64 { // the stream sg of the group
    const int lam = sg / 2;
    if (sg == 2 * Lam) return (g + 1) * Lam == s.L ? V.get("x_last", t, j) : V.get("x." + std::to_string((g + 1) * Lam), t, j);
    return V.get(L(sg % 2 ? "x_mid" : "x", lam), t, j);
  };
  const auto &ks = m.layers;
  // The calls' segments and their columns.
  auto seg_len = [&](const std::string &n) -> u64 {
    static const std::map<std::string, int> none;
    if (n == "G" || n == "U" || n == "UT" || n == "GH" || n == "UH" || n == "F" || n == "G0" || n == "CG") return f;
    if (n == "QK" || n == "QKH" || n == "CQK") return s.nqk;
    if (n == "N1" || n == "N1H" || n == "N2" || n == "N2H" || n == "O" || n == "D" || n == "XR" || n == "OH" || n == "DH" || n == "CN1" || n == "CN2") return d;
    if (n == "V" || n == "VH" || n == "CV") return s.dkv;
    if (n == "LR" || n == "LT" || n == "DD" || n == "LH" || n == "SEL" || n == "L3" || n == "CLR") return NE;
    if (n == "ROW") return 2 * s.NR;
    if (n == "MR") return 1;
    throw std::runtime_error("no such segment " + n);
  };
  auto routed = [](const std::string &n) { return n == "XR" || n == "LR" || n == "LT" || n == "DD" || n == "LH" || n == "SEL" || n == "MR" || n == "L3" || n == "CLR"; };
  R.calls.resize(pol.call.size()), R.place.resize(pol.call.size()), R.lJ.resize(pol.call.size());
  for (size_t c = 0; c < pol.call.size(); c++) {
    struct Piece { std::string seg; u64 first, size; };
    std::vector<Piece> pieces;
    for (const std::string &n : pol.call[c].segs) {
      const u64 len = seg_len(n);
      u64 first = 0;
      for (int b = 62; b >= 0; b--)
        if ((len >> b) & 1) pieces.push_back({n, first, 1ull << b}), first += 1ull << b;
      R.place[c][n].n = len, R.place[c][n].routed = routed(n);
    }
    std::stable_sort(pieces.begin(), pieces.end(), [](const Piece &a, const Piece &b) { return a.size > b.size; });
    u64 col = 0;
    for (const Piece &p : pieces) {
      R.place[c][p.seg].pieces.push_back({p.first, p.size}), R.place[c][p.seg].col.push_back(col);
      col += p.size;
    }
    int lJ = clog2u(col);
    lJ = std::max(lJ, s.nmin - s.ll - s.lt);
    R.lJ[c] = lJ;
    LookupCall &K = R.calls[c];
    K.n = s.ll + lJ + s.lt;
    const std::string &kind = pol.call[c].kind;
    K.k = kind == "R8" ? 8 : 16;
    if (kind == "FC") K.table = tabs.FC, K.t0 = 32768;
    else if (kind == "FL") K.table = tabs.FL, K.t0 = 32768;
    else if (kind == "FS") K.table = tabs.FS, K.t0 = 32768;
    K.keys.assign(1ull << K.n, 0), K.padding.assign(1ull << K.n, 1);
  }
  auto key_put = [&](const std::string &seg, int lam, u64 x, int t, u64 key) {
    for (size_t c = 0; c < pol.call.size(); c++) {
      auto it = R.place[c].find(seg);
      if (it == R.place[c].end()) continue;
      const u64 idx = (((u64)lam << R.lJ[c]) + it->second.pos(x)) * T + t;
      R.calls[c].keys[idx] = key, R.calls[c].padding[idx] = 0;
      return;
    }
    throw std::runtime_error("no call holds " + seg);
  };
  auto has = [&](const std::string &seg) { for (auto &m2 : R.place) if (m2.count(seg)) return true; return false; };
  auto SK = [&](__int128 A, I64 Q, int k, int h, I64 rh) { return (u64)((((A + ((__int128)1 << (k - 1)) - ((__int128)Q << k)) << lift(k))) - (h ? ((__int128)(rh + ofs_of(h)) << 16) : 0)); };
  auto HK = [&](I64 rh, int k, int h) { const int top = h == 1 ? 24 : 32; return (u64)((rh + ofs_of(h)) << (top - std::max(k, 16))); };
  auto RH = [&](__int128 A, I64 Q, int k, int h) -> I64 { const __int128 rem = A + ((__int128)1 << (k - 1)) - ((__int128)Q << k); return h ? (I64)(rem >> 16) - ofs_of(h) : 0; };
  // Streams.
  for (int t = 0; t < T; t++) {
    for (int j = 0; j < d; j++) {
      PutTo{R.xbI, Tensor2{0, s.XL, 1, d}}.put(0, j, t, str(0, t, j));
      PutTo{R.xbO, Tensor2{0, s.XL, 1, d}}.put(0, j, t, str(2 * Lam, t, j));
      for (int ix = 0; ix < 2 * Lam - 1; ix++) AF("xs").put(ix, j, t, str(ix + 1, t, j));
    }
  }
  // Norms.
  R.rows.assign(2 * Lam, std::vector<NormRow>(T));
  for (int i = 0; i < 2 * Lam; i++) {
    const int lam = i / 2, n = i % 2;
    const LayerScales &k = ks[g * Lam + lam];
    const int M = n ? k.M2 : k.M1, K = n ? k.K2 : k.K1;
    const I64 eps = n ? m.eps2 : m.eps1;
    for (int t = 0; t < T; t++) {
      std::vector<int32_t> lim((size_t)d * s.XL);
      for (int j = 0; j < d; j++) limbs_of(str(i, t, j), s.XL, &lim[(size_t)j * s.XL]);
      NormRow r = norm_row(s.XL, lim, d, eps, M);
      for (int j = 0; j < s.NR; j++) {
        AR("row").m.cell(R.tf.at("row").feat(0, i, j), t) = r.D[j] - 128;
        key_put("ROW", lam, (u64)n * s.NR + j, t, row_mul(j) * r.D[j]);
      }
      for (int j = 0; j < d; j++) {
        const char *wn = n ? "mw" : "aw", *rn = n ? "rm" : "ra", *cn = n ? "mn" : "an";
        const __int128 Aval = (__int128)str(i, t, j) * r.R;
        const I64 wv = V.get(L(n ? "m_w" : "a_w", lam), t, j), cv = V.get(L(n ? "m" : "a", lam), t, j);
        const I64 rh = RH(Aval, wv, K, 2);
        AF(wn).put(lam, j, t, wv), AF(rn).put(lam, j, t, rh), AD(cn).put(lam, j, t, cv);
        key_put(n ? "N2" : "N1", lam, j, t, SK(Aval, wv, K, 2, rh));
        key_put(n ? "N2H" : "N1H", lam, j, t, HK(rh, K, 2));
        key_put(n ? "CN2" : "CN1", lam, j, t, (u64)(wv + 32768) + ((u64)(cv + 2048) << 28));
      }
      R.rows[i][t] = std::move(r);
    }
  }
  for (int lam = 0; lam < Lam; lam++) {
    const LayerScales &k = ks[g * Lam + lam];
    const bool ex = NE && (lam & 1);
    const int lamM = (lam - 1) / 2;
    for (int t = 0; t < T; t++) {
      for (int x = 0; x < s.nqk; x++) {
        const bool isq = x < s.dq;
        const int kk = isq ? k.kq : k.kk;
        const I64 w = isq ? V.get(L("qw", lam), t, x) : V.get(L("kw", lam), t, x - s.dq), cl = isq ? V.get(L("q", lam), t, x) : V.get(L("k", lam), t, x - s.dq);
        const __int128 Aq = isq ? V.get(L("sum_q", lam), t, x) : V.get(L("sum_k", lam), t, x - s.dq);
        const I64 rh = RH(Aq, w, kk, s.hq);
        AF("qkw").put(lam, x, t, w);
        if (s.hq) AF("rqk").put(lam, x, t, rh);
        if (isq) QA("q").put(lam, x, t, cl); else QA("kx").put(lam, x - s.dq, t, cl);
        key_put("QK", lam, x, t, SK(Aq, w, kk, s.hq, rh));
        if (s.hq) key_put("QKH", lam, x, t, HK(rh, kk, s.hq));
        key_put("CQK", lam, x, t, (u64)(w + 32768) + ((u64)(cl + 2048) << 28));
      }
      for (int x = 0; x < s.dkv; x++) {
        const I64 w = V.get(L("vw", lam), t, x), cl = V.get(L("v", lam), t, x);
        const __int128 Av = V.get(L("sum_v", lam), t, x);
        const I64 rh = RH(Av, w, k.kv, s.hv);
        AF("vw").put(lam, x, t, w);
        if (s.hv) AF("rv").put(lam, x, t, rh);
        QA("v").put(lam, x, t, cl);
        key_put("V", lam, x, t, SK(Av, w, k.kv, s.hv, rh));
        if (s.hv) key_put("VH", lam, x, t, HK(rh, k.kv, s.hv));
        key_put("CV", lam, x, t, (u64)(w + 32768) + ((u64)(cl + 2048) << 28));
      }
      for (int x = 0; x < s.dq; x++) QA("o").put(lam, x, t, V.get(L("o", lam), t, x));
      for (int j = 0; j < d; j++) {
        const I64 Q1 = str(2 * lam + 1, t, j) - str(2 * lam, t, j), Q2 = str(2 * lam + 2, t, j) - str(2 * lam + 1, t, j);
        const __int128 Ao = V.get(L("sum_o", lam), t, j), Ad = V.get(L("sum_d", lam), t, j);
        const I64 r1 = RH(Ao, Q1, k.ko, s.ho), r2 = RH(Ad, Q2, k.kd, s.hdD);
        if (s.ho) AF("ro").put(lam, j, t, r1);
        if (s.hdD) AF("rd").put(lam, j, t, r2);
        key_put("O", lam, j, t, SK(Ao, Q1, k.ko, s.ho, r1));
        key_put("D", lam, j, t, SK(Ad, Q2, k.kd, s.hdD, r2));
        if (s.ho) key_put("OH", lam, j, t, HK(r1, k.ko, s.ho));
        if (s.hdD) key_put("DH", lam, j, t, HK(r2, k.kd, s.hdD));
      }
      for (int x = 0; x < f; x++) {
        const bool rt = ex && x >= fe;
        const int kg = rt ? k.kgR : k.kg, ku = rt ? k.kuR : k.ku;
        const I64 gw = V.get(L("gw", lam), t, x), up = V.get(L("u", lam), t, x), sl = V.get(L("s", lam), t, x);
        const __int128 Ag = V.get(L("sum_g", lam), t, x), Au = V.get(L("sum_u", lam), t, x);
        const I64 rg = RH(Ag, gw, kg, s.hgD), ru = RH(Au, up, ku, s.huD);
        AF("sl").put(lam, x, t, sl), AF("up").put(lam, x, t, up);
        if (s.hgD) AF("rg").put(lam, x, t, rg);
        if (s.huD) AF("ru").put(lam, x, t, ru);
        key_put("G", lam, x, t, SK(Ag, gw, kg, s.hgD, rg));
        key_put("U", lam, x, t, SK(Au, up, ku, s.huD, ru));
        if (s.hgD) key_put("GH", lam, x, t, HK(rg, kg, s.hgD));
        if (s.huD) key_put("UH", lam, x, t, HK(ru, ku, s.huD));
        key_put("UT", lam, x, t, (u64)(up + 32768));
        I64 arg = gw;
        if (s.gcl) {
          const I64 g0 = ((gw + 128) & 255) - 128, gh = (gw - g0) >> 8, gc = std::min<I64>(std::max<I64>(gh, -127), 127);
          arg = g0 + 256 * gc;
          AF("g0").put(lam, x, t, g0), AF("gc").put(lam, x, t, gc), AF("gh").put(lam, x, t, gh);
          key_put("G0", lam, x, t, (u64)(g0 + 128));
          key_put("CG", lam, x, t, (u64)(gh + 49152) + ((u64)(gc + 128) << 28) + (1ull << 40));
        } else {
          AF("gi").put(lam, x, t, gw);
        }
        key_put("F", lam, x, t, (u64)(arg + 32768) + ((u64)(sl + 4096) << 28));
      }
      if (ex) {
        const I64 chosen = V.get(L("e_star", lam), t, 0), sc = V.get(L("sc", lam), t, 0);
        I64 mr = 0;
        for (int e = 0; e < NE; e++) {
          const I64 lr = V.get(L("lr", lam), t, e);
          const I64 sel = e == chosen;
          if (sel) mr = lr;
          AR("lr").put(lamM, e, t, lr), AR("sel").put(lamM, e, t, sel);
          key_put("SEL", lam, e, t, 128 * sel);
          key_put("LT", lam, e, t, (u64)(2 * (lr + 16384)));
        }
        AR("sc").put(lamM, 0, t, sc);
        for (int e = 0; e < NE; e++) {
          const I64 lr = V.get(L("lr", lam), t, e);
          const __int128 Ar = V.get(L("sum_r", lam), t, e);
          // lw from lr: with the clamp lw is 8 ly + l3 of the unclamped logit.
          const I64 lw = (I64)((Ar + ((__int128)1 << (k.kr - 1))) >> k.kr);
          const I64 rl = RH(Ar, lw, k.kr, s.hr);
          if (s.hr) { AR("rl").put(lamM, e, t, rl); key_put("LH", lam, e, t, HK(rl, k.kr, s.hr)); }
          key_put("LR", lam, e, t, SK(Ar, lw, k.kr, s.hr, rl));
          if (s.rcl) {
            const I64 ly = lw >> 3, l3 = lw & 7;
            AR("ly").put(lamM, e, t, ly), AR("l3").put(lamM, e, t, l3);
            key_put("L3", lam, e, t, 32 * l3);
            key_put("CLR", lam, e, t, (u64)(ly + 32768) + ((u64)(lr - l3) << 25) + (1ull << 39));
          }
          I64 after = 0;
          for (int e2 = e + 1; e2 < NE; e2++) after += e2 == chosen;
          key_put("DD", lam, e, t, (u64)(mr - lr - after));
        }
        key_put("MR", lam, 0, t, s.gcl ? (u64)(mr + 16384) + ((u64)sc << 17) : (u64)(mr + 32768) + ((u64)sc << 17));
        for (int j = 0; j < d; j++) {
          const I64 mn = V.get(L("m", lam), t, j), xr = V.get(L("xr", lam), t, j);
          AD("xr").put(lamM, j, t, xr);
          key_put("XR", lam, j, t, (u64)(sc * mn + 32768 - 65536 * xr));
        }
      }
    }
  }
  return R;
}
}
