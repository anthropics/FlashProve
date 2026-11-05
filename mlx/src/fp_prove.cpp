// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0
// Proves a request on MLX: the native counterpart of cp_prove. Its companion, a build of the verifier
// (mlx/DESIGN_PROVER.md), supplies each segment's equation; the final proof is for the stock cp_verify.
// usage: fp_prove <config directory> <ivk directory> <reference directory> <seed file> <request file> <y> <commitment id> <out proof> <companion>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <spawn.h>
#include <sstream>
#include <sys/stat.h>
#include "cp_configured.h"
#include "engine/sumcheck_cpu.h"
#include "engine_mlx.h"
#include "frame/check.h"
#include "frame/check_entry.h"
#include "frame/check_key.h"
#include "index_mlx.h"
#include "seed_coins.h"
#include "seg_ff_proof.h"
using namespace blk;
using namespace fpmlx;
extern char **environ;
static std::string slurp(const std::string &p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) { fprintf(stderr, "fp_prove: cannot read %s\n", p.c_str()); exit(2); }
  std::stringstream s;
  s << f.rdbuf();
  return s.str();
}
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
int main(int argc, char **argv) {
  if (argc != 10) { fprintf(stderr, "usage: fp_prove <config> <ivk> <reference> <seed> <request> <y> <id> <out proof> <companion>\n"); return 2; }
  setvbuf(stdout, nullptr, _IOLBF, 0);
  const std::string cfg = argv[1], ivk = argv[2], ref = argv[3], seedf = argv[4], reqf = argv[5], id = argv[7], outp = argv[8], comp = argv[9];
  const u32 y = (u32)atoi(argv[6]);
  const std::string pol_t = slurp(cfg + "/policy.txt"), lay = slurp(cfg + "/layout.txt"), kl = slurp(cfg + "/KL.bin");
  std::vector<uint32_t> KL(kl.size() / 4);
  memcpy(KL.data(), kl.data(), kl.size());
  const std::string raw = slurp(ref + "/matrix.i8"), req = slurp(reqf);
  // The stream, the marker file and the companion.
  const std::string dir = outp + ".work";
  mkdir(dir.c_str(), 0700);
  const std::string spath = dir + "/stream.bin", mark = dir + "/mark.proof", yf = dir + "/y";
  uint8_t *m = fpstream::map(spath.c_str(), true);
  if (!m) { fprintf(stderr, "fp_prove: no stream\n"); return 2; }
  { std::ofstream f(yf, std::ios::binary); f.write((const char *)&y, 4); }
  {
    Writer h;
    write_head(h, TAG_PROOF, 0);
    std::vector<u8> b = h.bytes();
    b.resize(fpstream::MARK_BYTES, 0);
    std::ofstream f(mark, std::ios::binary);
    f.write((const char *)b.data(), b.size());
  }
  setenv("FP_STREAM", spath.c_str(), 1);
  const std::string log = dir + "/companion.log";
  posix_spawn_file_actions_t fa;
  posix_spawn_file_actions_init(&fa);
  posix_spawn_file_actions_addopen(&fa, 1, log.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
  posix_spawn_file_actions_adddup2(&fa, 1, 2);
  const char *cargs[] = {comp.c_str(), cfg.c_str(), ivk.c_str(), reqf.c_str(), yf.c_str(), mark.c_str(), "-id", id.c_str(), nullptr};
  pid_t pid = 0;
  if (posix_spawn(&pid, comp.c_str(), &fa, nullptr, (char *const *)cargs, environ)) { fprintf(stderr, "fp_prove: no companion\n"); return 2; }
  return cp_with_configured(pol_t, lay, KL, 0, true, [&](const Configured &c) -> int {
    const double t0 = now();
    Companion io;
    io.m = m, io.pid = pid;
    const KeyFixed f = key_fixed(c);
    const LinShape &S = f.lin;
    const u64 columns = u64(1) << S.k_col();
    const int8_t *cells = (const int8_t *)raw.data();
    CellMatrix M{S.matrix_rows, columns, cells};
    const CoinKey seed = load_or_make_seed(seedf.c_str());
    auto leaf_coins = seeded_coins(seed, COINS_LEAF), coins = seeded_coins(seed, COINS_REQUEST);
    EngineMlx engine;
    const NonceSearch search = [&](const ChainPrefix &p, unsigned bits) { return engine.search_nonce(p, bits); };
    IndexMlx table(S, M, *leaf_coins, c.configuration());
    printf("leaf table: %.1f s\n", now() - t0);
    // The model.
    const ModelShape shape = read_shape(ref + "/params.txt");
    const Tensors W = [&] { auto l = mx::load_safetensors(ref + "/weights.safetensors"); Tensors t; for (auto &kv : l.first) t.emplace(kv.first, kv.second); return t; }();
    const Policy pol = Policy::read(pol_t);
    std::vector<u32> tokens(shape.T);
    memcpy(tokens.data(), req.data(), 4 * shape.T);
    std::vector<int32_t> tk(tokens.begin(), tokens.end());
    const mx::array tok_a(tk.data(), {(int)tk.size()}, mx::int32);
    const double t1 = now();
    const Values V = capture_forward(shape, W, tok_a);
    const CallTables tabs = call_tables(W);
    printf("forward: %.1f s\n", now() - t1);
    const WeightInts wi(W);
    const Units units = Units::of(shape, S.k_col());
    // The statement, from the walk.
    const std::vector<u8> st = [&] {
      Writer none; io.file = &none; return io.take(fpstream::STATEMENT);
    }();
    RecordReader sr(st);
    const u64 na = sr.u();
    const std::vector<u8> request(sr.p, sr.p + na);
    sr.p += na;
    const u64 nb = sr.u();
    const std::vector<u8> file_words(sr.p, sr.p + nb);
    Writer file;
    io.file = &file;
    write_head(file, TAG_PROOF, 0);
    const IndexName name{table.commitment(), table.record()};
    Binder b(LABEL_FIRST_CHAIN, c.proof().chains.at(0).plan);
    RequestPublic x = bind_request_parameters(b, c);
    bind_request_statement(b, x, name, 0, request, file_words);
    u8 salt[DIGEST_BYTES];
    coins->bytes(salt, sizeof salt);
    Digest random_bytes;
    digest_from_bytes(salt, random_bytes);
    ProverRounds ch(b, random_bytes, file, search);
    ProverContext P(x.lin.codes, ch, engine, *coins);
    ReqProver frame(P, x.shape);
    ReqRows<ProverContext> rows(P, frame, x.layout);
    const ReqShape &sh = ch.get(x.shape);
    const ReqLayout &ly = ch.get(x.layout);
    Chan chan{ch};
    chan.next_public = [&] { const std::vector<u8> v = io.take(fpstream::PUBLIC); u64 w[3]; memcpy(w, v.data(), 24); return E{{F{w[0]}, F{w[1]}, F{w[2]}}}; };
    FFInput in{pol, shape, V, wi, units, cells, columns, tokens};
    // Group 0.
    double ts = now();
    FFGroup G = ff_build(pol, shape, V, tabs, 0);
    FFProver pr(in, G, tabs);
    pr.prepare(true);
    SegMask mask = make_mask(P, sh, 0);
    std::vector<OracleDraft> at_start;
    at_start.push_back(lin_draft(P, sh.at_start[0].shape, false, sh.at_start[0].leaf_class, {}));
    HeldArrays held;
    auto hold = [&](size_t link) { for (auto &kv : pr.arrays) held[{link, kv.first}] = kv.second; };
    hold(0);
    const std::vector<TableId> t0s = place_tables(P, ly, sh, 0, 0, pr.arrays);
    (void)frame.first(t0s, std::move(at_start), std::move(mask.table));
    printf("G0: first message sent, %.1f s; %zu bytes\n", now() - ts, file.size());
    MaskSlot cslot = P.st.masks(frame.at_start().at(0)).take();
    const E cm = P.mask_value(cslot);
    u32 slot = 0;
    pr.before_u3(chan, io, mask.v.data(), slot, cm);
    printf("G0: to the merge, %.1f s, %u scalars\n", now() - ts, slot);
    pr.moment_u3(io);
    hold(0);
    (void)frame.later(place_tables(P, ly, sh, 0, 1, pr.arrays));
    if (!pr.after_u3(ch)) { fprintf(stderr, "fp_prove: a combination leaves its window\n"); return 3; }
    hold(0);
    (void)frame.later(place_tables(P, ly, sh, 0, 2, pr.arrays));
    pr.after_digits(chan, io, mask.v.data(), slot);
    printf("G0: all rounds, %.1f s, %u scalars\n", now() - ts, slot);
    close_segment(P, frame, rows, io, &held, &mask.v);
    printf("G0: closed, %.1f s, %zu bytes\n", now() - ts, file.size());
    return 0;
  });
}
