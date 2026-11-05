// Every key of a feed-forward segment's lookup is a row of its call's table, on the reference's model.
// usage: ff_keys_test <directory of make_reference.py> <policy.txt>
#include <cstdio>
#include "seg_ff_build.h"
using namespace fpmlx;
static std::string slurp(const char *p) { std::ifstream f(p, std::ios::binary); std::stringstream s; s << f.rdbuf(); return s.str(); }
int main(int c, char **v) {
  const std::string dir = v[1];
  const ModelShape m = read_shape(dir + "/params.txt");
  auto w = mx::load_safetensors(dir + "/weights.safetensors").first;
  for (auto &[k, a] : w) if (k.rfind("table", 0) == 0) a = mx::astype(a, mx::int64);
  const Policy pol = Policy::read(slurp(v[2]));
  const Values V = capture_forward(m, w, w.at("tokens"));
  const CallTables tabs = call_tables(w);
  const int groups = m.L >> pol.get("ll");
  int bad = 0;
  for (int g = 0; g < groups; g++) {
    FFGroup G = ff_build(pol, m, V, tabs, g);
    for (size_t i = 0; i < G.calls.size(); i++) {
      std::vector<LookupCall> one{G.calls[i]};
      LookupSizes sz = lookup_sizes(one, 28);
      std::vector<std::vector<u32>> mult;
      const bool ok = lookup_multiplicities(one, sz, mult);
      u64 live = 0; for (char p : G.calls[i].padding) live += !p;
      printf("group %d call %s: 2^%u keys, %llu live: %s\n", g, pol.call[i].name.c_str(), G.calls[i].n, (unsigned long long)live, ok ? "every key is in the table" : "A KEY IS IN NO ROW");
      bad += !ok;
    }
  }
  printf("ff_keys_test: %s\n", bad ? "FAILED" : "all keys in their tables");
  return bad != 0;
}
