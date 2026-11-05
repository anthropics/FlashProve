// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0
#include "mlx_kernels.h"
#include <fstream>
#include <map>
#include <sstream>
#ifndef FPMLX_KERNEL_DIR
#error "FPMLX_KERNEL_DIR: the directory of mlx/kernels"
#endif
namespace fpmlx {
std::string read_kernel_header(const char *name) {
  std::ifstream f(std::string(FPMLX_KERNEL_DIR) + "/" + name);
  if (!f) throw std::runtime_error(std::string("no kernel file ") + name);
  std::stringstream s;
  s << f.rdbuf();
  std::string t = s.str(), out;
  std::istringstream lines(t);
  for (std::string l; std::getline(lines, l);)
    if (l.rfind("#pragma once", 0)) out += l + "\n";
  return out;
}
static mx::array run1(const std::string &name, const std::string &body, const std::vector<mx::array> &in, const std::vector<std::string> &names, mx::Shape shape, size_t threads) {
  static std::map<std::string, mx::fast::CustomKernelFunction> kernels;
  auto it = kernels.find(name);
  if (it == kernels.end())
    it = kernels.emplace(name, mx::fast::metal_kernel(name, names, {"out"}, body, read_kernel_header("field.h.metal"))).first;
  return it->second(in, {shape}, {mx::uint64}, {(int)threads, 1, 1}, {(int)std::min<size_t>(threads, 256), 1, 1}, {}, std::nullopt, false, {})[0];
}
mx::array f_binary(const char *op, const mx::array &a, const mx::array &b) {
  const std::string o(op);
  return run1("fp_f_" + o, "uint i = thread_position_in_grid.x; out[i] = gl_" + o + "(a[i], b[i]);", {a, b}, {"a", "b"}, a.shape(), a.size());
}
mx::array e_mul(const mx::array &a, const mx::array &b) {
  return run1("fp_e_mul",
              "uint i = thread_position_in_grid.x; GlE r = e_mul(GlE{a[3*i], a[3*i+1], a[3*i+2]}, GlE{b[3*i], b[3*i+1], b[3*i+2]});"
              "out[3*i] = r.c0; out[3*i+1] = r.c1; out[3*i+2] = r.c2;",
              {a, b}, {"a", "b"}, a.shape(), a.size() / 3);
}
}
