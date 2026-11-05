// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0
// Field arithmetic on MLX arrays of uint64. Elements of F are canonical words; an array of E has a
// last axis of 3 coordinates.
#pragma once
#include <mlx/mlx.h>
#include <string>
namespace fpmlx {
namespace mx = mlx::core;
std::string read_kernel_header(const char *name);
mx::array f_binary(const char *op, const mx::array &a, const mx::array &b); // "add" "sub" "mul"
mx::array e_mul(const mx::array &a, const mx::array &b);
}
