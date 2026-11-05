#!/bin/bash
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
# Builds mlx/build/fp_companion: verifier/src/cp_verify.cpp, unchanged, with the shadow headers first
# on the include path. It is no verifier to rely on: fp_prove uses it to walk a proof as it is written.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH=/opt/homebrew/opt/coreutils/libexec/gnubin:/opt/homebrew/opt/findutils/libexec/gnubin:$PATH
O=$PWD/mlx/build; mkdir -p "$O/shadow"
python3 mlx/tools/make_shadow.py "$PWD" "$O/shadow"
word=$(cd verifier/src && export LC_ALL=C && eval "$(grep '^build_word=' ../build.sh)" && echo "$build_word")
cd verifier/src
clang++ -O2 -std=c++17 -pthread -w -I"$O/shadow" -I../../mlx/src -I. -I../../blocks -I../../prover -include ../../mlx/src/macos_compat.h -DBLK_FIELD=2 -DBLK_COUNT_HASH_CALLS=1 -DSERVING_BUILD=1 -DDIGEST_WORDS=8 -DVF_SOURCE_DIGEST="\"$word\"" cp_verify.cpp -o "$O/fp_companion"
