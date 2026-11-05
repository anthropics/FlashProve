# MLX port: status

Branch `mlx-port`. **The port is not finished: no model proof has been made with it yet.** Neither
milestone of the task (synthetic model, SmolLM2) is done. Everything below is measured on an Apple
M5 Pro (64 GB, macOS 27), MLX 0.32.4.

## Design

Native port: `mlx/` holds C++ on `mlx::core` arrays with hand-written Metal kernels
(`mx::fast::metal_kernel`). `blocks/` (proof library) and `verifier/` are used as they are, apart
from the macOS link flags in `verifier/build.sh`. `prover/` (CUDA) is not compiled.

The branch `cuda-shim-parked` holds an earlier CUDA-to-Metal compatibility layer. It is not this
port and nothing here depends on it.

## Done and tested

| Piece | Where | Result |
|---|---|---|
| `cp_verify` on macOS | `verifier/build.sh` | builds with g++ 16; stored example gives `YES` |
| F and E arithmetic | `mlx/kernels/field.h.metal` | `field_test`: add, sub, mul, E mul equal to `blocks/field` on about 2^20 inputs each |
| `EngineMlx`: encoding, BLAKE2b leaves and tree, evaluation, folds | `mlx/src/engine_mlx.h`, `mlx_kernels.cpp` | `engine_test`: root, openings, evaluation, fold rounds, folded rows equal to `EngineCpu` |
| Whole toy protocols | `mlx/tests/toy_mlx.cpp`, `segments_mlx.cpp` | accepted, proof digests byte-identical to the CPU engine at seed 42 |
| Integer forward pass, with experts | `mlx/src/integer_forward.cpp` | `forward_test`: 191 tensors, 5,246,656 values equal to `model.py` on `target_small_f2_roomy` synthetic weights; one altered reference value makes it fail |

## Library tests with EngineMlx in place of EngineCpu (`mlx/run_library_tests.sh`)

37 test programs: 19 pass, 12 fail, 6 do not build.

- The 12 failures pass with `EngineCpu` on this Mac, so they are specific to the MLX runs.
- Likely cause, **not confirmed for all of them**: they `fork()` to check that misuse stops the
  process, and Metal does not work in a forked child. `lkp_test` says so directly; most other
  failing lines are such checks.
- **Open:** `fold_base_test` reports 60,556 of 60,616 refused; the other 60 are unexplained.
  `send_pad_test` line 224 is also unexplained.
- The 6 that do not build need Linux's `cpu_set_t` in test helpers (`blocks/tests/damage.h`,
  `blocks/tests/hiding/exact.h`).

## Not done

- Native `cp_commit`: the weight table, key file and leaf files.
- Native `cp_prove`: the sub-proofs of a request (feed-forward, attention, head, lookups, range
  checks), specified in `docs/format/11-*.md`. This is the largest remaining part.
- A proof accepted by `cp_verify`.
- SmolLM2: checkpoint conversion, and a separate verifier variant. It needs padding (360M: `H=16`,
  `G=16`, `d=1024`) because `G` must be a power of two. `theta0=100000` and `NE=0` lie outside the
  standard parameter form, so the unmodified verifier cannot end in `YES`.
- CLI, demo script, benchmarks.

## Known weak spots in what exists

- `search_nonce` still runs on the CPU.
- `exact_product` is a plain per-output kernel, not tiled.
- Tables live on the host and are uploaded per use; `Form` expansion is on the CPU.
- `encode_rows` needs fewer than 2^31 threads per call: no chunking yet, so large tables will fail.
- The forward pass does not yet test the range conditions of `docs/model.md` section 9.
- Kernels read their headers from `mlx/kernels` at run time, by a path fixed at build time.

## Next

1. Explain the open test failures, using a harness that does not fork after Metal starts.
2. Native key file and leaf files through `wf_prove`, `rng_prove`, `link_prove`, using
   `blocks/iop/tests/*_world.h` as templates, checked by `cp_verify`.
3. Native sub-proofs, one segment kind at a time.
