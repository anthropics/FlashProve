# MLX port: status

Branch `mlx-port`. **Milestone 1's chain works: a request on the bundled synthetic model is proved by
the native prover and the unmodified `bin/cp_verify` answers `YES`.** It is not a GPU-only prover yet:
the sumchecks and lookups still run on CPU threads (below). SmolLM2 is no longer the end goal, by the
owner's decision of 2026-11-05, and nothing of it is built. Everything below is measured on an Apple M5
Pro (64 GB, macOS 27), MLX 0.32.4.

## The result

`mlx/demo.sh` runs it all from synthetic weights: reference, placing, commitment, proof, the stock
`cp_verify` in a separate process, four rejected changes. Last run, commitment ID
`240e59038f03ecbde1af337de677312ccdad2e97f126a66858ae291c7daf6705`:

| Step | Measured |
|---|---|
| Commitment (`fp_commit`) | about 4 min; the key step runs on the CPU (`WfCpu`) |
| Proof (`fp_prove`, leaf table rebuilt, equation test on) | 49.3 s; 46.4 s with `FP_NO_EQUATION_TEST=1` |
| Segments | G0 9.2 s, A0 4.8 s, G1 10.3 s, A1 5.6 s, H 6.6 s |
| Proof size | 4,158,700 bytes |
| `cp_verify` | 0.71 s, exit 0, last line `YES` |
| Memory | maximum resident 23.9 GB; `peak memory footprint` 62.5 GB, not yet explained |
| CPU time of a proof | 122.7 s user in 47.1 s real: most of the proof is on CPU threads |

`cp_verify`'s lines on a run of this prover: `covered: r=15 e=64 T=64`, `proof of work: up to 32 bits
counted in 38 rounds; with it the least round has 128.00 bits, without it 96.23 bits`, these bits being
no security level (`guide/known-limits.md`). This `YES` is `cp_verify`'s, on a small model with synthetic
weights. It says nothing of a provider's model, and it does not show that the weights are hidden
(`guide/weight-hiding.md`).

Refused, each exit 1: a changed token (`NO: reply-differs`), a changed request, flipped bytes early, in
the middle and late, a cut proof (all `NO: proof-refused`), another commitment ID (`NO: model-statement`).
Two proofs of one request are both accepted and differ from byte 13 on.

## Limits of this result

- **CPU, not GPU:** on the GPU are the integer forward products, the encoding, the Merkle trees, the
  nonce search and the folds. The lookups, sumchecks, witnesses and the key step are host code.
- **The companion:** `fp_prove` needs a patched build of the verifier running beside it, for each
  segment's equation and public parts. It is a helper of the prover. It decides nothing: the judge is
  the stock `cp_verify`, built without those patches.
- **The table's budget is not kept:** a table allows a limited number of batches, and `lin_begin` counts
  them in memory. `fp_prove` rebuilds the table each run, so the count starts again. Too many proofs on
  one leaf table would open more columns than its masks cover. Fine for synthetic weights only.
- **The seed file** re-derives the leaf table's random entries and is secret.
- Paths that this model never reaches are untested: `e < T`, a cost above C, `v >= V` padding, more than
  one part of `hf`. Five limbs alone are served.

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
| Native commitment | `mlx/src/fp_commit.cpp`, `index_mlx.h`, `mlx/tools/place_weights.py` | on `target_small_f2_roomy`: key and leaf files pass the unmodified `cp_verify` (`PART PASSED`); 4 tampered files refused (`key-refused`, `entry-refused`). 4 min 17 s, of which the key file 198 s (range proof on the CPU). Writes no `isk` yet |
| Verifier companion | `mlx/build_companion.sh`, `mlx/tools/make_shadow.py`, `mlx/tests/stream_test.py` | the stored example proof, streamed 4 KB at a time through shared memory: `YES`, 5 segment equations exported |
| Masked sumcheck prover | `mlx/src/psc_prover.h`, `mlx/tests/psc_test.cpp` | 66 configurations (equality, plain, mixed rounds, shown values, bit orders): the claim `Left(C; S0; K) = 0` holds |
| Integer forward pass, with experts | `mlx/src/integer_forward.cpp` | `forward_test`: 191 tensors, 5,246,656 values equal to `model.py` on `target_small_f2_roomy` synthetic weights; one altered reference value makes it fail |

## Library tests with EngineMlx in place of EngineCpu (`mlx/run_library_tests.sh`)

37 test programs, all pass. `EngineMlx` runs on the GPU in the parent process. Tests that `fork()`
to check that a misuse ends the process run the engine's host path in the child (Metal does not
survive a fork), so those children exercise the engine's guards, not its kernels.

## Not done

- Moving the lookups and sumchecks to Metal, then `WfCpu` and the witnesses, by profile.
- Keeping a table's batch count between runs.
- Explaining the 62.5 GB footprint.
- A CLI beyond `fp_commit`, `fp_prove` and `mlx/demo.sh`; benchmarks over repeated runs.
- SmolLM2: dropped as the end goal. A separate verifier variant would be needed for it.
