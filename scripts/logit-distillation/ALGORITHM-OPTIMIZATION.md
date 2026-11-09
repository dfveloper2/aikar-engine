# Exact distillation algorithm optimizations

Local validation: 2026-11-08, Intel Xeon E5-2696 v3. Target workflow: `/mnt/openwebui/AIKAR/Lumen-3.8-Pulsar/agentic-distill.py`, backed by `examples/qlora_training/extract-logits.cpp` and `logit-distillation.h`.

## Changes

- Teacher top-k selection keeps a heap of k token IDs and scans logits in token order. It replaces a vocabulary-sized permuted index array with k candidates. Equal logits still prefer the lower token ID. At vocabulary 262144 and k=512, index scratch drops from 1048576 to 2048 bytes per encoder. Including the existing selection mask and FP16 probabilities, encoder scratch drops from 1311744 to 265216 bytes, excluding container overhead.
- Teacher-cache ID validation uses a bitmap instead of copying and sorting each row's IDs. Duplicate and out-of-range IDs are still rejected. The bitmap is cleared before each row, including after a failed read. At vocabulary 262144 this uses 32768 bytes, versus 2048 bytes for the previous top-512 sorting buffer.
- Extraction skips complete microbatches after the last supervised prediction in each window. Remaining microbatch shapes are preserved. Shortening the final microbatch was tested and rejected because it changed quantized model results. Empty-supervision windows retain their input/mask records and need no teacher decode.

The cache format, supervised positions, top-k, temperature, context, quantization, optimizer settings, probability calculations and KL objective remain unchanged. Heap selection still has O(V log k) worst-case cost; its improvement comes from sequential access and a smaller working set. Bitmap validation costs O(V/64 + k) rather than O(k log k).

## Measurements

The isolated encoding/read benchmark uses 64 deterministic normal-distributed rows, vocabulary 262144, top-512 and temperature 1. It compares the original and changed headers in one executable, alternates encoding order across five repetitions, and verifies identical encoded bytes. Read timings cover 16384 row reads, including validation and probability renormalization, from a warm file cache. Timings are local wall-clock observations with other build/inference activity, not statistical performance guarantees.

| Operation | Original median | Optimized median | Speed ratio |
| --- | ---: | ---: | ---: |
| Encode 64 teacher rows | 280.832 ms | 266.762 ms | 1.05x |
| Read/validate 16384 rows | 452.815 ms | 164.973 ms | 2.74x |

A separate selection-only experiment took 137.983 ms with the original partial sort and 72.249 ms with the bounded heap. That 1.91x ratio applies only to selection; full encoding still spends most of its time evaluating the full-vocabulary probabilities. An nth-element alternative took 431.233 ms and was not adopted.

The real teacher smoke test uses the local `gemma4-v2-Q8_0.gguf` and Lumen student vocabulary, eight short conversations, context 256, microbatch 128, eight CPU threads, one training window and one validation window. Both runs export eight supervised rows. Complete trailing microbatch skipping reduces decoded tokens from 512 to 384, while all 512 input/mask records remain in the cache.

| Measurement | Original | Optimized |
| --- | ---: | ---: |
| Extraction loop | 75.107 s | 55.613 s |
| Supervised rows | 8 | 8 |
| Cache SHA-256 | `23d7b4df6959fde728137679f7b87123d9a169282462343dfd11f0a19ae8a8dd` | Same |

This is a single padding-heavy smoke comparison, about 26% less extraction time. It does not establish full-corpus, GPU or end-to-end training throughput. Full-supervision windows have no trailing microbatches to skip.

## CUDA kernel optimization

The CUDA KL kernel now uses 512 threads per row for vocabularies of at least 4096 tokens, instead of one warp per row. Small vocabularies retain one warp. A shared bitmap replaces hash probing for vocabularies up to 262144 tokens; it requires at most 32 KiB. Larger vocabularies retain the original sparse hash table. Existing block reduction helpers combine partial maxima and sums. Separate reduction buffers avoid shared-memory reuse races, and a block barrier protects the final sparse gradient correction.

The full-vocabulary and tail maxima/sums remain separate. The optimization does not subtract the head probability from a rounded total, change temperature/top-k/precision, or add a probability approximation. Parallel summation changes rounding, so GPU losses and gradients are compared numerically rather than required to be byte-identical.

The selected block size was compared with 128, 256 and 1024 threads on V100. 512 threads gave a good balance for 128-row batches; 1024 was faster for eight rows but slightly slower for the 128-row backward pass. This is an internal kernel choice, with no training-option changes.

The final direct-kernel comparison uses a single SM70 executable containing the original and optimized kernels, deterministic normal-distributed logits, top-512, temperature 1, a masked first row, and the same CUDA compiler math flags. CUDA event medians cover five interleaved trials of five launches per implementation on physical GPU1. The user's separate extraction process was still active, so these timings are observations under concurrent load.

| Rows | Operation | Original | Optimized | Speed ratio |
| --- | --- | ---: | ---: | ---: |
| 8 | Forward | 10.163 ms | 0.498 ms | 20.4x |
| 8 | Backward | 15.862 ms | 0.805 ms | 19.7x |
| 128 | Forward | 10.373 ms | 0.612 ms | 17.0x |
| 128 | Backward | 18.303 ms | 1.111 ms | 16.5x |

These are KL-kernel timings at vocabulary 262144, not end-to-end training or teacher-extraction speedups. The direct comparison also checks gradients/losses against the old kernel and a double-precision reference, including tiny tails, one-row batches, word boundaries, top-1 and the hash fallback immediately above the bitmap limit. Compute Sanitizer reports zero racecheck hazards and zero memcheck errors for the final kernel configuration.

The GPU investigation found an independent production constraint: GPU0 reported HBM temperature 86 C with a reported memory operating limit of 85 C, active SW Thermal Slowdown, and SM clock 135 MHz. GPU1 reported approximately 997 MHz and no active thermal slowdown. GPU0 kernel timing was therefore excluded from performance claims. No temperature protection, power limits, clocks, resets or running production processes were changed. The current production extraction is using the older `build-v100/bin` executable.

Raw GPU evidence is in `/tmp/pulsar-distill-opt/`: `gpu-benchmark.cu`, `distillation-before.cu`, `gpu1-block-benchmark.log`, `gpu1-bitmap-benchmark.log`, `gpu1-tune.log`, `gpu1-final-benchmark.log`, `gpu-final-racecheck.log`, `gpu-final-memcheck.log`, and `gpu0-thermal.log`.

## Verification and builds

```bash
cmake --build build-distill-cpu --target llama-extract-logits llama-finetune-qlion test-logit-distillation -j 4
ctest --test-dir build-distill-cpu -R '^test-logit-distillation$' --output-on-failure
cmake -S . -B build-distill-cuda -DCMAKE_CUDA_ARCHITECTURES=70 -DGGML_CUDA_FA=ON -DGGML_CUDA_FA_QUANTS=f16-f16
cmake --build build-distill-cuda --target llama-extract-logits llama-finetune-qlion test-logit-distillation test-backend-ops -j 8
CUDA_VISIBLE_DEVICES=0,1 LD_LIBRARY_PATH="$PWD/build-distill-cuda/bin" build-distill-cuda/bin/test-backend-ops test -b CUDA1 -o DISTILLATION
CUDA_VISIBLE_DEVICES=0,1 GGML_CUDA_ALLREDUCE=none LD_LIBRARY_PATH="$PWD/build-distill-cuda/bin" build-distill-cuda/bin/test-logit-distillation --cuda
```

The CPU test passes. It checks selection against a full-sort reference across tied, ascending, descending and random inputs, multiple vocabulary sizes and temperatures, and reused encoders. It checks exact stored probabilities and tail mass, non-finite logits, duplicate/invalid cache IDs, bitmap word boundaries and row reuse. Existing KL reference, finite-difference gradient, mask, temperature, torn-cache and tensor-parallel CPU checks also pass. A real native extraction resumed from the original binary's completed first window plus a torn second window, retained six rows, regenerated the remaining two rows, and reproduced the same cache SHA-256. Diff whitespace, line endings and added ASCII checks pass.

CPU and CUDA extraction/training executables were rebuilt. After GPU access became available, the final CUDA tools were built for SM70 with the same Flash Attention build profile as the production V100 launcher. The earlier SM61/80 build passed the isolated operator tests but failed the tiny Gemma integration with an asynchronous CUDA launch failure; the SM70 rebuild completed that extraction successfully. Neither build profile changes the running production process.

Nine forward operator cases pass against CPU on GPU1, including top-4096, a non-word-aligned bitmap, the large-vocabulary hash fallback, masks and extreme logits. The small-case CPU finite-difference backend suite passes all four selected cases. The expanded native `test-logit-distillation --cuda` passes on both V100s, checking the actual forward/backward graph against a double reference and sampled finite differences at vocabularies 8, 4097, 262144 and 262145, temperatures 1 and 2, masks and extreme logits, followed by the existing two-GPU copy/accumulation checks.

A 30-layer tiny Gemma/Lumen fixture exercises the actual tools on GPU1 without competing for the full model's VRAM. Original and new teacher extraction produce identical 21-row caches with SHA-256 `6a38d51984c917c41b600def309116ce6866eb186a2e38afc6ad3071857b89b7`. Both trainers complete one optimizer step and save adapter/optimizer checkpoints using the same cache, context 256, microbatch 128, eight Q/output targets and LR 1e-5. Training KL is 0.01400141 originally and 0.01400135 after optimization; both print validation KL 0.00991590. This fixture checks execution and numerical consistency, not general model quality or sustained full-model throughput. Integration logs are `gpu-tiny-extract-before.log`, `gpu-tiny-extract-sm70.log`, `gpu-tiny-train-before.log` and `gpu-tiny-train-sm70.log`; final backend logs are `sm70-ops-tests.log` and `sm70-gradient-tests.log` in the evidence directory.

The optimization/validation work preserved the existing deployment artifacts and model-directory scripts/cache manifests. A later live check found the wrapper and production extraction using `build/bin`, rather than the earlier `build-v100/bin`. Switching `BIN_DIR` changes the recorded extraction-binary identity; an existing `cache.inputs.json` will reject that switch even though the tested native cache format is compatible. Do not bypass that provenance check or delete a partial production cache merely to use the new tools.

## Live two-V100 utilization check

A later 20-second observation at 2026-11-08 10:25 confirmed that production PID 198963 uses both V100s with `-dev CUDA0,CUDA1 -sm tensor -ts 1,1`. Each device holds approximately 14.3 GiB for the same extraction process. The second V100 is physical GPU1; physical GPU2 is the separate 2 GiB Quadro P620.

| Device | Mean SM utilization | Sampled range | SM clock | HBM temperature |
| --- | ---: | ---: | ---: | ---: |
| First V100, GPU0 | 94.7% | 12-100% | 135 MHz | 89-90 C |
| Second V100, GPU1 | 13.1% | 0-21% | 495 MHz | 52 C |

GPU0 reports active SW Thermal Slowdown and a memory operating limit of 85 C. GPU1 reports no thermal slowdown. Together with the tensor-parallel synchronization, this is evidence that the second V100 is waiting for the first, rather than being excluded from the computation. The user's stated 500 MHz operating cap was left unchanged; no clock or thermal-protection settings were changed.

A metadata-only allocation simulation was also run without loading another copy of the weights. Its meta-backend memory report is aggregated and does not establish a safe per-device allocation for a skewed split. The roughly 2 GiB remaining on each production device is not enough evidence to apply a large redistribution safely. No split change or production restart was performed during this check. Evidence: `live-two-v100.log`, `split-memory.cpp`, and `memory-split-1-1.log` under `/tmp/pulsar-distill-opt/`.

Raw local evidence and the side-by-side benchmark harness are in `/tmp/pulsar-distill-opt/`: `benchmark.cpp`, `benchmark.log`, `extract-before.log`, `extract-final.log`, `before.dat`, `final.dat`, and build logs. The `after.dat` intermediate experiment used shortened microbatches and is not the final implementation or a validation artifact.
