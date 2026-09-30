# Dwarf Violetto

Optimize local inference for Paradigma's Limite 1B Violetto on this Apple M4 MacBook Pro, 16 GB unified memory and 10 GPU cores. The user prefers speed with minimal measured mathematical quality loss.

## Architecture and workflow

Python 3.12, MLX and Metal. A reviewed local Limite architecture module loads official Safetensors through MLX-LM. Keep the original BF16 checkpoint as the reference. Benchmark quantized variants and runtime changes separately. Store reproducible measurements in `results/` and keep large weights in ignored `models/`.

Run `./violetto "Problem"` or `./violetto --interactive`. Setup is `./setup`; dependencies are pinned in `requirements.lock`. The default uses Q8, packed decode projections, shared RoPE tables, compiled XSA/residual operations and an equivalent simplified nucleus sampler. Each problem starts a fresh cache. No server is required.

Run commands with `.venv/bin/python`. GPU execution may need sandbox escalation to access Metal. Do not interpret a sandbox GPU failure as an unavailable GPU.

Validation commands: `python3 -m unittest discover -s tests -v`, `python3 scripts/check_repository.py`, `.venv/bin/python scripts/verify_fixtures.py`, `PYTHONPATH=. .venv/bin/python scripts/verify_runtime.py`, and `PYTHONPATH=. .venv/bin/python scripts/verify_sampling.py`. The optional independent reference uses `.venv/bin/python scripts/torch_reference.py`; install `requirements-reference.txt` for PyTorch 2.13.0 and setuptools 83.0.0. The historical reference used PyTorch 2.11.0. README.md documents benchmarks and evaluations. `scripts/report.py` rebuilds BENCHMARKS.md from the saved results. GitHub Actions checks CPU contracts, fixtures, artifact consistency, syntax and report freshness; it does not validate Metal inference.

## Status

Implemented and measured on September 25-26, 2026. Final default sampling measured 53.9 tokens/s versus BF16's 30.3. Greedy Q8 measured 71.8 versus 39.8 tokens/s; peak MLX allocations were 1.22 versus 2.16 GB. Q6 and Q4-HQ are available but have larger measured distribution changes. Q8 remains the quality-conscious default.

Runtime packing, fusion, tokenizer and cache checks passed. The simplified sampler matched stock masks and random draws at 359 positions and three seeds. The independent CPU reference is a limited cross-backend comparison, not CUDA bitwise equivalence. Keep source attribution in THIRD_PARTY.md and both license notices.

Additional decode experiments completed on September 30, 2026. Compiled RoPE, compiled MUDD and their combination matched full capped logits and final KV arrays at 960 forced decode positions across prefixes of 1, 128, 1100, 8192 and 32768 tokens. See `results/decode-candidate-correctness.json`. Settled RoPE comparisons measured +0.5%, +0.9% and -1.1% throughput at 128, 8192 and 32768 prompt tokens. Its process CPU time fell 6.5% in the 128-prompt/512-output workload. These small, mixed throughput changes do not justify a new default. MUDD, custom Metal RoPE, compiled MLP and grouped XSA also remain experimental. See `results/decode-optimization-summary.json`; exclude the three variable timing files listed there from selection evidence. Mathematical generation and the default Q8 path remain unchanged.

Final speed measurements are `results/final-*.json`; earlier measurements use different protocols and are retained as experiment history. The user rejected the 2048/4096-token output caps used in the first math checks because they interrupted reasoning. Fresh runs in `results/*-math-eos.json` completed all 12 cases correctly for both BF16 and Q8, with actual EOS ID 151643 and no truncation. BF16 generated 28370 tokens in 782.7 seconds, with a longest response of 7715 tokens. Q8 generated 26574 in 379.6 seconds, with a longest response of 6981 tokens. These are single-run end-to-end timings with different response lengths. They supersede the capped answer checks, not the fixed-work speed benchmarks. No competition-level quality preservation has been established.

The public repository is https://github.com/Zer0codestuff/dwarf-violetto, with `main` tracking `origin/main`. The first hosted CPU CI passed on publication commit `3037bd1`. Model weights, numerical reference arrays, environment and downloaded research are ignored by Git. Publication includes an English quick start, contribution guidance, Apache-2.0 NOTICE, retained MIT sampler license, CPU CI with pinned Actions and dependency updates. Private vulnerability reporting, Dependabot security updates, secret scanning and push protection are enabled. Keep evidence in AUDIT.md and `results/publication-*.json`. Historical script-path keys were normalized without changing numerical evidence or raw completions. Public history uses the GitHub noreply identity. Original history is preserved locally in `codex/local-history` and an ignored `.publication/local-history.bundle`; never push these archives.

## Known limits and next work

- CLI and evaluator now run until EOS by default. Only the configured context of 131072 tokens, including the prompt, bounds generation. `--max-tokens` is optional and explicit. Report `context_limit` separately from EOS and a requested cap.
- Evaluations save partial progress every 30 seconds. `--resume` skips completed cases and restarts an interrupted case with its original seed after checking settings, source, package versions, weight bytes, tokenizer and fixtures. Records from before this stronger fingerprint cannot resume with the current code. Do not edit runtime code during a run.
- Full configured 131072-token context, GPU power, ANE execution and speculative decoding were not validated. Original speed tests covered 128 and 4096 prompt tokens; additional alternating decode comparisons covered 8192 and 32768. GPU snapshots are point samples, not continuous utilization, frequency or power measurements.
- A larger held-out mathematical evaluation with multiple seeds is needed before claiming unchanged mathematical ability.
- MLX-LM applies top-p before temperature. Preserve that documented local behavior unless intentionally changing and revalidating sampling.
- `scripts/compare_decode.py` compares reversible candidates in forward/reverse blocks, records memory and GPU snapshots, and supports longer warmup and alternate seeds. `scripts/verify_decode_candidates.py` checks full logits and cache arrays against the current Q8 path. Run both with `PYTHONPATH=.`.
- Manual long-context prefill in verification and profiling must call `mx.clear_cache()` after each block, matching MLX-LM generation. Otherwise allocator buffers accumulate and can exhaust available system memory despite a small reported MLX allocation peak. The initial interrupted verification is preserved as history.
- The corrected synchronized profile is `results/decode-profile-cleared.json`. At 32K it measured 56.3 instrumented ms/token, 1.97 GB peak MLX allocations and 0.49 GB allocator cache. The old 372.7 ms/token result reflected allocator accumulation. These component timings change scheduling and are not production throughput. The CLI already used cache clearing.
- `scripts/throughput_feasibility.py` and `results/throughput-feasibility.json` contain a short synchronous pilot without sampling. Batch 4 reached about 213 aggregate tokens/s versus a 61-token/s serial control. Eight-token verification reduced target-pass cost by about 5.3x. Batch/verification full logits changed by up to 0.424/0.393, despite matching all 64 top predictions. This is not an AIME result or an actual speculative speedup. Future batching needs variable-length scheduling and quality validation. MTP needs a compatible trained drafter and safe rollback. The rotating local cache is trimmable only while its offset is below its 1025-key capacity; the pilot recorded a fresh cache. Neither path is enabled in the CLI or evaluator.
- Setup checks native Apple Silicon, uv and Python 3.12, and verifies the official BF16 weight SHA256. Quantization publishes a staged directory only after validating tokenizer, config and Safetensors payloads. Do not overwrite existing checkpoints or derive recipes from already quantized weights.
- The final-answer grader handles the last complete box and simple fractions only. It is not a symbolic verifier. Unknown case IDs and invalid sampling parameters fail before generation. Repository checks regrade all 24 saved EOS completions from their text and distinguish them from capped history.

## Constraints

- Preserve the checkpoint chat template, BF16 rounding sites, ordered MUDD sums, sigmoid logit cap, and inclusive 1025-key local window.
- Measure prompt processing and decode separately, with warmup and repetitions. Record memory and exact configurations.
- Mathematical answer checks and logit fidelity are separate evidence. Small local checks do not establish benchmark equivalence.
- Keep artifacts and code in English. No em dash characters.

## Do not

- Do not silently replace Violetto with another model or port Dwarf Star's unrelated architecture.
- Do not claim best possible speed or preserved quality without comparative measurements.
- Do not impose a small default output budget on mathematical evaluation or infer errors from truncated reasoning. Keep fixed-work speed measurements separate.
- Do not change global macOS settings, disable reasoning, truncate attention, or lower precision silently.
- Do not fuse multi-token prefill GEMM shapes: that changed BF16 rounding. Preserve partitioned prefill projections and only pack single-token decode.
- Do not use shapeless compilation for XSA: its repeat/reshape path needs shape-specific compilation.
- Do not enable the experimental broad attention correction/gating fusion by default. It changed a sampled trace after 184 tokens in the saved comparison.
- Do not publish to a different destination without the user's instruction. Push only the intended public branch or tag, never all local references or the archived history.
- Do not merge automated runtime dependency updates based only on CPU CI. Revalidate the affected Metal behavior and numerical comparisons before changing tested pins.
- Do not overwrite historical measurements when rerunning publication checks; use a new output path. Do not claim that CPU CI establishes GPU correctness or that dependency scanning guarantees security.
