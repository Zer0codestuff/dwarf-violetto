# Dwarf Violetto

Optimize local inference for Paradigma's Limite 1B Violetto on this Apple M4 MacBook Pro, 16 GB unified memory and 10 GPU cores. The user prefers speed with minimal measured mathematical quality loss.

## Architecture and workflow

Python 3.12, MLX and Metal. A reviewed local Limite architecture module loads official Safetensors through MLX-LM. Keep the original BF16 checkpoint as the reference. Benchmark quantized variants and runtime changes separately. Store reproducible measurements in `results/` and keep large weights in ignored `models/`.

Run `./violetto "Problem"` or `./violetto --interactive`. Setup is `./setup`; dependencies are pinned in `requirements.lock`. The default uses Q8, packed decode projections, shared RoPE tables, compiled XSA/residual operations and an equivalent simplified nucleus sampler. Each problem starts a fresh cache. No server is required.

Run commands with `.venv/bin/python`. GPU execution may need sandbox escalation to access Metal. Do not interpret a sandbox GPU failure as an unavailable GPU.

Validation commands: `.venv/bin/python scripts/verify_fixtures.py`, `PYTHONPATH=. .venv/bin/python scripts/verify_runtime.py`, and `PYTHONPATH=. .venv/bin/python scripts/verify_sampling.py`. The optional independent reference uses `.venv/bin/python scripts/torch_reference.py` with PyTorch 2.11.0. README.md documents benchmarks and evaluations. `scripts/report.py` rebuilds BENCHMARKS.md from the saved results.

## Status

Implemented and measured on September 25-26, 2026. Final default sampling measured 53.9 tokens/s versus BF16's 30.3. Greedy Q8 measured 71.8 versus 39.8 tokens/s; peak MLX allocations were 1.22 versus 2.16 GB. Q6 and Q4-HQ are available but have larger measured distribution changes. Q8 remains the quality-conscious default.

Runtime packing, fusion, tokenizer and cache checks passed. The simplified sampler matched stock masks and random draws at 359 positions and three seeds. The independent CPU reference is a limited cross-backend comparison, not CUDA bitwise equivalence. Keep source attribution in THIRD_PARTY.md and both license notices.

Final speed measurements are `results/final-*.json`; earlier measurements use different protocols and are retained as experiment history. Local math checks completed correctly on 9/12 BF16 cases and 10/12 Q8 cases within 4096 tokens. Remaining cases were incomplete. No competition-level quality preservation has been established.

The repository has no remote. Model weights, numerical reference arrays, environment and downloaded research are ignored by Git.

## Known limits and next work

- Long reasoning can exceed the default 4096-token output budget even on simple problems. Report truncation; users can explicitly increase `--max-tokens`.
- Full configured 131072-token context, GPU power, ANE execution and speculative decoding were not validated. Speed was tested at 128 and 4096 prompt tokens.
- A larger held-out mathematical evaluation with multiple seeds is needed before claiming unchanged mathematical ability.
- MLX-LM applies top-p before temperature. Preserve that documented local behavior unless intentionally changing and revalidating sampling.

## Constraints

- Preserve the checkpoint chat template, BF16 rounding sites, ordered MUDD sums, sigmoid logit cap, and inclusive 1025-key local window.
- Measure prompt processing and decode separately, with warmup and repetitions. Record memory and exact configurations.
- Mathematical answer checks and logit fidelity are separate evidence. Small local checks do not establish benchmark equivalence.
- Keep artifacts and code in English. No em dash characters.

## Do not

- Do not silently replace Violetto with another model or port Dwarf Star's unrelated architecture.
- Do not claim best possible speed or preserved quality without comparative measurements.
- Do not change global macOS settings, disable reasoning, truncate attention, or lower precision silently.
- Do not fuse multi-token prefill GEMM shapes: that changed BF16 rounding. Preserve partitioned prefill projections and only pack single-token decode.
- Do not use shapeless compilation for XSA: its repeat/reshape path needs shape-specific compilation.
- Do not publish or push this new project without a specified target.
