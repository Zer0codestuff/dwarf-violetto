# Dwarf Violetto

Local inference and measured optimizations for [Paradigma's Limite 1B Violetto](https://huggingface.co/paradigma-inc/limite-1b-violetto), targeting an Apple M4 MacBook Pro with 16 GB of unified memory. Python orchestrates execution; MLX runs the tensor operations on Metal.

This is a model-specific MLX runtime, inspired by Dwarf Star's focus on measuring a narrow implementation. It is not a fork of Dwarf Star. The Limite architecture starts from Piergiorgio Di Pasquale's MLX port. See [THIRD_PARTY.md](THIRD_PARTY.md) for attribution and pinned revisions.

The final M4 benchmark measured **53.9 tokens/s with default sampling**, versus 30.3 for BF16. Greedy decoding reached 71.8 tokens/s, versus 39.8 for BF16. Peak MLX allocations were 1.22 GB and 2.16 GB. At a 4096-token prompt, Q8 decoded at 66.9 tokens/s with greedy decoding. See [BENCHMARKS.md](BENCHMARKS.md) for methodology, accuracy checks and limitations.

## Use

The environment and checkpoints are already installed on this Mac.

```sh
./violetto "How many positive divisors does 360 have?"
./violetto --interactive
./violetto --model models/original "Find the area of a triangle with sides 13, 14, 15."
```

The default is Q8. Each interactive problem is independent, as intended by Violetto. `/quit` exits. Ctrl+C interrupts a generation. The default sampling is temperature 0.6 and top-p 0.95, as recommended by Paradigma. The original mathematical chat template is preserved. Both end tokens stop generation.

Violetto writes long reasoning traces. The default output budget is 4096 tokens; `--max-tokens 8192` increases it. Reaching the limit is reported explicitly. A model answer is not verified merely because generation completed.

Generation uses only local weights and tokenizer files. Internet access is needed for initial setup, not for solving problems.

## Set up another Apple Silicon Mac

Install `uv`, then run:

```sh
./setup
```

This creates a project-local Python 3.12 environment, downloads the pinned official checkpoint, and derives Q8 locally. It does not execute Python from the model repository. Keep several GB free for the original weights, converted weights and environment.

Optional variants:

```sh
.venv/bin/python -m dwarf_violetto.prepare --recipe q6
.venv/bin/python -m dwarf_violetto.prepare --recipe q4-hq
```

All quantization uses groups of 64. Q6 uses 6-bit projection weights and 8-bit embedding tables. Q4-HQ uses 4-bit projections and 8-bit embedding tables. Both tables are protected, including the input embedding that also serves as the output head. These are project recipes, not the similarly named community quantizations.

## Measure

```sh
.venv/bin/python -m dwarf_violetto.bench --model models/original --output results/bf16.json
.venv/bin/python -m dwarf_violetto.bench --model models/q8 --optimized --fused --output results/q8.json
.venv/bin/python -m dwarf_violetto.bench --model models/q8 --optimized --fused --context 4096 --output results/q8-4096.json
.venv/bin/python -m dwarf_violetto.evaluate --model models/q8 --optimized --fused --max-tokens 4096 --output results/math.json
```

Speed tests use fixed work, a warmup and three repetitions. They ignore EOS deliberately, count 127 decode intervals for 128 generated tokens, and report prompt processing separately from decoding. Prompt throughput includes time to the first token and is not a pure matrix-multiplication metric. The synthetic speed prompt is repeated and cut to the specified token count. Do not infer answer quality from it.

Mathematical checks use the checkpoint chat template, the recommended sampling settings and fixed seeds. All completions are saved. The 12 hand-authored problems are local regression checks, not an AIME score or evidence of unchanged competition-level ability. Token-distribution fidelity is a separate metric.

For numerical fidelity:

```sh
.venv/bin/python -m dwarf_violetto.fidelity --reference
.venv/bin/python -m dwarf_violetto.fidelity --model models/q8 --optimized --fused --output results/fidelity.json
```

The fixed corpus samples every fourth position after the first eight tokens, spanning prompts and worked solutions. It compares the full capped probability distribution across the vocabulary, not raw uncapped logits. Reference arrays stay local and are ignored by Git.

For the independent CPU comparison and focused runtime checks:

```sh
uv pip install --python .venv/bin/python torch==2.11.0
.venv/bin/python scripts/torch_reference.py
PYTHONPATH=. .venv/bin/python scripts/verify_runtime.py
.venv/bin/python scripts/verify_fixtures.py
```

PyTorch CPU and Metal do not use identical rounding. This checks agreement with an independent translation of the official formulas; it is not a live comparison with CUDA vLLM.

## Runtime choices

- Fold learned projection scales once, preserving BF16 rounding.
- Pack Q/K/V and gate/up projections for single-token decoding. During prefill, retain the original GEMM shapes because changing them altered BF16 rounding.
- Reuse position tables across local attention layers.
- Use Metal attention, full global KV caches and bounded 1025-key local caches.
- Compute only the last vocabulary projection in the inference interface. The analysis interface retains all positions.
- Keep MUDD accumulation order, dtype-dependent RMS normalization, partial RoPE and sigmoid logit capping.

The CLI also enables compiled XSA and residual operations. `--no-fused` disables them for comparison. Benchmark and evaluation commands require `--fused` explicitly. The compiled path matched all 359 sampled token distributions exactly and preserved the 128-token greedy trace in the short speed test.

The default sampler scatters the top-p retention mask directly to vocabulary order. It preserves the original MLX-LM masks and random draws; its measured speed gain is small. `--stock-sampler` restores the original implementation. To measure the default CLI sampling path:

```sh
.venv/bin/python -m dwarf_violetto.bench --model models/q8 --optimized --fused --temp 0.6 --top-p 0.95 --fast-sampler --output results/default-sampling.json
PYTHONPATH=. .venv/bin/python scripts/verify_sampling.py
```

The sampler validation reads the local Q8 reference arrays. Regenerate them with `python -m dwarf_violetto.fidelity --model models/q8 --optimized --reference --reference-dir results/q8-reference` inside the activated environment if needed.

No CPU/GPU global settings are changed. There is no pruning, fine-tuning, shortened attention window, or reduced reasoning budget hidden in the default. Speculative decoding has not been enabled: this checkpoint provides no trained draft head, and another full 48-layer pass needs a measured benefit to justify it. SSD offload is unnecessary at this model size.

## Limits

Results are specific to this M4, its power state, software versions and tested contexts. The configured 131072-token context is not a claim that this project has validated that full length. MLX peak allocation and process RSS are different measurements; neither is a GPU utilization or power reading.

Weights and caches are ignored by Git. No repository or service is published by setup or inference.
