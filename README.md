# Dwarf Violetto

Local inference and measured optimizations for [Paradigma's Limite 1B Violetto](https://huggingface.co/paradigma-inc/limite-1b-violetto), targeting an Apple M4 MacBook Pro with 16 GB of unified memory. Python orchestrates execution; MLX runs the tensor operations on Metal.

This is a model-specific MLX runtime, inspired by Dwarf Star's focus on measuring a narrow implementation. It is not a fork of Dwarf Star. The Limite architecture starts from Piergiorgio Di Pasquale's MLX port. See [THIRD_PARTY.md](THIRD_PARTY.md) for attribution and pinned revisions.

The September 25-26 M4 benchmark measured **53.9 tokens/s with default sampling**, versus 30.3 for BF16. Greedy decoding reached 71.8 tokens/s, versus 39.8 for BF16. Peak MLX allocations were 1.22 GB and 2.16 GB. At a 4096-token prompt, Q8 decoded at 66.9 tokens/s with greedy decoding. See [BENCHMARKS.md](BENCHMARKS.md) for methodology, accuracy checks and limitations.

## Quick start

Requirements: an Apple Silicon Mac, [uv](https://docs.astral.sh/uv/getting-started/installation/)
and about 8 GB of free disk space for the original checkpoint, Q8 and the Python
environment. The tested machine is an M4 MacBook Pro with 16 GB memory, running
macOS 27.0. Other Apple Silicon configurations are not benchmarked here.

Clone or download this repository, enter its directory, then run:

```sh
./setup
./violetto "How many positive divisors does 360 have?"
./violetto --interactive
./violetto --model models/original "Find the area of a triangle with sides 13, 14, 15."
```

The default is Q8. Each interactive problem is independent, as intended by Violetto. `/quit` exits. Ctrl+C interrupts a generation. The default sampling is temperature 0.6 and top-p 0.95, as recommended by Paradigma. The original mathematical chat template is preserved. Both end tokens stop generation.

Violetto writes long reasoning traces. Generation now continues until EOS by default, with no artificial output cap. The configured context holds 131072 tokens including the prompt. Exhausting it is reported as `context_limit`, not a completed answer. `--max-tokens 8192` is an optional, explicitly requested cap. A model answer is not verified merely because generation completed.

Generation uses only local weights and tokenizer files. Internet access is needed for initial setup, not for solving problems.

## Set up another Apple Silicon Mac

Install `uv`, then run:

```sh
./setup
```

This creates a project-local Python 3.12 environment, downloads the pinned official
checkpoint, checks its SHA256, and derives Q8 locally. It does not execute Python
from the model repository. Conversions use a temporary directory and expose the
finished checkpoint only after validating its files. Existing checkpoints are
preserved. If setup finds an incomplete Q8 directory from an older interrupted
conversion, rename that directory before running setup again.

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
.venv/bin/python -m dwarf_violetto.evaluate --model models/original --output results/bf16-math-eos.json
.venv/bin/python -m dwarf_violetto.evaluate --model models/q8 --optimized --fused --output results/q8-math-eos.json
```

Speed tests use fixed work, a warmup and three repetitions. They ignore EOS deliberately, count 127 decode intervals for 128 generated tokens, and report prompt processing separately from decoding. Prompt throughput includes time to the first token and is not a pure matrix-multiplication metric. The synthetic speed prompt is repeated and cut to the specified token count. Do not infer answer quality from it.

Mathematical checks use the checkpoint chat template, the recommended sampling settings and fixed seeds. All completions are saved. The 12 hand-authored problems are local regression checks, not an AIME score or evidence of unchanged competition-level ability. Token-distribution fidelity is a separate metric.

Run the two evaluations sequentially to avoid GPU contention. They have no output cap or time limit before EOS, other than the model context. The evaluator saves progress every 30 seconds and records the actual stop token, token count, elapsed time and peak MLX allocation. `--resume` skips completed cases and restarts an interrupted case with its original seed. It checks code, settings, package versions, weights, tokenizer and fixtures. Old records made before weight/tokenizer fingerprinting cannot be resumed with the new code. Use a different output path for a new run.

The previous 2048/4096-token answer checks are retained as historical, truncated runs. Their completion counts must not be interpreted as mathematical error rates. Fixed-length speed tests still use a token budget so each runtime performs the same amount of work.

For numerical fidelity:

```sh
.venv/bin/python -m dwarf_violetto.fidelity --reference
.venv/bin/python -m dwarf_violetto.fidelity --model models/q8 --optimized --fused --output results/fidelity.json
```

The fixed corpus samples every fourth position after the first eight tokens, spanning prompts and worked solutions. It compares the full capped probability distribution across the vocabulary, not raw uncapped logits. Reference arrays stay local and are ignored by Git.

For the independent CPU comparison and focused runtime checks:

```sh
uv pip install --python .venv/bin/python -r requirements-reference.txt
.venv/bin/python scripts/torch_reference.py
PYTHONPATH=. .venv/bin/python scripts/verify_runtime.py
.venv/bin/python scripts/verify_fixtures.py
```

PyTorch CPU and Metal do not use identical rounding. This checks agreement with an independent translation of the official formulas; it is not a live comparison with CUDA vLLM. The historical CPU check used PyTorch 2.11.0; the publication review repeats it with 2.13.0. PyTorch is optional and is not imported by inference.

CPU checks run without MLX, weights or a Mac:

```sh
python3 -m unittest discover -s tests -v
python3 scripts/verify_fixtures.py
python3 scripts/check_repository.py
python3 scripts/report.py
```

GitHub Actions runs these checks and verifies that the generated report is current.
It does not run Metal inference. On Apple Silicon, regenerate local Q8 arrays and
run the sampler check after setup:

```sh
.venv/bin/python -m dwarf_violetto.fidelity --model models/q8 --optimized --reference --reference-dir results/q8-reference
PYTHONPATH=. .venv/bin/python scripts/verify_sampling.py
```

The grader matches the last complete `\boxed{...}` answer, strips whitespace and
normalizes simple LaTeX fractions. It is a conservative fixture matcher, not a
symbolic mathematical verifier. A matching box in a truncated response does not
count as a completed correct answer.

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

The sampler validation reads the local Q8 reference arrays generated above.

No CPU/GPU global settings are changed. There is no pruning, fine-tuning, shortened attention window, or reduced reasoning budget hidden in the default. Speculative decoding has not been enabled: this checkpoint provides no trained draft head, and another full 48-layer pass needs a measured benefit to justify it. SSD offload is unnecessary at this model size.

## Additional decode experiments

The September 30 tests kept the existing Q8 default. Compiled RoPE, compiled MUDD and their combination matched full logits and final cache arrays at 960 forced decode positions, including 8K and 32K prefixes. The settled RoPE speed comparisons measured +0.5% at 128 prompt tokens, +0.9% at 8192 and -1.1% at 32768. Process CPU time fell 6.5% in the 128-prompt/512-output workload, including prefill. The small, mixed throughput changes do not establish a general speedup. Custom Metal RoPE and MLP compilation also failed to establish a clear throughput benefit. The scripts keep these candidates reversible and separate from the default path.

```sh
PYTHONPATH=. .venv/bin/python scripts/verify_decode_candidates.py
PYTHONPATH=. .venv/bin/python scripts/compare_decode.py --warmup 512 --modes baseline compiled_rope --output results/paired-check.json
PYTHONPATH=. .venv/bin/python scripts/compare_decode.py --context 32768 --tokens 256 --repeat 1 --output results/paired-long-check.json
PYTHONPATH=. .venv/bin/python scripts/profile_decode.py
```

Run inference measurements sequentially while other local inference and computer-use tasks are idle. The timing files record memory, code hashes, seeds, all generated tokens and GPU snapshots. Earlier variable series are retained and excluded from default selection. Component profiling synchronizes each operation and changes scheduling, so its times are diagnostic rather than production throughput.

Manual prefill loops clear the MLX allocator cache after each block, matching MLX-LM's generation policy. The first interrupted long-context verification lacked this step and accumulated temporary buffers. The corrected verification completed all checks. See `results/decode-optimization-summary.json` and the additional table in [BENCHMARKS.md](BENCHMARKS.md).

`PYTHONPATH=. .venv/bin/python scripts/throughput_feasibility.py` runs a short synchronous batch/verification pilot. Four identical sequences reached about 213 aggregate tokens/s versus 61 for its serial control. Verifying eight supplied tokens together reduced target-pass cost by about 5.3x. These are fixed-work computation rates without sampling or completed answers, not measured AIME or MTP generation speedups. Full logits changed, despite matching all 64 top predictions. A variable-length evaluator needs quality validation; speculative decoding also needs a trained drafter and safe local-cache rollback. The rotating cache permits trimming while its offset is below 1025, then becomes non-trimmable. The pilot's `true` value described a fresh cache.

`experiment_post.py` intentionally raises if its sampled trace changes. That
candidate was rejected and remains disabled. The other `experiment_*.py` files
are research scripts, not supported inference entry points. See
[results/README.md](results/README.md) for historical, excluded and current records.

## Limits

Results are specific to this M4, its power state, software versions and tested contexts. The configured 131072-token context is not a claim that this project has validated that full length. MLX peak allocation and process RSS are different measurements; neither is a GPU utilization or power reading.

Weights and caches are ignored by Git. No repository or service is published by setup or inference.

## License and contributions

Source code is licensed under [Apache-2.0](LICENSE). The sampler adapted from
MLX-LM retains its [MIT notice](licenses/MLX-LM-MIT.txt). Paradigma's model weights
are downloaded separately under Apache-2.0. This is an independent project; see
[THIRD_PARTY.md](THIRD_PARTY.md) and [NOTICE](NOTICE) for attribution.

See [CONTRIBUTING.md](CONTRIBUTING.md) for validation and measurement requirements.
