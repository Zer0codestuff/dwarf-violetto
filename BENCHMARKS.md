# M4 measurements

Final isolated speed measurements on an Apple M4 MacBook Pro, 10 GPU cores, 16 GB unified memory, connected to AC power. macOS 27.0, MLX 0.32.2, MLX-LM 0.31.3. Dates and source hashes are recorded in each JSON result.

With greedy decoding, the selected Q8 runtime generated **71.8 tokens/s**, versus **39.8 tokens/s** for BF16. That is **1.80x** throughput. Peak MLX allocation fell from **2.16 GB** to **1.22 GB**.

## Speed and memory

128 prompt tokens, 128 output tokens, greedy decoding, 512-token prefill chunks. One 16-token warmup and three measured repetitions per configuration. Medians are shown. EOS is ignored for fixed work. The benchmark uses MLX-LM's per-process wired-memory context, matching the CLI generation path.

| Configuration | Decode tokens/s | Prompt tokens/s | First token, s | Peak MLX, GB | RSS, GB |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 | 39.8 | 963.6 | 0.133 | 2.16 | 1.03 |
| Q8 eager | 67.5 | 1117.1 | 0.115 | 1.19 | 1.65 |
| Q8 optimized | 71.8 | 1164.6 | 0.110 | 1.22 | 1.10 |
| Q6 optimized | 83.7 | 1202.6 | 0.106 | 1.02 | 1.03 |
| Q4-HQ optimized | 102.7 | 1268.5 | 0.101 | 0.82 | 0.96 |

## Default sampling

Temperature 0.6 and top-p 0.95, using the same fixed-work protocol. These numbers include the cost of nucleus filtering and random sampling.

| Configuration | Decode tokens/s | Peak MLX, GB |
| --- | ---: | ---: |
| BF16, stock sampler | 30.3 | 2.16 |
| Q8, stock sampler | 52.9 | 1.22 |
| Q8, simplified sampler | 53.9 | 1.22 |

The simplified sampler produced identical top-p masks at all 359 tested positions and identical sampled tokens for three seeds. It preserves MLX-LM's order of operations, including top-p filtering before temperature scaling. This is not a claim of identical sampling behavior to vLLM.

The CLI uses the simplified sampler by default: **53.9 tokens/s** versus **30.3 tokens/s** for BF16 with stock sampling, or **1.78x**. Its 1.9% gain over Q8 with stock sampling is small and may overlap normal run-to-run variation. The three 128-token sampled traces were identical. `--stock-sampler` restores the original implementation.

At the same Q8 precision, the runtime changes added **6.3%** decode throughput. Quantization and runtime gains are measured separately.

Process CPU time for the fixed workload fell from 1.91 seconds to 1.24 seconds. This is CPU execution time, not whole-machine CPU utilization or electrical energy.

## Longer prompt

| Configuration | Prompt tokens | Decode tokens/s | Prompt tokens/s | First token, s | Peak MLX, GB |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 | 4096 | 38.5 | 1475.8 | 2.776 | 2.44 |
| Q8 optimized | 4096 | 66.9 | 1392.3 | 2.942 | 1.56 |

BF16 processed the longer prompt slightly faster. Q8's benefit is in token generation and memory, which matter more for long mathematical reasoning traces.

## Numerical fidelity

359 fixed positions from 12 mathematical prompts and worked solutions. KL compares complete capped next-token distributions, in nats. Top-1 agreement is not mathematical answer accuracy.

| Quantization | Mean KL against BF16 | Top-1 agreement |
| --- | ---: | ---: |
| Q8 | 0.001434 | 98.89% |
| Q6 with Q8 embeddings | 0.004753 | 94.43% |
| Q4 with Q8 embeddings | 0.047854 | 90.53% |

Q8 is the default because it stays closest to BF16. The faster Q6 and Q4 variants remain available explicitly.

Compiled XSA and residual operations matched the unfused Q8 distributions exactly at all 359 positions. Packed prefill projections matched the eager Q8 logits exactly on the runtime test. Earlier packing of full prefill GEMMs changed rounding and was rejected. Single-token decode keeps the packed projections.

The cache test crossed the 1025-key local window at position 1100 and preserved the top prediction. An independent PyTorch CPU translation agreed with MLX BF16 on 20 of 21 top predictions, with mean KL 0.001496. This is a limited cross-backend check, not bitwise equivalence with CUDA vLLM.

## Mathematical answers

With up to 4096 output tokens, BF16 completed 9/12 cases and answered 9/12 correctly. Q8 completed 10/12 and answered 10/12 correctly.

| Case | Expected | BF16 | Q8 |
| --- | --- | --- | --- |
| remainder | 4 | Correct | Correct |
| divisors | 24 | Correct | Correct |
| quadratic | 133 | Correct | Correct |
| geometry | 84 | Correct | Correct |
| combinatorics | 35 | Correct | Correct |
| probability | 5/18 | Correct | Correct |
| diophantine | 15 | Incomplete | Correct |
| recurrence | 6817 | Correct | Correct |
| factorial | 313 | Incomplete | Incomplete |
| lattice | 132 | Incomplete | Incomplete |
| crt | 23 | Correct | Correct |
| polynomial | 123 | Correct | Correct |

Sampling used temperature 0.6, top-p 0.95 and seed 42 plus the original case index. BF16 responses that hit the initial 2048-token budget were rerun with the same seeds and a 4096-token budget. Eight Q8 cases used the unfused optimized runtime; four used the compiled path. That distinction is retained in the raw records. Compilation equivalence was checked separately.

These 12 hand-authored checks are too small and easy to establish unchanged competition-level mathematical accuracy. They do not justify a claim of zero quality loss. The independent answer-key check corrected one fixture before the final report; saved BF16 completions were regraded without changing their text.

## Scope and limitations

Only the final-* speed files represent the final benchmark protocol. Earlier files are experiment history, some with different packing and memory-residency settings. The Mac was not otherwise isolated from normal applications. Earlier measurements changed under memory pressure; final runs were sequential with no other project inference process running.

MLX allocation, RSS, CPU time, first-token latency and system swap are recorded. GPU utilization and electrical power were not measured. The 131072-token configured maximum has not been validated; speed was measured at 128 and 4096 prompt tokens.

No trained compatible draft head is included in this checkpoint. Speculative decoding, ANE execution and a standalone C/Metal engine were not implemented. There is no evidence here that those alternatives would improve this small model on this M4. This is the fastest measured low-loss configuration among the implementations tested, not a claim of a hardware-wide optimum.

Run commands and source attribution are in README.md and THIRD_PARTY.md. Raw completions, measurements and code hashes are in results/.
