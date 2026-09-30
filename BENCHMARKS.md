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

Fresh runs without an artificial output cap: BF16 completed 12/12 cases at EOS and answered 12/12 correctly. Q8 completed 12/12 at EOS and answered 12/12 correctly. Only the configured 131072-token context, including the prompt, bounded generation. No time limit or reasoning shortcut was used.

| Case | Expected | BF16 | BF16 tokens | BF16 seconds | Q8 | Q8 tokens | Q8 seconds |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: |
| remainder | 4 | Correct | 1551 | 51.7 | Correct | 811 | 12.3 |
| divisors | 24 | Correct | 789 | 26.9 | Correct | 851 | 13.2 |
| quadratic | 133 | Correct | 750 | 25.6 | Correct | 549 | 8.0 |
| geometry | 84 | Correct | 722 | 24.7 | Correct | 679 | 9.9 |
| combinatorics | 35 | Correct | 1730 | 58.4 | Correct | 1950 | 28.6 |
| probability | 5/18 | Correct | 1243 | 36.1 | Correct | 797 | 11.6 |
| diophantine | 15 | Correct | 4203 | 108.5 | Correct | 1755 | 25.7 |
| recurrence | 6817 | Correct | 1019 | 25.9 | Correct | 1023 | 15.0 |
| factorial | 313 | Correct | 4267 | 109.9 | Correct | 6981 | 102.5 |
| lattice | 132 | Correct | 7715 | 201.5 | Correct | 6209 | 85.6 |
| crt | 23 | Correct | 1531 | 39.4 | Correct | 1020 | 13.6 |
| polynomial | 123 | Correct | 2850 | 74.0 | Correct | 3949 | 53.7 |

| Generation measure | BF16 | Q8 |
| --- | ---: | ---: |
| Total tokens | 28370 | 26574 |
| Median tokens per answer | 1541.0 | 1021.5 |
| Longest answer, tokens | 7715 | 6981 |
| Answers longer than 4096 tokens | 3 | 2 |
| Total elapsed seconds | 782.7 | 379.6 |
| Peak MLX allocation, GB | 2.24 | 1.26 |

Sampling used the stock MLX-LM sampler with temperature 0.6, top-p 0.95 and seed 42 plus the original case index. Both models reran every problem from a fresh cache. All Q8 cases used packed decode and compiled operations. The evaluator records the actual EOS ID and saves raw completions. Counts include the final EOS token, following MLX-LM's accounting. Elapsed times include the prompt and full response, exclude model loading, and are single runs with different response lengths, not fixed-work speed comparisons.

The earlier capped checks reported 9/12 completed correct BF16 answers and 10/12 Q8 answers. Their remaining answers were truncated, not demonstrated mathematical errors. These EOS runs supersede those quality results; the original records remain available as experiment history.

The [official release article](https://paradigma.inc/blog/limite-1b-violetto/) describes a context of up to 131k tokens. The output token counts above were measured locally.

These 12 hand-authored checks are too small and easy to establish unchanged competition-level mathematical accuracy. They do not justify a claim of zero quality loss. The independent answer-key check corrected one fixture before the final report; saved BF16 completions were regraded without changing their text.

## Scope and limitations

Only the final-* speed files represent the final benchmark protocol. Earlier files are experiment history, some with different packing and memory-residency settings. The Mac was not otherwise isolated from normal applications. Earlier measurements changed under memory pressure; final runs were sequential with no other project inference process running.

MLX allocation, RSS, CPU time, first-token latency and system swap are recorded. Continuous GPU utilization, frequency and electrical power were not measured. The 131072-token configured maximum has not been validated; the original speed tests covered 128 and 4096 prompt tokens. Additional decode comparisons at 8192 and 32768 tokens are reported separately below.

No trained compatible draft head is included in this checkpoint. Speculative decoding, ANE execution and a standalone C/Metal engine were not implemented. There is no evidence here that those alternatives would improve this small model on this M4. This is the fastest measured low-loss configuration among the implementations tested, not a claim of a hardware-wide optimum.

Run commands and source attribution are in README.md and THIRD_PARTY.md. Raw completions, measurements and code hashes are in results/.

## Publication review

The September 30 rerun with PyTorch 2.13.0 kept 20 of 21 top predictions and measured mean KL 0.002068 against MLX BF16. The historical CPU reference above used 2.11.0. Runtime packing and fusion again matched full logits exactly; local-cache behavior was checked on both sides of its 1025-key capacity. See AUDIT.md for the other checks.

## Additional decode experiments

Keep the existing Q8 default. The additional candidates did not establish a clear throughput benefit across the tested contexts. Compiled RoPE measured +0.5% at 128 prompt tokens, +0.9% at 8192 and -1.1% at 32768. These small changes overlap run variation. At 128 prompt tokens and 512 generated tokens, compiled RoPE reduced process CPU time by 6.5%, including prefill. RoPE, MUDD, the custom Metal kernel and MLP compilation remain reversible experiments.

Fresh alternating measurements on September 30, 2026, after the user closed another local model. Output counts and seeds are listed below. Warmups used 128 or 512 generated tokens per mode, with temperature 0.6, top-p 0.95 and forward/reverse mode blocks. These are separate from the older 128-output-token measurements above.

| Prompt tokens | Output tokens | Seed | Candidate | Baseline tokens/s | Candidate tokens/s | Gain | Peak MLX, GB |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 128 | 512 | 42 | Compiled MLX RoPE | 69.53 | 69.89 | +0.5% | 1.22 |
| 8192 | 256 | 42 | Compiled MLX RoPE | 61.92 | 62.47 | +0.9% | 1.62 |
| 32768 | 256 | 42 | Compiled MLX RoPE | 47.86 | 47.32 | -1.1% | 1.97 |
| 128 | 512 | 42 | Custom Metal RoPE | 69.67 | 70.14 | +0.7% | 1.33 |
| 128 | 512 | 42 | Compiled MLP + Metal RoPE | 69.67 | 69.98 | +0.5% | 1.33 |
| 128 | 512 | 42 | Compiled MLP | 69.67 | 66.97 | -3.9% | 1.33 |

The exactness checks covered 960 forced decode positions across RoPE, MUDD, and their combination, at prefixes of 1, 128, 1100, 8192 and 32768 tokens. Full capped logits and final KV cache arrays matched exactly. The long prefixes repeat a saved mathematical trace. This does not establish an AIME score.

The timing files record code hashes, memory availability, swap, all generated tokens and whether sampled traces matched. Earlier variable timings are retained as experiment history. The initial manual verification accumulated allocator buffers between prefill chunks; it was stopped and rerun with MLX-LM's normal cache-clearing policy. Its times are not speed evidence.

Broader attention correction/gating fusion changed a sampled trace and remains experimental. The grouped XSA implementation did not show a stable throughput benefit in the initial alternating run. No trained MTP head was added.

### Corrected component profile

Each component is synchronized, so these timings do not predict production tokens/s. The corrected prefill loop clears allocator buffers between chunks, matching the CLI.

| Context | Instrumented ms/token | Attention ms/token | MLP ms/token | MUDD ms/token | Peak MLX, GB | Allocator cache, GB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 45.55 | 17.67 | 17.98 | 1.08 | 1.22 | 0.02 |
| 8192 | 59.45 | 28.07 | 20.01 | 1.16 | 1.63 | 0.20 |
| 16384 | 50.55 | 22.88 | 18.34 | 0.95 | 1.75 | 0.30 |
| 32768 | 56.32 | 27.45 | 18.77 | 0.95 | 1.97 | 0.49 |

The old 32K profile reported 372.71 ms/token while its prefill loop accumulated allocator buffers. The corrected run measured 56.32 ms/token with 1.97 GB peak MLX allocations and 0.49 GB allocator cache. This fixes the diagnostic script; the production CLI already cleared these buffers. Attention dominates the corrected long-context component timings; MUDD is a small share.

### Batch and verification feasibility

A separate synchronous pilot used an identical 128-token prefix and 64 forced tokens, with a 16-token warmup and two measured runs per mode in forward/reverse order. It omits sampling, EOS, varying sequence lengths and completed answers. Verification computes all intermediate vocabulary distributions. Its rates must not be compared directly with the sampled generation tests above.

| Mode | Aggregate tokens/s | Tokens/s per sequence | Ratio to serial pilot | Max capped-logit difference |
| --- | ---: | ---: | ---: | ---: |
| Serial control | 60.83 | 60.83 | 1.00x | 0.000 |
| Batch of 2 | 116.76 | 58.38 | 1.92x | 0.416 |
| Batch of 4 | 212.90 | 53.23 | 3.50x | 0.424 |
| Verify 2 tokens | 116.92 | 116.92 | 1.92x | 0.332 |
| Verify 4 tokens | 231.22 | 231.22 | 3.80x | 0.332 |
| Verify 8 tokens | 324.48 | 324.48 | 5.33x | 0.393 |

The batch-of-four pilot suggests a useful way to reduce the total compute time of a future multi-problem evaluation. It duplicates one prefix at a shared length; it is not yet an AIME batch evaluator. Peak MLX allocations include retained logits for comparison.

All 64 forced positions kept the same top prediction, but batch and multi-token modes changed full logits. Therefore these results do not establish identical stochastic sampling or preserved mathematical accuracy.

The faster multi-token verification is a target-pass cost measurement, not an MTP generation speedup. A compatible trained drafter, acceptance/rejection sampling and safe cache rollback are still required. The rotating local cache is trimmable before its 1025-key capacity is reached, then reports `is_trimmable=False`; the pilot recorded a fresh cache only. No batch or speculative path was enabled in the CLI or evaluator.
