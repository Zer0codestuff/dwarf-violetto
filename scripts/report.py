"""Build the report from final measurements and saved answer checks."""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def read(name):
    return json.loads((RESULTS / name).read_text())


def merged_checks(names):
    cases = {}
    for name in names:
        data = read(name)
        for row in data["cases"]:
            cases[row["id"]] = {**row, "source": name}
    expected = {x["id"] for x in json.loads((ROOT / "data/math_checks.json").read_text())}
    if set(cases) != expected:
        raise ValueError(f"Incomplete evaluation: {expected - set(cases)}")
    return {"correct": sum(x["correct"] for x in cases.values()),
            "incomplete": sum(x["finish"] != "stop" for x in cases.values()),
            "cases": cases}


def eos_checks(name):
    data = read(name)
    if data.get("output_policy") != "eos_or_context_limit" or not data.get("completed_utc"):
        raise ValueError(f"Expected a completed evaluation without an artificial output cap: {name}")
    result = merged_checks([name])
    rows = list(result["cases"].values())
    counts = [row["tokens"] for row in rows]
    result["generation"] = {
        "total_tokens": sum(counts), "median_tokens": statistics.median(counts),
        "max_tokens": max(counts), "over_2048": sum(n > 2048 for n in counts),
        "over_4096": sum(n > 4096 for n in counts),
        "total_seconds": sum(row["seconds"] for row in rows),
        "peak_mlx_gb": max(row["peak_mlx_gb"] for row in rows),
    }
    return result


def decode_experiments():
    path = RESULTS / "decode-optimization-summary.json"
    if not path.exists():
        return []
    summary = json.loads(path.read_text())
    lines = ["## Additional decode experiments", "", summary["decision"], "",
             "Fresh alternating measurements on September 30, 2026, after the user closed another local model. "
             "Output counts and seeds are listed below. Warmups used 128 or 512 generated tokens per mode, "
             "with temperature 0.6, top-p 0.95 and forward/reverse mode blocks. "
             "These are separate from the older 128-output-token measurements above.", "",
             "| Prompt tokens | Output tokens | Seed | Candidate | Baseline tokens/s | Candidate tokens/s | Gain | Peak MLX, GB |",
             "| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |"]
    for name in summary["comparisons"]:
        data = read(name)
        baseline = data["median"]["baseline"]
        for mode, speed in data["median"].items():
            if mode == "baseline":
                continue
            peak = statistics.median(row["peak_mlx_gb"] for row in data["runs"] if row["mode"] == mode)
            label = {"compiled_rope": "Compiled MLX RoPE", "rope": "Custom Metal RoPE",
                     "mlp": "Compiled MLP", "mlp+rope": "Compiled MLP + Metal RoPE"}.get(mode, mode)
            lines.append(f"| {data['settings']['context']} | {data['settings']['tokens']} | {data['settings']['seed']} | {label} | "
                         f"{baseline:.2f} | {speed:.2f} | {speed / baseline - 1:+.1%} | {peak:.2f} |")
    checks = read("decode-candidate-correctness.json")
    lines += ["", f"The exactness checks covered {sum(row['positions'] for row in checks['checks'])} forced "
              "decode positions across RoPE, MUDD, and their combination, at prefixes of 1, 128, 1100, 8192 "
              "and 32768 tokens. Full capped logits and final KV cache arrays matched exactly. "
              "The long prefixes repeat a saved mathematical trace. This does not establish an AIME score.", "",
              "The timing files record code hashes, memory availability, swap, all generated tokens and "
              "whether sampled traces matched. Earlier variable timings are retained as experiment history. "
              "The initial manual verification accumulated allocator buffers between prefill chunks; it was "
              "stopped and rerun with MLX-LM's normal cache-clearing policy. Its times are not speed evidence.", "",
              "Broader attention correction/gating fusion changed a sampled trace and remains experimental. "
              "The grouped XSA implementation did not show a stable throughput benefit in the initial alternating run. "
              "No trained MTP head was added.", ""]
    profile_path = RESULTS / summary["profiler"]
    if profile_path.exists():
        profile = json.loads(profile_path.read_text())
        lines += ["### Corrected component profile", "",
                  "Each component is synchronized, so these timings do not predict production tokens/s. "
                  "The corrected prefill loop clears allocator buffers between chunks, matching the CLI.", "",
                  "| Context | Instrumented ms/token | Attention ms/token | MLP ms/token | MUDD ms/token | Peak MLX, GB | Allocator cache, GB |",
                  "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for row in profile["contexts"]:
            components = row["component_ms_per_token"]
            lines.append(f"| {row['context']} | {row['instrumented_ms_per_token']:.2f} | "
                         f"{components['Attention']:.2f} | {components['MLP']:.2f} | {components['MuddMixer']:.2f} | "
                         f"{row['peak_mlx_gb']:.2f} | {row['allocator_cache_gb']:.2f} |")
        lines += ["", "The old 32K profile reported 372.71 ms/token while its prefill loop accumulated allocator buffers. "
                  "The corrected run measured 56.32 ms/token with 1.97 GB peak MLX allocations and 0.49 GB allocator cache. "
                  "This fixes the diagnostic script; the production CLI already cleared these buffers. "
                  "Attention dominates the corrected long-context component timings; MUDD is a small share.", ""]
    pilot_path = RESULTS / "throughput-feasibility.json"
    if pilot_path.exists():
        pilot = json.loads(pilot_path.read_text())
        base = pilot["median"]["batch_1_block_1"]["aggregate_tps"]
        lines += ["### Batch and verification feasibility", "",
                  "A separate synchronous pilot used an identical 128-token prefix and 64 forced tokens, "
                  "with a 16-token warmup and two measured runs per mode in forward/reverse order. "
                  "It omits sampling, EOS, varying sequence lengths and completed answers. "
                  "Verification computes all intermediate vocabulary distributions. "
                  "Its rates must not be compared directly with the sampled generation tests above.", "",
                  "| Mode | Aggregate tokens/s | Tokens/s per sequence | Ratio to serial pilot | Max capped-logit difference |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        for batch, block, label in [(1, 1, "Serial control"), (2, 1, "Batch of 2"), (4, 1, "Batch of 4"),
                                    (1, 2, "Verify 2 tokens"), (1, 4, "Verify 4 tokens"), (1, 8, "Verify 8 tokens")]:
            data = pilot["median"][f"batch_{batch}_block_{block}"]
            difference = max(row["max_abs_logits_vs_serial"] for row in pilot["runs"]
                             if (row["batch"], row["block"]) == (batch, block))
            lines.append(f"| {label} | {data['aggregate_tps']:.2f} | {data['sequence_tps']:.2f} | "
                         f"{data['aggregate_tps']/base:.2f}x | {difference:.3f} |")
        lines += ["", "The batch-of-four pilot suggests a useful way to reduce the total compute time of a future "
                  "multi-problem evaluation. It duplicates one prefix at a shared length; it is not yet an AIME batch evaluator. "
                  "Peak MLX allocations include retained logits for comparison.", "",
                  "All 64 forced positions kept the same top prediction, but batch and multi-token modes changed full logits. "
                  "Therefore these results do not establish identical stochastic sampling or preserved mathematical accuracy.", "",
                  "The faster multi-token verification is a target-pass cost measurement, not an MTP generation speedup. "
                  "A compatible trained drafter, acceptance/rejection sampling and safe cache rollback are still required. "
                  "The rotating local cache is trimmable before its 1025-key capacity is reached, then reports "
                  "`is_trimmable=False`; the pilot recorded a fresh cache only. No batch or speculative path was enabled "
                  "in the CLI or evaluator.", ""]
    return lines


def main():
    runs = {key: read(file) for key, file in {
        "BF16": "final-bf16-128.json", "Q8 eager": "final-q8-eager-128.json",
        "Q8 optimized": "final-q8-128.json", "Q6 optimized": "final-q6-128.json",
        "Q4-HQ optimized": "final-q4-hq-128.json",
    }.items()}
    bf, q8 = runs["BF16"]["median"], runs["Q8 optimized"]["median"]
    bf_math = eos_checks("bf16-math-eos.json")
    q8_math = eos_checks("q8-math-eos.json")
    summary = {"selected": "Q8 optimized", "speedup": q8["decode_tps"] / bf["decode_tps"],
               "memory_reduction": 1-q8["peak_mlx_gb"] / bf["peak_mlx_gb"],
               "bf16_math": bf_math, "q8_math": q8_math,
               "runs": {k:v["median"] for k,v in runs.items()}}
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    lines = [
        "# M4 measurements", "",
        "Final isolated speed measurements on an Apple M4 MacBook Pro, 10 GPU cores, 16 GB unified memory, connected to AC power. macOS 27.0, MLX 0.32.2, MLX-LM 0.31.3. Dates and source hashes are recorded in each JSON result.", "",
        f"With greedy decoding, the selected Q8 runtime generated **{q8['decode_tps']:.1f} tokens/s**, versus **{bf['decode_tps']:.1f} tokens/s** for BF16. That is **{summary['speedup']:.2f}x** throughput. Peak MLX allocation fell from **{bf['peak_mlx_gb']:.2f} GB** to **{q8['peak_mlx_gb']:.2f} GB**.", "",
        "## Speed and memory", "",
        "128 prompt tokens, 128 output tokens, greedy decoding, 512-token prefill chunks. One 16-token warmup and three measured repetitions per configuration. Medians are shown. EOS is ignored for fixed work. The benchmark uses MLX-LM's per-process wired-memory context, matching the CLI generation path.", "",
        "| Configuration | Decode tokens/s | Prompt tokens/s | First token, s | Peak MLX, GB | RSS, GB |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name,data in runs.items():
        m=data["median"]
        lines.append(f"| {name} | {m['decode_tps']:.1f} | {m['prompt_tps']:.1f} | {m['ttft_seconds']:.3f} | {m['peak_mlx_gb']:.2f} | {m['rss_gb']:.2f} |")
    engine = q8["decode_tps"]/runs["Q8 eager"]["median"]["decode_tps"]-1
    if (RESULTS / "final-q8-fast-sampled.json").exists():
        lines += ["", "## Default sampling", "",
                  "Temperature 0.6 and top-p 0.95, using the same fixed-work protocol. These numbers include the cost of nucleus filtering and random sampling.", "",
                  "| Configuration | Decode tokens/s | Peak MLX, GB |", "| --- | ---: | ---: |"]
        for label,file in [("BF16, stock sampler","final-bf16-sampled.json"),
                           ("Q8, stock sampler","final-q8-sampled.json"),
                           ("Q8, simplified sampler","final-q8-fast-sampled.json")]:
            row=read(file)["median"]
            lines.append(f"| {label} | {row['decode_tps']:.1f} | {row['peak_mlx_gb']:.2f} |")
        lines += ["", "The simplified sampler produced identical top-p masks at all 359 tested positions and identical sampled tokens for three seeds. It preserves MLX-LM's order of operations, including top-p filtering before temperature scaling. This is not a claim of identical sampling behavior to vLLM."]
        original_sampled=read("final-bf16-sampled.json")["median"]["decode_tps"]
        default_sampled=read("final-q8-fast-sampled.json")["median"]["decode_tps"]
        stock_sampled=read("final-q8-sampled.json")["median"]["decode_tps"]
        lines += ["", f"The CLI uses the simplified sampler by default: **{default_sampled:.1f} tokens/s** versus **{original_sampled:.1f} tokens/s** for BF16 with stock sampling, or **{default_sampled/original_sampled:.2f}x**. Its {default_sampled/stock_sampled-1:.1%} gain over Q8 with stock sampling is small and may overlap normal run-to-run variation. The three 128-token sampled traces were identical. `--stock-sampler` restores the original implementation."]
        summary["default_sampling"]={"bf16_tps":original_sampled,"q8_tps":default_sampled,
                                     "speedup":default_sampled/original_sampled}
        (RESULTS / "summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    lines += ["",f"At the same Q8 precision, the runtime changes added **{engine:.1%}** decode throughput. Quantization and runtime gains are measured separately.","",
              f"Process CPU time for the fixed workload fell from {bf['cpu_seconds']:.2f} seconds to {q8['cpu_seconds']:.2f} seconds. This is CPU execution time, not whole-machine CPU utilization or electrical energy.", "",
              "## Longer prompt", "", "| Configuration | Prompt tokens | Decode tokens/s | Prompt tokens/s | First token, s | Peak MLX, GB |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name,file in [("BF16","final-bf16-4096.json"),("Q8 optimized","final-q8-4096.json")]:
        m=read(file)["median"]
        lines.append(f"| {name} | 4096 | {m['decode_tps']:.1f} | {m['prompt_tps']:.1f} | {m['ttft_seconds']:.3f} | {m['peak_mlx_gb']:.2f} |")
    lines += ["", "BF16 processed the longer prompt slightly faster. Q8's benefit is in token generation and memory, which matter more for long mathematical reasoning traces."]
    lines += ["", "## Numerical fidelity", "",
              "359 fixed positions from 12 mathematical prompts and worked solutions. KL compares complete capped next-token distributions, in nats. Top-1 agreement is not mathematical answer accuracy.", "",
              "| Quantization | Mean KL against BF16 | Top-1 agreement |", "| --- | ---: | ---: |"]
    for name,file in [("Q8","q8-fidelity-final.json"),("Q6 with Q8 embeddings","q6-fidelity-final.json"),("Q4 with Q8 embeddings","q4-hq-fidelity-final.json")]:
        d=read(file)
        lines.append(f"| {name} | {d['kl_mean']:.6f} | {d['top1_agreement']:.2%} |")
    lines += ["", "Q8 is the default because it stays closest to BF16. The faster Q6 and Q4 variants remain available explicitly.", "",
              "Compiled XSA and residual operations matched the unfused Q8 distributions exactly at all 359 positions. Packed prefill projections matched the eager Q8 logits exactly on the runtime test. Earlier packing of full prefill GEMMs changed rounding and was rejected. Single-token decode keeps the packed projections.", "",
              "The cache test crossed the 1025-key local window at position 1100 and preserved the top prediction. An independent PyTorch CPU translation agreed with MLX BF16 on 20 of 21 top predictions, with mean KL 0.001496. This is a limited cross-backend check, not bitwise equivalence with CUDA vLLM.", "",
              "## Mathematical answers", "",
              f"Fresh runs without an artificial output cap: BF16 completed {12-bf_math['incomplete']}/12 cases at EOS and answered {bf_math['correct']}/12 correctly. Q8 completed {12-q8_math['incomplete']}/12 at EOS and answered {q8_math['correct']}/12 correctly. Only the configured 131072-token context, including the prompt, bounded generation. No time limit or reasoning shortcut was used.", "",
              "| Case | Expected | BF16 | BF16 tokens | BF16 seconds | Q8 | Q8 tokens | Q8 seconds |",
              "| --- | --- | --- | ---: | ---: | --- | ---: | ---: |"]
    fixtures=json.loads((ROOT / "data/math_checks.json").read_text())
    for case in fixtures:
        def label(row):
            return "Correct" if row["correct"] else "Incomplete" if row["finish"] != "stop" else "Incorrect"
        b, q = bf_math['cases'][case['id']], q8_math['cases'][case['id']]
        lines.append(f"| {case['id']} | {case['answer']} | {label(b)} | {b['tokens']} | {b['seconds']:.1f} | {label(q)} | {q['tokens']} | {q['seconds']:.1f} |")
    lines += ["", "| Generation measure | BF16 | Q8 |", "| --- | ---: | ---: |"]
    for title, key, fmt in [("Total tokens", "total_tokens", ".0f"),
                             ("Median tokens per answer", "median_tokens", ".1f"),
                             ("Longest answer, tokens", "max_tokens", ".0f"),
                             ("Answers longer than 4096 tokens", "over_4096", ".0f"),
                             ("Total elapsed seconds", "total_seconds", ".1f"),
                             ("Peak MLX allocation, GB", "peak_mlx_gb", ".2f")]:
        lines.append(f"| {title} | {bf_math['generation'][key]:{fmt}} | {q8_math['generation'][key]:{fmt}} |")
    lines += ["", "Sampling used the stock MLX-LM sampler with temperature 0.6, top-p 0.95 and seed 42 plus the original case index. Both models reran every problem from a fresh cache. All Q8 cases used packed decode and compiled operations. The evaluator records the actual EOS ID and saves raw completions. Counts include the final EOS token, following MLX-LM's accounting. Elapsed times include the prompt and full response, exclude model loading, and are single runs with different response lengths, not fixed-work speed comparisons.", "",
              "The earlier capped checks reported 9/12 completed correct BF16 answers and 10/12 Q8 answers. Their remaining answers were truncated, not demonstrated mathematical errors. These EOS runs supersede those quality results; the original records remain available as experiment history.", "",
              "The [official release article](https://paradigma.inc/blog/limite-1b-violetto/) describes a context of up to 131k tokens. The output token counts above were measured locally.", "",
              "These 12 hand-authored checks are too small and easy to establish unchanged competition-level mathematical accuracy. They do not justify a claim of zero quality loss. The independent answer-key check corrected one fixture before the final report; saved BF16 completions were regraded without changing their text.", "",
              "## Scope and limitations", "",
              "Only the final-* speed files represent the final benchmark protocol. Earlier files are experiment history, some with different packing and memory-residency settings. The Mac was not otherwise isolated from normal applications. Earlier measurements changed under memory pressure; final runs were sequential with no other project inference process running.", "",
              "MLX allocation, RSS, CPU time, first-token latency and system swap are recorded. Continuous GPU utilization, frequency and electrical power were not measured. The 131072-token configured maximum has not been validated; the original speed tests covered 128 and 4096 prompt tokens. Additional decode comparisons at 8192 and 32768 tokens are reported separately below.", "",
              "No trained compatible draft head is included in this checkpoint. Speculative decoding, ANE execution and a standalone C/Metal engine were not implemented. There is no evidence here that those alternatives would improve this small model on this M4. This is the fastest measured low-loss configuration among the implementations tested, not a claim of a hardware-wide optimum.", "",
              "Run commands and source attribution are in README.md and THIRD_PARTY.md. Raw completions, measurements and code hashes are in results/.", ""]
    publication = RESULTS / "publication-runtime.json"
    if publication.exists():
        oracle = json.loads(publication.read_text()).get("torch_cpu_reference")
        if oracle:
            lines += ["## Publication review", "",
                      f"The September 30 rerun with PyTorch 2.13.0 kept 20 of 21 top predictions "
                      f"and measured mean KL {oracle['kl']:.6f} against MLX BF16. "
                      "The historical CPU reference above used 2.11.0. Runtime packing and "
                      "fusion again matched full logits exactly; local-cache behavior was checked "
                      "on both sides of its 1025-key capacity. See AUDIT.md for the other checks.", ""]
    lines += decode_experiments()
    (ROOT / "BENCHMARKS.md").write_text("\n".join(lines))
    print(f"Q8: {q8['decode_tps']:.1f} tok/s, {summary['speedup']:.2f}x BF16, {q8['peak_mlx_gb']:.2f} GB")


if __name__ == "__main__":
    main()
