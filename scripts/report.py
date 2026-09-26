"""Build the report from final measurements and saved answer checks."""
import json
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


def main():
    runs = {key: read(file) for key, file in {
        "BF16": "final-bf16-128.json", "Q8 eager": "final-q8-eager-128.json",
        "Q8 optimized": "final-q8-128.json", "Q6 optimized": "final-q6-128.json",
        "Q4-HQ optimized": "final-q4-hq-128.json",
    }.items()}
    bf, q8 = runs["BF16"]["median"], runs["Q8 optimized"]["median"]
    bf_math = merged_checks(["bf16-math.json", "bf16-math-4096-retry.json"])
    q8_math = merged_checks(["q8-math.json", "q8-math-remaining.json"])
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
              f"With up to 4096 output tokens, BF16 completed {12-bf_math['incomplete']}/12 cases and answered {bf_math['correct']}/12 correctly. Q8 completed {12-q8_math['incomplete']}/12 and answered {q8_math['correct']}/12 correctly.", "",
              "| Case | Expected | BF16 | Q8 |", "| --- | --- | --- | --- |"]
    fixtures=json.loads((ROOT / "data/math_checks.json").read_text())
    for case in fixtures:
        def label(row):
            return "Correct" if row["correct"] else "Incomplete" if row["finish"] != "stop" else "Incorrect"
        lines.append(f"| {case['id']} | {case['answer']} | {label(bf_math['cases'][case['id']])} | {label(q8_math['cases'][case['id']])} |")
    lines += ["", "Sampling used temperature 0.6, top-p 0.95 and seed 42 plus the original case index. BF16 responses that hit the initial 2048-token budget were rerun with the same seeds and a 4096-token budget. Eight Q8 cases used the unfused optimized runtime; four used the compiled path. That distinction is retained in the raw records. Compilation equivalence was checked separately.", "",
              "These 12 hand-authored checks are too small and easy to establish unchanged competition-level mathematical accuracy. They do not justify a claim of zero quality loss. The independent answer-key check corrected one fixture before the final report; saved BF16 completions were regraded without changing their text.", "",
              "## Scope and limitations", "",
              "Only the final-* speed files represent the final benchmark protocol. Earlier files are experiment history, some with different packing and memory-residency settings. The Mac was not otherwise isolated from normal applications. Earlier measurements changed under memory pressure; final runs were sequential with no other project inference process running.", "",
              "MLX allocation, RSS, CPU time, first-token latency and system swap are recorded. GPU utilization and electrical power were not measured. The 131072-token configured maximum has not been validated; speed was measured at 128 and 4096 prompt tokens.", "",
              "No trained compatible draft head is included in this checkpoint. Speculative decoding, ANE execution and a standalone C/Metal engine were not implemented. There is no evidence here that those alternatives would improve this small model on this M4. This is the fastest measured low-loss configuration among the implementations tested, not a claim of a hardware-wide optimum.", "",
              "Run commands and source attribution are in README.md and THIRD_PARTY.md. Raw completions, measurements and code hashes are in results/.", ""]
    (ROOT / "BENCHMARKS.md").write_text("\n".join(lines))
    print(f"Q8: {q8['decode_tps']:.1f} tok/s, {summary['speedup']:.2f}x BF16, {q8['peak_mlx_gb']:.2f} GB")


if __name__ == "__main__":
    main()
