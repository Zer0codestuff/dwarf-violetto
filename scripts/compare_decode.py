"""Alternating fixed-work comparisons for exact decode candidates."""
import argparse
import hashlib
import json
import re
import statistics
import subprocess
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import psutil

from dwarf_violetto.bench import measure
from dwarf_violetto.provenance import provenance, script_hashes
from dwarf_violetto.generation import validate_sampling
from dwarf_violetto.runtime import load, prompt_tokens
from dwarf_violetto.sampling import make_sampler
from decode_candidates import DecodeCandidates


def gpu_snapshot():
    output = subprocess.run(["ioreg", "-r", "-c", "AGXAccelerator", "-l"],
                            capture_output=True, text=True, check=True).stdout
    match = re.search(r'"PerformanceStatistics" = \{([^\n]+)\}', output)
    if match is None:
        return None
    stats = {name: int(value) for name, value in re.findall(r'"([^"]+)"=(-?\d+)', match.group(1))}
    return {name: stats[name] for name in (
        "Device Utilization %", "Renderer Utilization %", "Tiler Utilization %", "In use system memory"
    ) if name in stats}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--output", required=True)
    parser.add_argument("--modes", nargs="+", choices=DecodeCandidates.modes,
                        default=["baseline", "compiled_rope"])
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--temp", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=128)
    args = parser.parse_args()
    try:
        validate_sampling(args.temp, 0.95)
    except ValueError as error:
        parser.error(str(error))
    if args.context < 1 or args.tokens < 2 or args.repeat < 1 or args.warmup < 2:
        parser.error("Use a positive context and repetition count, and at least two output tokens")
    model, tokenizer, _ = load("models/q8", optimized=True, fused=True)
    if args.context + max(args.tokens, args.warmup) > model.args.max_position_embeddings:
        parser.error("The prompt and output budget exceed the configured model context")
    ids = prompt_tokens(tokenizer, "Find the last three digits of 1! + 2! + ... + 20!.")
    ids = (ids * (args.context // len(ids) + 1))[:args.context]
    sampler = make_sampler(args.temp, top_p=0.95)
    scripts = [Path(__file__), Path("scripts/decode_candidates.py"), Path("scripts/experiment_xsa.py")]
    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "packages": {name: version(name) for name in ("mlx", "mlx-lm", "transformers")},
        "method": "Each mode warms up with the configured number of generated tokens. Forward and reverse blocks alternate. "
                  "EOS is ignored for fixed work. Synthetic repeated prompt, not an answer quality test.",
        "settings": vars(args), "sha256": provenance("models/q8"),
        "script_sha256": script_hashes(scripts),
        "system_memory_before_warmup": {
            "available_gb": psutil.virtual_memory().available / 1e9,
            "swap_used_gb": psutil.swap_memory().used / 1e9,
        },
        "gpu_before_warmup": gpu_snapshot(),
        "runs": [],
    }
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with DecodeCandidates(model) as candidates:
        for mode in args.modes:
            candidates.enable(mode)
            measure(model, ids, args.warmup, 512, sampler, args.seed)
        for mode in (args.modes + list(reversed(args.modes))) * args.repeat:
            candidates.enable(mode)
            available_before = psutil.virtual_memory().available / 1e9
            swap_before = psutil.swap_memory().used / 1e9
            gpu_before = gpu_snapshot()
            row = measure(model, ids, args.tokens, 512, sampler, args.seed)
            row.update(mode=mode, available_gb=psutil.virtual_memory().available / 1e9,
                       swap_used_gb=psutil.swap_memory().used / 1e9,
                       available_before_gb=available_before, swap_before_gb=swap_before,
                       gpu_before=gpu_before, gpu_after=gpu_snapshot())
            result["runs"].append(row)
            dest.write_text(json.dumps(result, indent=2) + "\n")
            print(mode, round(row["decode_tps"], 2), "peak", round(row["peak_mlx_gb"], 2), flush=True)
    result["sampled_trace_identical"] = all(
        row["tokens"] == result["runs"][0]["tokens"] for row in result["runs"]
    )
    result["median"] = {
        mode: statistics.median(row["decode_tps"] for row in result["runs"] if row["mode"] == mode)
        for mode in args.modes
    }
    dest.write_text(json.dumps(result, indent=2) + "\n")
    print(result["median"], "identical:", result["sampled_trace_identical"], flush=True)


if __name__ == "__main__":
    main()
