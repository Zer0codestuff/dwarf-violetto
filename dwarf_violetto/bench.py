"""Fixed-token speed tests. EOS is intentionally ignored for comparable work."""
import argparse
import json
import platform
import statistics
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import mlx.core as mx
import psutil
from mlx_lm.generate import generate_step, generation_stream, wired_limit
from mlx_lm.sample_utils import make_sampler
from .runtime import load, prompt_tokens
from .provenance import provenance


def measure(model, tokens, count, chunk, sampler=None, seed=42):
    mx.random.seed(seed)
    mx.synchronize()
    mx.reset_peak_memory()
    proc = psutil.Process()
    cpu0 = proc.cpu_times()
    start = time.perf_counter()
    first = None
    generated = []
    generator = generate_step(mx.array(tokens), model, max_tokens=count,
                              prefill_step_size=chunk, sampler=sampler)
    with wired_limit(model, [generation_stream]):
        for token, _ in generator:
            generated.append(int(token))
            if first is None:
                first = time.perf_counter()
    end = time.perf_counter()
    cpu1 = proc.cpu_times()
    return {
        "prompt_tokens": len(tokens), "generated_tokens": len(generated),
        "ttft_seconds": first - start,
        "prompt_tps": len(tokens) / (first - start),
        "decode_tps": (len(generated) - 1) / (end - first),
        "wall_seconds": end - start,
        "cpu_seconds": cpu1.user + cpu1.system - cpu0.user - cpu0.system,
        "peak_mlx_gb": mx.get_peak_memory() / 1e9,
        "rss_gb": proc.memory_info().rss / 1e9,
        "tokens": generated,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/original")
    p.add_argument("--optimized", action="store_true")
    p.add_argument("--fused", action="store_true")
    p.add_argument("--tokens", type=int, default=128)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--context", type=int, default=128)
    p.add_argument("--chunk", type=int, default=512)
    p.add_argument("--temp", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fast-sampler", action="store_true")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    if args.tokens < 2 or args.repeat < 1 or args.context < 1:
        p.error("Use at least 2 output tokens, 1 repetition and 1 context token")
    model, tokenizer, _ = load(args.model, args.optimized, args.fused)
    ids = prompt_tokens(tokenizer, "Find all positive integers n such that n squared plus 3n plus 2 is divisible by 12. Explain each step.")
    ids = (ids * ((args.context // len(ids)) + 1))[:args.context]
    if args.fast_sampler:
        from .sampling import make_sampler as sampler_factory
    else:
        sampler_factory = make_sampler
    sampler = sampler_factory(args.temp, top_p=args.top_p)
    measure(model, ids, 16, args.chunk, sampler, args.seed)
    runs = []
    for i in range(args.repeat):
        row = measure(model, ids, args.tokens, args.chunk, sampler, args.seed)
        runs.append(row)
        print(f"{i + 1}: prefill={row['prompt_tps']:.1f} decode={row['decode_tps']:.1f} tok/s peak={row['peak_mlx_gb']:.2f} GB", flush=True)
    result = {"timestamp": datetime.now(timezone.utc).isoformat(),
              "platform": platform.platform(), "chip": "Apple M4, 10-core GPU, 16 GB",
              "system_memory": {"available_gb": psutil.virtual_memory().available / 1e9,
                                "swap_used_gb": psutil.swap_memory().used / 1e9},
              "packages": {x: version(x) for x in ["mlx", "mlx-lm", "transformers"]},
              "settings": vars(args), "sha256": provenance(args.model), "runs": runs,
              "median": {k: statistics.median(x[k] for x in runs)
                         for k in ["prompt_tps", "decode_tps", "ttft_seconds", "peak_mlx_gb", "rss_gb", "cpu_seconds"]}}
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
