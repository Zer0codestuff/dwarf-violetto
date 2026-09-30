"""Fixed-work batch and multi-token verification pilots, not answer-quality tests."""
import hashlib
import argparse
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm.generate import wired_limit

from dwarf_violetto.limite import Model
from dwarf_violetto.provenance import provenance
from dwarf_violetto.runtime import load, prompt_tokens


def measure(model, prefix, suffix, batch, block):
    cache = model.make_cache()
    mx.synchronize()
    mx.reset_peak_memory()
    start = time.perf_counter()
    logits = model(mx.array([prefix] * batch), cache)
    mx.eval(logits)
    prefill_seconds = time.perf_counter() - start
    mx.clear_cache()
    outputs = []
    start = time.perf_counter()
    for offset in range(0, len(suffix), block):
        # Verification needs every vocabulary distribution, including intermediate tokens.
        logits = Model.__call__(model, mx.array([suffix[offset:offset + block]] * batch), cache)
        mx.eval(logits)
        outputs.append(logits)
    seconds = time.perf_counter() - start
    return {
        "batch": batch, "block": block, "prefix_tokens": len(prefix),
        "forced_tokens_per_sequence": len(suffix), "prefill_seconds": prefill_seconds,
        "decode_seconds": seconds, "sequence_tps": len(suffix) / seconds,
        "aggregate_tps": batch * len(suffix) / seconds,
        "peak_mlx_gb": mx.get_peak_memory() / 1e9,
    }, mx.concatenate(outputs, axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/throughput-feasibility.json")
    args = parser.parse_args()
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    model, tokenizer, _ = load("models/q8", optimized=True, fused=True)
    problem = "Compute the last three digits of 1! + 2! + ... + 20!."
    cases = json.loads(Path("results/q8-math-eos.json").read_text())["cases"]
    text = next(case["text"] for case in cases if case["id"] == "factorial")
    ids = prompt_tokens(tokenizer, problem) + tokenizer.encode(text, add_special_tokens=False)
    prefix, suffix = ids[:128], ids[128:192]
    modes = [(1, 1), (2, 1), (4, 1), (1, 2), (1, 4), (1, 8)]
    local_cache = model.make_cache()[next(i for i in range(len(model.layers)) if i not in model.model._global)]
    result = {
        "sha256": provenance("models/q8"),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "method": "Synchronous forced decoding after an identical 128-token prefix. Two measured runs "
                  "per mode in forward/reverse order, after a 16-token warmup. No sampling or EOS. "
                  "Batch rows duplicate the same prefix at a shared length. Verification computes all "
                  "intermediate vocabulary logits. This is not a variable-length batch evaluator or MTP decoder.",
        "local_cache_trimmable": local_cache.is_trimmable(),
        "local_cache_trimmable_scope": "Fresh cache only. RotatingKVCache.is_trimmable() becomes false at max_size.",
        "runs": [],
    }
    reference = None
    with wired_limit(model):
        for batch, block in modes:
            _, output = measure(model, prefix, suffix[:16], batch, block)
            del output
            mx.clear_cache()
        for batch, block in modes + list(reversed(modes)):
            row, output = measure(model, prefix, suffix, batch, block)
            mx.eval(output)
            if reference is None:
                reference = output
            row["max_abs_logits_vs_serial"] = float(mx.max(mx.abs(output - reference)))
            row["top1_agreement_vs_serial"] = float(mx.mean(
                (mx.argmax(output, axis=-1) == mx.argmax(reference, axis=-1)).astype(mx.float32)
            ))
            result["runs"].append(row)
            dest.write_text(json.dumps(result, indent=2) + "\n")
            print(row, flush=True)
            del output
            mx.clear_cache()
    result["median"] = {
        f"batch_{batch}_block_{block}": {
            key: statistics.median(row[key] for row in result["runs"]
                                   if (row["batch"], row["block"]) == (batch, block))
            for key in ("decode_seconds", "aggregate_tps", "sequence_tps", "peak_mlx_gb")
        }
        for batch, block in modes
    }
    dest.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
