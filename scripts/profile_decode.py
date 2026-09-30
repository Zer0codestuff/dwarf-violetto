"""Synchronized component timings, distinct from uninterrupted generation speed."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import mlx.core as mx
from mlx_lm.generate import wired_limit
from dwarf_violetto.runtime import load, prompt_tokens
from dwarf_violetto.limite import Attention, MLP, MuddMixer
from dwarf_violetto.sampling import make_sampler
from dwarf_violetto.provenance import provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/decode-profile-cleared.json")
    args = parser.parse_args()
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    model, tokenizer, _ = load("models/q8", optimized=True, fused=True)
    source = json.loads(Path("results/q8-math-eos.json").read_text())
    trace = next(x["text"] for x in source["cases"] if x["id"] == "factorial")
    ids = prompt_tokens(tokenizer, "Compute the last three digits of 1! + 2! + ... + 20!.")
    ids += tokenizer.encode(trace, add_special_tokens=False)
    sampler = make_sampler(0.6, top_p=0.95)
    result = {"sha256": provenance("models/q8"),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "method":
              "Eight forced decode steps per context. Synchronize after each component. "
              "Instrumentation changes scheduling; component times are diagnostic, not production throughput. "
              "Long contexts repeat a saved mathematical trace. Clear the allocator cache after each "
              "prefill block, matching MLX-LM generation.", "contexts": []}
    originals = [(Attention, "__call__"), (MLP, "__call__"), (MuddMixer, "combine")]
    originals = [(cls, name, getattr(cls, name)) for cls, name in originals]
    with wired_limit(model):
        for length in [128, 8192, 16384, 32768]:
            cache = model.make_cache()
            mx.reset_peak_memory()
            tokens = (ids * (length // len(ids) + 1))[:length]
            for offset in range(0, length, 512):
                logits = model(mx.array([tokens[offset:offset+512]]), cache)
                mx.eval(logits)
                mx.clear_cache()
            totals = {"Attention": 0.0, "MLP": 0.0, "MuddMixer": 0.0, "sampler": 0.0}
            calls = {k: 0 for k in totals}
            def wrap(function, key):
                def measured(*args, **kwargs):
                    mx.synchronize()
                    start = time.perf_counter()
                    value = function(*args, **kwargs)
                    mx.eval(value)
                    totals[key] += time.perf_counter() - start
                    calls[key] += 1
                    return value
                return measured
            for cls, name, fn in originals:
                setattr(cls, name, wrap(fn, cls.__name__))
            start = time.perf_counter()
            try:
                for token in tokens[-8:]:
                    logits = model(mx.array([[token]]), cache)
                    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
                    mx.eval(logprobs)
                    wrap(sampler, "sampler")(logprobs)
            finally:
                for cls, name, fn in originals:
                    setattr(cls, name, fn)
            elapsed = time.perf_counter() - start
            row = {"context": length, "instrumented_ms_per_token": elapsed*1000/8,
                   "component_ms_per_token": {k:v*1000/8 for k,v in totals.items()},
                   "calls_per_token": {k:v/8 for k,v in calls.items()},
                   "peak_mlx_gb": mx.get_peak_memory()/1e9,
                   "allocator_cache_gb": mx.get_cache_memory()/1e9}
            result["contexts"].append(row)
            dest.write_text(json.dumps(result, indent=2)+"\n")
            print(json.dumps(row), flush=True)
            del cache, logits
            mx.clear_cache()


if __name__ == "__main__":
    main()
