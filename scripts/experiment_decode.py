"""Compare reversible decode candidates against the current Q8 runtime."""
import argparse
import hashlib
import json
import statistics
from functools import partial
from pathlib import Path
import mlx.core as mx
import psutil
from dwarf_violetto.runtime import load, prompt_tokens
from dwarf_violetto.limite import MLP
from dwarf_violetto.optimize import SharedRope
from dwarf_violetto.kernels import rope_decode
from dwarf_violetto.bench import measure
from dwarf_violetto.sampling import make_sampler
from dwarf_violetto.provenance import provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--output", default="results/decode-metal-mlp-refreshed.json")
    args = parser.parse_args()
    if args.tokens < 2:
        parser.error("Use at least two output tokens")
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    model, tokenizer, _ = load("models/q8", optimized=True, fused=True)
    mlp_call = MLP.__call__
    compiled = {id(layer.mlp): mx.compile(partial(mlp_call, layer.mlp)) for layer in model.layers}
    rope_call = SharedRope.__call__
    active = set()
    def mlp(self, x):
        return compiled[id(self)](x) if "mlp" in active and x.shape[-2] == 1 else mlp_call(self, x)
    def rope(self, x, offset):
        if "rope" not in active or x.shape[-2] != 1:
            return rope_call(self, x, offset)
        key = (offset, 1, x.dtype)
        if key != self.key:
            pos = mx.arange(offset, offset + 1, dtype=mx.float32)[:, None]
            theta = pos * self.freq[None, :]
            cos = mx.cos(theta).astype(mx.bfloat16).astype(x.dtype)[None, None]
            sin = (mx.sin(theta).astype(mx.bfloat16) * self.sign).astype(x.dtype)[None, None]
            self.key, self.tables = key, (cos, sin)
        return rope_decode(x, *self.tables)
    MLP.__call__, SharedRope.__call__ = mlp, rope
    prompt = prompt_tokens(tokenizer, "Find the last three digits of 1! + 2! + ... + 20!.")
    tokens = (prompt * (128 // len(prompt) + 1))[:128]
    sampler = make_sampler(.6, top_p=.95)
    result = {"note": "Fixed-work experiments, not mathematical answer tests. Two measurements per mode.",
              "settings": {"context": 128, "seed": 42, "tokens": args.tokens, "temp": 0.6},
              "sha256": provenance("models/q8"),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "runs": []}
    try:
        # Paired decoding on an identical forced prefix, including local eviction.
        for length in [1, 128, 1100]:
            caches = [model.make_cache(), model.make_cache()]
            ids = (prompt * (length // len(prompt) + 1))[:length]
            for offset in range(0, length, 256):
                for mode, cache in zip([set(), {"mlp", "rope"}], caches):
                    active.clear(); active.update(mode)
                    output = model(mx.array([ids[offset:offset+256]]), cache)
                    mx.eval(output)
                mx.clear_cache()
            for token in ids[:16]:
                active.clear()
                a = model(mx.array([[token]]), caches[0]); mx.eval(a)
                active.update(["mlp", "rope"])
                b = model(mx.array([[token]]), caches[1]); mx.eval(b)
                assert bool(mx.array_equal(a, b)), (length, float(mx.max(mx.abs(a-b))))
            print(f"Exact logits at {length}-token prefix, 16 decode steps", flush=True)
            del caches, output, a, b
            mx.clear_cache()
        for mode in [set(), {"rope"}, {"mlp"}, {"rope", "mlp"}, {"rope", "mlp"}, {"mlp"}, {"rope"}, set()]:
            active.clear(); active.update(mode)
            measure(model, tokens, 128, 512, sampler)
            row = measure(model, tokens, args.tokens, 512, sampler)
            row["mode"] = "+".join(sorted(mode)) or "baseline"
            row.update(available_gb=psutil.virtual_memory().available/1e9,
                       swap_used_gb=psutil.swap_memory().used/1e9)
            result["runs"].append(row)
            dest.write_text(json.dumps(result, indent=2)+"\n")
            print(row["mode"], round(row["decode_tps"], 2), flush=True)
        reference = result["runs"][0]["tokens"]
        result["sampled_trace_identical"] = all(x["tokens"] == reference for x in result["runs"])
        result["median"] = {mode: statistics.median(row["decode_tps"] for row in result["runs"] if row["mode"] == mode)
                            for mode in {row["mode"] for row in result["runs"]}}
        dest.write_text(json.dumps(result, indent=2)+"\n")
        assert result["sampled_trace_identical"], "Sampled trace changed"
    finally:
        MLP.__call__, SharedRope.__call__ = mlp_call, rope_call


if __name__ == "__main__":
    main()
