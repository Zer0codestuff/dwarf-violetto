"""Compare full logits and cache state on identical forced mathematical prefixes."""
import argparse
import hashlib
import json
from pathlib import Path

import mlx.core as mx
from mlx_lm.generate import wired_limit

from dwarf_violetto.provenance import provenance, script_hashes
from dwarf_violetto.runtime import load, prompt_tokens
from decode_candidates import DecodeCandidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", nargs="+", type=int, default=[1, 128, 1100, 8192, 32768])
    parser.add_argument("--positions", type=int, default=64)
    parser.add_argument("--modes", nargs="+", choices=DecodeCandidates.modes,
                        default=["compiled_rope", "mudd", "compiled_rope+mudd"])
    parser.add_argument("--output", default="results/decode-candidate-correctness.json")
    args = parser.parse_args()
    if min(args.contexts) < 1 or args.positions < 1:
        parser.error("Contexts and decode positions must be positive")
    model, tokenizer, _ = load("models/q8", optimized=True, fused=True)
    if max(args.contexts) + args.positions > model.args.max_position_embeddings:
        parser.error("The prefix and decode positions exceed the configured model context")
    problem = "Compute the last three digits of 1! + 2! + ... + 20!."
    trace = json.loads(Path("results/q8-math-eos.json").read_text())
    text = next(case["text"] for case in trace["cases"] if case["id"] == "factorial")
    source = prompt_tokens(tokenizer, problem) + tokenizer.encode(text, add_special_tokens=False)
    scripts = [Path(__file__), Path("scripts/decode_candidates.py")]
    result = {"settings": vars(args), "sha256": provenance("models/q8"),
              "script_sha256": script_hashes(scripts),
              "trace_sha256": hashlib.sha256(text.encode()).hexdigest(),
              "method": "Repeat a saved mathematical trace for long prefixes. Compare every capped logit "
                        "and the final KV state. Multi-token prefill uses the unchanged eager path. "
                        "Clear the allocator cache after each prefill block, matching MLX-LM generation.",
              "checks": []}
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with DecodeCandidates(model) as candidates, wired_limit(model):
        for length in args.contexts:
            ids = (source * ((length + args.positions) // len(source) + 1))[:length + args.positions]
            for mode in args.modes:
                caches = [model.make_cache(), model.make_cache()]
                prefill_exact = True
                for offset in range(0, length, 512):
                    outputs = []
                    for setting, cache in zip(("baseline", mode), caches):
                        candidates.enable(setting)
                        logits = model(mx.array([ids[offset:min(offset + 512, length)]]), cache)
                        mx.eval(logits)
                        outputs.append(logits)
                    prefill_exact &= bool(mx.array_equal(*outputs))
                    mx.clear_cache()
                max_abs = 0.0
                for token in ids[length:]:
                    candidates.enable("baseline")
                    a = model(mx.array([[token]]), caches[0])
                    mx.eval(a)
                    candidates.enable(mode)
                    b = model(mx.array([[token]]), caches[1])
                    mx.eval(b)
                    max_abs = max(max_abs, float(mx.max(mx.abs(a - b))))
                cache_exact = all(
                    a.offset == b.offset == length + args.positions and
                    all(bool(mx.array_equal(x, y)) for x, y in zip(a.state, b.state))
                    for a, b in zip(*caches)
                )
                row = {"mode": mode, "context": length, "positions": args.positions,
                       "prefill_exact": prefill_exact, "decode_max_abs": max_abs,
                       "cache_exact": cache_exact, "peak_mlx_gb": mx.get_peak_memory() / 1e9}
                result["checks"].append(row)
                dest.write_text(json.dumps(result, indent=2) + "\n")
                print(row, flush=True)
                if not prefill_exact or max_abs != 0 or not cache_exact:
                    raise AssertionError(row)
                del caches, outputs, logits, a, b
                mx.clear_cache()
    result["all_exact"] = True
    dest.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
