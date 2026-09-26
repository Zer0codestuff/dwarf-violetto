"""Compare capped next-token distributions against fixed BF16 traces."""
import argparse
import gc
import json
from pathlib import Path
import mlx.core as mx
import numpy as np
from .runtime import load, prompt_tokens
from .provenance import provenance


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/original")
    p.add_argument("--reference", action="store_true")
    p.add_argument("--optimized", action="store_true")
    p.add_argument("--fused", action="store_true")
    p.add_argument("--output", default="results/fidelity.json")
    p.add_argument("--reference-dir", default="results")
    args = p.parse_args()
    model, tokenizer, _ = load(args.model, optimized=args.optimized, fused=args.fused)
    cases = json.loads(Path("data/math_checks.json").read_text())
    rows = []
    for case in cases:
        ids = prompt_tokens(tokenizer, case["problem"])
        ids += tokenizer.encode(case["solution"], add_special_tokens=False)
        # Full logits are required for this analysis, including on optimized models.
        h = model.model(mx.array([ids]))
        raw = model.model.embed_tokens.as_linear(h)
        logits = model.softcap_a * mx.sigmoid((raw.astype(mx.float32) + model.softcap_b) / model.softcap_c)
        lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        # Sample every fourth position. All cases and positions are fixed before comparing.
        lp = lp[0, 8::4].astype(mx.float32)
        mx.eval(lp)
        arr = np.asarray(lp)
        refpath = Path(args.reference_dir) / f"reference-{case['id']}.npz"
        if args.reference:
            refpath.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(refpath, logprobs=arr, ids=ids)
        else:
            with np.load(refpath) as ref:
                if list(ref["ids"]) != ids:
                    raise ValueError("Tokenizer changed relative to the reference")
                old = ref["logprobs"]
            kl = np.sum(np.exp(old) * (old-arr), axis=-1)
            agreement = np.argmax(old, axis=-1) == np.argmax(arr, axis=-1)
            rows.append({"id": case["id"], "positions": len(arr),
                         "kl_mean": float(kl.mean()), "kl_max": float(kl.max()),
                         "top1_agreement": float(agreement.mean()),
                         "max_logprob_difference": float(np.max(np.abs(old-arr)))})
        del h, raw, logits, lp, arr
        gc.collect()
        mx.clear_cache()
        print(case["id"], flush=True)
    if not args.reference:
        n = sum(x["positions"] for x in rows)
        result = {"settings": vars(args), "sha256": provenance(args.model), "positions": n, "cases": rows,
                  "kl_mean": sum(x["kl_mean"] * x["positions"] for x in rows) / n,
                  "top1_agreement": sum(x["top1_agreement"] * x["positions"] for x in rows) / n}
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k != "cases"}))


if __name__ == "__main__":
    main()
