"""Focused checks for numerical packing, tokenizer, and the local window."""
import json
from pathlib import Path
import mlx.core as mx
import numpy as np
from tokenizers import Tokenizer
from dwarf_violetto.runtime import load, prompt_tokens
from dwarf_violetto.optimize import optimize
from dwarf_violetto.fused import xsa, residual


def distribution(a, b):
    la = a.astype(mx.float32) - mx.logsumexp(a.astype(mx.float32), axis=-1, keepdims=True)
    lb = b.astype(mx.float32) - mx.logsumexp(b.astype(mx.float32), axis=-1, keepdims=True)
    return {"top1": float(mx.mean((mx.argmax(a,axis=-1)==mx.argmax(b,axis=-1)).astype(mx.float32))),
            "kl": float(mx.mean(mx.sum(mx.exp(la)*(la-lb),axis=-1))),
            "max_abs": float(mx.max(mx.abs(a-b)))}


def last_logits(model, ids, cache=None):
    h = model.model(mx.array([ids]), cache=cache)[:, -1:, :]
    raw = model.model.embed_tokens.as_linear(h)
    return 23 * mx.sigmoid((raw.astype(mx.float32)+5)/7.5)


def main():
    model, tokenizer, _ = load("models/q8")
    raw = Tokenizer.from_file("models/original/tokenizer.json")
    text = "Solve 7^2026 mod 13.\nα + β = 3; perché?"
    assert tokenizer.encode(text, add_special_tokens=False) == raw.encode(text, add_special_tokens=False).ids
    ids = prompt_tokens(tokenizer, "How many positive divisors does 360 have?")
    before = model(mx.array([ids]))
    mx.eval(before)
    optimize(model)
    after = model(mx.array([ids]))
    mx.eval(after)
    packing = distribution(before, after)
    assert packing["top1"] >= 0.98 and packing["kl"] < 0.001, packing
    for layer in model.layers:
        layer.self_attn._fused_xsa = xsa
        layer._fused_residual = residual
    compiled = model(mx.array([ids]))
    mx.eval(compiled)
    fusion = distribution(after, compiled)
    assert fusion["max_abs"] == 0, fusion

    # Compare a complete prefill with chunked cache updates across local eviction.
    long_ids = (ids * (1100//len(ids)+1))[:1100]
    full = last_logits(model, long_ids)
    mx.eval(full)
    cache = model.make_cache()
    for start in range(0, 1097, 256):
        stop = min(start+256,1097)
        last_logits(model,long_ids[start:stop],cache)
        mx.eval([c.state for c in cache])
    for token in long_ids[1097:]:
        incremental = last_logits(model,[token],cache)
        mx.eval(incremental)
    window = distribution(full,incremental)
    assert window["top1"] == 1.0 and window["kl"] < 0.005, window
    assert all(c.offset == 1100 for c in cache)
    assert max(c.keys.shape[-2] for i,c in enumerate(cache) if i not in model.model._global) <= 1025
    result = {"packing": packing, "fusion": fusion, "cache_at_1100": window, "tokenizer_matches_raw": True}

    if Path("results/torch-reference.npz").exists():
        model,_,_ = load("models/original")
        ref = np.load("results/torch-reference.npz")
        actual = model(mx.array([ref["ids"].tolist()]))[0]
        oracle = distribution(mx.array(ref["logits"]),actual)
        result["torch_cpu_reference"] = oracle
        # This is cross-backend agreement, not bitwise CUDA equivalence.
        assert oracle["top1"] >= 0.95 and oracle["kl"] < 0.01, oracle
    Path("results/correctness.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
