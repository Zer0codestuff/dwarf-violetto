"""Test compiled attention correction and gate against unfused output gating."""
import json
from pathlib import Path
import mlx.core as mx
from dwarf_violetto.runtime import load, prompt_tokens
from dwarf_violetto.bench import measure
from dwarf_violetto.sampling import make_sampler
from dwarf_violetto.provenance import provenance


def compiled_post(attn):
    def post(y, v, x):
        vf = mx.repeat(v, attn.repeats, axis=2).astype(mx.float32)
        norm = mx.maximum(mx.sqrt(mx.sum(vf * vf, axis=-1, keepdims=True)), attn.xsa_eps)
        vn = vf / norm
        proj = mx.sum(y.astype(mx.float32) * vn, axis=-1, keepdims=True)
        a = mx.tanh(attn.xsa_alpha.astype(mx.float32)).reshape(1, 1, -1, 1)
        y = y - (a * proj * vn).astype(y.dtype)
        gate_w = attn.attn_gate.astype(x.dtype)
        gate = attn.attn_gate_scale * mx.sigmoid(x[..., :attn.attn_gate_channels] @ gate_w.T)
        return y * gate[..., None].astype(y.dtype)
    return mx.compile(post)


def main():
    model, tokenizer, _ = load("models/q8", optimized=True, fused=True)
    functions = [compiled_post(layer.self_attn) for layer in model.layers]
    def enable(value):
        for layer, fn in zip(model.layers, functions):
            layer.self_attn._fused_post = fn if value else None
    prompt = prompt_tokens(tokenizer, "Find the last three digits of 1! + 2! + ... + 20!.")
    ids = (prompt * (1100//len(prompt)+1))[:1100]
    caches = [model.make_cache(), model.make_cache()]
    for offset in range(0, 1100, 256):
        for mode, cache in zip([False, True], caches):
            enable(mode)
            logits = model(mx.array([ids[offset:offset+256]]), cache); mx.eval(logits)
        mx.clear_cache()
    for token in ids[:64]:
        enable(False); a = model(mx.array([[token]]), caches[0]); mx.eval(a)
        enable(True); b = model(mx.array([[token]]), caches[1]); mx.eval(b)
        assert bool(mx.array_equal(a,b)), float(mx.max(mx.abs(a-b)))
    print("64 decode steps after 1100-token prefix: exact logits", flush=True)
    del caches, logits, a, b
    mx.clear_cache()
    result = {"sha256": provenance("models/q8"), "exact_decode_positions":64, "runs":[]}
    sampler = make_sampler(.6, top_p=.95)
    for mode in [False, True, True, False, False, True]:
        enable(mode)
        measure(model,ids[:128],32,512,sampler)
        row=measure(model,ids[:128],256,512,sampler)
        row['mode']='post' if mode else 'baseline'
        result['runs'].append(row)
        Path('results/decode-post-candidate.json').write_text(json.dumps(result,indent=2)+'\n')
        print(row['mode'],round(row['decode_tps'],2),flush=True)
    assert all(r['tokens']==result['runs'][0]['tokens'] for r in result['runs'])


if __name__ == '__main__':
    main()
