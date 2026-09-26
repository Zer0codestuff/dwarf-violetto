"""Independent eager PyTorch translation of the official inference formulas.

Uses CPU BF16, not vLLM CUDA. Backend rounding differences are expected.
Only the original Safetensors are loaded. No remote Python is executed.
"""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from tokenizers import Tokenizer

torch.set_num_threads(2)
ROOT = Path(__file__).resolve().parents[1]


@torch.inference_mode()
def main():
    folder = ROOT / "models/original"
    cfg = json.loads((folder / "config.json").read_text())
    w = load_file(str(folder / "model.safetensors"))
    tok = Tokenizer.from_file(str(folder / "tokenizer.json"))
    text = "<|endoftext|><|im_start|>user\nCompute the number of divisors of 360.<|im_end|>\n<|im_start|>assistant\n"
    ids = tok.encode(text).ids
    t = torch.tensor([ids])
    norm = lambda x: F.rms_norm(x, (x.shape[-1],))
    emb = w["model.embed_tokens.weight"]
    h = norm(F.embedding(t, emb))
    ve = F.embedding(t, w["model.value_embeds.weight"]).reshape(1, len(ids), 2, 128)
    history = {0: h}
    freq = ((1 / cfg["rope_base_local"]) ** torch.linspace(0, 1, cfg["rope_n_pairs"])).repeat_interleave(2)
    theta = torch.arange(len(ids), dtype=torch.float32)[:, None] * freq[None]
    cos = theta.cos().to(torch.bfloat16)[None, None]
    sin = theta.sin().to(torch.bfloat16)
    sin[:, 1::2] *= -1
    sin = sin[None, None]

    def rotate(x):
        a = x[..., :64]
        flipped = a.reshape(*a.shape[:-1], 32, 2).flip(-1).reshape(a.shape)
        return torch.cat([cos*a + sin*flipped, x[..., 64:]], dim=-1)

    def mix(x, taps, layer, residual=False):
        key = "model.mudd."
        inner = F.gelu(F.linear(norm(x), w[key+"dense1"].to(x.dtype)))
        suffix = "_mlp" if residual else ""
        weights = F.linear(inner, w[key+"dense2"+suffix][layer, :len(taps)].to(x.dtype))
        weights = weights + w[key+"bias"+suffix][layer, :len(taps)].to(x.dtype)
        out = weights[..., 0:1] * history[taps[0]]
        for j in range(1, len(taps)):
            out = out + weights[..., j:j+1] * history[taps[j]]
        return out

    for i in range(48):
        prefix = f"model.layers.{i}."
        att = prefix + "self_attn."
        taps = cfg["mudd_tap_idx"].get(str(i))
        a = norm(mix(h, taps, i)) if taps else norm(h)
        residual = mix(h, taps, i, True) if taps else h
        qkv = []
        for name, heads in [("q_proj", 10), ("k_proj", 2), ("v_proj", 2)]:
            weight = w[att+name+".weight"]
            weight = w[att+"qkv_scale"].to(weight.dtype) * weight
            qkv.append(F.linear(a, weight).reshape(1, len(ids), heads, 128))
        q,k,v = qkv
        if i in cfg["ve_layers"]:
            gate = 2 * torch.sigmoid(F.linear(a[..., :12], w[att+"ve_gate"].to(a.dtype)))
            v = v + gate[..., None] * ve
        q,k = norm(q).transpose(1,2), norm(k).transpose(1,2)
        if i not in cfg["global_layers"]:
            q,k = rotate(q), rotate(k)
        pos = torch.arange(len(ids))
        mask = pos[:,None] >= pos[None,:]
        if i not in cfg["global_layers"]:
            mask &= pos[None,:] >= pos[:,None] - cfg["sliding_window"]
        y = F.scaled_dot_product_attention(q, k.repeat_interleave(5,1),
            v.transpose(1,2).repeat_interleave(5,1), attn_mask=mask, scale=0.1).transpose(1,2)
        vn = F.normalize(v.repeat_interleave(5,2).float(), dim=-1, eps=1e-4)
        proj = (y.float()*vn).sum(-1,keepdim=True)
        alpha = torch.tanh(w[att+"xsa_alpha"].float()).reshape(1,1,10,1)
        y = y - (alpha*proj*vn).to(y.dtype)
        gate = 2 * torch.sigmoid(F.linear(a[..., :128], w[att+"attn_gate"].to(a.dtype)))
        y = y * gate[...,None]
        ow = w[att+"o_proj.weight"]
        branch = F.linear(y.reshape(1,len(ids),1280), w[att+"o_scale"].to(ow.dtype)*ow)
        h = w[prefix+"resid_lambda_attn"].to(h.dtype)*residual + w[prefix+"post_lambda_attn"].to(h.dtype)*branch
        z = norm(h)
        gate = F.linear(z,w[prefix+"mlp.gate_proj.weight"])
        up = F.linear(z,w[prefix+"mlp.up_proj.weight"])
        branch = F.linear(F.silu(gate)*up,w[prefix+"mlp.down_proj.weight"])
        h = w[prefix+"resid_lambda_mlp"].to(h.dtype)*h + w[prefix+"post_lambda_mlp"].to(h.dtype)*branch
        history[i+1] = h
    logits = 23 * torch.sigmoid((F.linear(norm(h),emb).float()+5)/7.5)
    dest = ROOT / "results/torch-reference.npz"
    np.savez_compressed(dest, ids=ids, logits=logits[0].numpy())
    print(f"Saved independent CPU reference for {len(ids)} positions: {dest}")


if __name__ == "__main__":
    main()
