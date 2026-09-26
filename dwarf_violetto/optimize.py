"""Pack existing projection rows without requantizing their values."""
import mlx.core as mx
import mlx.nn as nn
from .limite import LimiteRope


def concatenate_linear(modules):
    first = modules[0]
    if isinstance(first, nn.QuantizedLinear):
        if not all(isinstance(m, nn.QuantizedLinear) and
                   (m.bits, m.group_size, m.mode) == (first.bits, first.group_size, first.mode)
                   for m in modules):
            raise ValueError("Packed projections must share their quantization format")
        input_dims = first.scales.shape[1] * first.group_size
        merged = nn.QuantizedLinear(input_dims, sum(m.weight.shape[0] for m in modules),
                                    bias=False, group_size=first.group_size,
                                    bits=first.bits, mode=first.mode)
        for name in ("weight", "scales", "biases"):
            merged[name] = mx.concatenate([m[name] for m in modules], axis=0)
    else:
        merged = nn.Linear(first.weight.shape[1], sum(m.weight.shape[0] for m in modules), bias=False)
        merged.weight = mx.concatenate([m.weight for m in modules], axis=0)
    return merged


class SharedRope(LimiteRope):
    """Reuse position tables across the 36 local layers in the current step."""
    def __init__(self, original):
        self.rotated, self.freq, self.sign = original.rotated, original.freq, original.sign
        self.key = None
        self.tables = None

    def __call__(self, x, offset):
        length = x.shape[-2]
        key = (offset, length, x.dtype)
        if key != self.key:
            pos = mx.arange(offset, offset + length, dtype=mx.float32)[:, None]
            theta = pos * self.freq[None, :]
            cos = mx.cos(theta).astype(mx.bfloat16).astype(x.dtype)[None, None]
            sin = (mx.sin(theta).astype(mx.bfloat16) * self.sign).astype(x.dtype)[None, None]
            self.key, self.tables = key, (cos, sin)
        cos, sin = self.tables
        head = x[..., :self.rotated]
        pairs = head.reshape(*head.shape[:-1], self.rotated // 2, 2)
        flipped = mx.flip(pairs, axis=-1).reshape(head.shape)
        rotated = cos * head + sin * flipped
        return mx.concatenate([rotated, x[..., self.rotated:]], axis=-1)


def optimize(model):
    rope = SharedRope(model.model._rope)
    model.model._rope = rope
    for layer in model.layers:
        attn = layer.self_attn
        attn._rope = rope
        attn.qkv_proj = concatenate_linear([attn.q_proj, attn.k_proj, attn.v_proj])
        for name in ("q_proj", "k_proj", "v_proj"):
            del attn[name]
        mlp = layer.mlp
        mlp.gate_up_proj = concatenate_linear([mlp.gate_proj, mlp.up_proj])
        del mlp["gate_proj"]
        del mlp["up_proj"]
    mx.eval(model.parameters())
    mx.clear_cache()
    return model
