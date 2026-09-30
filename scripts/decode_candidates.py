"""Reversible decode candidates shared by verification and timing scripts."""
from functools import partial

import mlx.core as mx

from dwarf_violetto.fused import xsa
from dwarf_violetto.limite import MuddMixer
from dwarf_violetto.optimize import SharedRope
from experiment_xsa import grouped_xsa


@mx.compile
def compiled_rope(x, cos, sin):
    rotated = cos.shape[-1]
    head = x[..., :rotated]
    pairs = head.reshape(*head.shape[:-1], rotated // 2, 2)
    flipped = mx.flip(pairs, axis=-1).reshape(head.shape)
    return mx.concatenate([cos * head + sin * flipped, x[..., rotated:]], axis=-1)


class DecodeCandidates:
    modes = ("baseline", "compiled_rope", "mudd", "compiled_rope+mudd", "grouped_xsa")

    def __init__(self, model):
        self.model = model
        self.active = "baseline"
        self.original_rope = SharedRope.__call__
        self.original_mudd = MuddMixer.combine
        self.compiled_mudd = {
            (index, residual): mx.compile(partial(
                self.original_mudd, model.model.mudd, layer_idx=index, r_way=residual
            ))
            for index in model.model._tap_idx
            for residual in (False, True)
        }

    def enable(self, mode):
        if mode not in self.modes:
            raise ValueError(f"Unknown decode candidate: {mode}")
        self.active = mode
        for layer in self.model.layers:
            layer.self_attn._fused_xsa = grouped_xsa if mode == "grouped_xsa" else xsa

    def __enter__(self):
        self.original_xsa = [layer.self_attn._fused_xsa for layer in self.model.layers]

        def rope(module, x, offset):
            if "compiled_rope" not in self.active or x.shape[-2] != 1:
                return self.original_rope(module, x, offset)
            key = (offset, 1, x.dtype)
            if key != module.key:
                pos = mx.arange(offset, offset + 1, dtype=mx.float32)[:, None]
                theta = pos * module.freq[None, :]
                cos = mx.cos(theta).astype(mx.bfloat16).astype(x.dtype)[None, None]
                sin = (mx.sin(theta).astype(mx.bfloat16) * module.sign).astype(x.dtype)[None, None]
                module.key, module.tables = key, (cos, sin)
            return compiled_rope(x, *module.tables)

        def mudd(module, values, x_cur, layer_idx, r_way=False):
            if "mudd" in self.active and x_cur.shape[-2] == 1:
                return self.compiled_mudd[(layer_idx, r_way)](values, x_cur)
            return self.original_mudd(module, values, x_cur, layer_idx, r_way)

        SharedRope.__call__, MuddMixer.combine = rope, mudd
        return self

    def __exit__(self, *exc):
        SharedRope.__call__, MuddMixer.combine = self.original_rope, self.original_mudd
        for layer, function in zip(self.model.layers, self.original_xsa):
            layer.self_attn._fused_xsa = function
