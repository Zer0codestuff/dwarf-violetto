# Copyright © 2026
# MLX port by Piergiorgio Di Pasquale (pierjoe).
# Source: https://huggingface.co/pierjoe/limite-1b-violetto-mlx-bf16
# Adapted for Dwarf Violetto. Licensed under Apache-2.0.
#
# MLX implementation of the `limite` architecture (paradigma-inc/limite-1b-violetto).
#
# Ported from the reference vLLM plugin at
# https://github.com/paradigma-inc/limite-violetto (Apache-2.0), files
# `src/limite_vllm/_components.py` and `src/limite_vllm/modeling.py`.
#
# The architecture deviates from a standard Llama-style decoder in ten ways, all
# of which are pinned in config.json and reproduced verbatim here:
#
#  1. RMSNorm has no learnable gain and no explicit epsilon: torch's default is
#     `finfo(x.dtype).eps`, i.e. 0.0078125 in bfloat16, not 1e-6.
#  2. Attention softmax scale is a literal 0.1, not head_dim**-0.5 (0.08838).
#  3. Layers 3, 7, ... 47 are full-attention AND rotary-free (`global_nope`);
#     all others use a sliding window that is inclusive of the query token,
#     hence a span of `sliding_window + 1` = 1025 keys.
#  4. Partial rotary: only the first 2 * rope_n_pairs = 64 lanes rotate, on
#     adjacent pairs, with the sine negated on odd lanes, and the cos/sin tables
#     rounded to bfloat16 before use.
#  5. Per-layer learned scalars `qkv_scale` / `o_scale` fold into the
#     projections in bfloat16 (see `sanitize`).
#  6. Value embeddings: a second embedding table is gated into `v` on 16 layers,
#     before QK-norm. The gate reads the first 12 channels of the layer input.
#  7. A per-head output gate `2 * sigmoid(...)` is applied before o_proj.
#  8. XSA: a cross-head correction that subtracts the component of the attention
#     output along the (normalised) value vector, computed in float32.
#  9. MUDD: at layers 24 and 47 both the attention input and the residual base
#     are input-dependent weighted sums of earlier residual states, accumulated
#     strictly left to right.
# 10. Logits are squashed: `23 * sigmoid((raw + 5) / 7.5)`, so they live in
#     [0, 23] rather than being raw dot products.

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from mlx_lm.models.cache import KVCache, RotatingKVCache

# torch.finfo(dtype).eps, which is what `F.rms_norm(x, shape)` uses when `eps`
# is left as None (`rms_norm_eps_mode: "torch_finfo_default"`).
_FINFO_EPS = {
    mx.bfloat16: 0.0078125,
    mx.float16: 0.0009765625,
    mx.float32: 1.1920928955078125e-07,
}


def rms_norm(x: mx.array) -> mx.array:
    """Gain-free RMSNorm with torch's dtype-dependent default epsilon."""
    return mx.fast.rms_norm(x, None, _FINFO_EPS[x.dtype])


def partitioned_projection(module, x, widths):
    """Keep prefill GEMM shapes unchanged to preserve BF16 rounding."""
    outputs = []
    start = 0
    for width in widths:
        stop = start + width
        if isinstance(module, nn.QuantizedLinear):
            outputs.append(mx.quantized_matmul(
                x, module.weight[start:stop], scales=module.scales[start:stop],
                biases=module.biases[start:stop], transpose=True,
                group_size=module.group_size, bits=module.bits, mode=module.mode,
            ))
        else:
            outputs.append(x @ module.weight[start:stop].T)
        start = stop
    return outputs


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "limite"
    # shape
    hidden_size: int = 1280
    num_hidden_layers: int = 48
    num_attention_heads: int = 10
    num_key_value_heads: int = 2
    head_dim: int = 128
    intermediate_size: int = 3328
    vocab_size: int = 151680
    max_position_embeddings: int = 131072
    tie_word_embeddings: bool = True
    mlp_type: str = "swiglu"
    # normalisation
    rms_norm_has_weight: bool = False
    rms_norm_eps_mode: str = "torch_finfo_default"
    qk_norm: str = "rms_pre_rope"
    # attention
    attention_softmax_scale: float = 0.1
    sliding_window: int = 1024
    global_window: int = -1
    global_layers: List[int] = field(default_factory=list)
    global_nope: bool = True
    attn_gate_channels: int = 128
    attn_gate_scale: float = 2.0
    attn_gate_applied: str = "per_head_before_o_proj"
    # rotary
    pos_mode: str = "rope"
    rope_base_local: float = 1024.0
    rope_n_pairs: int = 32
    rope_per_layer: bool = False
    rope_style: str = "interleaved_pairs_odd_lane_sign_flip"
    # value embeddings
    ve_dim: int = 128
    ve_layers: List[int] = field(default_factory=list)
    ve_gate_channels: int = 12
    ve_gate_scale: float = 2.0
    ve_stored_heads: int = 2
    ve_head_slice: str = "first_num_key_value_heads"
    # cross-head correction
    xsa: bool = True
    xsa_layers: List[int] = field(default_factory=list)
    xsa_normalize_eps: float = 1e-4
    # dynamic dense residual mixing
    mudd: bool = True
    mudd_layers: List[int] = field(default_factory=list)
    mudd_taps: int = 3
    mudd_inter: int = 32
    mudd_tap_idx: Dict[str, List[int]] = field(default_factory=dict)
    mudd_mlp: bool = True
    mudd_accumulation: str = "ordered_left_to_right"
    # head
    softcap_logits: Dict[str, Any] = field(
        default_factory=lambda: {"kind": "sigmoid", "a": 23.0, "b": 5.0, "c": 7.5}
    )
    final_softcap: float = 0.0
    lm_head_precision_mode: str = "oracle_exact"

    def __post_init__(self):
        # Every one of these is a fork in the numerics. A silent fallback would
        # produce a plausible-looking model with wrong numbers, so refuse loudly
        # exactly where the reference does.
        unsupported = {
            "mlp_type": (self.mlp_type, {"swiglu"}),
            "pos_mode": (self.pos_mode, {"rope"}),
            "qk_norm": (self.qk_norm, {"rms_pre_rope"}),
            "rms_norm_eps_mode": (self.rms_norm_eps_mode, {"torch_finfo_default"}),
            "rope_style": (
                self.rope_style,
                {"interleaved_pairs_odd_lane_sign_flip"},
            ),
            "mudd_accumulation": (self.mudd_accumulation, {"ordered_left_to_right"}),
            "ve_head_slice": (self.ve_head_slice, {"first_num_key_value_heads"}),
            "lm_head_precision_mode": (
                self.lm_head_precision_mode,
                {"oracle_exact", "fp32_accumulate"},
            ),
            "attn_gate_applied": (
                self.attn_gate_applied,
                {"per_head_before_o_proj"},
            ),
        }
        for name, (value, allowed) in unsupported.items():
            if value not in allowed:
                raise NotImplementedError(
                    f"limite: unsupported {name}={value!r}; implemented: {sorted(allowed)}"
                )
        if self.rms_norm_has_weight:
            raise NotImplementedError("limite: RMSNorm carries no learnable gain.")
        if self.final_softcap:
            raise NotImplementedError("limite: pre-head tanh cap is not implemented.")
        if not self.global_nope:
            raise NotImplementedError("limite: global layers must be rotary-free.")
        if self.global_window != -1:
            raise NotImplementedError("limite: global layers must be unwindowed.")
        if self.rope_per_layer:
            raise NotImplementedError("limite: per-layer rotary bases not implemented.")
        if self.softcap_logits.get("kind") != "sigmoid":
            raise NotImplementedError(
                f"limite: only sigmoid logit softcapping, got {self.softcap_logits!r}"
            )
        if self.attn_gate_channels and self.attn_gate_scale != 2.0:
            raise NotImplementedError(
                "limite: the attention gate is only implemented as 2 * sigmoid(...)."
            )
        if self.num_attention_heads * self.head_dim != self.hidden_size:
            raise ValueError("limite: num_attention_heads * head_dim != hidden_size")
        if self.ve_dim > self.head_dim:
            raise ValueError("limite: ve_dim must not exceed head_dim")
        self.mudd_tap_idx = {int(k): [int(i) for i in v] for k, v in self.mudd_tap_idx.items()}
        if self.mudd and sorted(self.mudd_tap_idx) != sorted(self.mudd_layers):
            raise ValueError(
                f"limite: mudd_tap_idx keys {sorted(self.mudd_tap_idx)} must match "
                f"mudd_layers {sorted(self.mudd_layers)}"
            )

    @property
    def local_window(self) -> int:
        """Visible span for a local layer, query token included."""
        return self.sliding_window + 1


class LimiteRope:
    """Partial, adjacent-pair-interleaved rotary with an odd-lane sign flip.

    Deliberately not an ``nn.Module``: the frequency table is derived from the
    config, not loaded, and must stay out of ``parameters()``.
    """

    def __init__(self, head_dim: int, n_pairs: int, base: float):
        self.rotated = 2 * n_pairs
        freq = (1.0 / base) ** mx.linspace(0, 1, n_pairs, dtype=mx.float32)
        # One frequency per pair, shared by both lanes of the pair.
        self.freq = mx.repeat(freq, 2)
        # sin is negated on the odd lane of every pair, which together with the
        # pairwise flip below gives (a, b) -> (a cos + b sin, b cos - a sin).
        sign = mx.tile(mx.array([1.0, -1.0], mx.float32), (n_pairs,))
        self.sign = sign

    def __call__(self, x: mx.array, offset: int) -> mx.array:
        # x: [B, H, L, head_dim]
        L = x.shape[-2]
        pos = mx.arange(offset, offset + L, dtype=mx.float32)[:, None]
        theta = pos * self.freq[None, :]
        # `rope_cos_sin_dtype: bfloat16` -- the rounding is part of the trained
        # function, so it happens before the multiply, not after.
        cos = mx.cos(theta).astype(mx.bfloat16).astype(x.dtype)[None, None]
        sin = (mx.sin(theta).astype(mx.bfloat16) * self.sign).astype(x.dtype)[None, None]

        # Lanes beyond 2 * n_pairs have frequency 0, so cos == 1 and sin == 0
        # there and the rotation is the identity; slicing is bit-identical.
        head = x[..., : self.rotated]
        pairs = head.reshape(*head.shape[:-1], self.rotated // 2, 2)
        flipped = mx.flip(pairs, axis=-1).reshape(head.shape)
        rotated = cos * head + sin * flipped
        if self.rotated == x.shape[-1]:
            return rotated
        return mx.concatenate([rotated, x[..., self.rotated :]], axis=-1)


class MuddMixer(nn.Module):
    """Dynamic dense residual mixing over a few earlier residual states."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        n, taps, inter = args.num_hidden_layers, args.mudd_taps, args.mudd_inter
        self.dense1 = mx.zeros((inter, args.hidden_size), mx.float32)
        self.dense2 = mx.zeros((n, taps, inter), mx.float32)
        self.bias = mx.zeros((n, taps), mx.float32)
        self.uses_r_way = bool(args.mudd_mlp)
        if self.uses_r_way:
            self.dense2_mlp = mx.zeros((n, taps, inter), mx.float32)
            self.bias_mlp = mx.zeros((n, taps), mx.float32)

    def combine(
        self,
        values: List[mx.array],
        x_cur: mx.array,
        layer_idx: int,
        r_way: bool = False,
    ) -> mx.array:
        count = len(values)
        dt = x_cur.dtype
        inner = nn.gelu(rms_norm(x_cur) @ self.dense1.astype(dt).T)
        dense2, bias = (
            (self.dense2_mlp, self.bias_mlp) if r_way else (self.dense2, self.bias)
        )
        w = inner @ dense2[layer_idx, :count].astype(dt).T
        w = w + bias[layer_idx, :count].astype(dt)
        # `mudd_accumulation: ordered_left_to_right`: seed with the first tap and
        # add the rest one at a time. A single stacked contraction is
        # mathematically identical and numerically is not.
        out = w[..., 0:1].astype(values[0].dtype) * values[0]
        for i in range(1, count):
            out = out + w[..., i : i + 1].astype(values[i].dtype) * values[i]
        return out


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int, rope: LimiteRope):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.repeats = self.n_heads // self.n_kv_heads
        self.head_dim = args.head_dim
        self.scale = args.attention_softmax_scale
        self.ve_dim = args.ve_dim
        self.ve_stored_heads = args.ve_stored_heads
        self.ve_gate_scale = args.ve_gate_scale
        self.ve_gate_channels = args.ve_gate_channels
        self.attn_gate_channels = args.attn_gate_channels
        self.attn_gate_scale = args.attn_gate_scale
        self.xsa_eps = args.xsa_normalize_eps

        self.uses_ve = layer_idx in set(args.ve_layers)
        self.uses_xsa = bool(args.xsa) and layer_idx in set(args.xsa_layers)
        self.is_global = layer_idx in set(args.global_layers)
        self.uses_rope = not (args.global_nope and self.is_global)
        self._rope = rope

        dim = args.hidden_size
        kv_dim = self.n_kv_heads * self.head_dim
        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, kv_dim, bias=False)
        self.v_proj = nn.Linear(dim, kv_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)

        # `qkv_scale` / `o_scale` are absent by design: `Model.sanitize` folds
        # them into the projections above, in bfloat16, exactly once.
        self.xsa_alpha = mx.zeros((self.n_heads,), mx.float32)
        if self.uses_ve:
            self.ve_gate = mx.zeros((self.ve_stored_heads, self.ve_gate_channels), mx.float32)
        if self.attn_gate_channels:
            self.attn_gate = mx.zeros((self.n_heads, self.attn_gate_channels), mx.float32)

    def __call__(
        self,
        attn_in: mx.array,
        ve: Optional[mx.array],
        mask: Optional[mx.array],
        cache: Optional[Any],
    ) -> mx.array:
        B, L, _ = attn_in.shape
        if "qkv_proj" in self:
            qdim, kvdim = self.n_heads * self.head_dim, self.n_kv_heads * self.head_dim
            if L > 1:
                q, k, v = partitioned_projection(self.qkv_proj, attn_in, [qdim, kvdim, kvdim])
            else:
                qkv = self.qkv_proj(attn_in)
                q, k, v = mx.split(qkv, [qdim, qdim + kvdim], axis=-1)
            q = q.reshape(B, L, self.n_heads, self.head_dim)
            k = k.reshape(B, L, self.n_kv_heads, self.head_dim)
            v = v.reshape(B, L, self.n_kv_heads, self.head_dim)
        else:
            q = self.q_proj(attn_in).reshape(B, L, self.n_heads, self.head_dim)
            k = self.k_proj(attn_in).reshape(B, L, self.n_kv_heads, self.head_dim)
            v = self.v_proj(attn_in).reshape(B, L, self.n_kv_heads, self.head_dim)

        if self.uses_ve:
            # The VE-adjusted `v` is both what attention consumes and what is
            # written to the KV cache.
            gate_w = self.ve_gate.astype(attn_in.dtype)
            gate = self.ve_gate_scale * mx.sigmoid(
                attn_in[..., : gate_w.shape[-1]] @ gate_w.T
            )
            v = v + gate[..., None] * ve

        # `qk_norm: rms_pre_rope` -- after the value-embedding step, before rotary.
        q = rms_norm(q).transpose(0, 2, 1, 3)
        k = rms_norm(k).transpose(0, 2, 1, 3)
        v_t = v.transpose(0, 2, 1, 3)

        if self.uses_rope:
            offset = cache.offset if cache is not None else 0
            q = self._rope(q, offset)
            k = self._rope(k, offset)

        if cache is not None:
            keys, vals = cache.update_and_fetch(k, v_t)
        else:
            keys, vals = k, v_t

        y = scaled_dot_product_attention(
            q, keys, vals, cache=cache, scale=self.scale, mask=mask
        )
        y = y.transpose(0, 2, 1, 3)  # [B, L, n_heads, head_dim]

        if self.uses_xsa and getattr(self, "_fused_xsa", None) is not None:
            y = self._fused_xsa(y, v, self.xsa_alpha)
        elif self.uses_xsa:
            # float32 throughout, and against this step's own `v`, not the cache.
            vf = mx.repeat(v, self.repeats, axis=2).astype(mx.float32)
            norm = mx.maximum(
                mx.sqrt(mx.sum(vf * vf, axis=-1, keepdims=True)), self.xsa_eps
            )
            vn = vf / norm
            proj = mx.sum(y.astype(mx.float32) * vn, axis=-1, keepdims=True)
            alpha = mx.tanh(self.xsa_alpha.astype(mx.float32)).reshape(1, 1, -1, 1)
            y = y - (alpha * proj * vn).astype(y.dtype)

        if self.attn_gate_channels:
            gate_w = self.attn_gate.astype(attn_in.dtype)
            gate = self.attn_gate_scale * mx.sigmoid(
                attn_in[..., : self.attn_gate_channels] @ gate_w.T
            )
            y = y * gate[..., None].astype(y.dtype)

        return self.o_proj(y.reshape(B, L, -1))


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim, inter = args.hidden_size, args.intermediate_size
        self.gate_proj = nn.Linear(dim, inter, bias=False)
        self.up_proj = nn.Linear(dim, inter, bias=False)
        self.down_proj = nn.Linear(inter, dim, bias=False)

    def __call__(self, h: mx.array) -> mx.array:
        # (silu(gate_proj(h)) * up_proj(h)) @ down_proj, no clamp on either factor
        if "gate_up_proj" in self:
            if h.shape[-2] > 1:
                width = self.gate_up_proj.weight.shape[0] // 2
                gate, up = partitioned_projection(self.gate_up_proj, h, [width, width])
            else:
                gate, up = mx.split(self.gate_up_proj(h), 2, axis=-1)
            return self.down_proj(nn.silu(gate) * up)
        return self.down_proj(nn.silu(self.gate_proj(h)) * self.up_proj(h))


class TransformerBlock(nn.Module):
    """One block. Neither join is a plain additive residual: both are
    `resid_lambda * stream + post_lambda * branch` with learned coefficients."""

    def __init__(self, args: ModelArgs, layer_idx: int, rope: LimiteRope):
        super().__init__()
        self.self_attn = Attention(args, layer_idx, rope)
        self.mlp = MLP(args)
        self.resid_lambda_attn = mx.zeros((), mx.float32)
        self.post_lambda_attn = mx.zeros((), mx.float32)
        self.resid_lambda_mlp = mx.zeros((), mx.float32)
        self.post_lambda_mlp = mx.zeros((), mx.float32)

    def __call__(
        self,
        x: mx.array,
        attn_in: mx.array,
        residual_base: mx.array,
        ve: Optional[mx.array],
        mask: Optional[mx.array],
        cache: Optional[Any],
    ) -> mx.array:
        attn_out = self.self_attn(attn_in, ve, mask, cache)
        dt = x.dtype
        # The fp32 coefficient is rounded down to the activation dtype *before*
        # multiplying, matching the reference.
        fused_residual = getattr(self, "_fused_residual", None)
        if fused_residual is not None:
            x = fused_residual(self.resid_lambda_attn, residual_base, self.post_lambda_attn, attn_out)
        else:
            x = (
                self.resid_lambda_attn.astype(dt) * residual_base
                + self.post_lambda_attn.astype(dt) * attn_out
            )
        mlp_out = self.mlp(rms_norm(x))
        if fused_residual is not None:
            return fused_residual(self.resid_lambda_mlp, x, self.post_lambda_mlp, mlp_out)
        return (
            self.resid_lambda_mlp.astype(dt) * x
            + self.post_lambda_mlp.astype(dt) * mlp_out
        )


class LimiteModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.value_embeds = nn.Embedding(
            args.vocab_size, args.ve_stored_heads * args.ve_dim
        )
        self._rope = LimiteRope(args.head_dim, args.rope_n_pairs, args.rope_base_local)
        self.mudd = MuddMixer(args)
        self.layers = [
            TransformerBlock(args, i, self._rope) for i in range(args.num_hidden_layers)
        ]

        self._global = set(args.global_layers)
        self._tap_idx = args.mudd_tap_idx
        # hist[0] = rms_norm(embedding); hist[j] = output of layer j-1
        self._retained = {t for taps in self._tap_idx.values() for t in taps}
        self._any_ve = bool(args.ve_layers)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        if input_embeddings is not None and self._any_ve:
            raise ValueError(
                "limite reads token ids on every value-embedding layer, so an "
                "embeddings-only input cannot reconstruct the value stream."
            )
        args = self.args
        h = rms_norm(
            self.embed_tokens(inputs) if input_embeddings is None else input_embeddings
        )

        ve = None
        if self._any_ve:
            B, L = inputs.shape
            ve = self.value_embeds(inputs).reshape(
                B, L, args.ve_stored_heads, args.ve_dim
            )
            if args.ve_dim < args.head_dim:
                pad = mx.zeros(
                    (B, L, args.ve_stored_heads, args.head_dim - args.ve_dim), ve.dtype
                )
                ve = mx.concatenate([ve, pad], axis=-1)
            if args.ve_stored_heads > args.num_key_value_heads:
                ve = ve[:, :, : args.num_key_value_heads]
            ve = ve.astype(h.dtype)

        if cache is None:
            cache = [None] * len(self.layers)

        gi = next((i for i in range(len(self.layers)) if i in self._global), None)
        li = next((i for i in range(len(self.layers)) if i not in self._global), None)
        global_mask = create_attention_mask(h, cache[gi]) if gi is not None else None
        local_mask = (
            create_attention_mask(h, cache[li], window_size=args.local_window)
            if li is not None
            else None
        )

        history = {0: h} if self._tap_idx else {}
        for i, (layer, c) in enumerate(zip(self.layers, cache)):
            taps = self._tap_idx.get(i)
            if taps is not None:
                values = [history[t] for t in taps]
                attn_in = rms_norm(self.mudd.combine(values, h, i))
                residual_base = (
                    self.mudd.combine(values, h, i, r_way=True)
                    if self.mudd.uses_r_way
                    else h
                )
            else:
                attn_in = rms_norm(h)
                residual_base = h
            mask = global_mask if i in self._global else local_mask
            h = layer(h, attn_in, residual_base, ve, mask, c)
            if (i + 1) in self._retained:
                history[i + 1] = h

        return rms_norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = LimiteModel(args)
        cap = args.softcap_logits
        self.softcap_a = float(cap["a"])
        self.softcap_b = float(cap["b"])
        self.softcap_c = float(cap["c"])

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        h = self.model(inputs, cache, input_embeddings)
        # `tie_word_embeddings` is required by this architecture, and
        # `lm_head_precision_mode: oracle_exact` means the GEMM runs in the
        # activation dtype; only the softcap is float32.
        raw = self.model.embed_tokens.as_linear(h)
        return self.softcap_a * mx.sigmoid(
            (raw.astype(mx.float32) + self.softcap_b) / self.softcap_c
        )

    def sanitize(self, weights):
        """Fold the per-layer projection scalars into the projections.

        The reference computes `scale.to(weight.dtype) * weight`, where `scale`
        is a 0-dim float32 tensor and `weight` is bfloat16. PyTorch treats a
        0-dim tensor as a wrapped scalar for promotion, so the product is
        bfloat16 and the scalar is rounded to bfloat16 first. Reproduced here.

        Idempotent: an already-converted MLX checkpoint carries no `*_scale`
        keys, so this is a no-op on reload.
        """
        folds = {"qkv_scale": ("q_proj", "k_proj", "v_proj"), "o_scale": ("o_proj",)}
        out = {k: v for k, v in weights.items() if k.rsplit(".", 1)[-1] not in folds}
        for key, value in weights.items():
            stem, _, leaf = key.rpartition(".")
            if leaf not in folds:
                continue
            for target in folds[leaf]:
                wk = f"{stem}.{target}.weight"
                w = out[wk]
                out[wk] = value.astype(w.dtype) * w
        return out

    def cast_predicate(self, key):
        """Keep the checkpoint's float32 auxiliary tensors in float32.

        `xsa_alpha` is consumed as `tanh(alpha.float())`, so narrowing it to
        bfloat16 at conversion time would change the numbers. The others are
        cast to the activation dtype at the point of use and so are unaffected,
        but they total under 0.5 MB, so there is nothing to gain by narrowing
        them and one less discrepancy by not.
        """
        return not key.endswith(
            (
                "xsa_alpha",
                "attn_gate",
                "ve_gate",
                "resid_lambda_attn",
                "post_lambda_attn",
                "resid_lambda_mlp",
                "post_lambda_mlp",
                "mudd.dense1",
                "mudd.dense2",
                "mudd.dense2_mlp",
                "mudd.bias",
                "mudd.bias_mlp",
            )
        )

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        window = self.args.local_window
        return [
            KVCache() if i in self.model._global else RotatingKVCache(max_size=window)
            for i in range(self.args.num_hidden_layers)
        ]
