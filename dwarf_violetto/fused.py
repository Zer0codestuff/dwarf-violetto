"""Compiled XSA and residual operations, checked against the eager path."""
from functools import partial
import mlx.core as mx


@mx.compile
def xsa(y, v, alpha):
    repeats = y.shape[-2] // v.shape[-2]
    vf = mx.repeat(v, repeats, axis=2).astype(mx.float32)
    norm = mx.maximum(mx.sqrt(mx.sum(vf * vf, axis=-1, keepdims=True)), 1e-4)
    vn = vf / norm
    proj = mx.sum(y.astype(mx.float32) * vn, axis=-1, keepdims=True)
    a = mx.tanh(alpha.astype(mx.float32)).reshape(1, 1, -1, 1)
    return y - (a * proj * vn).astype(y.dtype)


@partial(mx.compile, shapeless=True)
def residual(a, x, b, branch):
    return a.astype(x.dtype) * x + b.astype(x.dtype) * branch
