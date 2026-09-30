"""Decode-only Metal candidates. Keep prefill arithmetic unchanged."""
import mlx.core as mx


_rope = mx.fast.metal_kernel(
    name="violetto_rope_decode",
    input_names=["x", "cos", "sin"],
    output_names=["out"],
    ensure_row_contiguous=False,
    source="""
        uint i = thread_position_in_grid.x;
        if (i >= N) return;
        uint d = i % 128;
        uint loc = elem_to_loc(i, x_shape, x_strides, x_ndim);
        if (d < 64) {
            int step = (d % 2 == 0) ? 1 : -1;
            uint other = loc + step * x_strides[x_ndim - 1];
            T a = T(float(x[loc]) * float(cos[d]));
            T b = T(float(x[other]) * float(sin[d]));
            out[i] = T(float(a) + float(b));
        } else {
            out[i] = x[loc];
        }
    """,
)


def rope_decode(x, cos, sin):
    if x.shape[-2] != 1 or x.shape[-1] != 128:
        raise ValueError("The decode RoPE kernel requires one token and head dimension 128")
    return _rope(inputs=[x, cos, sin], template=[("T", x.dtype), ("N", x.size)],
                 grid=(x.size, 1, 1), threadgroup=(256, 1, 1),
                 output_shapes=[x.shape], output_dtypes=[x.dtype])[0]
