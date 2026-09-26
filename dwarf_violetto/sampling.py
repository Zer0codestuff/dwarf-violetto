# Copyright © 2023-2024 Apple Inc.
# Adapted from MLX-LM sample_utils.py. See licenses/MLX-LM-MIT.txt.
"""Stock MLX-LM nucleus semantics with one fewer permutation of the vocabulary.

The retention mask is scattered directly back to vocabulary order. This
replaces the stock inverse-permutation construction and cumulative-probability
gather. Temperature order and random categorical sampling remain unchanged.
"""
from functools import partial
import mlx.core as mx


@mx.compile
def top_p_filter(logprobs, top_p):
    probabilities = mx.exp(logprobs)
    indices = mx.argsort(logprobs, axis=-1)
    ordered_probabilities = mx.take_along_axis(probabilities, indices, axis=-1)
    cumulative = mx.cumsum(ordered_probabilities, axis=-1)
    keep = mx.put_along_axis(mx.zeros(logprobs.shape, dtype=mx.bool_),
                             indices, cumulative > 1-top_p, axis=-1)
    return mx.where(keep, logprobs, -float("inf"))


def make_sampler(temp=0.6, top_p=0.95):
    if temp == 0:
        return lambda x: mx.argmax(x, axis=-1)

    @partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
    def sample(logprobs):
        if 0 < top_p < 1:
            logprobs = top_p_filter(logprobs, top_p)
        return mx.random.categorical(logprobs * (1/temp))

    return sample
