"""Load local weights with an explicit, reviewed architecture."""
import json
from pathlib import Path

import mlx.core as mx
from mlx_lm.utils import load_model, load_tokenizer
from transformers import PreTrainedConfig
from .limite import Model, ModelArgs
from .optimize import optimize
from .checkpoint import checkpoint_files


class TokenizerConfig(PreTrainedConfig):
    model_type = "limite"


class InferenceModel(Model):
    """Avoid computing unused vocabulary logits during prompt processing."""
    def __call__(self, inputs, cache=None, input_embeddings=None):
        h = self.model(inputs, cache, input_embeddings)
        raw = self.model.embed_tokens.as_linear(h[:, -1:, :])
        return self.softcap_a * mx.sigmoid(
            (raw.astype(mx.float32) + self.softcap_b) / self.softcap_c
        )


def load(path, optimized=False, fused=False):
    path = Path(path).resolve()
    checkpoint_files(path)
    config = json.loads((path / "config.json").read_text())
    if config.get("model_file"):
        raise ValueError("External model code is not supported. Use the pinned checkpoint.")
    cls = InferenceModel if optimized else Model
    model, config = load_model(path, get_model_classes=lambda config: (cls, ModelArgs))
    if optimized:
        optimize(model)
    if fused:
        if model.args.xsa_normalize_eps != 1e-4:
            raise ValueError("Compiled XSA requires the checkpoint's epsilon of 1e-4")
        from .fused import xsa, residual
        for layer in model.layers:
            layer.self_attn._fused_xsa = xsa
            layer._fused_residual = residual
    tokenizer = load_tokenizer(path, tokenizer_config_extra={
        "fix_mistral_regex": False, "local_files_only": True,
        "config": TokenizerConfig.from_dict(config),
    },
                               eos_token_ids=[151643, 151645])
    return model, tokenizer, config


def prompt_tokens(tokenizer, problem):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": problem}],
        add_generation_prompt=True, tokenize=True, return_dict=False,
    )
