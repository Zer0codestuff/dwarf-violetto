"""Derive quantized checkpoints locally from the original BF16 weights."""
import argparse
import json
import shutil
from pathlib import Path
from mlx_lm.utils import quantize_model, save_model
from .runtime import load
from .checkpoint import staged_checkpoint


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="models/original")
    p.add_argument("--recipe", choices=["q8", "q6", "q4-hq"], required=True)
    args = p.parse_args()
    dest = Path("models") / args.recipe
    if dest.exists():
        raise SystemExit(f"Refusing to overwrite {dest}")
    source_config = json.loads((Path(args.source) / "config.json").read_text())
    if source_config.get("quantization") or source_config.get("quantization_config"):
        p.error("Derive each recipe from the original BF16 checkpoint, not quantized weights")
    model, _, config = load(args.source)
    bits = {"q8": 8, "q6": 6, "q4-hq": 4}[args.recipe]

    def predicate(path, module):
        # Protect both lookup tables, especially the tied output projection.
        selected = max(bits, 8) if "embed" in path else bits
        return {"bits": selected, "group_size": 64, "mode": "affine"}

    model, config = quantize_model(model, config, 64, bits, quant_predicate=predicate)
    with staged_checkpoint(dest) as staging:
        save_model(staging, model)
        for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                     "chat_template.jinja", "generation_config.json", "source.json", "LICENSE", "NOTICE"):
            src = Path(args.source) / name
            if src.exists():
                shutil.copy2(src, staging / name)
        config["dwarf_recipe"] = args.recipe
        (staging / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(dest)


if __name__ == "__main__":
    main()
