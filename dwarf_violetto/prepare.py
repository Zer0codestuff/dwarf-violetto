"""Derive quantized checkpoints locally from the original BF16 weights."""
import argparse
import json
import shutil
from pathlib import Path
from mlx_lm.utils import quantize_model, save_model
from .runtime import load


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="models/original")
    p.add_argument("--recipe", choices=["q8", "q6", "q4-hq"], required=True)
    args = p.parse_args()
    dest = Path("models") / args.recipe
    if dest.exists():
        raise SystemExit(f"Refusing to overwrite {dest}")
    model, _, config = load(args.source)
    bits = {"q8": 8, "q6": 6, "q4-hq": 4}[args.recipe]

    def predicate(path, module):
        # Protect both lookup tables, especially the tied output projection.
        selected = max(bits, 8) if "embed" in path else bits
        return {"bits": selected, "group_size": 64, "mode": "affine"}

    model, config = quantize_model(model, config, 64, bits, quant_predicate=predicate)
    save_model(dest, model)
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                 "chat_template.jinja", "generation_config.json", "source.json"):
        src = Path(args.source) / name
        if src.exists():
            shutil.copy2(src, dest / name)
    config["dwarf_recipe"] = args.recipe
    (dest / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(dest)


if __name__ == "__main__":
    main()
