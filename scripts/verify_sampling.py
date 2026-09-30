"""Check exact nucleus masks and RNG behavior against stock MLX-LM."""
import json
import argparse
from pathlib import Path
import numpy as np
import mlx.core as mx
from mlx_lm.sample_utils import apply_top_p, make_sampler as stock
from dwarf_violetto.sampling import top_p_filter, make_sampler

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--reference-dir", default="results/q8-reference")
parser.add_argument("--output", default="results/sampling-correctness.json")
args = parser.parse_args()
paths = sorted(Path(args.reference_dir).glob("reference-*.npz"))
if len(paths) != 12:
    parser.error("Expected 12 Q8 reference arrays; generate them with dwarf_violetto.fidelity --reference")
positions = 0
for path in paths:
    with np.load(path) as data:
        lp = mx.array(data["logprobs"])
    original = apply_top_p(lp, 0.95)
    changed = top_p_filter(lp, 0.95)
    assert bool(mx.array_equal(original, changed)), path.name
    for seed in [0, 42, 123]:
        mx.random.seed(seed)
        a = stock(0.6, top_p=0.95)(lp)
        mx.eval(a)
        mx.random.seed(seed)
        b = make_sampler(0.6, top_p=0.95)(lp)
        mx.eval(b)
        assert bool(mx.array_equal(a,b)), (path.name, seed)
    positions += len(lp)
assert positions == 359
result = {"positions": positions, "top_p": 0.95, "temperature": 0.6,
          "seeds": [0,42,123], "exact_filter_match": True, "exact_token_match": True}
dest = Path(args.output)
dest.parent.mkdir(parents=True, exist_ok=True)
dest.write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result))
