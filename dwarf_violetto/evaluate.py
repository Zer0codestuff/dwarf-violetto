"""Local answer checks with saved completions. Not a competition benchmark."""
import argparse
import json
import re
import time
from pathlib import Path
import mlx.core as mx
from mlx_lm import stream_generate
from mlx_lm.sample_utils import make_sampler
from .runtime import load, prompt_tokens
from .provenance import provenance


def boxed(text):
    starts = list(re.finditer(r"\\boxed\s*\{", text))
    if not starts:
        return None
    start = starts[-1].end()
    depth = 1
    for end in range(start, len(text)):
        depth += (text[end] == "{") - (text[end] == "}")
        if depth == 0:
            return text[start:end]
    return None


def normalize(value):
    if value is None:
        return None
    value = re.sub(r"\\[dt]?frac\{(-?\d+)\}\{(\d+)\}", r"\1/\2", value)
    return re.sub(r"[\s,$]", "", value)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/q8")
    p.add_argument("--optimized", action="store_true")
    p.add_argument("--fused", action="store_true")
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--limit", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True)
    p.add_argument("--only", help="Comma-separated case IDs; original seed indices are preserved")
    args = p.parse_args()
    cases = json.loads(Path("data/math_checks.json").read_text())[:args.limit]
    model, tokenizer, _ = load(args.model, args.optimized, args.fused)
    result = {"settings": vars(args), "sha256": provenance(args.model),
              "sampling": {"temperature": 0.6, "top_p": 0.95},
              "suite": "12 hand-authored local checks, not held-out competition evaluation", "cases": []}
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    for i, case in enumerate(cases):
        if args.only and case["id"] not in args.only.split(","):
            continue
        mx.random.seed(args.seed + i)
        text = ""
        start = time.perf_counter()
        for response in stream_generate(model, tokenizer, prompt_tokens(tokenizer, case["problem"]),
                                        max_tokens=args.max_tokens, prefill_step_size=512,
                                        sampler=make_sampler(0.6, top_p=0.95)):
            text += response.text
        answer = boxed(text)
        row = {"id": case["id"], "expected": case["answer"], "answer": answer,
               "correct": response.finish_reason == "stop" and normalize(answer) == normalize(case["answer"]),
               "finish": response.finish_reason, "tokens": response.generation_tokens,
               "seconds": time.perf_counter() - start, "text": text}
        result["cases"].append(row)
        result["correct"] = sum(x["correct"] for x in result["cases"])
        result["truncated"] = sum(x["finish"] == "length" for x in result["cases"])
        dest.write_text(json.dumps(result, indent=2) + "\n")
        print(f"{case['id']}: answer={answer} correct={row['correct']} tokens={row['tokens']} finish={row['finish']}", flush=True)


if __name__ == "__main__":
    main()
