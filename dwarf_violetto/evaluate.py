"""Local answer checks with saved completions. Not a competition benchmark."""
import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
import mlx.core as mx
from mlx_lm import stream_generate
from mlx_lm.sample_utils import make_sampler
from .runtime import load, prompt_tokens
from .provenance import provenance
from .generation import output_budget, finish_reason


def save(dest, result):
    """Keep the last complete JSON record if generation is interrupted."""
    temporary = dest.with_suffix(dest.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(dest)


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
    p.add_argument("--max-tokens", type=int, help="Optional output cap; default: EOS within the model context")
    p.add_argument("--limit", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True)
    p.add_argument("--only", help="Comma-separated case IDs; original seed indices are preserved")
    p.add_argument("--resume", action="store_true", help="Skip saved completed cases; restart an interrupted case with its original seed")
    args = p.parse_args()
    if args.max_tokens is not None and args.max_tokens < 1:
        p.error("An explicit output budget must be positive")
    cases = json.loads(Path("data/math_checks.json").read_text())[:args.limit]
    settings = {k: v for k, v in vars(args).items() if k not in {"resume", "output"}}
    result = {"settings": settings, "sha256": provenance(args.model),
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "sampling": {"temperature": 0.6, "top_p": 0.95},
              "suite": "12 hand-authored local checks, not held-out competition evaluation", "cases": []}
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if not args.resume:
            p.error("Output already exists; choose a new file or use --resume")
        previous = json.loads(dest.read_text())
        if previous["settings"] != settings or previous["sha256"] != result["sha256"]:
            p.error("Cannot resume with different settings, source, model config or fixtures")
        result = previous
    model, tokenizer, _ = load(args.model, args.optimized, args.fused)
    result["context_length"] = model.args.max_position_embeddings
    result["output_policy"] = "eos_or_context_limit" if args.max_tokens is None else "explicit_cap"
    completed = {row["id"] for row in result["cases"]}
    for i, case in enumerate(cases):
        if args.only and case["id"] not in args.only.split(","):
            continue
        if case["id"] in completed:
            continue
        tokens = prompt_tokens(tokenizer, case["problem"])
        budget = output_budget(len(tokens), model.args.max_position_embeddings, args.max_tokens)
        mx.random.seed(args.seed + i)
        mx.reset_peak_memory()
        text = ""
        start = time.perf_counter()
        checkpoint = start
        first = None
        result["in_progress"] = {"id": case["id"], "seed": args.seed + i, "tokens": 0,
                                 "prompt_tokens": len(tokens), "effective_output_ceiling": budget,
                                 "text": "", "seconds": 0}
        save(dest, result)
        print(f"Starting {case['id']}: seed={args.seed+i}, output_policy={result['output_policy']}", flush=True)
        for response in stream_generate(model, tokenizer, tokens,
                                        max_tokens=budget, prefill_step_size=512,
                                        sampler=make_sampler(0.6, top_p=0.95)):
            text += response.text
            now = time.perf_counter()
            if first is None:
                first = now
            if now - checkpoint >= 30:
                result["in_progress"].update(tokens=response.generation_tokens, text=text, seconds=now-start)
                save(dest, result)
                print(f"{case['id']}: {response.generation_tokens} tokens, {now-start:.1f}s, still generating", flush=True)
                checkpoint = now
        answer = boxed(text)
        finish = finish_reason(response.finish_reason, len(tokens), model.args.max_position_embeddings, args.max_tokens)
        row = {"id": case["id"], "expected": case["answer"], "answer": answer,
               "correct": finish == "stop" and normalize(answer) == normalize(case["answer"]),
               "finish": finish, "stop_token_id": response.token if finish == "stop" else None,
               "tokens": response.generation_tokens, "prompt_tokens": len(tokens),
               "effective_output_ceiling": budget, "seed": args.seed + i,
               "seconds": time.perf_counter() - start, "ttft_seconds": first-start,
               "peak_mlx_gb": response.peak_memory, "text": text}
        result["cases"].append(row)
        result.pop("in_progress", None)
        result["correct"] = sum(x["correct"] for x in result["cases"])
        result["truncated"] = sum(x["finish"] != "stop" for x in result["cases"])
        save(dest, result)
        print(f"{case['id']}: answer={answer} correct={row['correct']} tokens={row['tokens']} finish={row['finish']}", flush=True)
    result["completed_utc"] = datetime.now(timezone.utc).isoformat()
    save(dest, result)


if __name__ == "__main__":
    main()
