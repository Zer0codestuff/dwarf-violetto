"""Single-turn mathematical inference, fully local after model download."""
import argparse
import sys
import time
import mlx.core as mx
from mlx_lm import stream_generate
from .sampling import make_sampler
from .runtime import load, prompt_tokens


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("problem", nargs="?")
    p.add_argument("--model", default="models/q8")
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--temp", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--interactive", action="store_true", help="Keep weights loaded; each problem is an independent turn")
    p.add_argument("--fused", action=argparse.BooleanOptionalAction, default=True,
                   help="Use verified compiled elementwise operations (default: enabled)")
    p.add_argument("--stock-sampler", action="store_true", help="Use the original MLX-LM sampler for comparison")
    args = p.parse_args()
    interactive = args.interactive or (not args.problem and sys.stdin.isatty())
    if args.max_tokens < 1 or args.temp < 0 or not 0 < args.top_p <= 1:
        p.error("Use a positive token budget, nonnegative temperature and 0 < top-p <= 1")
    problem = args.problem or (None if interactive else sys.stdin.read().strip())
    if not interactive and not problem:
        p.error("Provide a problem")
    model, tokenizer, _ = load(args.model, optimized=True, fused=args.fused)
    sampler_factory = make_sampler
    if args.stock_sampler:
        from mlx_lm.sample_utils import make_sampler as sampler_factory
    while True:
        if not problem:
            try:
                problem = input("Problem (or /quit): ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if problem == "/quit":
                break
            if not problem:
                continue
        tokens = prompt_tokens(tokenizer, problem)
        if len(tokens) + args.max_tokens > model.args.max_position_embeddings:
            p.error("Prompt and output budget exceed the model context")
        mx.random.seed(args.seed)
        start = time.perf_counter()
        first = None
        try:
            for response in stream_generate(model, tokenizer, tokens, max_tokens=args.max_tokens,
                                            sampler=sampler_factory(args.temp, top_p=args.top_p),
                                            prefill_step_size=512):
                if first is None:
                    first = time.perf_counter()
                print(response.text, end="", flush=True)
            elapsed = time.perf_counter() - start
            print()
            print(f"\n{response.generation_tokens} tokens | {elapsed:.2f}s total | "
                  f"{first-start:.2f}s to first token | {response.peak_memory:.2f} GB MLX | "
                  f"finish={response.finish_reason}", file=sys.stderr)
            if response.finish_reason == "length":
                print("Token budget reached. The answer may be incomplete.", file=sys.stderr)
        except KeyboardInterrupt:
            print("\nGeneration interrupted.", file=sys.stderr)
        if not interactive:
            break
        problem = None


if __name__ == "__main__":
    main()
