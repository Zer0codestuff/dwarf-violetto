# Result files

These records are measurements and generated completions, not model weights.
BENCHMARKS.md is rebuilt from them with `python3 scripts/report.py`.

- `final-*.json`: the September 25-26 fixed-work speed protocol.
- `bf16-math-eos.json`, `q8-math-eos.json`: all 12 local answers completed at EOS.
- `*-fidelity-final.json`, `fusion-fidelity.json`: complete capped distributions
  at 359 fixed positions. This is separate from mathematical answer accuracy.
- `decode-candidate-correctness.json`: 960 forced decode comparisons through
  32768-token prefixes, including full logits and final cache state.
- `paired-decode-settled-*.json`: later alternating comparisons used to assess
  compiled RoPE. The additional candidates remain disabled by default.
- `decode-optimization-summary.json`: selection decision, source files and
  excluded timing series.
- `decode-profile-cleared.json`: synchronized diagnostic timings with allocator
  cache clearing. Component synchronization changes execution scheduling.
- `throughput-feasibility.json`: forced batch/verification pilot. It is not a
  stochastic batch evaluator or speculative generation implementation.
- `publication-*.json`: checks repeated for the publication review.

Other files are experiment history. In particular, capped math runs contain
unfinished reasoning; `decode-candidate-correctness-initial.json` is a partial
run; `decode-profile.json` accumulated allocator buffers; the three excluded
speed series are listed in `decode-optimization-summary.json`.

The original measurements predate the stronger weight/tokenizer fingerprint.
Their source hashes are retained as recorded. Absolute script-path keys in
September 30 records were normalized to repository-relative paths for public
distribution; hashes, timing values, tokens and completion text were preserved.
Full reference arrays (`*.npz`) stay local and can be regenerated from the pinned
checkpoint. Fresh-cache trimmability in the feasibility pilot does not establish
rollback safety after the rotating cache fills.
