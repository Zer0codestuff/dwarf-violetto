# Contributing

This project targets Limite 1B Violetto on Apple Silicon. Keep changes focused
and include the model recipe, hardware, dependency versions and command used
to reproduce a problem or measurement.

Run the CPU checks before opening a pull request:

```sh
python3 -m unittest discover -s tests -v
python3 scripts/verify_fixtures.py
python3 scripts/check_repository.py
```

Runtime changes also need the Metal checks listed in README.md. Measure
prefill and decode separately, with warmup, repetitions, memory and source
hashes. Keep existing result files as historical evidence and write new
measurements to a new path. Small timing gains need an alternating comparison.

Preserve BF16 rounding sites, ordered MUDD sums, the 1025-key local window,
the mathematical chat template and EOS stopping. A change to sampling or
precision needs numerical and answer-quality evidence. Do not infer answer
accuracy from a truncated response or a matching top prediction.

Do not commit weights, environments, numerical reference arrays, credentials
or personal filesystem paths. Use English for source, documentation and UI.

Contributions use the repository's Apache-2.0 license. Retain the separate MIT
notice when modifying the sampler adapted from MLX-LM.
