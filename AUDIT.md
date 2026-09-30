# Publication review

Reviewed on September 30, 2026, on the project's Apple M4 MacBook Pro.
The project is suitable for publication as an independent, experimental
MLX runtime with reproducible measurements for Limite 1B Violetto.

The review covered every source module and script, launchers, setup,
dependency manifests, fixtures, result accounting, documentation, licenses
and the existing Git history. Research candidates remain disabled by default.

## Verified behavior

| Check | Result | Evidence |
| --- | --- | --- |
| Fresh setup | Python 3.12.14, all 35 pinned runtime packages installed; Q8 weight bytes, config, tokenizer and chat template matched the existing checkpoint | [Clean setup](results/publication-clean-setup.json) |
| Official BF16 weights | SHA256 matched the pinned Hugging Face revision; the clean setup reused cached weight files | `scripts/download.py`, `data/sources.json` |
| Offline CLI | Correct answer at EOS, 851 generated tokens, identical to the historical seeded divisor trace | [Clean setup](results/publication-clean-setup.json) |
| Interactive CLI | Two independent problems with the same prompt and seed produced the same complete response | [Interactive check](results/publication-interactive.json) |
| Evaluator resume | Completed case preserved without duplicates; changed seed refused; real evaluation completed at EOS | [Resume check](results/publication-resume.json), [Evaluation](results/publication-evaluator.json) |
| Packing and compiled operations | Full logits matched exactly; tokenizer matched raw encoding; local-window boundary crossed at 1100 tokens | [Runtime check](results/publication-runtime.json) |
| Decode candidates | Full logits and final cache state matched at 960 forced positions across prefixes of 1, 128, 1100, 8192 and 32768 tokens | [Decode checks](results/publication-decode-correctness.json) |
| Q8 numerical fidelity | Same result as historical measurements: mean KL 0.001434, 355/359 top predictions matched BF16 | [Q8 fidelity](results/publication-q8-fidelity.json) |
| Optional quantizations | Q6 and Q4-HQ distributions rechecked against newly generated BF16 arrays; both retain their larger measured deviations | [Q6](results/publication-q6-fidelity.json), [Q4-HQ](results/publication-q4-hq-fidelity.json) |
| Fusion fidelity | Exact full distributions at all 359 positions, using newly generated Q8 reference arrays | [Fusion fidelity](results/publication-fusion-fidelity.json) |
| Simplified sampler | Exact stock masks and sampled tokens at 359 positions and seeds 0, 42 and 123, using new Q8 arrays | [Sampler check](results/publication-sampling.json) |
| Independent CPU reference | PyTorch 2.13.0 matched 20/21 top predictions; mean KL 0.002068 against MLX BF16 | [Runtime check](results/publication-runtime.json) |
| Mathematical fixtures | All 12 answer keys independently checked; all 24 saved BF16/Q8 EOS completions regraded from their text | `scripts/verify_fixtures.py`, `scripts/check_repository.py` |
| CPU contracts | Eight tests passed, also with site packages disabled; covers output limits, grading, interrupted conversion/save, checkpoint validation and resume fingerprints | `tests/test_contracts.py` |

Historical speed medians were recomputed from the individual runs. Fixed-work
token accounting and weighted fidelity aggregates were checked. BENCHMARKS.md
and `results/summary.json` were regenerated and checked for repeatability.
The publication checks do not replace the historical speed protocol.

## Corrections

- Setup now checks native Apple Silicon, uv and Python 3.12. It verifies the
  official weight digest and checks an existing Q8 checkpoint for incomplete files.
- Quantization stages the output and validates it before renaming. An interrupted
  conversion leaves no incomplete final directory. Recipes cannot overwrite
  existing checkpoints or use already quantized source weights.
- Resume fingerprints now include weight bytes, tokenizer files, code, fixtures,
  dependency versions and the lock file. Old evaluation records cannot resume
  with the stronger fingerprint.
- Invalid sampling values, prefill chunks and case selections fail before
  generation. The CLI resets peak memory for each independent problem.
- Unsupported untied embeddings and FP32-head configuration now fail explicitly.
  Compiled XSA also checks its expected epsilon. Default model arithmetic is unchanged.
- The rotating cache is trimmable before its 1025-key capacity is reached, then
  becomes non-trimmable. The feasibility pilot recorded a fresh cache. Earlier
  documentation incorrectly treated that limitation as unconditional.
- Manual prefill in the remaining research scripts now clears allocator buffers.
- Public result metadata uses relative script paths. Numerical values, hashes,
  tokens and raw historical completions were preserved.
- The quick start now works for a new clone. Contribution guidance, license
  attribution and CPU CI are included. Actions are pinned to verified commits.

## Dependencies, provenance and distribution

`pip-audit` 2.10.1 found no known vulnerabilities in the 40 installed runtime
and optional-reference packages after upgrading optional PyTorch to 2.13.0 and
setuptools to 83.0.0. The 35 core runtime pins did not change. See
[dependency audit](results/publication-dependencies.json). The relevant advisories
are [PyTorch](https://github.com/advisories/GHSA-rrmf-rvhw-rf47) and
[setuptools](https://github.com/advisories/GHSA-h35f-9h28-mq5c).

Gitleaks 8.30.1 scanned both original commits and distributable source files.
Only public fixture/tokenizer SHA256 values triggered its generic-key rule.
The configuration excludes those exact verified values, with file restrictions;
default detection remains enabled. A synthetic credential was detected by a
separate positive control. No actual credential was found. Scans do not guarantee
the absence of all vulnerabilities or secrets.

The official model and reviewed MLX port declare Apache-2.0. The official source
includes LICENSE and no NOTICE. The sampler's separate Apple copyright and MIT
license are retained. Root LICENSE, NOTICE and THIRD_PARTY.md describe the source
distribution. Model weights, numerical arrays, research downloads, caches and
environments remain outside Git. The [Apache redistribution conditions](https://www.apache.org/licenses/LICENSE-2.0)
require retaining applicable notices and identifying modified files.

The original local commits contain a personal author email. Preserve that local
history and use GitHub's noreply identity in the reviewed publication copy.
The repository destination and visibility still need the user's explicit choice.
No GitHub repository has been created by this review.

## Remaining limits

This review does not establish competition-level mathematical accuracy, zero
quantization loss, CUDA bitwise equivalence or a hardware-wide speed optimum.
The configured 131072-token maximum remains unvalidated in full. Continuous GPU
power/frequency, ANE execution, a variable-length batch evaluator and safe
speculative rollback remain untested or unimplemented.

The 12 local math problems are small regression fixtures. The fresh end-to-end
check reused one of them; the 24 complete historical responses were regraded,
not all regenerated. Numerical comparisons cover a fixed 359-position corpus
and repeated long-prefix traces. They do not replace a held-out evaluation.

GitHub-hosted CI has been prepared and its commands passed locally without MLX.
Its first hosted run can be checked after publication. CPU CI does not validate
Metal inference. The experimental broad post-attention fusion remains rejected
because its sampled trace changed.
