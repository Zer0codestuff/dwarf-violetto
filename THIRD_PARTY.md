# Sources and attribution

The model and original inference formulas are by Paradigma, under Apache-2.0.

- Model: https://huggingface.co/paradigma-inc/limite-1b-violetto
- Model revision: `9402e422e87b220507963cd42c997459c1e87b43`
- Official implementation: https://github.com/paradigma-inc/limite-violetto
- Reviewed source revision: `d9f8fea11c79d596bd8c8bbff4b78020b2d01ad6`

`dwarf_violetto/limite.py` adapts Piergiorgio Di Pasquale's MLX port, published under Apache-2.0:

- https://huggingface.co/pierjoe/limite-1b-violetto-mlx-bf16
- Reviewed revision: `ffa57c444fafdba0019c55d4f502bc6b9ddcc123`
- Local changes: explicit MLX-LM imports, optional packed decode projections, prefill projections with original shapes, and optional compiled operations for experiments.

MLX and MLX-LM are by Apple and their contributors, under MIT. They provide the Metal kernels, quantization, cache implementations, sampling, and token streaming. Their license files are installed with the packages.

`dwarf_violetto/sampling.py` adapts the MLX-LM 0.31.3 nucleus filter. The original Apple copyright notice and MIT license are retained in the source and `licenses/MLX-LM-MIT.txt`.

Dwarf Star by Salvatore Sanfilippo and its contributors inspired the model-specific measurement approach. No Dwarf Star code is included.

- https://github.com/antirez/ds4

The local math checks are hand-authored project fixtures. They are not an official benchmark dataset.
