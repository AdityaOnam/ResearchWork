# `qssm/models/`: network definitions

> 🔒 = withheld until publication (placeholder with the real interface; see [WITHHELD.md](../../WITHHELD.md)).

| File | Contents |
|---|---|
| `qs_mamba.py` 🔒 | **The QSSM encoder**: selective-scan layers, spatial scan and q-space scan |
| `encoders.py` | `build_encoder(kind)` for the factorial: `ViTEncoder` (A0), `QHybridViT` (A1), `QSMambaEncoder` (A2/A3) |
| `qspace.py` 🔒 | ordering of the gradient directions for the q-space scan |
| `ssm_ops.py` | backend dispatch: fused `mamba-ssm` CUDA kernel or pure-PyTorch scan, same weights; `--self-test` |
| `rdt_net.py` | `ConvGlobalSkipModel`: encoder + PDE decoder (the full network), `SimpleViT`, slice preprocessing |
| `global_layer_rxn_diff.py` | `GlobalFeatureBlock_Diffusion`: the reaction–diffusion PDE block (K Euler steps, adaptive/`dt_live` step) |
| `pde_block.py` | PDE defaults (`get_default_pde_args`) and shared PDE layers |
| `building_blocks.py`, `operations.py`, `utils.py`, `utils_pde.py` | convolutional building blocks (DSConv, SE, activations, drop-path) |

## Building a model

```python
import sys; sys.path.insert(0, "qssm"); import paths          # puts models/ on sys.path
from rdt_net import ConvGlobalSkipModel

a3 = ConvGlobalSkipModel(in_channels=91, K=5, encoder="mamba")                        # QSSM
a2 = ConvGlobalSkipModel(in_channels=91, K=5, encoder="mamba", enc_kw={"use_qscan": False})
a1 = ConvGlobalSkipModel(in_channels=91, K=5, encoder="hybrid")
a0 = ConvGlobalSkipModel(in_channels=91, K=5, encoder="vit")

y = a3(x)                  # x: (B, 91, 174, 145)  ->  y: (B, 1, 174, 145), free-water fraction
y = a3(x, order=perm)      # optional explicit q-space traversal (default: canonical)
```

Add `enc_kw={"kernel": "torch"}` to force the pure-PyTorch scan (CPU), or `"fused"` to
require the CUDA kernel.

## Parameter budget (at 174×145)

| Arm | Encoder | Total | Positional embedding | Q-branch |
|---|---:|---:|---:|---:|
| A0 ViT | 392,384 | 509,383 | 99,072 | 0 |
| A1 hybrid | 396,592 | 513,591 | 99,072 | 4,208 |
| A2 scan | 291,776 | 408,775 | 0 | 0 |
| A3 QSSM | 295,984 | 412,983 | 0 | 4,208 |

The ViT's positional embedding is created lazily at the first forward pass and scales
with the input frame, so its count only holds at 174×145. The scan encoders have no
positional embedding.
