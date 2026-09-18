# Withheld until publication

QSSM is currently under review. This public release shows the **full repository
structure, documentation, infrastructure and baseline**. The files that make up the
**novel method and the unpublished analyses** are placeholders.

Each placeholder keeps the real module's **public interface**: file name, class and
function names, signatures and one-line docstrings. Imports, the folder layout and the
READMEs therefore stay accurate. Calling a withheld function raises `WithheldError`
(from [qssm/_withheld.py](qssm/_withheld.py)), and running a withheld script prints a
notice and exits.

The full implementation will be released on publication. This repository is shared for
resume review; the author can walk through the withheld parts in an interview.

## Withheld files 🔒

| File | Component | Why |
|---|---|---|
| `qssm/models/qs_mamba.py` | QSSM encoder: selective-scan layers, spatial scan and q-space scan | core contribution |
| `qssm/models/qspace.py` | ordering of gradient directions for the q-space scan | core contribution |
| `qssm/data/synthetic_brain.py` | synthetic free-water brain generator | novel training data construction |
| `qssm/data/corpus.py` | real + synthetic slice corpus and losses | novel training data construction |
| `qssm/train/train_mamba.py` | trainer for the 2×2 encoder factorial | training recipe |
| `qssm/ablations/ablate_a3.py` | component knockouts of the trained QSSM | unpublished analysis |
| `qssm/ablations/ablate_gate_se.py` | gate / SE saturation analysis | unpublished analysis |
| `qssm/evaluate/evaluate_directions.py` | direction-subsampling robustness | unpublished analysis |
| `qssm/evaluate/evaluate_fa_md.py` | downstream FA/MD accuracy | unpublished analysis |
| `qssm/stats/fa_md_stats.py` | FA/MD statistics and tables | unpublished analysis |

## What works in this release

| Runs as-is | |
|---|---|
| `qssm/tests/test_shapes.py` | builds arm **A0** (ViT + PDE) and checks its parameter count; the other arms are reported as withheld |
| `qssm/efficiency/bench.py --cpu-only` | parameter and FLOP counts for A0 |
| `qssm/models/rdt_net.py`, `encoders.py` (ViT), `global_layer_rxn_diff.py`, `pde_block.py`, `building_blocks.py`, `ssm_ops.py` | the RDT-Net network and the reaction–diffusion PDE decoder |
| `qssm/evaluate/evaluate.py`, `evaluate_snr.py`, `qssm/data/noise_snr.py` | evaluation of any compatible checkpoint, Rician-noise robustness |
| `qssm/stats/stats.py` | the full statistics protocol (Wilcoxon, 2×2 interaction, TOST, Holm) |
| `rdtnet_baseline/` | RDT-Net training |
| `tractography/` | free-water-corrected CSD / SS3T-CSD tractography and TractSeg |
| `hpc/` | preprocessing, SLURM jobs, adaptive-compute controller, CUDA build script |

The numbers quoted in the READMEs for A1–A3 (parameters, FLOPs) come from the full
implementation and will be reproducible from it on release.
