# `qssm/`: Q-Space Selective-Scan Models

This folder is the main research contribution: a state-space (Mamba/S6) encoder that
scans diffusion MRI **across gradient directions**. It is studied in a controlled 2×2
factorial against the attention encoder it replaces.

## The idea in three sentences

1. Free water is **isotropic**: its signal is the same in every gradient direction, so it
   lives in the angular DC component of q-space.
2. The ViT baseline collapses the 91 directions with one linear convolution before any
   learned interaction, so it never models the angular profile.
3. QSSM adds a **state-space scan across the gradient directions**, so the angular
   structure is modelled explicitly, and replaces attention with a spatial scan. It does
   this with **−18.9% parameters and −12.2% FLOPs** compared with the ViT baseline.

High-level overview: [docs/architecture.md](../docs/architecture.md).

> 🔒 **Public release:** the QSSM encoder, q-space ordering, synthetic-data corpus,
> trainer and unpublished analyses are placeholders. See [WITHHELD.md](../WITHHELD.md).

## The 2×2 factorial

|  | **no q-scan** | **q-scan** |
|---|---|---|
| **attention** | A0 `--encoder vit` (control = RDT-Net) | A1 `--encoder hybrid` |
| **spatial scan** | A2 `--encoder mamba --no-qscan` | **A3 `--encoder mamba` (QSSM)** |

All four arms share the corpus, loss, optimiser, schedule, seed, decoder and warm start.
**Only the encoder changes.** Each main effect is therefore measured twice (A1−A0 and
A3−A2 for q-space; A2−A0 and A3−A1 for the spatial operator).

## Folder map

| Folder | What it contains | Start with |
|---|---|---|
| [models/](models) | encoders, q-space ordering, S6 kernels, PDE decoder | `encoders.py`, `qs_mamba.py` |
| [data/](data) | 70/30 real:synthetic slice corpus, synthetic-brain generator, Rician noise | `corpus.py` |
| [train/](train) | a single trainer for every arm | `train_mamba.py` |
| [evaluate/](evaluate) | test-set MAE, SNR sweep, direction subsampling, downstream FA/MD | `evaluate.py` |
| [ablations/](ablations) | pre-flight arm check and inference-time knockouts | `verify_arms.py` |
| [efficiency/](efficiency) | params / FLOPs / memory / latency | `bench.py` |
| [stats/](stats) | paired tests, 2×2 interaction, equivalence, multiple-comparison correction | `stats.py` |
| [tests/](tests) | forward pass of every arm on random tensors (no data) | `test_shapes.py` |
| [paths.py](paths.py) | the only place paths are resolved (environment variables) | — |

## End-to-end workflow

```bash
# 0. sanity (no data)
python tests/test_shapes.py
python models/ssm_ops.py --self-test          # fused kernel == PyTorch fallback (if mamba-ssm is installed)

# 1. pre-flight on your data: q-space order, arm param counts, warm-start behaviour
python ablations/verify_arms.py

# 2. train (A0 first, alone: it validates the pipeline against the known baseline)
python train/train_mamba.py --encoder vit    --real-frac 0.7   # A0
python train/train_mamba.py --encoder hybrid --real-frac 0.7   # A1
python train/train_mamba.py --encoder mamba  --real-frac 0.7 --no-qscan   # A2
python train/train_mamba.py --encoder mamba  --real-frac 0.7   # A3 (QSSM)

# 3. evaluate
python evaluate/evaluate.py --all --save-maps
python evaluate/evaluate_snr.py --all
python evaluate/evaluate_directions.py --all
python evaluate/evaluate_fa_md.py --cohort hcp

# 4. ablate and measure cost
python ablations/ablate_a3.py
python efficiency/bench.py --json $QSSM_RESULTS_DIR/bench.json --md $QSSM_RESULTS_DIR/bench.md

# 5. statistics (every number in a table comes from here)
python stats/stats.py --md $QSSM_RESULTS_DIR/stats.md
```

`hpc/slurm/` wraps steps 2–5 as SLURM jobs.

## Conventions used throughout

- **One metric.** `MAE = mean(|pred·mask − gt·mask|)` over the full 174×145 frame. The
  same quantity is the training loss term, the validation/selection criterion and the
  reported number. In-mask MAE is written as a diagnostic column only.
- **Masked input.** The corpus zeroes the input outside the brain, so evaluation does the
  same by default (`--mask-input`). A checkpoint records its own setting, and that
  setting takes precedence.
- **Deterministic.** The seed is fixed, the DataLoader has its own generator (so batch
  order doesn't depend on how many random draws model construction made), and noise is
  seeded per (subject, SNR).
- **Nothing hard-coded.** Paths come from `paths.py`, and subject lists from `splits/`.
