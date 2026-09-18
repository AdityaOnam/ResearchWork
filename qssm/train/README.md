# `qssm/train/`: training

> 🔒 `train_mamba.py` is withheld until publication. It is a placeholder with the real
> interface; see [WITHHELD.md](../../WITHHELD.md).

A single trainer covers every arm of the 2×2 factorial. Only the encoder changes between
arms. Data, decoder, loss, optimiser, schedule and seed are shared.

```bash
python train_mamba.py --encoder vit    --real-frac 0.7              # A0  ViT baseline
python train_mamba.py --encoder hybrid --real-frac 0.7              # A1  attention + q-space scan
python train_mamba.py --encoder mamba  --real-frac 0.7 --no-qscan   # A2  spatial scan
python train_mamba.py --encoder mamba  --real-frac 0.7              # A3  QSSM
```

| Flag | Meaning |
|---|---|
| `--encoder {vit,hybrid,mamba}` | encoder family |
| `--no-qscan` | drop the q-space scan (A2) |
| `--real-frac` | fraction of real (vs synthetic) training slices |
| `--ssm-kernel {auto,fused,torch}` | selective-scan backend (see `INSTALL.md`) |
| `--seed`, `--epochs` | reproducibility and schedule length |
| `--subjects N` | smoke test on N subjects |

All arms warm-start their PDE decoder from the RDT-Net checkpoint (`$QSSM_WARM_START`).

**Outputs** (git-ignored): logs and metrics in `$QSSM_RESULTS_DIR/<run-tag>/`, and the
best checkpoint in `$QSSM_CKPT_DIR/`.

On a SLURM cluster, `hpc/slurm/train_arms.sbatch` trains the four arms as one array job.
