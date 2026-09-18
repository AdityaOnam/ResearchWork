# `qssm/efficiency/`: cost benchmark

`bench.py` measures, for each arm: total, encoder and per-module parameters, forward
FLOPs, peak training and inference memory, per-slice latency and per-subject inference
time.

```bash
python bench.py --cpu-only                                   # params + FLOPs: any machine, any time
python bench.py --json out/bench.json --md out/bench.md      # + GPU memory / latency (needs an idle GPU)
```

## Static costs (deterministic, from `--cpu-only`)

| Arm | Params | vs A0 | GFLOPs / slice | vs A0 |
|---|---:|---:|---:|---:|
| A0 ViT | 509,383 | — | 8.33 | — |
| A1 hybrid | 513,591 | +0.8% | 8.68 | +4.2% |
| A2 scan | 408,775 | −19.8% | 6.97 | −16.4% |
| **A3 QSSM** | **412,983** | **−18.9%** | **7.32** | **−12.2%** |

These are properties of the architecture, not of a training run, so the command above
reproduces them exactly.

## Measurement protocol (enforced by the script)

- **No timing on a shared GPU.** If another process holds the GPU, timings are skipped.
  `--force` runs them anyway and stamps the output `UNRELIABLE`. Timings taken during
  concurrent jobs are not comparable. On SLURM use `hpc/slurm/bench.sbatch`, which
  requests `--exclusive`.
- **Memory** is `torch.cuda.max_memory_allocated` after a clean reset, with
  `cudnn.benchmark` off. Autotuning reserves a large one-time workspace that would
  otherwise be recorded as a real requirement.
- **Latency** is measured after warm-up, as the mean over `--iters` runs bracketed by
  CUDA synchronisation.
- FLOPs are counted with `torch.utils.flop_counter` at batch 1, 91 × 174 × 145.
