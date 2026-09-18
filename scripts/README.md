# `scripts/`: repository-level utilities

| Script | What it does |
|---|---|
| `setup_check.py` | First-run verification without data. Checks Python/PyTorch/CUDA versions, required and optional packages (`mamba-ssm`, DIPY, TractSeg), external tools on `PATH` (MRtrix3, SS3T-CSD, FSL), the `QSSM_*` path variables, the subject split files, and runs a forward pass of every buildable model arm. |

```bash
python scripts/setup_check.py
```

Missing optional tools are reported as `warn`. The script exits non-zero only when a
required check fails.
