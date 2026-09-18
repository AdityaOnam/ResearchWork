# Installation

## 1. Python environment

Python 3.10+ and PyTorch 2.1+ are required. The code was developed with PyTorch 2.12
(CUDA 13.0) on an NVIDIA L40S (sm_89).

```bash
python -m venv .venv && source .venv/bin/activate
# install the torch build that matches your CUDA driver first: https://pytorch.org
pip install -r requirements.txt
```

Check the install (no data or GPU needed):

```bash
python qssm/tests/test_shapes.py        # all four arms build, exact parameter counts
python qssm/efficiency/bench.py --cpu-only
```

## 2. Fused selective-scan kernel (optional, recommended for training)

Without `mamba-ssm`, the selective scan runs on a portable pure-PyTorch associative scan
(`qssm/models/qs_mamba.py`). With it, `qssm/models/ssm_ops.py` dispatches to the fused
CUDA kernel. Both backends use **the same weights**: a checkpoint moves between them
unchanged, and they agree to ~1e-7.

| Backend | Where it runs | Relative speed |
|---|---|---|
| `--ssm-kernel torch` | CPU or GPU | 1× |
| `--ssm-kernel fused` | CUDA only | ~28× faster at this model's shapes |
| `--ssm-kernel auto` (default) | fused if importable, else torch | — |

A3 runs 16 spatial scans per forward pass (4 directions × 4 blocks). Training it on the
pure-PyTorch path takes about three times as long, so build the kernel for any real
training run.

**Try a wheel first:**

```bash
pip install causal-conv1d mamba-ssm transformers
python qssm/models/ssm_ops.py --self-test
```

**If no wheel matches your torch/CUDA/GPU**, build from source with
[hpc/env/build_mamba.sh](hpc/env/build_mamba.sh):

```bash
CUDA_HOME=/path/to/cuda-toolkit TORCH_CUDA_ARCH_LIST=8.9 \
    bash hpc/env/build_mamba.sh causal-conv1d mamba-ssm
```

The script's header covers the usual failures: an `nvcc` that can't target the GPU, a
missing `Python.h`, a conda `gcc` shadowing the system one, and the hidden
`transformers` dependency. On a machine without root, a conda-forge `cuda-toolkit` works
as `CUDA_HOME`.

## 3. Neuroimaging tools (tractography and preprocessing only)

| Tool | Used by |
|---|---|
| [MRtrix3](https://www.mrtrix.org) ≥ 3.0 | `tractography/`, `hpc/preprocessing/` |
| [MRtrix3Tissue](https://3tissue.github.io) (`ss3t_csd_beta1`) | `tractography/ss3t_tractseg/`. Needs Python ≤ 3.11 (it imports the removed `imp` module) |
| [TractSeg](https://github.com/MIC-DKFZ/TractSeg) | `pip install TractSeg` |
| FSL (topup/eddy) and ANTs (N4) | `hpc/preprocessing/dwi_preprocess.py` |

The QSSM model code needs none of these.
