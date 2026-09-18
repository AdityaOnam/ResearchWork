#!/usr/bin/env bash
# Build causal-conv1d + mamba-ssm from source against the torch already installed in
# the active Python. Use this when no prebuilt wheel matches your torch/CUDA/GPU.
#
#   CUDA_HOME=/path/to/cuda  TORCH_CUDA_ARCH_LIST=8.9  bash hpc/env/build_mamba.sh causal-conv1d mamba-ssm
#
# Gotchas this handles:
#   1. nvcc must be able to target your GPU (e.g. sm_89 needs CUDA >= 11.8) and should
#      match torch.version.cuda. Without root, a conda-forge toolkit works:
#          micromamba create -p ~/envs/cuda -c conda-forge --override-channels cuda-toolkit=<ver>
#   2. conda-forge puts headers under targets/x86_64-linux/include -> added to CPATH.
#   3. If the interpreter has no Python.h (no pythonX.Y-dev, no sudo), point PY_INCLUDE
#      at any matching-version conda python's include/pythonX.Y directory.
#   4. A conda gcc earlier on PATH can break nvcc; /usr/bin is put first.
#   5. mamba_ssm/__init__ imports MambaLMHeadModel, which needs `transformers`.
set -euo pipefail

: "${CUDA_HOME:?set CUDA_HOME to a CUDA toolkit whose nvcc targets your GPU}"
PY="${PY:-python}"
export PATH="$CUDA_HOME/bin:/usr/bin:$PATH"
export CPATH="$CUDA_HOME/targets/x86_64-linux/include${PY_INCLUDE:+:$PY_INCLUDE}${CPATH:+:$CPATH}"
export LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/targets/x86_64-linux/lib/stubs${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CC=/usr/bin/gcc CXX=/usr/bin/g++ CUDAHOSTCXX=/usr/bin/g++
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"   # one arch keeps the build short
export MAX_JOBS="${MAX_JOBS:-12}"
export CAUSAL_CONV1D_FORCE_BUILD=TRUE MAMBA_FORCE_BUILD=TRUE

echo "gcc   $(gcc --version | head -1)"
echo "nvcc  $(nvcc --version | tail -2 | head -1)"
echo "torch $($PY -c 'import torch;print(torch.__version__, torch.version.cuda)')"
echo "arch  $TORCH_CUDA_ARCH_LIST   jobs $MAX_JOBS"

$PY -m pip install transformers
for pkg in "$@"; do
  echo "=== building $pkg ==="
  $PY -m pip install --no-deps --no-build-isolation --no-cache-dir "$pkg"
done

# Verify: the fused kernel must agree with the pure-PyTorch fallback
$PY qssm/models/ssm_ops.py --self-test
