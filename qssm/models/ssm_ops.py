#!/usr/bin/env python3
"""
ssm_ops — one selective-scan op, two backends, one set of weights.

The dispatch is at the *operation* level, not the module level: `S6` keeps its exact
parameterisation (`in_proj / conv1d / x_proj / dt_proj / out_proj / A_log / D`) and only the
inner computation swaps. A checkpoint trained with the fused CUDA kernel therefore loads
bit-for-bit into the pure-PyTorch model and back, which is what lets a run be reproduced on
a machine without `nvcc` — including by a reviewer.

Swapping in `mamba_ssm.modules.mamba_simple.Mamba` instead would have been less code and
would have thrown that property away.

Backends
--------
`fused`  `mamba_ssm.ops.selective_scan_interface.selective_scan_fn` + `causal_conv1d_fn`.
         Build instructions: INSTALL.md and hpc/env/build_mamba.sh. Agrees with the
         pure path to ~1e-7 and is substantially faster.

`torch`  The log-depth associative scan in `qs_mamba.py`. Portable, differentiable,
         validated against the kernel, but much slower for the scan arms.

`auto` picks fused when the import succeeded and the tensors are CUDA floats.

Three things about the fused call that silently corrupt results if got wrong, all of them
learned the hard way and all of them asserted by `--self-test`:

  1. `delta` must be passed **pre-softplus and without the bias**. The kernel applies both
     internally via `delta_bias=` and `delta_softplus=True`. Passing
     `F.softplus(dt_proj(dt))` type-checks, runs, and gives a wrong answer.
  2. `z` is fused into the kernel, so the gating `y * F.silu(z)` must be *dropped* on the
     fused path rather than applied afterwards.
  3. `b`, `c` and `z` must be `.contiguous()` after their transposes; the kernel rejects
     the strided views without a useful message.

    python ssm_ops.py --self-test
"""

from __future__ import annotations

import os

import torch

__all__ = ["HAVE_FUSED", "FUSED_IMPORT_ERROR", "resolve_kernel", "s6_forward", "self_test"]

try:
    from causal_conv1d import causal_conv1d_fn
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    HAVE_FUSED = True
    FUSED_IMPORT_ERROR = None
except Exception as exc:                                  # pragma: no cover - env dependent
    causal_conv1d_fn = selective_scan_fn = None
    HAVE_FUSED = False
    FUSED_IMPORT_ERROR = exc


def resolve_kernel(kernel: str, ref: torch.Tensor) -> str:
    """Map `auto|fused|torch` to the backend actually used.

    `fused` that cannot be honoured is a hard error, never a silent downgrade: a run that
    quietly takes three times longer than budgeted is worse than one that fails at second 1.
    """
    if kernel not in ("auto", "fused", "torch"):
        raise ValueError(f"unknown kernel {kernel!r}; expected auto, fused or torch")
    ok = (HAVE_FUSED and ref.is_cuda
          and ref.dtype in (torch.float32, torch.float16, torch.bfloat16))
    if kernel == "fused":
        if not HAVE_FUSED:
            raise RuntimeError(
                f"--ssm-kernel fused but mamba_ssm/causal_conv1d did not import: "
                f"{FUSED_IMPORT_ERROR!r}. Build them with ~/envs/build_mamba.sh, or pass "
                f"--ssm-kernel torch.")
        if not ok:
            raise RuntimeError(
                f"--ssm-kernel fused requires CUDA float tensors; got device "
                f"{ref.device}, dtype {ref.dtype}.")
        return "fused"
    return "fused" if (kernel == "auto" and ok) else "torch"


def s6_forward(mod, x, kernel="auto"):
    """`S6.forward` for both backends. `x: (B, L, D)` -> `(B, L, D)`.

    `mod` is the `S6` module; every parameter is read off it, so the two paths cannot drift
    into different parameterisations.
    """
    which = resolve_kernel(kernel, x)
    l = x.shape[1]
    xs, z = mod.in_proj(x).chunk(2, dim=-1)
    xs = xs.transpose(1, 2)                                    # (B, d_inner, L)

    if which == "torch":
        from qs_mamba import selective_scan                    # local, avoids a cycle
        xs = torch.nn.functional.silu(mod.conv1d(xs)[:, :, :l])
        proj = mod.x_proj(xs.transpose(1, 2))
        dt, b, c = torch.split(proj, [mod.dt_rank, mod.d_state, mod.d_state], dim=-1)
        dt = torch.nn.functional.softplus(mod.dt_proj(dt)).transpose(1, 2)
        y = selective_scan(xs, dt, mod.A_log, b.transpose(1, 2), c.transpose(1, 2),
                           mod.D, chunk=mod.chunk, checkpoint=mod.checkpoint)
        y = y.transpose(1, 2) * torch.nn.functional.silu(z)
        return mod.out_proj(y)

    # -- fused
    xs = causal_conv1d_fn(xs.contiguous(), mod.conv1d.weight.squeeze(1),
                          mod.conv1d.bias, activation="silu")
    proj = mod.x_proj(xs.transpose(1, 2))
    dt, b, c = torch.split(proj, [mod.dt_rank, mod.d_state, mod.d_state], dim=-1)
    # Pre-softplus, bias excluded -- the kernel applies dt_proj.bias and softplus itself.
    dt = (mod.dt_proj.weight @ dt.transpose(1, 2)).contiguous()
    y = selective_scan_fn(
        xs, dt, -torch.exp(mod.A_log.float()),
        b.transpose(1, 2).contiguous(), c.transpose(1, 2).contiguous(),
        mod.D.float(), z=z.transpose(1, 2).contiguous(),
        delta_bias=mod.dt_proj.bias.float(), delta_softplus=True)
    # z was gated inside the kernel; do NOT apply silu(z) again here.
    return mod.out_proj(y.transpose(1, 2))


def self_test(tol=1e-5, verbose=True):
    """Assert the two backends agree at the shapes this project actually runs.

    Cheap insurance against a future `mamba-ssm` upgrade changing argument semantics --
    every failure mode in the module docstring is a silent wrong answer, not a crash.
    """
    from qs_mamba import S6

    if not HAVE_FUSED:
        raise RuntimeError(f"fused kernels unavailable: {FUSED_IMPORT_ERROR!r}")
    if not torch.cuda.is_available():
        raise RuntimeError("self-test needs a GPU")

    dev = torch.device("cuda")
    worst = 0.0
    # (d_model, L, B): the spatial scan (embed_dim 64 over the 43x36 patch grid) and the
    # q-space scan (d_model 16 over 91 directions, one sequence per coarse position).
    for name, d_model, l, b in (("spatial", 64, 1548, 2), ("q-space", 16, 91, 836)):
        torch.manual_seed(0)
        mod = S6(d_model, d_state=16 if d_model == 64 else 8, expand=2).to(dev).eval()
        x = torch.randn(b, l, d_model, device=dev)
        with torch.no_grad():
            diff = (s6_forward(mod, x, "torch") - s6_forward(mod, x, "fused")).abs().max()
        worst = max(worst, float(diff))
        if verbose:
            print(f"  {name:<8} d_model {d_model:>3}  L {l:>4}  B {b:>3}   "
                  f"max|diff| {float(diff):.3e}   {'ok' if diff < tol else 'FAIL'}")
    if worst >= tol:
        raise AssertionError(f"fused/pure disagreement {worst:.3e} exceeds {tol:g}")
    return worst


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--bench", action="store_true")
    args = ap.parse_args()

    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    print(f"fused available: {HAVE_FUSED}"
          + ("" if HAVE_FUSED else f"  ({FUSED_IMPORT_ERROR!r})"))

    if args.self_test:
        print("self-test:")
        print(f"  worst {self_test():.3e}  -- PASS")

    if args.bench:
        import time
        from qs_mamba import S6
        dev = torch.device("cuda")
        mod = S6(64, d_state=16, expand=2).to(dev).eval()
        x = torch.randn(2, 1548, 64, device=dev)

        def bench(k, n=20):
            with torch.no_grad():
                for _ in range(3):
                    s6_forward(mod, x, k)
                torch.cuda.synchronize()
                t = time.time()
                for _ in range(n):
                    s6_forward(mod, x, k)
                torch.cuda.synchronize()
            return (time.time() - t) / n * 1e3

        print(f"  torch  {bench('torch'):6.2f} ms")
        print(f"  fused  {bench('fused'):6.2f} ms")
