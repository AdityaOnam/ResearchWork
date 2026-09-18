#!/usr/bin/env python3
"""Cost benchmark for the QS-Mamba arms — the paper's headline efficiency table.

Measures, per arm: parameter counts (total / encoder / decoder / per-module),
FLOPs, peak TRAINING memory, peak INFERENCE memory, per-slice latency, and
per-subject inference time.

Policy this file enforces (see README.md in this folder):

  * **No cost number is quoted from a contended GPU.** Timings taken while other
    jobs share the device are not comparable. This script refuses to take GPU timings while other processes hold the GPU,
    unless you pass --force (which stamps the output as UNRELIABLE).
  * Parameter and FLOP counts are deterministic and device-independent, so they
    run anywhere, any time — use --cpu-only for those alone.
  * Memory is reported as torch.cuda.max_memory_allocated around a clean reset,
    with cudnn.benchmark OFF, because autotuning a shape used once reserved a
    6.2 GB workspace that then stood as the process high-water mark for a whole
    run and got logged as if it were a real requirement.

Usage:
    python3 bench.py --cpu-only --json out/bench_static.json   # anytime
    python3 bench.py --json out/bench.json --md out/bench.md   # sole occupancy
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

# --- wire in the model tree -------------------------------------------------
MODELS = Path(__file__).resolve().parents[1] / "models"
sys.path.insert(0, str(MODELS))

from rdt_net import ConvGlobalSkipModel  # noqa: E402

IN_CH, H, W = 91, 174, 145
K = 5
AXIAL_SLICES_PER_SUBJECT = 145   # volumes are 145 x 174 x 145 x 91

ARMS: dict[str, tuple[str, dict]] = {
    "A0": ("vit", {}),
    "A1": ("hybrid", {}),
    "A2": ("mamba", dict(use_qscan=False)),
    "A3": ("mamba", {}),
}
ARM_NAMES = {
    "A0": "A0 ViT (attention)",
    "A1": "A1 hybrid (attention + q-scan)",
    "A2": "A2 scan (cross-scan)",
    "A3": "A3 QS-Mamba (cross-scan + q-scan)",
}
EXPECTED_PARAMS = {"A0": 509_383, "A1": 513_591, "A2": 408_775, "A3": 412_983}


# ---------------------------------------------------------------------------
# Occupancy guard

def gpu_occupancy() -> list[dict]:
    """Other processes currently holding the GPU."""
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-compute-apps=pid,used_memory,process_name",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20,
        ).stdout.strip()
    except Exception:
        return []
    procs = []
    me = os.getpid()
    for line in out.splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        try:
            pid = int(parts[0])
        except (ValueError, IndexError):
            continue
        if pid == me:
            continue
        procs.append({"pid": pid,
                      "used_mib": parts[1] if len(parts) > 1 else "?",
                      "name": parts[2] if len(parts) > 2 else "?"})
    return procs


# ---------------------------------------------------------------------------
# Static cost — deterministic, device-independent

def build(arm: str, dt_live: bool = False) -> torch.nn.Module:
    kind, kw = ARMS[arm]
    return ConvGlobalSkipModel(in_channels=IN_CH, K=K, encoder=kind,
                               enc_kw=dict(kw), dt_live=dt_live)


def count(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def static_costs(arm: str) -> dict:
    model = build(arm)
    total = count(model)

    # The encoder is `in_vit`; everything else is the shared decoder stack.
    encoder = getattr(model, "in_vit", None)
    enc = count(encoder) if encoder is not None else 0

    per_module = {}
    for name, child in model.named_children():
        per_module[name] = count(child)

    # The q-space branch, when present.
    qbranch = 0
    if encoder is not None:
        for n, sub in encoder.named_children():
            if n in {"qscan", "q_merge"}:
                qbranch += count(sub)

    # pos_emb is created lazily inside SimpleViT.forward(), so it does not
    # appear in named_parameters() until one forward has run. Materialise it.
    with torch.no_grad():
        model.eval()
        _ = model(torch.zeros(1, IN_CH, H, W))
    total_after = count(model)
    pos_emb = total_after - total

    res = {
        "arm": arm,
        "name": ARM_NAMES[arm],
        "params_total": total_after,
        "params_encoder": count(encoder) if encoder is not None else 0,
        "params_qbranch": qbranch,
        "params_pos_emb": pos_emb,
        "params_by_module": {k: count(getattr(model, k))
                             for k in per_module},
        "params_expected": EXPECTED_PARAMS[arm],
        "params_match_expected": total_after == EXPECTED_PARAMS[arm],
    }
    del model
    return res


def flops(arm: str) -> int | None:
    """Forward FLOPs at batch 1. Returns None if the counter is unavailable."""
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except Exception:
        return None
    model = build(arm).eval()
    x = torch.zeros(1, IN_CH, H, W)
    with torch.no_grad():
        _ = model(x)                      # materialise pos_emb first
        counter = FlopCounterMode(display=False)
        try:
            with counter:
                _ = model(x)
            return int(counter.get_total_flops())
        except Exception:
            return None
        finally:
            del model


# ---------------------------------------------------------------------------
# Dynamic cost — needs a quiet GPU

def gpu_costs(arm: str, device: torch.device, train_batch: int = 2,
              warmup: int = 5, iters: int = 30) -> dict:
    torch.backends.cudnn.benchmark = False    # see module docstring
    model = build(arm).to(device)

    # --- inference: peak memory + latency, batch 1, no grad ---------------
    model.eval()
    x1 = torch.zeros(1, IN_CH, H, W, device=device)
    with torch.no_grad():
        _ = model(x1)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x1)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = model(x1)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
    infer_peak = torch.cuda.max_memory_allocated(device)
    per_slice_s = (t1 - t0) / iters

    # --- training: peak memory + step time, batch 2, fwd+bwd -------------
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    xb = torch.zeros(train_batch, IN_CH, H, W, device=device)
    yb = torch.zeros(train_batch, 1, H, W, device=device)
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        loss = (model(xb) - yb).abs().mean()
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        opt.zero_grad(set_to_none=True)
        loss = (model(xb) - yb).abs().mean()
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    train_peak = torch.cuda.max_memory_allocated(device)
    per_step_s = (t1 - t0) / iters

    del model, opt, xb, yb, x1
    torch.cuda.empty_cache()

    return {
        "infer_peak_bytes": int(infer_peak),
        "infer_peak_gb": infer_peak / 2**30,
        "train_peak_bytes": int(train_peak),
        "train_peak_gb": train_peak / 2**30,
        "latency_per_slice_ms": per_slice_s * 1e3,
        "slices_per_sec": 1.0 / per_slice_s,
        "latency_per_subject_s": per_slice_s * AXIAL_SLICES_PER_SUBJECT,
        "train_step_ms": per_step_s * 1e3,
        "train_batch": train_batch,
        "iters": iters,
    }


# ---------------------------------------------------------------------------

def to_markdown(res: dict) -> str:
    L = ["# QS-Mamba — cost benchmark\n"]
    if res["gpu"]["measured"]:
        rel = "UNRELIABLE — GPU was contended" if res["gpu"]["contended"] \
              else "sole occupancy"
        L.append(f"Device: `{res['gpu']['name']}` · torch {res['torch']} · "
                 f"**{rel}**\n")
    else:
        L.append(f"Static costs only (no GPU measurement). torch "
                 f"{res['torch']}\n")
        if res["gpu"]["contended"]:
            L.append(f"GPU held by {len(res['gpu']['procs'])} other process(es); "
                     f"timings deliberately skipped.\n")

    L.append("\n## Size\n")
    L.append("| arm | total params | encoder | q-branch | pos_emb | vs A0 |")
    L.append("|---|---:|---:|---:|---:|---:|")
    a0 = res["arms"]["A0"]["params_total"]
    for k in ARMS:
        a = res["arms"][k]
        L.append(f"| {a['name']} | {a['params_total']:,} | "
                 f"{a['params_encoder']:,} | {a['params_qbranch']:,} | "
                 f"{a['params_pos_emb']:,} | "
                 f"{100*(a['params_total']-a0)/a0:+.1f}% |")

    if any(res["arms"][k].get("flops") for k in ARMS):
        L.append("\n## Compute\n")
        L.append("| arm | forward GFLOPs (batch 1) | vs A0 |")
        L.append("|---|---:|---:|")
        f0 = res["arms"]["A0"].get("flops")
        for k in ARMS:
            f = res["arms"][k].get("flops")
            if not f:
                continue
            rel = f"{100*(f-f0)/f0:+.1f}%" if f0 else "—"
            L.append(f"| {res['arms'][k]['name']} | {f/1e9:.2f} | {rel} |")

    if res["gpu"]["measured"]:
        L.append("\n## Memory and latency\n")
        L.append("| arm | train peak | vs A0 | infer peak | vs A0 | "
                 "ms/slice | s/subject | ms/train step |")
        L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        t0 = res["arms"]["A0"]["train_peak_gb"]
        i0 = res["arms"]["A0"]["infer_peak_gb"]
        for k in ARMS:
            a = res["arms"][k]
            L.append(
                f"| {a['name']} | {a['train_peak_gb']:.2f} GB | "
                f"{100*(a['train_peak_gb']-t0)/t0:+.1f}% | "
                f"{a['infer_peak_gb']:.3f} GB | "
                f"{100*(a['infer_peak_gb']-i0)/i0:+.1f}% | "
                f"{a['latency_per_slice_ms']:.2f} | "
                f"{a['latency_per_subject_s']:.2f} | "
                f"{a['train_step_ms']:.1f} |"
            )
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cpu-only", action="store_true",
                    help="static costs only; never touch the GPU")
    ap.add_argument("--force", action="store_true",
                    help="measure on the GPU even if contended (marks output "
                         "UNRELIABLE)")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--md", type=Path)
    args = ap.parse_args()

    # Public release: arms whose encoder is withheld cannot be built; benchmark the rest.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from qssm._withheld import WithheldError
    for arm in list(ARMS):
        try:
            build(arm)
        except WithheldError:
            print(f"[bench] {arm}: encoder withheld in the public release (see WITHHELD.md); "
                  f"skipped", file=sys.stderr)
            del ARMS[arm]

    procs = gpu_occupancy()
    contended = len(procs) > 0
    want_gpu = (not args.cpu_only) and torch.cuda.is_available()
    measure = want_gpu and (not contended or args.force)

    res: dict = {
        "torch": torch.__version__,
        "gpu": {
            "available": torch.cuda.is_available(),
            "name": torch.cuda.get_device_name(0)
            if torch.cuda.is_available() else None,
            "contended": contended,
            "procs": procs,
            "measured": measure,
        },
        "axial_slices_per_subject": AXIAL_SLICES_PER_SUBJECT,
        "arms": {},
    }

    if contended and want_gpu and not args.force:
        print(f"[bench] GPU held by {len(procs)} other process(es). "
              f"Skipping all timing and memory measurement.", file=sys.stderr)
        for p in procs:
            print(f"        pid {p['pid']:>7}  {p['used_mib']:>7} MiB  "
                  f"{p['name']}", file=sys.stderr)
        print("[bench] Re-run under sole occupancy, or pass --force to take "
              "numbers you must not publish.", file=sys.stderr)

    for arm in ARMS:
        print(f"[bench] {arm} …", file=sys.stderr)
        d = static_costs(arm)
        f = flops(arm)
        if f:
            d["flops"] = f
        if measure:
            d.update(gpu_costs(arm, torch.device("cuda"), iters=args.iters))
        res["arms"][arm] = d

    bad = [k for k, v in res["arms"].items() if not v["params_match_expected"]]
    if bad:
        print(f"[bench] WARNING: parameter counts differ from verify_arms.py "
              f"for {bad}", file=sys.stderr)

    md = to_markdown(res)
    print(md)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(res, indent=2))
        print(f"[bench] wrote {args.json}", file=sys.stderr)
    if args.md:
        args.md.parent.mkdir(parents=True, exist_ok=True)
        args.md.write_text(md)
        print(f"[bench] wrote {args.md}", file=sys.stderr)


if __name__ == "__main__":
    main()
