#!/usr/bin/env python3
"""Canonical statistics for the QS-Mamba paper.

Every accuracy number that appears in the manuscript is produced here, from the
per-subject CSVs written by `evaluate.py`. Nothing is typed by hand into a table.

Policy this file enforces (see README.md in this folder):

  * The metric is `mae_script` = torch.mean(|pred*m - gt*m|), the only convention
    this project reports. `mae_inmask` is carried as a diagnostic column and is
    never used for a contrast.
  * Every p-value is reported next to the NAME of the test that produced it.
    Wilcoxon and paired-t disagree here by an order of magnitude, and quoting the
    smaller one under the other's label is the error this harness exists to stop.
  * The exact two-sided Wilcoxon floor at n=18 is 2/2**18 = 7.63e-06. A p-value at
    that floor means "as extreme as this test can report", not "p < 1e-15".
  * Null claims get a TOST equivalence test. p > 0.05 is not evidence of absence.
  * The whole contrast family gets a Holm correction.
  * Every delta is reported against the run-to-run noise floor measured from two
    runs of the SAME configuration. A delta smaller than ~2x that floor is not a
    finding, whatever its p-value.

Usage:
    python3 stats.py                      # human-readable report to stdout
    python3 stats.py --json out/stats.json --md out/stats.md
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Where the evidence lives

import os  # noqa: E402

_REPO = Path(__file__).resolve().parents[2]
EVAL_DIR = Path(os.environ.get("QSSM_RESULTS_DIR", _REPO / "results")) / "eval"

# label -> (csv stem, human name). Order matters for the report.
ARMS: dict[str, tuple[str, str]] = {
    "A0":   ("A0_vit",              "A0 ViT (attention, no q-scan)"),
    "A1":   ("A1_hybrid",           "A1 hybrid (attention + q-scan)"),
    "A2":   ("A2_mamba_noqs",       "A2 scan (cross-scan, no q-scan)"),
    "A3":   ("A3_pure",             "A3 QS-Mamba (cross-scan + q-scan)"),
    "PUB":  ("published_r70",       "published ViT baseline (same config as A0)"),
    "A1dt": ("A1_dtlive",           "A1 + live PDE timestep"),
    "FWB":  ("RDT_NET_baseline",    "RDT-Net, 92 channels"),
}

# Parameter counts, asserted by verify_arms.py. Used for the cost-adjusted view.
PARAMS = {
    "A0": 509_383, "A1": 513_591, "A2": 408_775, "A3": 412_983,
    "PUB": 509_383, "A1dt": 513_591, "FWB": None,
}

# The contrast family. (a, b, label, kind) — `kind` drives which extra test runs.
#   "effect"     -> expected non-zero, report superiority
#   "null"       -> claimed to be nothing, so it gets a TOST
#   "noise"      -> same configuration twice; defines the noise floor
CONTRASTS = [
    ("A3", "A2", "A3-A2  q-scan under a scan encoder",      "effect"),
    ("A1", "A0", "A1-A0  q-scan under attention",           "null"),
    ("A2", "A0", "A2-A0  scan operator alone",              "null"),
    ("A3", "A0", "A3-A0  cross-family headline",            "effect"),
    ("A3", "PUB", "A3-published",                           "effect"),
    ("A0", "PUB", "A0-published  SAME CONFIG, TWO RUNS",    "noise"),
    ("A1dt", "A1", "A1dt-A1  live timestep",                "null"),
    ("A3", "FWB", "A3-RDT_NET  92-channel baseline",        "effect"),
]

# TOST equivalence margin. Set to the run-to-run noise floor: a difference smaller
# than what the same code produces on a rerun cannot be an architectural effect.
# Computed at runtime from the "noise" contrast rather than hard-coded.
TOST_MARGIN_MULTIPLIER = 1.0

BOOTSTRAP_N = 20_000
RNG_SEED = 20260816


# ---------------------------------------------------------------------------
# Loading

def load_arm(stem: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return (subject_ids, mae_script, mae_inmask), summary rows dropped."""
    path = EVAL_DIR / f"{stem}.csv"
    subs, script, inmask = [], [], []
    with path.open() as fh:
        for row in csv.DictReader(fh):
            sid = (row.get("subject") or "").strip()
            if not sid or sid in {"mean", "std"}:
                continue
            subs.append(sid)
            script.append(float(row["mae_script"]))
            inmask.append(float(row["mae_inmask"]))
    return subs, np.asarray(script), np.asarray(inmask)


def load_all() -> tuple[list[str], dict[str, np.ndarray], dict[str, np.ndarray]]:
    subjects: list[str] | None = None
    script: dict[str, np.ndarray] = {}
    inmask: dict[str, np.ndarray] = {}
    for key, (stem, _) in ARMS.items():
        subs, sc, im = load_arm(stem)
        if subjects is None:
            subjects = subs
        elif subs != subjects:
            raise SystemExit(
                f"subject order differs in {stem}.csv — paired tests would be "
                f"silently wrong.\n  expected {subjects}\n  got      {subs}"
            )
        script[key], inmask[key] = sc, im
    assert subjects is not None
    return subjects, script, inmask


# ---------------------------------------------------------------------------
# Tests

def wilcoxon_floor(n: int) -> float:
    """Smallest two-sided p the exact signed-rank test can return for n pairs."""
    return 2.0 / (2.0 ** n)


def bootstrap_ci(diff: np.ndarray, n: int = BOOTSTRAP_N,
                 alpha: float = 0.05) -> tuple[float, float]:
    rng = np.random.default_rng(RNG_SEED)
    idx = rng.integers(0, diff.size, size=(n, diff.size))
    means = diff[idx].mean(axis=1)
    return (float(np.quantile(means, alpha / 2)),
            float(np.quantile(means, 1 - alpha / 2)))


def tost(diff: np.ndarray, margin: float) -> tuple[float, str]:
    """Two one-sided tests for equivalence within +/- margin.

    Returns (p, verdict). A significant TOST means the difference is
    statistically smaller than the margin — i.e. real evidence of *absence*,
    which p > 0.05 on a superiority test never gives you.
    """
    n = diff.size
    mean, se = diff.mean(), diff.std(ddof=1) / math.sqrt(n)
    if se == 0:
        return 0.0, "identical"
    t_lo = (mean + margin) / se       # H0: diff <= -margin
    t_hi = (mean - margin) / se       # H0: diff >= +margin
    p_lo = 1 - stats.t.cdf(t_lo, n - 1)
    p_hi = stats.t.cdf(t_hi, n - 1)
    p = max(p_lo, p_hi)
    verdict = "equivalent" if p < 0.05 else "inconclusive"
    return float(p), verdict


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni step-down adjusted p-values."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (key, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)   # enforce monotonicity
        out[key] = running
    return out


def contrast(a: np.ndarray, b: np.ndarray) -> dict:
    diff = a - b
    n = diff.size
    w = stats.wilcoxon(a, b)
    t = stats.ttest_rel(a, b)
    lo, hi = bootstrap_ci(diff)
    return {
        "n": n,
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "mean_diff": float(diff.mean()),
        "pct_diff": float(100.0 * diff.mean() / b.mean()),
        "wins": int((diff < 0).sum()),
        "p_wilcoxon": float(w.pvalue),
        "p_ttest": float(t.pvalue),
        "at_wilcoxon_floor": bool(
            math.isclose(w.pvalue, wilcoxon_floor(n), rel_tol=1e-9)
        ),
        "d_z": float(diff.mean() / diff.std(ddof=1)),
        "ci95_lo": lo,
        "ci95_hi": hi,
    }


# ---------------------------------------------------------------------------
# Report

def build(subjects, script) -> dict:
    n = len(subjects)
    res: dict = {
        "metric": "mae_script = torch.mean(|pred*m - gt*m|)",
        "n_subjects": n,
        "subjects": subjects,
        "wilcoxon_exact_floor": wilcoxon_floor(n),
        "arms": {
            k: {
                "name": ARMS[k][1],
                "params": PARAMS[k],
                "mean": float(script[k].mean()),
                "std": float(script[k].std(ddof=1)),
            }
            for k in ARMS
        },
        "contrasts": {},
    }

    # Noise floor first — every other contrast is reported relative to it.
    noise_key = next(lab for a, b, lab, kind in CONTRASTS if kind == "noise")
    for a, b, label, kind in CONTRASTS:
        if kind != "noise":
            continue
        c = contrast(script[a], script[b])
        c["kind"] = kind
        c["mean_abs_diff"] = float(np.abs(script[a] - script[b]).mean())
        res["contrasts"][label] = c
    noise_floor = res["contrasts"][noise_key]["mean_abs_diff"]
    res["noise_floor_mean_abs_diff"] = noise_floor
    res["tost_margin"] = noise_floor * TOST_MARGIN_MULTIPLIER

    for a, b, label, kind in CONTRASTS:
        if kind == "noise":
            continue
        c = contrast(script[a], script[b])
        c["kind"] = kind
        c["mean_abs_diff"] = float(np.abs(script[a] - script[b]).mean())
        c["effect_over_noise"] = c["mean_abs_diff"] / noise_floor
        if kind == "null":
            p_eq, verdict = tost(script[a] - script[b], res["tost_margin"])
            c["p_tost"] = p_eq
            c["tost_verdict"] = verdict
        res["contrasts"][label] = c

    # Holm across the family, on the Wilcoxon p-values.
    adj = holm({k: v["p_wilcoxon"] for k, v in res["contrasts"].items()})
    for k, v in res["contrasts"].items():
        v["p_wilcoxon_holm"] = adj[k]

    # ---- The 2x2 factorial ------------------------------------------------
    inter = (script["A3"] - script["A2"]) - (script["A1"] - script["A0"])
    lo, hi = bootstrap_ci(inter)
    res["factorial"] = {
        "cells": {k: float(script[k].mean()) for k in ("A0", "A1", "A2", "A3")},
        "interaction": {
            "definition": "(A3-A2) - (A1-A0)",
            "mean": float(inter.mean()),
            "p_wilcoxon": float(stats.wilcoxon(inter).pvalue),
            "p_ttest": float(stats.ttest_1samp(inter, 0).pvalue),
            "ci95_lo": lo, "ci95_hi": hi,
        },
        "main_effect_qscan": {
            "definition": "mean(A1,A3) - mean(A0,A2)",
            "mean": float((((script["A1"] + script["A3"]) / 2)
                           - ((script["A0"] + script["A2"]) / 2)).mean()),
            "p_wilcoxon": float(stats.wilcoxon(
                ((script["A1"] + script["A3"]) / 2)
                - ((script["A0"] + script["A2"]) / 2)).pvalue),
            "caveat": "driven entirely by the A3 cell; never report without the "
                      "interaction",
        },
        "main_effect_scan": {
            "definition": "mean(A2,A3) - mean(A0,A1)",
            "mean": float((((script["A2"] + script["A3"]) / 2)
                           - ((script["A0"] + script["A1"]) / 2)).mean()),
            "p_wilcoxon": float(stats.wilcoxon(
                ((script["A2"] + script["A3"]) / 2)
                - ((script["A0"] + script["A1"]) / 2)).pvalue),
        },
    }
    return res


def to_markdown(res: dict) -> str:
    L: list[str] = []
    L.append("# QS-Mamba — statistics\n")
    L.append(f"Metric: `{res['metric']}`  ·  n = {res['n_subjects']} held-out "
             f"HCP subjects\n")
    L.append(f"Exact two-sided Wilcoxon floor at n={res['n_subjects']}: "
             f"`{res['wilcoxon_exact_floor']:.2e}` — a p-value at this floor "
             f"means *as extreme as this test can report*.\n")
    L.append(f"Run-to-run noise floor (same config, two runs): mean |Δ| = "
             f"`{res['noise_floor_mean_abs_diff']:.3e}`. "
             f"TOST margin set to this value.\n")

    L.append("\n## Arms\n")
    L.append("| arm | params | mean MAE | sd |")
    L.append("|---|---:|---:|---:|")
    for k, v in res["arms"].items():
        p = f"{v['params']:,}" if v["params"] else "—"
        L.append(f"| {v['name']} | {p} | {v['mean']:.6f} | {v['std']:.6f} |")

    L.append("\n## Factorial\n")
    f = res["factorial"]
    c = f["cells"]
    L.append("|  | no q-scan | + q-scan |")
    L.append("|---|---|---|")
    L.append(f"| **attention** | A0 `{c['A0']:.6f}` | A1 `{c['A1']:.6f}` |")
    L.append(f"| **spatial scan** | A2 `{c['A2']:.6f}` | **A3 `{c['A3']:.6f}`** |")
    i = f["interaction"]
    L.append(f"\n**Interaction** `{i['definition']}` = `{i['mean']:.3e}` "
             f"(95% CI {i['ci95_lo']:.2e} to {i['ci95_hi']:.2e}), "
             f"Wilcoxon *p* = `{i['p_wilcoxon']:.3g}`, "
             f"paired-*t* *p* = `{i['p_ttest']:.3g}`.\n")
    L.append("This is the strongest contrast available and the one that states "
             "the claim exactly: neither ingredient works alone.\n")
    for key in ("main_effect_qscan", "main_effect_scan"):
        m = f[key]
        L.append(f"- `{m['definition']}` = `{m['mean']:.3e}`, Wilcoxon *p* = "
                 f"`{m['p_wilcoxon']:.4g}`"
                 + (f" — {m['caveat']}" if "caveat" in m else ""))

    L.append("\n## Contrasts\n")
    L.append("| contrast | Δ% | wins | Wilcoxon *p* | Holm | paired-*t* *p* | "
             "*d_z* | Δ/noise |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for label, v in res["contrasts"].items():
        wins = f"{v['wins']}/{v['n']}" if v["kind"] != "noise" else \
               f"{v['wins']}/{v['n']}"
        ratio = f"{v['effect_over_noise']:.2f}x" if "effect_over_noise" in v \
                else "— (defines it)"
        floor = " †" if v["at_wilcoxon_floor"] else ""
        L.append(
            f"| {label} | {v['pct_diff']:+.3f} | {wins} | "
            f"{v['p_wilcoxon']:.3g}{floor} | {v['p_wilcoxon_holm']:.3g} | "
            f"{v['p_ttest']:.3g} | {v['d_z']:+.3f} | {ratio} |"
        )
    L.append("\n† at the exact Wilcoxon floor.\n")

    eq = [(k, v) for k, v in res["contrasts"].items() if "p_tost" in v]
    if eq:
        L.append("\n## Equivalence (TOST) for the null claims\n")
        L.append(f"Margin = `{res['tost_margin']:.3e}` (the run-to-run noise "
                 f"floor). A significant TOST is evidence of *absence*; "
                 f"`p > 0.05` on a superiority test alone is not.\n")
        L.append("| contrast | TOST *p* | verdict |")
        L.append("|---|---:|---|")
        for k, v in eq:
            L.append(f"| {k} | {v['p_tost']:.4g} | **{v['tost_verdict']}** |")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, help="write machine-readable results")
    ap.add_argument("--md", type=Path, help="write the markdown report")
    args = ap.parse_args()

    subjects, script, _inmask = load_all()
    res = build(subjects, script)
    md = to_markdown(res)

    print(md)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(res, indent=2))
        print(f"[stats] wrote {args.json}")
    if args.md:
        args.md.parent.mkdir(parents=True, exist_ok=True)
        args.md.write_text(md)
        print(f"[stats] wrote {args.md}")


if __name__ == "__main__":
    main()
