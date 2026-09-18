# `qssm/stats/`: statistics

> 🔒 = withheld until publication (placeholder with the real interface; see [WITHHELD.md](../../WITHHELD.md)).

Every accuracy number that goes into a table is produced here, from the per-subject CSVs
written by `evaluate/`. Nothing is typed into a table by hand.

| Script | Produces |
|---|---|
| `stats.py` | The main contrast family over the four arms: paired tests, effect sizes, 2×2 interaction, equivalence tests, Holm correction, noise-floor comparison |
| `fa_md_stats.py` 🔒 | Downstream tables (FW MAE, ΔFA, ΔMD) for the test and external cohorts, with a "not significantly worse than best" bolding rule |

```bash
python stats.py                                   # report to stdout
python stats.py --json out/stats.json --md out/stats.md
python fa_md_stats.py --json out/fa_md.json
```

## Protocol

| Rule | Why |
|---|---|
| **Paired Wilcoxon signed-rank**, exact, two-sided, per subject | Few subjects, non-normal per-subject errors, and the same subjects under every arm |
| Paired t-test and bootstrap CI reported alongside, **always labelled** | The two can differ by an order of magnitude; a p-value is never quoted without the name of its test |
| **Exact Wilcoxon floor**: at n = 18 the smallest possible two-sided p is 2/2¹⁸ = 7.63 × 10⁻⁶ | A p at the floor means "as extreme as this test can report", nothing more |
| **2×2 interaction**: (A3 − A2) − (A1 − A0) per subject | Tests whether the q-space effect depends on the spatial operator |
| **TOST equivalence** for every "no difference" claim | p > 0.05 is not evidence of absence |
| **Holm correction** over the whole contrast family | Several contrasts are tested at once |
| **Noise floor**: two runs of the *same* configuration | A difference smaller than about 2× the floor isn't an architectural effect, whatever its p-value |

## Contrasts

`A3−A2` (q-scan under scan) · `A1−A0` (q-scan under attention) · `A2−A0` (spatial
operator) · `A3−A0` (headline) · `A0−A0′` (same configuration twice: the noise floor) ·
`A1dt−A1` (live PDE step) · `A3−RDT-Net` (92-channel baseline).

Input CSVs are read from `$QSSM_RESULTS_DIR/eval/` (written by `evaluate/evaluate.py`).
