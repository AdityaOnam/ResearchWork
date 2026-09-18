# `qssm/evaluate/`: evaluation

> 🔒 = withheld until publication (placeholder with the real interface; see [WITHHELD.md](../../WITHHELD.md)).

All four scripts use `evaluate.py`'s loader, preprocessing, masking and scoring. A number
from any of them comes from the same code path as the clean test-set number.

| Script | Question | Output (under `$QSSM_RESULTS_DIR/eval/`) |
|---|---|---|
| `evaluate.py` | Test-set free-water MAE per subject, per arm | `<label>.csv`, `summary.csv`, maps with `--save-maps` |
| `evaluate_snr.py` | How does error grow as SNR falls (60 → 10)? | `snr_sweep.csv` |
| `evaluate_directions.py` 🔒 | How does error grow with fewer gradient directions (91 → 46 → 30 → 15)? | `direction_sweep.csv` |
| `evaluate_fa_md.py` 🔒 | After free-water correction, how accurate are FA and MD? | `$QSSM_RESULTS_DIR/fa_md/*.csv` |

```bash
python evaluate.py --all                                  # every arm checkpoint found in $QSSM_CKPT_DIR
python evaluate.py --checkpoint $QSSM_CKPT_DIR/QSM_...pth --label A3_pure --save-maps
python evaluate_snr.py --all                              # or: --model A3_pure --snr 30 20 10 --subjects 4
python evaluate_directions.py --all                       # or: --model A3_pure --keep 45 29
python evaluate_fa_md.py --cohort hcp                     # needs --save-maps output first
python evaluate_fa_md.py --cohort btc                     # optional external cohort (splits/external.txt)
```

## Details

- **Architecture is read from the checkpoint.** The keys identify the encoder
  (`scan_blocks.*` means scan, `qscan.*` means q-branch), and 91- vs 92-channel inputs are
  handled by one code path. The exception is **`--dt-live`**, which can't be inferred from
  the weights (Softplus has no parameters), so pass it explicitly.
- **Masked input by default**, matching training. `--no-mask-input` exists only for
  checkpoints trained unmasked.
- **Ground truth** is the DIPY multi-shell bi-tensor free-water map. The MAE convention is
  described in [../README.md](../README.md#conventions-used-throughout).
- **Direction subsampling** keeps b0 plus an angularly uniform (linspace) subset of the
  b = 1000 directions and zero-pads the rest. The input shape stays fixed, so no retraining
  is needed.
- **FA/MD** refits a tensor to the free-water-corrected signal (`--fit wls`, or `nls` for
  the slower signal-space fit) and compares with the multi-shell reference. It also
  reports a floor: the error you would get with a *perfect* free-water map.
