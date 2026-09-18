# Subject splits

The scripts read subject IDs from plain-text files in this folder. You can point
`QSSM_SPLITS_DIR` at another folder instead.

| File | Used by |
|---|---|
| `train.txt` | `qssm/train/train_mamba.py`, `qssm/models/qspace.py` (canonical q-space order) |
| `test.txt` | every script in `qssm/evaluate/`, `qssm/ablations/`, `tractography/` |
| `external.txt` | `qssm/evaluate/evaluate_fa_md.py --cohort btc` |
| `all.txt` | `hpc/slurm/preprocess_array.sbatch` |

Format: one ID per line, and `#` starts a comment. See [example.txt](example.txt).

**Subject lists are git-ignored on purpose.** Keep them local, and do not commit
identifiers of clinical subjects.

The setup these scripts were developed on: 18 HCP training subjects (3 held out for
validation) and 18 HCP test subjects.
