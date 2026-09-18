# Contributing and maintenance

This repository is maintained by the QSSM authors at IIT Patna. External contributions
are welcome through issues and pull requests once the project is public. Until then,
please read [CONFIDENTIALITY.md](CONFIDENTIALITY.md) first.

## Before opening a pull request

```bash
python scripts/setup_check.py          # environment + data-free model test
python -m compileall -q .              # every script byte-compiles
python qssm/efficiency/bench.py --cpu-only
```

CI runs the same checks on every push and pull request (`.github/workflows/ci.yml`).

## Rules of the repository

1. **No data, weights or results in git.** `.gitignore` blocks NIfTI, MRtrix, tractogram
   and checkpoint files, and CI fails if any are tracked. Share those through the lab's
   storage, never through the repository.
2. **No hard-coded paths.** Resolve every location through `qssm/paths.py` (environment
   variables) or a script argument. Never commit machine names, usernames or cluster paths.
3. **No subject identifiers.** Subject lists live in `splits/*.txt`, which are git-ignored.
   Clinical-cohort IDs must never appear in code, comments, logs or issues.
4. **No unpublished numbers.** Code comments and READMEs describe *what* is measured, not
   the values obtained. Results belong in the manuscript.
5. **Withheld files stay placeholders** until the release is approved (see `WITHHELD.md`).
6. **Every folder keeps a README** that says what the folder does, how to run it and what it
   expects. Update it in the same pull request as the code.
7. **Record user-visible changes** in `CHANGELOG.md`.

## Style

- Python ≥ 3.10, 4-space indentation, lines ≤ 100 characters.
- Every script starts with a docstring stating its purpose and a usage example.
- Command-line scripts use `argparse` and print a one-line summary on completion.
- Shell scripts use `set -euo pipefail` and must pass `bash -n`.

## Reporting a problem

Open an issue with: the command you ran, the full error, the output of
`python scripts/setup_check.py`, and your OS / GPU / CUDA version. **Do not attach
imaging data or subject identifiers.**
