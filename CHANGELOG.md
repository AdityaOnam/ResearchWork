# Changelog

All notable changes to this repository are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Planned
- Release the withheld QSSM implementation and analyses on publication (see `WITHHELD.md`).
- Finalize `CITATION.cff` (authors, DOI, repository URL).

## [0.1.0] — 2026-09-18

### Added
- Public repository structure: `qssm/`, `rdtnet_baseline/`, `tractography/`, `hpc/`, `docs/`.
- RDT-Net baseline (ViT + reaction–diffusion PDE decoder) and its trainer.
- Evaluation (test-set MAE, Rician-noise robustness) and the statistics protocol.
- Free-water-corrected tractography: single-tissue CSD pipeline and SS3T-CSD + TractSeg.
- SLURM job templates, DWI preprocessing, adaptive-compute controller, CUDA build script.
- Environment-variable path configuration (`qssm/paths.py`); no hard-coded paths.
- `scripts/setup_check.py`, data-free model test, CI workflow.

### Withheld
- QSSM encoder, q-space ordering, synthetic-data corpus, trainer and unpublished
  analyses: placeholders with the real interface until publication.
