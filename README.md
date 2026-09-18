# QSSM: Q-Space Selective-Scan Models for Free-Water Estimation in Diffusion MRI

![status](https://img.shields.io/badge/status-under%20review-orange)
![purpose](https://img.shields.io/badge/purpose-resume%20validation-blueviolet)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![pytorch](https://img.shields.io/badge/pytorch-2.1%2B-ee4c2c)
![license](https://img.shields.io/badge/license-MIT-green)

> **Purpose: validating the work on my resume ahead of publication.**
> This repository backs the claims in my resume entry **Systems & AI Research Intern
> (Medical Image Processing), Indian Institute of Technology (IIT) Patna, May 2026 â€“
> present**. The manuscript (targeted at *IEEE TMI*) is not yet published, so the core
> method is ðŸ”’ **withheld under lab confidentiality**. The code around it is here so the
> work can be checked: architecture, baselines, evaluation and statistics code,
> tractography and HPC pipelines.

**QSSM** is a state-space (Mamba-style) encoder for free-water estimation from diffusion
MRI. It models the diffusion signal **across gradient directions (q-space)** as well as
across the image. It builds on **RDT-Net**, a hybrid Vision Transformer + reactionâ€“diffusion
PDE network, and cuts parameters by **18.9%** and FLOPs by **12.2%** compared with it.

## Resume claims â†’ evidence

| Resume claim | Evidence in this repo | How to check |
|---|---|---|
| **RDT-Net**: hybrid ViT + reactionâ€“diffusion PDE framework for noise-robust free-water quantification | âœ… full model and trainer: [`rdtnet_baseline/`](rdtnet_baseline), [`qssm/models/rdt_net.py`](qssm/models/rdt_net.py), [`global_layer_rxn_diff.py`](qssm/models/global_layer_rxn_diff.py) | `python qssm/tests/test_shapes.py` builds it (509,383 params) |
| RDT-Net accuracy vs classical AMICO fitting | ðŸ“„ evaluation code present: [`qssm/evaluate/`](qssm/evaluate) | numbers reported in the manuscript |
| **QSSM**: 509K â†’ 413K parameters (âˆ’18.9%), âˆ’12.2% FLOPs | ðŸ”’ encoder withheld; the 2Ã—2 design, cost benchmark and parameter budget are documented: [`docs/architecture.md`](docs/architecture.md), [`qssm/efficiency/`](qssm/efficiency) | `python qssm/efficiency/bench.py --cpu-only` (reproduces all four arms on the full code) |
| QSSM beats the ViT baseline in a 2Ã—2 ablation (Wilcoxon) | âœ… statistics protocol: [`qssm/stats/stats.py`](qssm/stats/stats.py) (exact Wilcoxon, 2Ã—2 interaction, TOST, Holm) | p-values reported in the manuscript |
| **Automated bundle segmentation** with TractSeg from FODs | âœ… complete pipelines: [`tractography/`](tractography) (FW correction â†’ SS3T-CSD â†’ TractSeg bundles, endings, TOMs) | runs on any HCP-style subject |
| **Data scale & optimisation**: SLURM processing, adaptive-compute controller | âœ… SLURM array jobs, preprocessing, SNR-conditioned iteration controller: [`hpc/`](hpc) | throughput figures reported in the manuscript |

âœ… code included Â· ðŸ“„ code included, needs data Â· ðŸ”’ withheld until publication ([WITHHELD.md](WITHHELD.md))

Quick check with no data needed: `python scripts/setup_check.py`.

---

## Project Overview

At a high level, the repository supports this end-to-end workflow:

1. start from HCP multishell diffusion MRI,
2. extract single-shell inputs (one b0 plus the b = 1000 shell),
3. train the **RDT-Net** baseline (ViT encoder + reactionâ€“diffusion PDE decoder),
4. train the **QSSM** encoder arms in a controlled 2Ã—2 factorial against it,
5. evaluate accuracy, noise robustness and statistical significance,
6. apply the predicted free-water maps to **free-water-corrected tractography** and
   **TractSeg** white-matter bundle segmentation,
7. run all of it at scale with **SLURM** array jobs.

## Highlights

- **Q-space state-space encoder.** A selective scan across the gradient directions of
  each voxel, plus a spatial scan that replaces self-attention.
- **Smaller and cheaper.** 509,383 â†’ 412,983 parameters (âˆ’18.9%) and âˆ’12.2% FLOPs
  compared with the ViT baseline.
- **Controlled 2Ã—2 design.** {attention, spatial scan} Ã— {no q-scan, q-scan}. Only the
  encoder varies, so each effect is measured twice.
- **Physics-informed decoder.** A reactionâ€“diffusion PDE decoder (RDT-Net) that
  regularises the free-water map spatially.
- **Downstream tractography.** Free-water-corrected FODs and TractSeg segmentation of
  72 white-matter bundles.
- **HPC-ready.** SLURM templates for preprocessing, the four-arm training sweep,
  evaluation and tractography, plus an adaptive-compute controller that predicts how many
  PDE refinement iterations each input needs.
- **Reproducible by construction.** No hard-coded paths, seeded data loading, a
  data-free setup check, and CI on every push.

## What This Repository Contains

| Folder | Contents |
|---|---|
| [`qssm/`](qssm) | â˜… the research: encoder arms, evaluation, statistics, efficiency benchmark |
| [`rdtnet_baseline/`](rdtnet_baseline) | RDT-Net, the ViT + reactionâ€“diffusion PDE baseline |
| [`tractography/`](tractography) | free-water-corrected CSD / SS3T-CSD tractography and TractSeg |
| [`hpc/`](hpc) | DWI preprocessing, SLURM jobs, adaptive-compute controller, CUDA build |
| [`scripts/`](scripts) | `setup_check.py`: first-run verification |
| [`docs/`](docs) | architecture overview, expected data layout |
| [`splits/`](splits) | subject-list templates (your lists stay local) |

Every folder has its own README explaining what it does, how to run it and what it
expects.

## Supported Entry Points

Start with these:

| Task | Command |
|---|---|
| Verify the setup (no data) | `python scripts/setup_check.py` |
| Build and check the models (no data) | `python qssm/tests/test_shapes.py` |
| Parameter / FLOP table (no data) | `python qssm/efficiency/bench.py --cpu-only` |
| Train RDT-Net | `python rdtnet_baseline/train_rxn_diff.py` |
| Train a QSSM arm ðŸ”’ | `python qssm/train/train_mamba.py --encoder mamba` |
| Evaluate checkpoints | `python qssm/evaluate/evaluate.py --all` |
| Noise robustness | `python qssm/evaluate/evaluate_snr.py --all` |
| Statistics | `python qssm/stats/stats.py --md results/stats.md` |
| Tractography + TractSeg | `bash tractography/ss3t_tractseg/run_fw_tractseg.sh` |
| Everything on SLURM | `sbatch hpc/slurm/train_arms.sbatch` (see [hpc/README.md](hpc/README.md)) |

ðŸ”’ = placeholder in this release (see [WITHHELD.md](WITHHELD.md)).

## HCP Data Access

HCP data is **not** redistributed in this repository. Obtain it from the official
Human Connectome Project portals:

- HCP homepage: <https://www.humanconnectome.org/>
- HCP data and software: <https://www.humanconnectome.org/software/>
- HCP Young Adult open-access data-use terms:
  <https://www.humanconnectome.org/study/hcp-young-adult/document/wu-minn-hcp-consortium-open-access-data-use-terms>

In practice:

1. create an HCP / NIH Data Archive account,
2. accept the applicable data-use terms,
3. download the diffusion MRI files (`data.nii.gz`, `bvals`, `bvecs`,
   `nodif_brain_mask.nii.gz`) for your subjects.

Clinical data is never part of this repository (see [CONFIDENTIALITY.md](CONFIDENTIALITY.md)).

## Compact Workflow

```text
HCP multishell dMRI
    |
    +--> hpc/preprocessing/shell_extractor.py
    |        -> dwi_b0_1000.nii.gz + bvals_b0_1000 / bvecs_b0_1000
    |
    +--> multishell FWDTI fit (DIPY, run separately) -> free_water_volume.nii.gz (ground truth)
                                        |
             +--------------------------+
             |
             +--> rdtnet_baseline/train_rxn_diff.py      RDT-Net (ViT + PDE)
             |        |
             |        +--> warm start
             |               |
             +--> qssm/train/train_mamba.py ðŸ”’             A0 Â· A1 Â· A2 Â· A3 (QSSM)
                      |
                      +--> qssm/evaluate/                  MAE Â· SNR sweep
                      +--> qssm/stats/                     Wilcoxon Â· 2Ã—2 Â· TOST Â· Holm
                      +--> qssm/efficiency/                params Â· FLOPs Â· memory
                      |
                      +--> tractography/                   FW-corrected FODs -> TractSeg bundles
```

### 1. Prepare single-shell inputs

`hpc/preprocessing/shell_extractor.py` converts multishell `data.nii.gz` into one-b0 +
one-shell volumes (`dwi_b0_{1000,2000,3000}.nii.gz` with matching gradients). The models
use `dwi_b0_1000.nii.gz`, which has 91 channels. For raw BIDS acquisitions,
`hpc/preprocessing/dwi_preprocess.py` runs denoising, Gibbs removal, topup/eddy, bias
correction and T1 registration first.

### 2. Train the RDT-Net baseline

RDT-Net is a ViT encoder followed by two reactionâ€“diffusion PDE blocks and a gated
decoder. It is trained on single-shell DWI plus a Stage-I ANN free-water prior, with
multishell ground truth as the target. See [rdtnet_baseline/README.md](rdtnet_baseline/README.md).

### 3. Train the QSSM arms ðŸ”’

The four arms share the RDT-Net decoder and differ only in the encoder:

|  | no q-scan | q-scan |
|---|---|---|
| **attention** | A0 Â· 509,383 params | A1 Â· 513,591 |
| **spatial scan** | A2 Â· 408,775 | **A3 QSSM Â· 412,983** |

High-level design: [docs/architecture.md](docs/architecture.md).

### 4. Evaluate and test

Per-subject free-water MAE on a held-out cohort, robustness across SNR 60 â†’ 10, and
paired statistics: exact Wilcoxon signed-rank, the 2Ã—2 interaction, TOST equivalence for
null claims, and Holm correction. See [qssm/README.md](qssm/README.md).

### 5. Tractography and bundle segmentation

The predicted free-water map is removed from the DWI signal. Then SS3T-CSD computes FODs,
`tckgen` produces whole-brain tractography, and TractSeg segments bundles, bundle
endings and tract orientation maps. See [tractography/README.md](tractography/README.md).

## Repository Layout

```text
ResearchWork/
â”œâ”€â”€ qssm/
â”‚   â”œâ”€â”€ models/          encoder arms, PDE decoder, selective-scan backends
â”‚   â”œâ”€â”€ data/            training-data construction, Rician noise
â”‚   â”œâ”€â”€ train/           trainer for all arms
â”‚   â”œâ”€â”€ evaluate/        test-set MAE, SNR sweep, downstream analyses
â”‚   â”œâ”€â”€ ablations/       pre-flight checks, component knockouts
â”‚   â”œâ”€â”€ efficiency/      params / FLOPs / memory / latency
â”‚   â”œâ”€â”€ stats/           statistical protocol
â”‚   â”œâ”€â”€ tests/           data-free model test
â”‚   â””â”€â”€ paths.py         environment-variable path resolution
â”œâ”€â”€ rdtnet_baseline/     RDT-Net trainer
â”œâ”€â”€ tractography/
â”‚   â”œâ”€â”€ ss3t_tractseg/   SS3T-CSD + TractSeg pipeline
â”‚   â””â”€â”€ csd_pipeline/    single-tissue CSD pipeline
â”œâ”€â”€ hpc/
â”‚   â”œâ”€â”€ preprocessing/   DWI preprocessing, shell extraction
â”‚   â”œâ”€â”€ slurm/           sbatch templates
â”‚   â”œâ”€â”€ adaptive_compute/  PDE-iteration controller
â”‚   â””â”€â”€ env/             mamba-ssm CUDA build
â”œâ”€â”€ scripts/             setup_check.py
â”œâ”€â”€ docs/                architecture.md, data_layout.md
â”œâ”€â”€ splits/              subject-list templates
â”œâ”€â”€ .github/workflows/   CI
â”œâ”€â”€ README.md Â· INSTALL.md Â· WITHHELD.md Â· CONFIDENTIALITY.md
â””â”€â”€ CONTRIBUTING.md Â· CHANGELOG.md Â· CITATION.cff Â· LICENSE
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/setup_check.py
```

The optional fused selective-scan CUDA kernel (`mamba-ssm`) and the neuroimaging tools
(MRtrix3, MRtrix3Tissue, TractSeg, FSL, ANTs) are covered in [INSTALL.md](INSTALL.md).

## Quickstart

With no data, verify the installation and the models:

```bash
python scripts/setup_check.py
python qssm/tests/test_shapes.py
python qssm/efficiency/bench.py --cpu-only
```

With data, point the code at it and list your subjects:

```bash
export QSSM_DATA_ROOT=/path/to/your/data      # layout: docs/data_layout.md
cp splits/example.txt splits/train.txt         # one subject ID per line
cp splits/example.txt splits/test.txt
python qssm/evaluate/evaluate.py --all
```

## Configuration

No path is hard-coded. Everything resolves through [qssm/paths.py](qssm/paths.py):

| Variable | Default | Meaning |
|---|---|---|
| `QSSM_DATA_ROOT` | `./data` | dataset root |
| `QSSM_TRAIN_SIGNAL`, `QSSM_TRAIN_GT` | `$QSSM_DATA_ROOT/training`, `/FWE` | training DWI, free-water ground truth |
| `QSSM_TEST_ROOT` | `$QSSM_DATA_ROOT/test` | held-out test cohort |
| `QSSM_PRIOR_ROOT` | `$QSSM_DATA_ROOT/ann_prior` | Stage-I ANN free-water priors |
| `QSSM_SPLITS_DIR` | `./splits` | `train.txt`, `test.txt`, `external.txt` |
| `QSSM_CKPT_DIR`, `QSSM_RESULTS_DIR` | `./checkpoints`, `./results` | outputs (git-ignored) |

## Confidentiality

This repository is shared **only to validate the work described in my resume** while the
paper is under preparation. It comes from ongoing,
**unpublished research at IIT Patna**.

- The QSSM core implementation and unpublished analyses are withheld under lab
  confidentiality until publication. I'm happy to walk through the withheld parts in an
  interview without sharing the source.
- No imaging data, subject identifiers, checkpoints or results are included. Clinical data
  is covered by separate agreements and must never be committed or shared.
- Do not redistribute withheld components or results obtained with them without the
  principal investigator's written permission.

The full notice is in [CONFIDENTIALITY.md](CONFIDENTIALITY.md).

## Maintenance

- **CI** (`.github/workflows/ci.yml`) byte-compiles every script, checks shell syntax,
  runs the data-free model test and benchmark, and fails if any data or weight file is
  tracked.
- **Contribution rules:** [CONTRIBUTING.md](CONTRIBUTING.md). Summary: no data, no
  paths, no subject IDs, no unpublished numbers, and keep every README current.
- **Changes:** [CHANGELOG.md](CHANGELOG.md).

## Publication status

Manuscript in preparation (target: *IEEE Transactions on Medical Imaging*). The full code
will be released on publication.

## Notes

- This repository does not include HCP data, patient data, checkpoints or generated outputs.
- ðŸ”’ files are placeholders until publication. Calling them raises a clear
  "withheld" error rather than failing silently.
- This is a portfolio snapshot, not a supported software release.

## Acknowledgements

Developed at the Indian Institute of Technology (IIT) Patna. Built on PyTorch,
`mamba-ssm`, DIPY, MRtrix3, MRtrix3Tissue and TractSeg. Data were provided in part by the
Human Connectome Project, WU-Minn Consortium (Principal Investigators: David Van Essen
and Kamil Ugurbil; 1U54MH091657) funded by the 16 NIH Institutes and Centers that support
the NIH Blueprint for Neuroscience Research, and by the McDonnell Center for Systems
Neuroscience at Washington University.

## License

Source code: [MIT](LICENSE). The license does not cover data, withheld components or
unpublished results (see [CONFIDENTIALITY.md](CONFIDENTIALITY.md)).
  
 
 
