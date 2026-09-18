# `qssm/ablations/`: verifying and dissecting the arms

> 🔒 = withheld until publication (placeholder with the real interface; see [WITHHELD.md](../../WITHHELD.md)).

| Script | When to run it | What it does |
|---|---|---|
| `verify_arms.py` | **before training** | (1) exact parameter count of every arm; (2) A1 at initialisation computes **bit-for-bit** the same function as A0 after the same warm start (`max|A1(x) − A0(x)| == 0`); (3) the warm start lands on every module except the new encoder parts, reporting missing/unexpected key names; (4) fused and PyTorch scan kernels agree. Rebuilds the canonical q-space order from the training bvecs and checks it against the pinned permutation. |
| `ablate_a3.py` 🔒 | after training A3 | Inference-time knockouts on the trained QSSM checkpoint (below). |
| `ablate_gate_se.py` 🔒 | any checkpoint | Checks whether `ChannelGate` and `SEBlock` still respond to their input or have saturated into constants, then knocks each one out. |

```bash
python verify_arms.py
python ablate_a3.py                 # --subjects N for a quick look
python ablate_gate_se.py --checkpoint $QSSM_CKPT_DIR/<checkpoint>.pth
```

## `ablate_a3.py` variants

| Variant | What changes | What it isolates |
|---|---|---|
| q-branch off | `q_merge` set to zero (A2 architecture at inference) | how much the trained model relies on the q-space path |
| q-order random | q-branch kept, directions scrambled | whether the direction ordering carries information |
| q-order identity | raw file order (what code that ignores bvecs would use) | cost of ignoring gradient geometry |

These are **inference-time** knockouts. They show how much the trained model *relies on*
a component, not whether a model trained without it would do worse. Answering that
second question needs retraining, for example `train_mamba.py --q-order random`.

## The 2×2 as an ablation

The factorial itself (see [../README.md](../README.md#the-22-factorial)) is the main
*trained* ablation. `stats/stats.py` computes both main effects and their interaction.
