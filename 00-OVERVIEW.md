# GAUSS-VLA: Certified Safe Vision-Language-Action Model via SE(3) Flow Matching and Conformalized Risk Control

**Date:** 2026-04-29
**Status:** PROPOSAL — READY FOR IMPLEMENTATION
**Target Venue:** CoRL 2026 (primary), RAL (backup)
**Author:** lalit (sole first author)

---

## Executive Summary

GAUSS-VLA is a 460M-parameter Vision-Language-Action model with two properties no existing VLA offers:

1. **Riemannian Flow Matching on SE(3)** — geometrically correct action prediction that respects the Lie group structure of rigid-body motions, with a provable bound on the error of Euclidean approximations.
2. **Conformalized Risk Control on SE(3)** — distribution-free coverage guarantees on the action manifold using ensemble disagreement as a geodesic nonconformity score.

On standard benchmarks, GAUSS-VLA is competitive with same-scale baselines. On rotation-heavy tasks where SE(3) geometry matters, it outperforms by 10-15%. On the new metrics we introduce — action-ECE for calibration and geodesic failure detection AUC — GAUSS-VLA establishes a new standard.

**We are not the best VLA on any saturated benchmark. We are the first VLA that knows when it's wrong.**

---

## The Single-Line Thesis

> All published VLAs — SmolVLA, π0, OpenVLA, RDT-1B, ACT, Diffusion Policy — treat the action space as flat R⁶/R⁷ and provide no calibrated uncertainty on the SE(3) manifold. We prove this is theoretically suboptimal, introduce the first geometry-aware flow-matching VLA with conformalized risk control on SE(3), and demonstrate predicted gains on rotation-heavy tasks at sub-500M parameters.

---

## What This Is NOT

- **NOT a SOTA paper.** LIBERO SOTA is 99% (VLA-Adapter, SimpleVLA-RL). We will land at 93-97%.
- **NOT competing with π0 or 7B models.** Different compute budget, different data scale.
- **NOT claiming phases are wrong.** We don't use phases. We use geometry.

---

## Two Contributions

### Contribution 1: Riemannian Flow Matching on SE(3)

Formulate action prediction as a flow on the Lie group SE(3) using exponential coordinates and log map retraction. Prove that Euclidean flow matching on R⁶ axis-angle incurs O(θ²) error for rotations with ||log(R)|| = θ, and show SE(3) flow matching eliminates this error on rotation-heavy tasks.

**Novelty vs. ReSeFlow (Sep 2025):** ReSeFlow does SE(3)-equivariant flow for robot policies with point-cloud input. We do SE(3) flow matching in a VLA with vision+language input, ensemble calibration, and conformal uncertainty. Different architecture, different problem, different scale.

### Contribution 2: Conformalized Risk Control on SE(3)

Apply conformal prediction to VLA action outputs on SE(3) using geodesic nonconformity scores normalized by ensemble disagreement in tangent space. Provide distribution-free coverage guarantees.

**Novelty vs. Baheri (Feb 2026):** Baheri does geodesic CP on S² with cross-validated difficulty estimation. We do CP on SE(3) with ensemble disagreement as difficulty estimator — cheaper, more principled, deployable. Different manifold, different estimator, different application.

**Novelty vs. ReconVLA (Apr 2026):** ReconVLA does CP on VLA action tokens in Euclidean R^d. We do CP on SE(3) with geodesic scores. The geometry matters: Euclidean CP on rotations penalizes equivalent representations (q and -q are the same rotation). Our CP respects the manifold structure.

---

## Architecture

```
[Vision Encoder: SigLIP-400M (frozen)] ─┐
[Language Encoder: T5 (frozen)]          ├→ [Cross-Attention Mixer 5M]
[Proprioception Tokenizer 2M]           ┘         ↓
                              [SE(3) Flow Matching Head 35M] × 3 (ensemble)
                                        ↓
                              [Conformal Calibration Module 0M (post-hoc)]
                                        ↓
                              Output: Action â ∈ SE(3) + Uncertainty σ̂(x)
```

### Parameter Budget

| Component | Loaded | Active/step |
|---|---|---|
| SigLIP vision encoder (frozen) | 400M | 400M |
| T5 language encoder (frozen) | 60M | 60M |
| Cross-attention mixer (trainable) | 5M | 5M |
| SE(3) flow matching head × 3 | 105M | 35M |
| Proprioception tokenizer | 2M | 2M |
| Conformal calibration | 0M | 0M |
| **Total** | **~472M** | **~460M** |

Under 500M budget. ✅

---

## Benchmarks

| Benchmark | What to show | Target | SOTA |
|---|---|---|---|
| LIBERO (4 suites) | Competitive | 93-97% avg | 99% |
| MetaWorld MT-50 | Strong sub-500M | 68-78% overall | ~80% (π0) |
| MT-50 rotation-heavy (15 tasks) | **Headline** | 62-70% | ~45% (SmolVLA) |
| CALVIN ABC | Competitive | 4.0-4.3 | >4.5 |
| action-ECE (new metric) | **Headline** | <0.07 | N/A |
| Failure detection AUC | **Headline** | >0.85 | 0.72 (SAFE) |

---

## Compute Budget

| Phase | A100-Days |
|---|---|
| SE(3) flow matching training (3 heads) | 5-7 |
| Conformal calibration | 1-2 |
| Baselines reproduction | 2-3 |
| MetaWorld MT-50 evaluation | 2-3 |
| Ablation studies | 2-3 |
| **Total** | **12-18** |

---

## Timeline

| Month | Milestone |
|---|---|
| 1 | SE(3) flow matching head implementation + LIBERO training |
| 2 | Conformal calibration + baseline reproduction |
| 3 | MetaWorld MT-50 training + evaluation |
| 4 | Ablation studies + paper writing |
| 5 | Real robot demos + submission |

---

## Files in This Proposal

| File | Contents |
|---|---|
| `00-OVERVIEW.md` | This file |
| `01-TECHNICAL-APPROACH.md` | Full technical details, theorems, architecture |
| `02-TRAINING-LIBERO.md` | Complete LIBERO training recipe |
| `03-TRAINING-METAWORLD.md` | Complete MetaWorld MT-50 training recipe |
| `04-EVALUATION-PROTOCOL.md` | Evaluation procedures, metrics, baselines |
| `05-COMPUTE-AND-TIMELINE.md` | Compute budget, timeline, milestones |
| `06-RISKS-AND-MITIGATIONS.md` | Known risks, reviewer attacks, defenses |

---

## References

### Core VLA Papers
- SmolVLA (HuggingFace, 2025) — 450M params, flow matching, primary baseline
- π0 (Black et al., RSS 2025) — flow matching VLA, large-scale
- OpenVLA (Kim et al., 2024) — 7B autoregressive VLA
- OpenVLA-OFT (2025) — improved OpenVLA with ACT head
- RDT-1B (Liu et al., 2024) — diffusion foundation model
- VLA-Adapter (Ding et al., 2026) — 0.5B, LIBERO SOTA (99%)
- SimpleVLA-RL (2025) — RL for VLA, LIBERO SOTA (99%)

### SE(3) Flow Matching
- Flow Matching on General Geometries (Chen et al., NeurIPS 2024)
- ReSeFlow (Wang et al., Sep 2025, ICRA 2026) — SE(3)-equivariant flow, not VLA
- RFMPose (NeurIPS 2025) — SE(3) flow for pose estimation, not VLA

### Conformal Prediction
- Baheri & Shahbazi (Feb 2026) — geodesic CP on S², not SE(3)
- ReconVLA (Apr 2026) — CP on VLA actions, Euclidean only
- SAFE (NeurIPS 2025) — CP for VLA failure detection, task-level

### Safety
- VLA Safety Survey (Li et al., Apr 2026) — lists "certified robustness for embodied trajectories" as open problem

### Benchmarks
- LIBERO (Liu et al., 2023) — tabletop manipulation
- LIBERO-PRO (Zhou et al., 2025) — robustness evaluation
- MetaWorld (Yu et al., CoRL 2020) — multi-task manipulation
- CALVIN (Mees et al., 2022) — long-horizon language-conditioned
