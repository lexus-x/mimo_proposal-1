# 05 — Compute Budget and Timeline

## 1. Compute Budget

### 1.1 Hardware Requirements

| Resource | Spec | Use |
|---|---|---|
| GPU | 1× A100 80GB (minimum) | Training + evaluation |
| GPU | 2× A100 80GB (recommended) | Parallel training + eval |
| CPU | 16+ cores | Data collection, preprocessing |
| RAM | 64GB+ | Dataset loading |
| Storage | 500GB+ SSD | Datasets + checkpoints |

### 1.2 Compute Breakdown

| Phase | Task | A100-Days | Notes |
|---|---|---|---|
| 1 | Data collection (MetaWorld MT-50) | 0.2 | CPU-only, parallel |
| 2 | Data preprocessing | 0.1 | CPU-only |
| 3 | SE(3) flow matching training (LIBERO) | 3-4 | 150K steps, batch 64 |
| 4 | Conformal calibration (LIBERO) | 0.1 | Post-hoc, fast |
| 5 | Full fine-tuning (LIBERO) | 1.5 | 50K steps |
| 6 | SE(3) flow matching training (MT-50) | 5-7 | 300K steps, batch 256 |
| 7 | Conformal calibration (MT-50) | 0.1 | Post-hoc |
| 8 | Baseline reproduction (SmolVLA, ACT) | 2-3 | Official checkpoints + eval |
| 9 | Evaluation (all benchmarks) | 2-3 | LIBERO + MT-50 + CALVIN |
| 10 | Ablation studies | 2-3 | Ensemble size, LoRA rank, etc. |
| 11 | Hyperparameter search | 2-3 | Key hyperparameters |
| **Total** | | **18-24** | |

### 1.3 Budget-Constrained Plan (13 A100-Days)

If compute is limited to ~13 A100-days:

| Phase | Task | A100-Days |
|---|---|---|
| 3 | SE(3) flow matching (LIBERO) | 3 |
| 5 | Fine-tuning (LIBERO) | 1.5 |
| 6 | SE(3) flow matching (MT-50) | 5 |
| 8 | Baselines | 2 |
| 9 | Evaluation | 1.5 |
| **Total** | | **13** |

**Cuts:** Skip ablations (do after submission), skip CALVIN (report from baselines), skip hyperparameter search (use defaults).

---

## 2. Timeline

### 2.1 Month 1: Foundation

| Week | Task | Deliverable |
|---|---|---|
| 1 | Environment setup, data collection | MT-50 demos collected |
| 2 | SE(3) flow matching head implementation | Code: `models/flow_head.py` |
| 3 | LIBERO training (Phase 1) | Checkpoint: `libero_phase1.pt` |
| 4 | Conformal calibration (LIBERO) | Checkpoint: `libero_calibrated.pt` |

### 2.2 Month 2: MetaWorld + Baselines

| Week | Task | Deliverable |
|---|---|---|
| 5 | MetaWorld MT-50 training | Checkpoint: `mt50_phase1.pt` |
| 6 | MetaWorld calibration + eval | Results: `mt50_results.json` |
| 7 | Baseline reproduction | Results: `baseline_results.json` |
| 8 | LIBERO fine-tuning + final eval | Results: `libero_final.json` |

### 2.3 Month 3: Ablations + Paper

| Week | Task | Deliverable |
|---|---|---|
| 9 | Ablation: ensemble size (2, 3, 5) | Ablation table |
| 10 | Ablation: LoRA rank, denoising steps | Ablation table |
| 11 | Paper writing (Sections 1-4) | Draft |
| 12 | Paper writing (Sections 5-7) + revision | Complete draft |

### 2.4 Month 4: Polish + Submit

| Week | Task | Deliverable |
|---|---|---|
| 13 | Internal review + revision | Revised paper |
| 14 | Real robot demo (optional) | Video + results |
| 15 | Final revision + supplementary | Final paper |
| 16 | Submit to CoRL 2026 | Submitted |

---

## 3. Cost Estimation

| Item | Cost | Notes |
|---|---|---|
| A100 cloud (24 days × $2/hr) | ~$1,152 | Lambda/CoreWeave/paperspace |
| A100 cloud (13 days, budget plan) | ~$624 | Minimum viable |
| Storage (500GB) | ~$50 | SSD |
| **Total (full)** | **~$1,200** | |
| **Total (budget)** | **~$700** | |

---

## 4. Reproducibility Checklist

- [ ] All random seeds fixed (42 default)
- [ ] All hyperparameters documented in config files
- [ ] All baseline checkpoints saved and shared
- [ ] Evaluation code released with paper
- [ ] Training curves logged to W&B
- [ ] Docker/conda environment file included
- [ ] Data collection scripts released
- [ ] Conformal calibration procedure documented
