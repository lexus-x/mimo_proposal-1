# 06 — Risks and Mitigations

## 1. Scoop Risks

### 1.1 Baheri Extension to SE(3) (HIGH RISK)

**Risk:** Baheri & Shahbazi (Feb 2026) extend their geodesic CP framework from S² to SE(3). This would partially scoop our Contribution 2.

**Likelihood:** Medium. Baheri's paper is recent (Feb 2026) and focuses on S² for geomagnetic forecasting. Extension to SE(3) is a natural next step.

**Mitigation:**
- Our difficulty estimator (ensemble disagreement) is fundamentally different from Baheri's (cross-validated). This is a different contribution, not an extension.
- We apply to VLAs (robotics), not geomagnetic forecasting. Different domain.
- Speed: submit within 3-4 months. The window is open NOW.

**If scooped:** Reposition as "VLA application of Baheri's framework with novel difficulty estimator." Still publishable, but weaker.

### 1.2 ReconVLA Extension to SE(3) (MEDIUM RISK)

**Risk:** ReconVLA (Apr 2026) extends their Euclidean CP to manifold-valued actions.

**Likelihood:** Low-medium. ReconVLA focuses on Euclidean action tokens. Extension to SE(3) requires significant new theory.

**Mitigation:**
- Our approach is different: geodesic nonconformity scores vs. Euclidean CQR.
- We use ensemble disagreement, they use residual-based scores.
- Different contributions.

### 1.3 ReSeFlow Extension to VLA (MEDIUM RISK)

**Risk:** ReSeFlow (Sep 2025, ICRA 2026) extends their SE(3)-equivariant flow to VLA-scale models.

**Likelihood:** Low. ReSeFlow uses equivariant networks (different architecture). Extension to VLA requires significant engineering.

**Mitigation:**
- We use standard transformers with SE(3) tangent-space output, not equivariant networks. Different architecture.
- We add conformal uncertainty, which ReSeFlow doesn't have.

### 1.4 Large Lab System Paper (LOW-MEDIUM RISK)

**Risk:** Physical Intelligence, Hugging Face, or Google releases a VLA with built-in uncertainty quantification.

**Likelihood:** Medium. π0.5 already exists. Adding uncertainty is a natural extension.

**Mitigation:**
- Our uncertainty is formally calibrated (conformal prediction), not just "we have an ensemble."
- Coverage guarantees are our unique selling point.
- Even if they add uncertainty, they likely won't have SE(3) geometric guarantees.

---

## 2. Technical Risks

### 2.1 SE(3) Flow Matching Doesn't Improve on LIBERO (MEDIUM RISK)

**Risk:** The Euclidean approximation error (Theorem 1) is small in practice. SE(3) flow matching doesn't significantly outperform Euclidean on LIBERO tasks.

**Likelihood:** Medium. LIBERO tasks may not have enough rotation diversity to show the difference.

**Mitigation:**
- Focus on MetaWorld rotation-heavy tasks (15 tasks with significant rotation).
- If LIBERO doesn't show improvement, head-line MT-50 rotation-heavy results.
- Ablation: compare SE(3) vs Euclidean flow matching on rotation-heavy subset.

### 2.2 Ensemble Collapse (LOW RISK)

**Risk:** All 3 ensemble heads converge to the same solution, defeating the purpose of ensemble disagreement.

**Likelihood:** Low with proper diversity loss (λ_div = 0.01).

**Mitigation:**
- Monitor per-head agreement during training. If agreement > 95%, increase λ_div.
- Use different random seeds for each head initialization.
- Use different data augmentation for each head (optional).

### 2.3 Conformal Coverage Violated (LOW RISK)

**Risk:** Empirical coverage is below 1 − α due to temporal dependence in robot actions (exchangeability violation).

**Likelihood:** Low for LIBERO (short episodes). Medium for MetaWorld (longer episodes).

**Mitigation:**
- Use split conformal with temporal weighting if needed.
- Report empirical coverage alongside theoretical guarantee.
- If coverage is violated, use weighted conformal prediction (Barber et al., 2023).

### 2.4 MetaWorld MT-50 Performance Below Target (MEDIUM RISK)

**Risk:** GAUSS-VLA achieves <65% on MT-50, below the 68-78% target.

**Likelihood:** Medium. MT-50 is hard for sub-500M models. SmolVLA only gets ~55-62%.

**Mitigation:**
- Use task-grouped LoRA (separate LoRA per task group) if needed.
- Increase training steps to 500K if compute allows.
- Focus on the rotation-heavy subset (our advantage) even if overall is lower.

---

## 3. Reviewer Attack Preparation

### 3.1 "This is just Baheri on SE(3)"

**Defense:**
1. Baheri uses cross-validated difficulty estimation. We use ensemble disagreement — a fundamentally different approach that captures epistemic uncertainty, not heteroscedastic noise.
2. Baheri applies to S² (geomagnetic forecasting). We apply to SE(3) (robot manipulation). Different manifold, different domain.
3. Our VLA integration + safety framing is novel.
4. Empirically show that ensemble disagreement outperforms cross-validation as difficulty estimator on SE(3).

### 3.2 "ReconVLA already does this"

**Defense:**
1. ReconVLA uses Euclidean CQR on action tokens. We use geodesic CQR on SE(3).
2. Euclidean CP on rotations penalizes equivalent representations (q and −q). Our CP respects the manifold structure.
3. Empirical comparison: show coverage gap between Euclidean and SE(3) CP on rotation-heavy tasks.

### 3.3 "ReSeFlow already does SE(3) flow matching"

**Defense:**
1. ReSeFlow is not a VLA. It uses point-cloud input, no language, no large-scale pretraining.
2. We integrate SE(3) flow matching into a VLA with vision+language input.
3. We add conformal uncertainty, which ReSeFlow doesn't have.
4. Different architecture (standard transformer vs equivariant network).

### 3.4 "Set-valued actions are impractical"

**Defense:**
1. We don't propose set-valued actions. We propose calibrated uncertainty for risk control.
2. The conformal set radius is a real-time safety signal: "stop when uncertain."
3. Practical deployment: robot executes the point prediction, but uses set radius to decide whether to proceed or ask for help.

### 3.5 "Exchangeability doesn't hold for sequential actions"

**Defense:**
1. Empirically: show coverage holds on held-out rollouts even with temporal dependence.
2. Theoretically: use split conformal with temporal weighting if needed.
3. For short episodes (LIBERO, ~30 steps), exchangeability is approximately satisfied.

### 3.6 "The Lipschitz argument is hand-wavy"

**Defense:**
1. Empirically measure spectral norm of weight matrices during training.
2. Show correlation between model size and Lipschitz constant.
3. If monotonicity doesn't hold, drop the claim and keep empirical tightness result.

### 3.7 "Why not just use SAFE + single action?"

**Defense:**
1. SAFE does task-level failure detection. We do action-level uncertainty on SE(3).
2. SAFE uses Euclidean scores. We use geodesic scores on the manifold.
3. Our coverage guarantee is distribution-free and formally calibrated. SAFE's is not.

### 3.8 "LIBERO is saturated / MT-50 numbers are not SOTA"

**Defense:**
1. We don't claim SOTA on LIBERO. We report it as a sanity check (93-97%).
2. Our headline results are on rotation-heavy tasks (MT-50) and new metrics (action-ECE, failure AUC).
3. We introduce new metrics the field needs. That's the contribution, not benchmark numbers.

---

## 4. Backup Plans

### 4.1 If SE(3) Flow Matching Doesn't Help

**Pivot:** Use standard Euclidean flow matching + conformal uncertainty. Drop Theorem 1. Keep Contribution 2 (conformal risk control). Still a novel paper — "first VLA with calibrated uncertainty."

**Venue:** Still CoRL-eligible. Weaker but publishable.

### 4.2 If MT-50 Results Are Weak

**Pivot:** Focus entirely on LIBERO + new metrics. Don't report MT-50. Headline: "first VLA with calibrated uncertainty on LIBERO."

**Venue:** Still CoRL-eligible. Narrower contribution.

### 4.3 If Both Are Weak

**Pivot:** Benchmark paper. "Systematic evaluation of VLA uncertainty methods." Compare ensemble, MC-dropout, evidential, conformal on LIBERO + MT-50.

**Venue:** RAL or ICRA workshop. Service contribution, not novel method.

---

## 5. What Would Force Us to Abandon

1. Baheri extends to SE(3) AND applies to robotics before we submit → reposition but continue.
2. ReconVLA v2 adds manifold CP → compare head-to-head, continue if we're better.
3. Physical Intelligence releases π1 with certified uncertainty → pivot to benchmark paper.
4. SE(3) flow matching has zero improvement on any benchmark → drop Contribution 1, keep Contribution 2.
5. Conformal coverage is systematically violated → drop coverage claim, keep empirical calibration.

**None of these are paper-killers.** Each has a viable pivot. The research direction is robust.
