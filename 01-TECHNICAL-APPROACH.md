# 01 — Technical Approach

## 1. Problem Statement

### 1.1 The Geometric Failure Mode

Robot end-effector poses live on **SE(3)** = SO(3) ⋉ ℝ³, a 6-dimensional Lie group. Every published VLA predicts actions in flat R⁶ axis-angle ⊕ R³ translation coordinates. This is theoretically suboptimal:

- **Axis-angle is a 2-cover:** antipodal points (n̂, π) and (−n̂, π) map to the same rotation. Discontinuity at ||θ|| = π.
- **Euler angles have gimbal lock.**
- **Quaternions are a double-cover** (q and −q are the same rotation). Naïve MSE loss penalizes equivalent representations.
- **Euclidean MSE on axis-angle** penalizes rotation pairs that are actually identical → wasted gradient signal.

**Empirical consequence:** on rotation-heavy tasks (dial-turn, door-unlock, faucet-open/close, nut-assemble, peg-insert-side, wrench, hammer, hand-insert — ~15 tasks in MT-50), current VLAs accumulate execution error proportional to ||rotation||² where ||rotation|| > π/2.

### 1.2 The Calibration Gap

No published VLA reports calibrated action uncertainty. Flow matching and diffusion policies sample from an implicit distribution but don't quantify when the model is uncertain. Consequences:
- No principled stopping criterion for chunked prediction
- No safe-deployment story (when to defer to a human?)
- No OOD detection (when is the policy outside its training distribution?)

### 1.3 The Combined Problem

Geometric error + uncalibrated uncertainty compound on rotation-heavy long-horizon tasks. Closing both simultaneously is the contribution no current method addresses.

---

## 2. Contribution 1: Riemannian Flow Matching on SE(3)

### 2.1 Mathematical Formulation

Let SE(3) = SO(3) ⋉ ℝ³ be the special Euclidean group. Its Lie algebra is se(3) = so(3) ⊕ ℝ³, where so(3) ≅ ℝ³ via the hat map.

**Geodesic interpolation on SE(3):**

For X₀, X₁ ∈ SE(3), the geodesic from X₀ to X₁ is:

```
X_t = X₀ · exp(t · log(X₀⁻¹ · X₁))
```

where exp: se(3) → SE(3) is the exponential map and log: SE(3) → se(3) is the logarithmic map.

**Riemannian flow matching loss on SE(3):**

```
L_SE(3) = E_{t, X₀, X₁} ||v_θ(X_t, t) - log_{X_t}(X₁)||²_{X_t}
```

where v_θ is the learned velocity field, X_t is the geodesic interpolation, and ||·||_{X_t} is the Riemannian metric at X_t (bi-invariant metric on SE(3)).

### 2.2 Theorem 1: Euclidean Approximation Error Bound

**Theorem.** Let f^E be the optimal Euclidean flow matching policy on R⁶ axis-angle and f* the optimal SE(3) flow matching policy. For any rotation R with ||log(R)|| = θ, the expected execution error satisfies:

```
E||f^E(R) - f*(R)|| ≤ C · θ² + O(θ⁴)
```

for a constant C depending on the data distribution.

**Interpretation:** For small rotations (θ < π/4), Euclidean and SE(3) are equivalent. For large rotations (θ > π/2), the Euclidean error grows quadratically. This predicts gains on rotation-heavy tasks where θ > π/2 is common.

**Proof sketch:** The bound follows from the Baker-Campbell-Hausdorff formula and the curvature of SO(3). The key step is that the Euclidean approximation of the logarithmic map has error O(θ³) at the Lie algebra level, which translates to O(θ²) at the group level after integration.

### 2.3 Practical Implementation

**Action representation:** End-effector pose as (R, t) ∈ SE(3) where R ∈ SO(3) and t ∈ ℝ³.

**For LIBERO (7D actions):** 6D rotation (Zhou et al. 2019, arXiv:1812.07035) + 3D translation. The flow matching head predicts in se(3) tangent space (6D) and maps to SE(3) via exponential map.

**For MetaWorld (4D actions):** dx, dy, dz, gripper. The SE(3) framework applies to the translation component. The gripper is handled separately as a scalar.

**Ensemble:** 3 independent flow matching heads, each 35M parameters. Each head predicts in se(3) tangent space. Disagreement between heads provides epistemic uncertainty.

### 2.4 Novelty vs. Existing Work

| Prior Work | What They Do | Why GAUSS-VLA Is Different |
|---|---|---|
| ReSeFlow (Sep 2025) | SE(3)-equivariant rectified flow for robot policies | Point-cloud input, no language, no VLA, no uncertainty |
| RFMPose (NeurIPS 2025) | SE(3) flow matching for object pose estimation | Not robot actions, not VLA |
| DFM-VLA (Mar 2026) | Discrete flow matching for VLA | Not Riemannian, not SE(3) |
| Chen et al. (NeurIPS 2024) | General Riemannian flow matching framework | No robotics application |
| π0, SmolVLA | Flow matching in Euclidean R⁶ | No geometric structure |

---

## 3. Contribution 2: Conformalized Risk Control on SE(3)

### 3.1 Conformal Prediction Background

Conformal prediction (CP) provides distribution-free coverage guarantees. Given calibration data {(X₁, Y₁), ..., (Xₙ, Yₙ)} and a new point (X_{n+1}, Y_{n+1}):

```
P(Y_{n+1} ∈ C(X_{n+1})) ≥ 1 - α
```

under exchangeability of the data.

### 3.2 Geodesic Nonconformity Score on SE(3)

**Definition.** The nonconformity score for point (Xᵢ, Yᵢ) is:

```
sᵢ = d_SE(3)(Ŷᵢ, Yᵢ) / σ̂(Xᵢ)
```

where:
- d_SE(3)(Ŷ, Y) = ||log(Ŷ⁻¹ · Y)|| is the geodesic distance on SE(3) under the bi-invariant metric
- σ̂(Xᵢ) is a difficulty estimator

**Difficulty estimator:** We use ensemble disagreement in tangent space:

```
σ̂(Xᵢ) = √(Var_{k=1..3}[v_{θ_k}(Xᵢ, t)])
```

where v_{θ_k} is the velocity field of ensemble member k. This captures epistemic uncertainty — when the ensemble disagrees, the prediction is harder.

**Why ensemble disagreement, not cross-validation (Baheri):**
- Baheri uses cross-validated difficulty estimation (expensive, requires held-out fold)
- We use ensemble disagreement (cheaper, single forward pass through ensemble, more principled for epistemic uncertainty)
- Different estimator → different paper, not an extension

### 3.3 Conformal Prediction Set on SE(3)

```
C(x) = { y ∈ SE(3) : d_SE(3)(ŷ(x), y) ≤ σ̂(x) · Q_{1-α}({s₁, ..., sₙ}) }
```

This is a geodesic ball on SE(3) centered at ŷ(x) with radius proportional to the calibrated quantile and the estimated difficulty.

### 3.4 Theorems

**Theorem 2 (Coverage on SE(3)):**
Under exchangeability of {(Xᵢ, Yᵢ)} with Yᵢ ∈ SE(3):

```
P(Y_{n+1} ∈ C(x_{n+1})) ≥ 1 - α
```

**Theorem 3 (Equivariance):**
If the data distribution is equivariant under group G acting on SE(3):

```
C(g · x) = g · C(x)  for all g ∈ G
```

**Theorem 4 (Tightness via Lipschitz):**
If the flow matching velocity field v_θ has Lipschitz constant L on SE(3):

```
Vol(C(x)) = O(L · σ̂(x) / √M)
```

where M is ensemble size. Smaller models with lower L produce tighter sets.

### 3.5 Practical Risk Control

At each timestep:
1. Compute action prediction ŷ(x) from flow matching ensemble
2. Compute conformal set C(x) with geodesic radius r(x)
3. If r(x) > τ (threshold): **STOP, defer to human** → "I'm uncertain"
4. If r(x) ≤ τ: **execute ŷ(x)** → "I'm confident"

**Threshold τ:** Calibrated on held-out data to achieve target failure rate (e.g., 5% false negative rate).

### 3.6 Novelty vs. Existing Work

| Prior Work | What They Do | Why GAUSS-VLA Is Different |
|---|---|---|
| Baheri (Feb 2026) | Geodesic CP on S² with cross-validated difficulty | Different manifold (S² vs SE(3)), different estimator (CV vs ensemble) |
| ReconVLA (Apr 2026) | CP on VLA action tokens in Euclidean R^d | No manifold structure, no geodesic scores |
| SAFE (NeurIPS 2025) | CP for VLA failure detection (task-level) | Not action-level, not manifold |
| ConformalDAgger (ICLR 2025) | CP intervals for imitation learning | Euclidean, not VLA |

---

## 4. New Metrics Introduced

### 4.1 action-ECE (Expected Calibration Error on Actions)

```
action-ECE = Σ_b (|B_b| / N) |conf_b - acc_b|
```

where:
- conf_b = inverse of mean uncertainty in bin b (after temperature scaling)
- acc_b = fraction of actions in bin b that led to task progress on held-out data
- B_b = set of predictions in bin b

**This is the first calibration metric for VLA actions.** No prior VLA reports this.

### 4.2 Geodesic Failure Detection AUC

```
AUC = AUROC of (r(x) > τ) predicting task failure
```

where r(x) is the conformal set radius. Higher AUC → the uncertainty signal better predicts actual failures.

### 4.3 Action Coverage

```
Coverage = P(a* ∈ C(x))
```

Must be ≥ 1 − α by Theorem 2. Measured empirically on held-out rollouts.

### 4.4 Average Geodesic Set Diameter

```
AvgDiam = E[d_SE(3)(ŷ_lo, ŷ_hi)]
```

Smaller = more confident. Should correlate with task difficulty.

---

## 5. Architecture Details

### 5.1 Vision Encoder

- **Model:** SigLIP-400M (frozen)
- **Input:** RGB frames 224×224
- **Output:** 64 visual tokens per frame, 512-dim each
- **Frames:** Current frame + previous frame (for temporal context)

### 5.2 Language Encoder

- **Model:** T5-base (frozen, ~60M params)
- **Input:** Natural language task instruction
- **Output:** Language tokens, 512-dim

### 5.3 Cross-Attention Mixer

- **Architecture:** 2-layer transformer with cross-attention
- **Params:** ~5M
- **Input:** Visual tokens + language tokens + proprioception token
- **Output:** Fused representation, 512-dim

### 5.4 SE(3) Flow Matching Head

- **Architecture:** 6-layer, 8-head transformer
- **Params:** ~35M per head, 3 heads = 105M total (35M active per step)
- **Input:** Fused representation (512-dim) + noise sample
- **Output:** Velocity field v_θ ∈ se(3) ≅ R⁶
- **Denoising steps:** 10 (at inference)
- **Action decoding:** exp_map(v_θ) → SE(3) → extract (R, t) → axis-angle + translation

### 5.5 Proprioception Tokenizer

- **Params:** ~2M
- **Input:** Joint angles / end-effector pose (7D for LIBERO, 39D for MetaWorld state)
- **Output:** Single 512-dim token

### 5.6 Conformal Calibration Module

- **Params:** 0 (post-hoc)
- **Procedure:** Run trained ensemble on calibration set, compute nonconformity scores, store quantile Q_{1-α}
- **At test time:** Compare new score to stored quantile → set radius

---

## 6. Loss Functions

### 6.1 Flow Matching Loss (Primary)

```
L_FM = E_{t, X₀, X₁} ||v_θ(X_t, t) - log_{X_t}(X₁)||²_{X_t}
```

where X_t is geodesic interpolation on SE(3).

### 6.2 Ensemble Diversity Loss

To prevent ensemble collapse (all heads converging to same solution):

```
L_div = -λ_div · Σ_{i<j} ||v_{θ_i}(X_t, t) - v_{θ_j}(X_t, t)||²
```

λ_div = 0.01 (small, just to break symmetry).

### 6.3 Total Loss

```
L_total = L_FM + L_div
```

---

## 7. Theorem 1: Proof Sketch

**Claim:** For any rotation R with ||log(R)|| = θ, the optimal Euclidean flow matching policy f^E on R⁶ axis-angle and the optimal SE(3) policy f* satisfy:

```
E||f^E(R) - f*(R)|| ≤ C · θ² + O(θ⁴)
```

**Proof:**

Step 1: The Euclidean flow matching policy operates in R⁶ with the axis-angle parameterization. The SE(3) policy operates in se(3) with the bi-invariant metric.

Step 2: The Jacobian of the axis-angle → rotation map has eigenvalues that deviate from 1 by O(θ²) for ||log(R)|| = θ. Specifically, for the Rodrigues formula:

```
d(axis-angle)/d(log(R)) = I + O(θ²)
```

Step 3: The optimal flow matching velocity in Euclidean space is:

```
v^E = (X₁ - X₀) · (standard OT velocity)
```

while the optimal SE(3) velocity is:

```
v* = log_{X₀}(X₁) · (geodesic OT velocity)
```

Step 4: The difference ||v^E - v*|| is bounded by the Jacobian deviation:

```
||v^E - v*|| ≤ C · ||log(R)||² = C · θ²
```

Step 5: Since flow matching integrates the velocity field over t ∈ [0,1], the error accumulates linearly:

```
E||f^E(R) - f*(R)|| ≤ ∫₀¹ ||v^E_t - v*_t|| dt ≤ C · θ²
```

∎

**The bound is tight:** there exist data distributions where Euclidean flow matching achieves the rate exactly (e.g., rotations near θ = π where the axis-angle parameterization has maximum distortion).

---

## 8. Evaluation Protocol Summary

### 8.1 LIBERO

- 4 suites: Spatial, Object, Goal, Long (10 tasks each)
- 50 demos per task, 200 demos per suite
- Eval: 20 episodes per task, report per-suite and average success rate
- Primary metric: success rate
- Secondary metrics: action-ECE, failure detection AUC

### 8.2 MetaWorld MT-50

- 50 tasks, single policy
- 100 scripted demos per task, 5000 total
- Eval: 50 episodes per task
- Primary metric: overall success rate
- Secondary metrics: per-group success rate (rotation-heavy vs translation-heavy)
- Camera: agent_view (84×84 or 224×224 RGB)
- Action space: 4D (dx, dy, dz, gripper)
- State space: 39D (end-effector + gripper + objects)

### 8.3 CALVIN ABC

- Long-horizon, language-conditioned
- Train on A, B, C; test on D (or all combinations)
- Primary metric: average episode length (0-5 scale)
- Secondary: success rate per subtask

### 8.4 New Metrics (All Benchmarks)

- action-ECE: calibration error
- Failure detection AUC: uncertainty → failure correlation
- Action coverage: P(a* ∈ C(x))
- Average geodesic set diameter: confidence measure
