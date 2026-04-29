# 02 — Training Recipe: LIBERO

## 1. Environment Setup

```bash
# Create environment
conda create -n gauss-vla python=3.10
conda activate gauss-vla

# Core dependencies
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.44.0 accelerate==0.33.0
pip install einops timm==1.0.9 wandb

# Robotics frameworks
pip install lerobot  # HuggingFace LeRobot framework
pip install gymnasium==0.26

# LIBERO
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO && pip install -e .

# MetaWorld (for cross-benchmark eval)
pip install metaworld

# Our codebase
cd gauss-vla && pip install -e .
```

---

## 2. Dataset Preparation

### 2.1 LIBERO Demonstrations

```bash
# Download all LIBERO suites
python scripts/download_libero.py \
  --suites spatial,object,goal,long \
  --save_dir data/libero \
  --n_demos 50

# Expected structure:
# data/libero/
#   spatial/    (10 tasks × 50 demos = 500 episodes)
#   object/     (10 tasks × 50 demos = 500 episodes)
#   goal/       (10 tasks × 50 demos = 500 episodes)
#   long/       (10 tasks × 50 demos = 500 episodes)
# Total: 2000 episodes
```

### 2.2 Data Format

Each episode contains:
- `images`: RGB frames (224×224) from third-person camera + wrist camera
- `actions`: 7D end-effector delta (dx, dy, dz, dRx, dRy, dRz, grip)
- `proprioception`: 7D joint state or end-effector pose
- `language_instruction`: Natural language task description
- `success`: Binary success label

### 2.3 Action Preprocessing for SE(3)

Convert 7D actions to SE(3) representation:

```python
import torch
from scipy.spatial.transform import Rotation

def action_to_se3(action):
    """Convert 7D action to SE(3) representation.
    
    Args:
        action: tensor of shape (..., 7) = (dx, dy, dz, dRx, dRy, dRz, grip)
    Returns:
        translation: (..., 3) = (dx, dy, dz)
        rotation_matrix: (..., 3, 3) from axis-angle (dRx, dRy, dRz)
        grip: (..., 1) gripper value
    """
    translation = action[..., :3]
    axis_angle = action[..., 3:6]
    grip = action[..., 6:7]
    
    # Convert axis-angle to rotation matrix
    angle = torch.norm(axis_angle, dim=-1, keepdim=True)  # (..., 1)
    axis = axis_angle / (angle + 1e-8)  # (..., 3)
    
    # Rodrigues formula
    K = skew_symmetric(axis)  # (..., 3, 3)
    R = torch.eye(3) + torch.sin(angle) * K + (1 - torch.cos(angle)) * K @ K
    
    return translation, R, grip

def skew_symmetric(v):
    """Convert vector to skew-symmetric matrix."""
    # v: (..., 3)
    zeros = torch.zeros_like(v[..., 0])
    return torch.stack([
        zeros, -v[..., 2], v[..., 1],
        v[..., 2], zeros, -v[..., 0],
        -v[..., 1], v[..., 0], zeros
    ], dim=-1).reshape(*v.shape[:-1], 3, 3)
```

### 2.4 Logarithmic Map (SE(3) → se(3))

```python
def log_map_se3(R, t):
    """Compute logarithmic map from SE(3) to se(3).
    
    Args:
        R: (..., 3, 3) rotation matrix
        t: (..., 3) translation
    Returns:
        omega: (..., 3) rotation vector (axis-angle)
        v: (..., 3) translation component in se(3)
    """
    # Rotation: axis-angle from rotation matrix
    angle = torch.acos(torch.clamp((torch.trace(R) - 1) / 2, -1 + 1e-7, 1 - 1e-7))
    # For small angles, use Taylor expansion
    # For larger angles, use the standard formula
    ...
    return omega, v
```

---

## 3. Training Pipeline

### 3.1 Phase 1: Pretrain Flow Matching Heads (Frozen Backbone)

**Goal:** Train the 3 SE(3) flow matching heads with frozen vision/language encoders.

```bash
python train_gauss.py \
  --config configs/libero_phase1.yaml \
  --data_dir data/libero \
  --output_dir checkpoints/libero_phase1 \
  --wandb_project gauss-vla \
  --wandb_run_name libero-phase1
```

**Config: `configs/libero_phase1.yaml`**
```yaml
model:
  vision_encoder:
    name: "google/siglip-base-patch16-224"
    frozen: true
  language_encoder:
    name: "google-t5/t5-base"
    frozen: true
  cross_attention:
    n_layers: 2
    hidden_dim: 512
    n_heads: 8
  proprioception_tokenizer:
    input_dim: 7
    hidden_dim: 512
  flow_matching_head:
    n_heads: 3           # ensemble size
    layers_per_head: 6
    hidden_dim: 512
    n_attn_heads: 8
    output_dim: 6        # se(3) = so(3) ⊕ R³ = 6D
    denoising_steps: 10  # at inference
  lora:
    enabled: true
    rank: 16
    alpha: 32
    target_modules: ["q_proj", "v_proj", "out_proj"]

training:
  steps: 150000
  batch_size: 64
  lr: 1.0e-4
  lr_schedule: cosine
  warmup_steps: 5000
  gradient_clip: 1.0
  weight_decay: 0.01
  
loss:
  lambda_fm: 1.0
  lambda_div: 0.01    # ensemble diversity loss
  lambda_grip: 0.1    # gripper prediction loss (MSE)

eval:
  eval_every: 5000
  n_eval_episodes: 20  # per task
  eval_suite: libero_long  # most informative suite
  save_best: true

compute:
  gpus: 1
  precision: bf16
  gradient_accumulation: 2
```

**Expected time:** ~3-4 days on 1× A100 80GB

### 3.2 Phase 2: Conformal Calibration

**Goal:** Calibrate the conformal prediction set using held-out data.

```bash
python calibrate_conformal.py \
  --checkpoint checkpoints/libero_phase1/best.pt \
  --calibration_data data/libero/calibration_split \
  --alpha 0.1 \                    # 90% coverage target
  --output configs/conformal_quantile.pt
```

**Procedure:**
1. Split training data 80/20 (train/calibration)
2. Run trained ensemble on calibration set
3. Compute nonconformity scores: sᵢ = d_SE(3)(ŷᵢ, yᵢ) / σ̂(xᵢ)
4. Store Q_{1-α} quantile

**Expected time:** ~2-3 hours on 1× A100

### 3.3 Phase 3: Full Fine-tuning (Optional)

**Goal:** Unfreeze backbone and fine-tune end-to-end for maximum performance.

```bash
python train_gauss.py \
  --config configs/libero_phase3.yaml \
  --resume checkpoints/libero_phase1/best.pt \
  --output_dir checkpoints/libero_phase3 \
  --wandb_project gauss-vla \
  --wandb_run_name libero-phase3
```

**Config changes from Phase 1:**
```yaml
model:
  vision_encoder:
    frozen: false      # unfreeze
  language_encoder:
    frozen: false      # unfreeze

training:
  steps: 50000
  lr: 5.0e-5          # lower LR for fine-tuning
  lr_vision: 1.0e-5   # even lower for vision encoder
```

**Expected time:** ~1.5 days on 1× A100

---

## 4. Hyperparameter Search

Key hyperparameters to search:

| Parameter | Range | Default |
|---|---|---|
| Learning rate | 5e-5 to 2e-4 | 1e-4 |
| LoRA rank | 8, 16, 32 | 16 |
| Ensemble size | 2, 3, 5 | 3 |
| Denoising steps | 5, 10, 20 | 10 |
| Batch size | 32, 64, 128 | 64 |
| λ_div | 0.001, 0.01, 0.1 | 0.01 |
| λ_grip | 0.05, 0.1, 0.2 | 0.1 |

---

## 5. Evaluation on LIBERO

```bash
python evaluate.py \
  --checkpoint checkpoints/libero_phase3/best.pt \
  --conformal_quantile configs/conformal_quantile.pt \
  --benchmark libero \
  --suites spatial,object,goal,long \
  --n_episodes 20 \
  --output results/libero_results.json \
  --compute_calibration_metrics true
```

**Metrics to report:**
1. Per-suite success rate (Spatial, Object, Goal, Long)
2. Average success rate
3. action-ECE (our new metric)
4. Failure detection AUC
5. Action coverage (should be ≥ 90%)
6. Average geodesic set diameter

---

## 6. Baselines to Reproduce

| Baseline | Source | How to Reproduce |
|---|---|---|
| SmolVLA-450M | HuggingFace | `lerobot/smolvla` checkpoint, eval on same data |
| ACT | LeRobot | `lerobot/act` checkpoint |
| OpenVLA-OFT | OpenVLA | Use official checkpoint + OFT code |

Run all baselines with identical evaluation protocol (same seeds, same episodes).
