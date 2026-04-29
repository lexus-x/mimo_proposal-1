# 07 — Code Structure

## Repository Layout

```
gauss-vla/
├── README.md                    # Project overview
├── LICENSE                      # MIT License
├── setup.py                     # Package installation
├── requirements.txt             # Dependencies
├── pyproject.toml               # Build config
│
├── configs/                     # Training configs
│   ├── libero_phase1.yaml
│   ├── libero_phase3.yaml
│   ├── metaworld_mt50.yaml
│   └── baselines/
│       ├── smolvla.yaml
│       └── act.yaml
│
├── data/                        # Data scripts (not data itself)
│   ├── download_libero.py
│   ├── collect_metaworld.py
│   ├── preprocess.py
│   └── dataloader.py
│
├── models/                      # Model implementations
│   ├── __init__.py
│   ├── gauss_vla.py            # Main GAUSS-VLA model
│   ├── vision_encoder.py       # SigLIP wrapper
│   ├── language_encoder.py     # T5 wrapper
│   ├── cross_attention.py      # Cross-attention mixer
│   ├── proprioception.py       # Proprioception tokenizer
│   ├── flow_head.py            # SE(3) flow matching head
│   ├── se3_utils.py            # SE(3) math utilities
│   │   ├── exp_map()           # Exponential map se(3) → SE(3)
│   │   ├── log_map()           # Logarithmic map SE(3) → se(3)
│   │   ├── geodesic_distance() # Geodesic distance on SE(3)
│   │   ├── geodesic_interp()   # Geodesic interpolation
│   │   └── skew_symmetric()    # Hat map
│   ├── conformal.py            # Conformal prediction module
│   │   ├── nonconformity_score()
│   │   ├── calibrate()
│   │   ├── prediction_set()
│   │   └── coverage_check()
│   └── ensemble.py             # Ensemble wrapper
│
├── training/                    # Training scripts
│   ├── train.py                # Main training loop
│   ├── train_libero.py         # LIBERO-specific training
│   ├── train_metaworld.py      # MetaWorld-specific training
│   ├── losses.py               # Loss functions
│   │   ├── flow_matching_loss()
│   │   ├── diversity_loss()
│   │   └── gripper_loss()
│   └── optimizer.py            # Optimizer + scheduler
│
├── evaluation/                  # Evaluation scripts
│   ├── evaluate.py             # Main evaluation loop
│   ├── evaluate_libero.py
│   ├── evaluate_metaworld.py
│   ├── evaluate_calvin.py
│   ├── metrics.py              # Metric implementations
│   │   ├── action_ece()
│   │   ├── failure_detection_auc()
│   │   ├── action_coverage()
│   │   └── avg_set_diameter()
│   └── baselines.py            # Baseline evaluation
│
├── scripts/                     # Utility scripts
│   ├── calibrate_conformal.py  # Conformal calibration
│   ├── run_ablation.py         # Ablation studies
│   ├── collect_metaworld.py    # Data collection
│   └── visualize.py            # Visualization tools
│
├── tests/                       # Unit tests
│   ├── test_se3_utils.py
│   ├── test_flow_head.py
│   ├── test_conformal.py
│   └── test_metrics.py
│
├── paper/                       # Paper materials
│   ├── main.tex
│   ├── figures/
│   └── tables/
│
└── checkpoints/                 # Model checkpoints (gitignored)
    └── .gitkeep
```

---

## Key Implementation Files

### 2.1 se3_utils.py — Core SE(3) Mathematics

```python
"""SE(3) Lie group utilities for GAUSS-VLA."""

import torch
import torch.nn.functional as F

def exp_map_se3(omega, v):
    """Exponential map from se(3) to SE(3).
    
    Args:
        omega: (..., 3) rotation vector (axis-angle)
        v: (..., 3) translation vector in se(3)
    Returns:
        R: (..., 3, 3) rotation matrix
        t: (..., 3) translation
    """
    angle = torch.norm(omega, dim=-1, keepdim=True)  # (..., 1)
    axis = omega / (angle + 1e-8)  # (..., 3)
    
    # Rodrigues formula
    K = skew_symmetric(axis)  # (..., 3, 3)
    I = torch.eye(3, device=omega.device).expand_as(K)
    
    sin_angle = torch.sin(angle).unsqueeze(-1)  # (..., 1, 1)
    cos_angle = torch.cos(angle).unsqueeze(-1)  # (..., 1, 1)
    
    R = I + sin_angle * K + (1 - cos_angle) * K @ K
    
    # Translation: V * v where V is the left Jacobian of SO(3)
    # V = I + (1-cos(angle))/angle² * K + (angle - sin(angle))/angle³ * K²
    angle_sq = angle ** 2 + 1e-8
    angle_cu = angle ** 3 + 1e-8
    
    V = (I 
         + ((1 - cos_angle) / angle_sq).unsqueeze(-1) * K 
         + ((angle - torch.sin(angle)) / angle_cu).unsqueeze(-1) * K @ K)
    
    t = (V @ v.unsqueeze(-1)).squeeze(-1)
    
    return R, t


def log_map_se3(R, t):
    """Logarithmic map from SE(3) to se(3).
    
    Args:
        R: (..., 3, 3) rotation matrix
        t: (..., 3) translation
    Returns:
        omega: (..., 3) rotation vector (axis-angle)
        v: (..., 3) translation component
    """
    # Rotation: axis-angle from rotation matrix
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]  # (...)
    angle = torch.acos(torch.clamp((trace - 1) / 2, -1 + 1e-7, 1 - 1e-7))  # (...)
    
    # Handle small angles
    small = angle < 1e-6
    omega = torch.zeros_like(t)
    
    # For non-small angles: omega = angle / (2 * sin(angle)) * (R - R^T)_vee
    R_diff = R - R.transpose(-1, -2)  # (..., 3, 3)
    omega[~small] = (
        angle[~small, None] / (2 * torch.sin(angle[~small, None]) + 1e-8) 
        * vee_map(R_diff[~small])
    )
    
    # For small angles: omega ≈ vee_map(R - I)
    omega[small] = vee_map(R[small] - torch.eye(3, device=R.device))
    
    # Translation: v = V^{-1} * t
    # V^{-1} = I - 0.5 * K + (1/angle² - (1+cos)/(2*angle*sin)) * K²
    K = skew_symmetric(omega / (angle[..., None] + 1e-8))
    I = torch.eye(3, device=R.device).expand_as(K)
    
    # Simplified for small angles
    V_inv = I - 0.5 * K
    v = (V_inv @ t.unsqueeze(-1)).squeeze(-1)
    
    return omega, v


def geodesic_distance_se3(R1, t1, R2, t2):
    """Geodesic distance on SE(3) under bi-invariant metric.
    
    Args:
        R1, R2: (..., 3, 3) rotation matrices
        t1, t2: (..., 3) translations
    Returns:
        dist: (...) geodesic distance
    """
    # Rotation distance: angle of R1^{-1} R2
    R_rel = R1.transpose(-1, -2) @ R2  # (..., 3, 3)
    trace = R_rel[..., 0, 0] + R_rel[..., 1, 1] + R_rel[..., 2, 2]
    rot_dist = torch.acos(torch.clamp((trace - 1) / 2, -1 + 1e-7, 1 - 1e-7))
    
    # Translation distance
    t_rel = t2 - t1  # (..., 3)
    trans_dist = torch.norm(t_rel, dim=-1)
    
    # Combined (weighted)
    dist = torch.sqrt(rot_dist ** 2 + trans_dist ** 2)
    
    return dist


def skew_symmetric(v):
    """Convert vector to skew-symmetric matrix (hat map).
    
    Args:
        v: (..., 3) vector
    Returns:
        K: (..., 3, 3) skew-symmetric matrix
    """
    zeros = torch.zeros_like(v[..., 0])
    K = torch.stack([
        zeros, -v[..., 2], v[..., 1],
        v[..., 2], zeros, -v[..., 0],
        -v[..., 1], v[..., 0], zeros
    ], dim=-1).reshape(*v.shape[:-1], 3, 3)
    return K


def vee_map(K):
    """Convert skew-symmetric matrix to vector (vee map).
    
    Args:
        K: (..., 3, 3) skew-symmetric matrix
    Returns:
        v: (..., 3) vector
    """
    return torch.stack([
        K[..., 2, 1], K[..., 0, 2], K[..., 1, 0]
    ], dim=-1)
```

### 2.2 flow_head.py — SE(3) Flow Matching Head

```python
"""SE(3) Flow Matching Head for GAUSS-VLA."""

import torch
import torch.nn as nn
from models.se3_utils import exp_map_se3, log_map_se3, geodesic_distance_se3

class SE3FlowMatchingHead(nn.Module):
    """Flow matching head that predicts in se(3) tangent space."""
    
    def __init__(self, hidden_dim=512, n_layers=6, n_heads=8, output_dim=6):
        super().__init__()
        self.output_dim = output_dim  # 6 for se(3) = so(3) ⊕ R³
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Flow matching transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output projection to se(3)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, t, context):
        """
        Args:
            x: (B, D) current noisy action in se(3)
            t: (B, 1) diffusion time
            context: (B, L, hidden_dim) context from VLM backbone
        Returns:
            v: (B, output_dim) predicted velocity in se(3)
        """
        # Time embedding
        t_emb = self.time_embed(t)  # (B, hidden_dim)
        
        # Combine: [t_emb, x_proj, context]
        x_proj = self.output_proj.weight @ x.T  # Simplified projection
        seq = torch.cat([
            t_emb.unsqueeze(1),
            x.unsqueeze(1),
            context
        ], dim=1)  # (B, 1+1+L, hidden_dim)
        
        # Transformer
        out = self.transformer(seq)  # (B, 1+1+L, hidden_dim)
        
        # Take the x position output
        v = self.output_proj(out[:, 1, :])  # (B, output_dim)
        
        return v
    
    def sample(self, context, n_steps=10):
        """Sample action by integrating the flow.
        
        Args:
            context: (B, L, hidden_dim) context
            n_steps: number of integration steps
        Returns:
            action: (B, output_dim) sampled action in se(3)
        """
        B = context.shape[0]
        
        # Start from noise
        x = torch.randn(B, self.output_dim, device=context.device)
        
        # Euler integration
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((B, 1), i * dt, device=context.device)
            v = self.forward(x, t, context)
            x = x + v * dt
        
        return x
```

### 2.3 conformal.py — Conformal Prediction Module

```python
"""Conformal Prediction for SE(3) action sets."""

import torch
import numpy as np
from models.se3_utils import geodesic_distance_se3

class ConformalPredictor:
    """Post-hoc conformal calibration for VLA actions on SE(3)."""
    
    def __init__(self, alpha=0.1):
        """
        Args:
            alpha: miscoverage level (1-alpha = coverage target)
        """
        self.alpha = alpha
        self.quantile = None
    
    def calibrate(self, ensemble, calibration_loader):
        """Calibrate using held-out data.
        
        Args:
            ensemble: list of 3 flow matching heads
            calibration_loader: DataLoader for calibration set
        """
        scores = []
        
        for batch in calibration_loader:
            images, proprio, actions_gt = batch
            
            # Get ensemble predictions
            predictions = []
            for head in ensemble:
                pred = head.sample(context=images)
                predictions.append(pred)
            
            # Compute ensemble disagreement (difficulty estimate)
            disagreement = torch.stack(predictions).std(dim=0)  # (B, D)
            sigma = disagreement.norm(dim=-1)  # (B,)
            
            # Compute nonconformity scores
            for i in range(len(images)):
                # Geodesic distance between prediction and ground truth
                pred_mean = torch.stack([p[i] for p in predictions]).mean(dim=0)
                dist = geodesic_distance_se3(
                    pred_mean[:9].reshape(3, 3), pred_mean[9:12],
                    actions_gt[i][:9].reshape(3, 3), actions_gt[i][9:12]
                )
                
                # Normalized score
                score = dist / (sigma[i] + 1e-8)
                scores.append(score.item())
        
        # Compute quantile
        scores = np.array(scores)
        n = len(scores)
        q_level = np.ceil((1 - self.alpha) * (n + 1)) / n
        self.quantile = np.quantile(scores, q_level)
        
        return self.quantile
    
    def prediction_set_radius(self, ensemble_prediction, ensemble_disagreement):
        """Compute conformal prediction set radius.
        
        Args:
            ensemble_prediction: (B, D) mean prediction
            ensemble_disagreement: (B,) ensemble std
        
        Returns:
            radius: (B,) geodesic radius of prediction set
        """
        return ensemble_disagreement * self.quantile
    
    def is_uncertain(self, radius, threshold):
        """Check if prediction is too uncertain to execute.
        
        Args:
            radius: (B,) prediction set radius
            threshold: float, maximum acceptable radius
        
        Returns:
            uncertain: (B,) boolean mask
        """
        return radius > threshold
```

---

## 3. Installation

```bash
# Clone repository
git clone https://github.com/lexus-x/gauss-vla.git
cd gauss-vla

# Install dependencies
pip install -e .

# Run tests
python -m pytest tests/

# Download data
python scripts/download_libero.py --save_dir data/libero
python scripts/collect_metaworld.py --save_dir data/metaworld_mt50

# Train
python training/train.py --config configs/libero_phase1.yaml

# Evaluate
python evaluation/evaluate.py --checkpoint checkpoints/best.pt
```
