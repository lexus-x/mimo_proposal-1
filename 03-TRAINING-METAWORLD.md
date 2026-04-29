# 03 — Training Recipe: MetaWorld MT-50

## 1. MetaWorld Environment Details

### 1.1 Robot

- **Robot:** Sawyer (7-DOF arm)
- **Control:** Cartesian displacement control (end-effector space)
- **Control frequency:** 10 Hz

### 1.2 Action Space

```
Box(-1.0, 1.0, (4,), float32)

Index 0: dx — End-effector displacement in x (meters)
Index 1: dy — End-effector displacement in y (meters)
Index 2: dz — End-effector displacement in z (meters)
Index 3: gripper — Gripper control (-1 = open, +1 = close)
```

**Note:** MetaWorld uses 4D actions (3D translation + gripper), NOT 7D end-effector with rotation. The SE(3) flow matching contribution applies to the translation component (ℝ³). The gripper is handled as a separate scalar prediction. For rotation-heavy tasks (dial-turn, door-unlock, faucet), the rotation is implicitly achieved through the 3D trajectory, not explicit rotation control.

### 1.3 State/Observation Space

```
39-dimensional vector:

Indices 0:3   — End-effector position (x, y, z)
Index  3      — Gripper state (how open/closed)
Indices 4:7   — Object 1 position (x, y, z)
Indices 7:11  — Object 1 quaternion (w, x, y, z)
Indices 11:14 — Object 2 position (x, y, z) [placeholder if single-object task]
Indices 14:18 — Object 2 quaternion (w, x, y, z) [placeholder if single-object task]
Indices 18:39 — Padding/one-hot task ID (for MT-10/MT-50 benchmarks)
```

**For VLA input:** We use the 39D state vector as proprioception input AND render RGB images from cameras.

### 1.4 Camera Setup

MetaWorld provides two camera views:

**Camera 1: Agent View (Primary)**
```python
camera_config = {
    'name': 'agentview',
    'type': 'fixed',
    'lookat': [0.0, 0.75, 0.15],    # Center of workspace
    'distance': 1.0,                  # Distance from lookat
    'elevation': -30,                 # Degrees below horizontal (looking down)
    'azimuth': 180,                   # Facing robot
    'fov': 45,                        # Field of view
}
# Resolution: 84×84 (default) or 224×224 (for VLA)
# This is the PRIMARY camera for VLA input
```

**Camera 2: Side View (Secondary)**
```python
camera_config = {
    'name': 'sideview', 
    'type': 'fixed',
    'lookat': [0.0, 0.75, 0.15],
    'distance': 1.5,
    'elevation': 0,                   # Horizontal
    'azimuth': 90,                    # Side view
    'fov': 45,
}
# Resolution: 84×84
# Optional: used for evaluation visualization, not primary input
```

**Camera 3: Wrist Camera (Simulated)**
```python
# MetaWorld doesn't natively have a wrist camera
# We simulate it by attaching a camera to the end-effector
camera_config = {
    'name': 'wrist',
    'type': 'track',
    'track_body': 'hand',             # Attached to end-effector
    'distance': 0.3,
    'elevation': -15,                 # Slightly looking down
    'azimuth': 0,
    'fov': 60,
}
# Resolution: 84×84
# Shows close-up of gripper and target object
```

### 1.5 Rendering Configuration

```python
render_config = {
    'offscreen': True,                 # No display needed
    'width': 224,                      # Match VLA input resolution
    'height': 224,
    'camera_id': 'agentview',          # Primary camera
}
```

---

## 2. Dataset Collection

### 2.1 Scripted Expert Demonstrations

MetaWorld provides built-in scripted policies for each task. Use these to collect demonstrations:

```python
import metaworld
import numpy as np
from metaworld.policies import SawyerReachV3Policy  # etc.

def collect_demos(task_name, n_demos=100, max_steps=500):
    """Collect scripted demonstrations for a single task.
    
    Args:
        task_name: e.g., 'reach-v3', 'push-v3', etc.
        n_demos: number of demonstrations per task
        max_steps: max steps per episode
        
    Returns:
        demos: list of dicts with keys:
            - observations: (T, 39) state vectors
            - actions: (T, 4) actions
            - images: (T, 224, 224, 3) RGB frames
            - success: bool
    """
    ml1 = metaworld.ML1(task_name, seed=42)
    env = ml1.train_classes[task_name]()
    task = ml1.train_tasks[0]
    
    demos = []
    for i in range(n_demos):
        env.set_task(task)
        obs = env.reset()
        
        episode_obs = []
        episode_actions = []
        episode_images = []
        
        policy = get_policy(task_name)  # Scripted policy
        
        for step in range(max_steps):
            action = policy.get_action(obs[:39])  # Use state, not image
            next_obs, reward, done, info = env.step(action)
            
            # Render image
            img = env.render(
                offscreen=True, 
                width=224, 
                height=224, 
                camera_name='agentview'
            )
            
            episode_obs.append(obs[:39])
            episode_actions.append(action)
            episode_images.append(img)
            
            obs = next_obs
            if info.get('success', False):
                break
        
        demos.append({
            'observations': np.array(episode_obs),
            'actions': np.array(episode_actions),
            'images': np.array(episode_images),
            'success': info.get('success', False)
        })
    
    env.close()
    return demos
```

### 2.2 MT-50 Task List with Groups

```python
MT50_TASKS = {
    # Group A — REACH/TOUCH (8 tasks)
    'reach': 'reach-v3',
    'reach_wall': 'reach-wall-v3',
    'button_press_topdown': 'button-press-topdown-v3',
    'button_press': 'button-press-v3',
    'button_press_topdown_wall': 'button-press-topdown-wall-v3',
    'button_press_wall': 'button-press-wall-v3',
    'coffee_button': 'coffee-button-v3',
    'handle_press': 'handle-press-v3',
    
    # Group B — PUSH/SLIDE (9 tasks)
    'push': 'push-v3',
    'push_wall': 'push-wall-v3',
    'push_back': 'push-back-v3',
    'soccer': 'soccer-v3',
    'coffee_push': 'coffee-push-v3',
    'shelf_place': 'shelf-place-v3',
    'plate_slide': 'plate-slide-v3',
    'plate_slide_side': 'plate-slide-side-v3',
    'plate_slide_back': 'plate-slide-back-v3',
    
    # Group C — PICK/PLACE (7 tasks)
    'pick_place': 'pick-place-v3',
    'pick_place_wall': 'pick-place-wall-v3',
    'pick_out_of_hole': 'pick-out-of-hole-v3',
    'assembly': 'assembly-v3',
    'disassemble': 'disassemble-v3',
    'box_close': 'box-close-v3',
    'bin_picking': 'bin-picking-v3',
    
    # Group D — OPEN/CLOSE (10 tasks)
    'door_open': 'door-open-v3',
    'door_close': 'door-close-v3',
    'drawer_open': 'drawer-open-v3',
    'drawer_close': 'drawer-close-v3',
    'window_open': 'window-open-v3',
    'window_close': 'window-close-v3',
    'faucet_open': 'faucet-open-v3',
    'faucet_close': 'faucet-close-v3',
    'handle_press_side': 'handle-press-side-v3',
    'door_lock': 'door-lock-v3',
    
    # Group E — COMPLEX/MULTI-STEP (16 tasks)
    'hammer': 'hammer-v3',
    'peg_insert_side': 'peg-insert-side-v3',
    'peg_unplug_side': 'peg-unplug-side-v3',
    'stick_push': 'stick-push-v3',
    'stick_pull': 'stick-pull-v3',
    'basketball': 'basketball-v3',
    'hand_insert': 'hand-insert-v3',
    'lever_pull': 'lever-pull-v3',
    'rope': 'rope-v3',
    'sweep': 'sweep-v3',
    'sweep_into': 'sweep-into-v3',
    'dial_turn': 'dial-turn-v3',
    'door_unlock': 'door-unlock-v3',
    'nut_assemble': 'nut-assemble-v3',
    'wrench': 'wrench-v3',
}
```

**Rotation-heavy tasks (Group D + select Group E, ~15 tasks):**
- door-open, door-close, drawer-open, drawer-close, window-open, window-close
- faucet-open, faucet-close, dial-turn, door-unlock, door-lock
- hammer, nut-assemble, wrench, handle-press-side

These tasks involve significant end-effector reorientation and are where SE(3) flow matching should show the largest gains.

### 2.3 Full Collection Script

```bash
python scripts/collect_metaworld.py \
  --benchmark MT50 \
  --n_demos_per_task 100 \
  --max_steps 500 \
  --resolution 224 \
  --camera agentview \
  --save_dir data/metaworld_mt50 \
  --n_workers 8  # parallel collection
```

**Expected output:**
```
data/metaworld_mt50/
  reach-v3/
    demo_000.npz    # observations, actions, images, success
    demo_001.npz
    ...
  push-v3/
    ...
  ... (50 task directories)
  metadata.json      # task list, groups, camera config
```

**Expected collection time:** ~4-6 hours on 8 CPU cores

---

## 3. Training

### 3.1 Data Preprocessing

```python
def preprocess_metaworld_episode(episode):
    """Convert MetaWorld episode to VLA training format.
    
    Args:
        episode: dict with 'observations', 'actions', 'images'
    Returns:
        processed: dict ready for VLA training
    """
    obs = episode['observations']     # (T, 39)
    actions = episode['actions']       # (T, 4)
    images = episode['images']         # (T, 224, 224, 3)
    
    # Proprioception: use end-effector + gripper (4D)
    proprio = obs[:, :4]               # (x, y, z, gripper)
    
    # Language instruction: task-specific
    instruction = TASK_INSTRUCTIONS[task_name]  # e.g., "push the puck to the goal"
    
    # Actions: 4D (dx, dy, dz, gripper)
    # Note: No explicit rotation in MetaWorld actions
    # SE(3) applies to translation component only
    
    return {
        'images': images,
        'proprio': proprio,
        'actions': actions,
        'language': instruction,
    }
```

### 3.2 Training Config for MetaWorld

```yaml
# configs/metaworld_mt50.yaml
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
    input_dim: 4            # MetaWorld: (x, y, z, gripper)
    hidden_dim: 512
  flow_matching_head:
    n_heads: 3
    layers_per_head: 6
    hidden_dim: 512
    n_attn_heads: 8
    output_dim: 4            # MetaWorld: (dx, dy, dz, gripper)
    denoising_steps: 10
    # Note: output_dim=4 for MetaWorld (not 6 as in LIBERO)
    # The flow matching operates in R⁴ for MetaWorld
    # SE(3) structure applies to translation (R³), gripper is separate
  lora:
    enabled: true
    rank: 16
    alpha: 32

training:
  steps: 300000              # More steps for 50-task multi-task
  batch_size: 256            # Larger batch for multi-task stability
  lr: 3.0e-4                 # Higher LR for multi-task
  lr_schedule: cosine
  warmup_steps: 10000
  gradient_clip: 1.0
  
  # Multi-task specific
  task_sampling: uniform     # Sample tasks uniformly
  task_id_embedding: true    # Append one-hot task ID to state
  
loss:
  lambda_fm: 1.0
  lambda_div: 0.01
  lambda_grip: 0.2           # Higher weight for gripper in MetaWorld

eval:
  eval_every: 10000
  n_eval_episodes: 50         # Per task
  save_best: true
  eval_metric: mt50_success   # Overall MT-50 success rate
```

### 3.3 Training Command

```bash
python train_gauss.py \
  --config configs/metaworld_mt50.yaml \
  --data_dir data/metaworld_mt50 \
  --benchmark metaworld \
  --output_dir checkpoints/mt50 \
  --wandb_project gauss-vla \
  --wandb_run_name mt50-phase1
```

**Expected time:** ~5-7 days on 1× A100 80GB

### 3.4 Conformal Calibration for MetaWorld

```bash
python calibrate_conformal.py \
  --checkpoint checkpoints/mt50/best.pt \
  --calibration_data data/metaworld_mt50/calibration_split \
  --alpha 0.1 \
  --benchmark metaworld \
  --output configs/mt50_conformal_quantile.pt
```

---

## 4. Evaluation

### 4.1 Standard MT-50 Evaluation

```python
import metaworld

def evaluate_mt50(checkpoint, n_episodes=50):
    """Evaluate on MT-50 benchmark.
    
    Returns:
        results: dict with per-task and overall success rates
    """
    ml50 = metaworld.MT50(seed=42)
    model = load_model(checkpoint)
    
    results = {}
    for task_name, env_cls in ml50.train_classes.items():
        env = env_cls()
        task = [t for t in ml50.train_tasks if t.env_name == task_name][0]
        
        successes = 0
        for ep in range(n_episodes):
            env.set_task(task)
            obs = env.reset()
            
            for step in range(500):
                # Get image observation
                img = env.render(offscreen=True, width=224, height=224, 
                               camera_name='agentview')
                
                # Get state observation
                state = obs[:39]
                
                # Predict action
                action = model.predict(img, state, task_instruction)
                
                obs, reward, done, info = env.step(action)
                if info.get('success', False):
                    successes += 1
                    break
        
        results[task_name] = successes / n_episodes
        env.close()
    
    # Compute group averages
    results['group_a_reach'] = np.mean([results[t] for t in GROUP_A])
    results['group_b_push'] = np.mean([results[t] for t in GROUP_B])
    results['group_c_pick'] = np.mean([results[t] for t in GROUP_C])
    results['group_d_open'] = np.mean([results[t] for t in GROUP_D])  # rotation-heavy
    results['group_e_complex'] = np.mean([results[t] for t in GROUP_E])
    results['overall'] = np.mean(list(results.values())[:50])
    results['rotation_heavy'] = np.mean([results[t] for t in ROTATION_HEAVY_TASKS])
    
    return results
```

### 4.2 Metrics to Report

| Metric | Description | Target |
|---|---|---|
| MT-50 Overall | Average success across 50 tasks | 68-78% |
| Rotation-Heavy (15 tasks) | Group D + select Group E | 62-70% |
| Translation-Heavy (20 tasks) | Group A + B | 70-78% |
| Mixed (15 tasks) | Group C + remaining E | 65-75% |
| action-ECE | Calibration error | <0.07 |
| Failure Detection AUC | Uncertainty → failure | >0.85 |

### 4.3 Per-Task Success Rate Table

Report per-task success rates for all 50 tasks. Group by:
1. Group (A-E)
2. Rotation-heavy vs translation-heavy
3. Single-object vs two-object tasks

---

## 5. MetaWorld-Specific Considerations

### 5.1 Task ID Conditioning

MT-50 observations include one-hot task IDs (indices 18:39 in the 39D state). Two approaches:

**Option A: Use task ID (standard)**
- Append one-hot task ID to proprioception input
- Simpler, standard approach
- Used by most MT-50 papers

**Option B: Language-only conditioning (VLA style)**
- Don't use task ID, rely only on language instruction
- More realistic for VLA setting
- Harder but more general

**Recommendation:** Report both. Option A for fair comparison with MT-50 baselines. Option B to show VLA capability.

### 5.2 Gripper Handling

MetaWorld tasks have different gripper requirements:
- Tasks that need gripper: pick-place, assembly, disassemble, etc.
- Tasks that don't: push, reach, button-press, etc.

**Strategy:** Always predict gripper. For tasks that don't need it, the gripper prediction is ignored (masked). The gripper loss is weighted by whether the task requires gripper action.

### 5.3 Reward Shaping

MetaWorld provides dense rewards for each task. For VLA training, we use:
- **Imitation loss** (flow matching on demonstrations) — primary
- **NOT using reward signal** — we're doing imitation learning, not RL

For evaluation, we use the environment's built-in success criterion.

### 5.4 Episode Length

MetaWorld episodes are max 500 steps (50 seconds at 10Hz). Most tasks complete in 50-200 steps. Scripted experts typically solve in 30-100 steps.

### 5.5 Seed and Reproducibility

```python
# Always use fixed seeds
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# MetaWorld environment seeding
env.reset(seed=SEED)
```

Report mean ± std over 3 random seeds for all results.

---

## 6. Baselines for MetaWorld

| Baseline | How to Run | Expected |
|---|---|---|
| ACT | LeRobot framework | ~50% |
| Diffusion Policy | LeRobot framework | ~55% |
| SmolVLA-450M | HuggingFace checkpoint | ~55-62% |
| OpenVLA-OFT | Official checkpoint | ~70% (7B, unfair comparison) |

**Fair comparison:** Only compare against sub-500M models in main table. Report 7B+ models in appendix.
