# 04 — Evaluation Protocol

## 1. Overview

GAUSS-VLA is evaluated on three benchmarks (LIBERO, MetaWorld MT-50, CALVIN) plus four new metrics. This document specifies the exact evaluation procedures to ensure reproducibility.

---

## 2. LIBERO Evaluation

### 2.1 Setup

```python
LIBERO_CONFIG = {
    'suites': ['spatial', 'object', 'goal', 'long'],
    'tasks_per_suite': 10,
    'total_tasks': 40,
    'eval_episodes_per_task': 20,
    'max_steps_per_episode': 300,     # LIBERO default
    'success_threshold': 'task_specific',  # Each task has its own success condition
    'camera': 'agentview',            # Third-person camera
    'resolution': [224, 224],
    'control_frequency': 10,          # Hz
}
```

### 2.2 Evaluation Procedure

```python
def evaluate_libero(checkpoint, conformal_quantile, suite):
    """Evaluate GAUSS-VLA on a LIBERO suite.
    
    For each task:
        1. Load task environment
        2. Run 20 episodes with different initial states
        3. Record: success, per-step uncertainty, conformal set radius
        4. Compute: success rate, action-ECE, failure detection AUC
    """
    results = {}
    for task in suite.tasks:
        task_results = {
            'successes': 0,
            'uncertainties': [],
            'set_radii': [],
            'failures_detected': 0,
            'actual_failures': 0,
        }
        
        for ep in range(20):
            obs = task.reset()
            episode_success = False
            episode_uncertainties = []
            episode_radii = []
            
            for step in range(300):
                # Get image + proprioception
                img = task.render(camera='agentview', resolution=[224, 224])
                proprio = obs[:7]
                
                # Predict action with uncertainty
                action, uncertainty, set_radius = model.predict_with_uncertainty(
                    img, proprio, task.instruction, conformal_quantile
                )
                
                episode_uncertainties.append(uncertainty)
                episode_radii.append(set_radius)
                
                # Execute
                obs, reward, done, info = task.step(action)
                
                if info.get('success', False):
                    episode_success = True
                    break
            
            task_results['successes'] += int(episode_success)
            task_results['uncertainties'].append(episode_uncertainties)
            task_results['set_radii'].append(episode_radii)
            
            # Failure detection: did uncertainty exceed threshold before failure?
            if not episode_success:
                task_results['actual_failures'] += 1
                if any(r > FAILURE_THRESHOLD for r in episode_radii):
                    task_results['failures_detected'] += 1
        
        results[task.name] = {
            'success_rate': task_results['successes'] / 20,
            'mean_uncertainty': np.mean([np.mean(u) for u in task_results['uncertainties']]),
            'mean_set_radius': np.mean([np.mean(r) for r in task_results['set_radii']]),
        }
    
    return results
```

### 2.3 LIBERO Metrics

| Metric | Formula | Target |
|---|---|---|
| Success Rate | successes / total_episodes | ≥93% avg |
| Per-Suite SR | successes_in_suite / episodes_in_suite | Spatial ≥95%, Object ≥95%, Goal ≥90%, Long ≥85% |
| action-ECE | See Section 5.1 | <0.07 |
| Failure Detection AUC | See Section 5.2 | >0.85 |
| Action Coverage | See Section 5.3 | ≥90% |
| Avg Geodesic Set Diameter | See Section 5.4 | Smaller = better |

---

## 3. MetaWorld MT-50 Evaluation

### 3.1 Setup

```python
METAWORLD_CONFIG = {
    'benchmark': 'MT50',
    'tasks': 50,
    'eval_episodes_per_task': 50,
    'max_steps_per_episode': 500,
    'success_threshold': 'task_specific',
    'camera': 'agentview',
    'resolution': [224, 224],
    'control_frequency': 10,
}
```

### 3.2 Task Grouping for Analysis

```python
ROTATION_HEAVY = [
    'door-open-v3', 'door-close-v3', 'drawer-open-v3', 'drawer-close-v3',
    'window-open-v3', 'window-close-v3', 'faucet-open-v3', 'faucet-close-v3',
    'dial-turn-v3', 'door-unlock-v3', 'door-lock-v3',
    'hammer-v3', 'nut-assemble-v3', 'wrench-v3', 'handle-press-side-v3'
]  # 15 tasks

TRANSLATION_HEAVY = [
    'reach-v3', 'reach-wall-v3', 'push-v3', 'push-wall-v3', 'push-back-v3',
    'soccer-v3', 'coffee-push-v3', 'shelf-place-v3', 'plate-slide-v3',
    'plate-slide-side-v3', 'plate-slide-back-v3',
    'button-press-v3', 'button-press-topdown-v3', 'button-press-wall-v3',
    'button-press-topdown-wall-v3', 'coffee-button-v3',
    'handle-press-v3', 'stick-push-v3', 'stick-pull-v3', 'sweep-v3'
]  # 20 tasks

MIXED = [
    'pick-place-v3', 'pick-place-wall-v3', 'pick-out-of-hole-v3',
    'assembly-v3', 'disassemble-v3', 'box-close-v3', 'bin-picking-v3',
    'basketball-v3', 'hand-insert-v3', 'lever-pull-v3', 'rope-v3',
    'sweep-into-v3'
]  # 15 tasks (remaining)
```

### 3.3 Evaluation Procedure

```python
def evaluate_mt50(checkpoint, conformal_quantile):
    """Evaluate on MT-50 benchmark.
    
    Returns:
        per_task: dict mapping task_name -> success_rate
        group_results: dict mapping group_name -> mean_success_rate
        overall: float, mean success across all 50 tasks
        metrics: dict with action-ECE, failure AUC, etc.
    """
    import metaworld
    
    ml50 = metaworld.MT50(seed=42)
    model = load_model(checkpoint)
    
    per_task = {}
    all_uncertainties = []
    all_radii = []
    all_outcomes = []  # 1 = success, 0 = failure
    
    for task_name, env_cls in ml50.train_classes.items():
        env = env_cls()
        task = [t for t in ml50.train_tasks if t.env_name == task_name][0]
        
        successes = 0
        task_uncertainties = []
        task_radii = []
        task_outcomes = []
        
        for ep in range(50):
            env.set_task(task)
            obs = env.reset()
            
            ep_success = False
            ep_uncertainties = []
            ep_radii = []
            
            for step in range(500):
                img = env.render(offscreen=True, width=224, height=224,
                               camera_name='agentview')
                state = obs[:39]
                
                action, uncertainty, set_radius = model.predict_with_uncertainty(
                    img, state, TASK_INSTRUCTIONS[task_name], conformal_quantile
                )
                
                ep_uncertainties.append(uncertainty)
                ep_radii.append(set_radius)
                
                obs, reward, done, info = env.step(action)
                if info.get('success', False):
                    ep_success = True
                    break
            
            successes += int(ep_success)
            task_uncertainties.append(ep_uncertainties)
            task_radii.append(ep_radii)
            task_outcomes.append(int(ep_success))
        
        per_task[task_name] = successes / 50
        all_uncertainties.extend([np.mean(u) for u in task_uncertainties])
        all_radii.extend([np.mean(r) for r in task_radii])
        all_outcomes.extend(task_outcomes)
        
        env.close()
    
    # Group results
    group_results = {
        'rotation_heavy': np.mean([per_task[t] for t in ROTATION_HEAVY]),
        'translation_heavy': np.mean([per_task[t] for t in TRANSLATION_HEAVY]),
        'mixed': np.mean([per_task[t] for t in MIXED]),
        'overall': np.mean(list(per_task.values())),
    }
    
    return per_task, group_results
```

---

## 4. CALVIN Evaluation

### 4.1 Setup

```python
CALVIN_CONFIG = {
    'split': 'ABC',           # Train on A,B,C; test on D
    'eval_episodes': 1000,
    'max_steps_per_episode': 360,
    'language_instruction': True,
    'metric': 'average_length',  # 0-5 scale
}
```

### 4.2 Evaluation

```python
def evaluate_calvin(checkpoint, split='ABC'):
    """Evaluate on CALVIN benchmark.
    
    Returns:
        avg_length: float, average episode length (0-5)
        per_task_sr: dict, success rate per subtask
    """
    # Use official CALVIN evaluation code
    # See: https://github.com/mees/calvin
    pass
```

---

## 5. New Metrics (GAUSS-VLA Specific)

### 5.1 action-ECE (Expected Calibration Error)

```python
def compute_action_ece(predictions, uncertainties, outcomes, n_bins=15):
    """Compute calibration error for action predictions.
    
    Args:
        predictions: (N, ...) predicted actions
        uncertainties: (N,) uncertainty scores (higher = more uncertain)
        outcomes: (N,) binary outcomes (1 = task progress, 0 = failure)
        n_bins: number of bins for calibration curve
    
    Returns:
        ece: float, expected calibration error
    """
    # Convert uncertainties to confidence (inverse)
    confidences = 1.0 / (1.0 + uncertainties)  # (N,)
    
    # Bin by confidence
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        mask = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i+1])
        if mask.sum() == 0:
            continue
        
        bin_conf = confidences[mask].mean()
        bin_acc = outcomes[mask].mean()
        bin_weight = mask.sum() / len(confidences)
        
        ece += bin_weight * abs(bin_conf - bin_acc)
    
    return ece
```

### 5.2 Failure Detection AUC

```python
def compute_failure_detection_auc(set_radii, outcomes):
    """Compute AUC for failure detection using conformal set radius.
    
    Args:
        set_radii: (N,) conformal set radii (higher = more uncertain)
        outcomes: (N,) binary outcomes (1 = success, 0 = failure)
    
    Returns:
        auc: float, area under ROC curve
    """
    from sklearn.metrics import roc_auc_score
    
    # Higher radius should predict failure (outcome = 0)
    # So we negate: failure = 1 when radius is high
    failure_labels = 1 - outcomes
    scores = set_radii  # Higher = more likely failure
    
    return roc_auc_score(failure_labels, scores)
```

### 5.3 Action Coverage

```python
def compute_action_coverage(true_actions, predicted_actions, set_radii, conformal_quantile):
    """Compute fraction of true actions within conformal set.
    
    Args:
        true_actions: (N, ...) ground truth actions
        predicted_actions: (N, ...) predicted actions
        set_radii: (N,) conformal set radii
    
    Returns:
        coverage: float, fraction of true actions in set
    """
    distances = geodesic_distance(predicted_actions, true_actions)  # (N,)
    in_set = distances <= set_radii
    return in_set.mean()
```

### 5.4 Average Geodesic Set Diameter

```python
def compute_avg_set_diameter(set_radii):
    """Average conformal set radius (proxy for set size).
    
    Args:
        set_radii: (N,) conformal set radii
    
    Returns:
        avg_diameter: float
    """
    return 2 * set_radii.mean()  # diameter = 2 × radius
```

---

## 6. Baselines

### 6.1 Baselines to Run

| Baseline | Params | Source | Notes |
|---|---|---|---|
| SmolVLA-450M | 450M | HuggingFace | Primary same-scale baseline |
| ACT | 80M | LeRobot | Fixed H=8, flow matching |
| Diffusion Policy | 30M | LeRobot | Per-step diffusion |
| OpenVLA-OFT | 7B | Official | Larger model, unfair comparison |

### 6.2 Evaluation Protocol for Baselines

Run ALL baselines with:
- Same evaluation episodes (same seeds)
- Same camera views and resolution
- Same success criteria
- Same number of evaluation episodes

For fairness, report baselines with and without our new metrics:
- action-ECE: baselines don't have conformal sets, so use ensemble variance as uncertainty proxy
- Failure detection AUC: same approach
- Coverage: N/A for baselines (no conformal sets)

---

## 7. Statistical Significance

- Report mean ± std over 3 random seeds for all metrics
- Use paired t-test or Wilcoxon signed-rank test for head-to-head comparisons
- Report p-values for key claims (rotation-heavy improvement, calibration improvement)
- Confidence intervals: 95% for all reported numbers
