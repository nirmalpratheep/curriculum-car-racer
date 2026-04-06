# Training the Curriculum Car Racer

> The agent starts on a wide oval, barely able to keep from flying off the road.
> By the end it threads two-car-width choke points at speed, brakes for hairpins,
> and finds the shortest racing line through complex polygon circuits — all from a
> single 64x64 image and seven sensor readings, learned entirely from reward.

There are four files that matter:

```
run_training.py <- single command: launches, monitors, auto-fixes, restarts
train.py        <- PPO training loop (called by run_training.py)
training.md     <- this file (guidance, tuning, monitoring)
monitor.py      <- standalone 50k-step health checks (optional, for manual runs)
```

---

## Quick Start — Fully Automated (Recommended)

Goal: master all 10 tracks with fast laps, short racing lines, zero crashes.
One command handles everything — monitoring, diagnosis, and automatic restarts
with tuned fixes when the agent stalls.

```bash
# Single command — trains, monitors, auto-fixes, advances through all 10 tracks
uv run python run_training.py

# With custom budget
uv run python run_training.py --total-steps 10_000_000

# Fresh start (ignore existing checkpoints)
uv run python run_training.py --fresh

# Dry run (print commands without executing)
uv run python run_training.py --dry-run
```

`run_training.py` does the following automatically:
1. Launches `train.py` with optimized hyperparameters
2. Polls W&B metrics every 60 seconds
3. Runs health checks at every 50K step boundary (PASS/FAIL)
4. Detects curriculum stalls (no advancement for 200K steps)
5. Diagnoses the failure mode (NOT_STEERING, NOT_PROGRESSING, etc.)
6. Kills training, applies the right fix, and resumes from the latest checkpoint
7. Stops when all 10 tracks are mastered or the step budget is exhausted

### Manual Training (Advanced)

If you need direct control over hyperparameters, run `train.py` directly
and `monitor.py` in a second terminal:

```bash
# Terminal 1 — Training
uv run python train.py \
    --total-steps 10_000_000 \
    --num-envs 8 \
    --subproc \
    --compile \
    --rollout-steps 4096 \
    --ppo-epochs 4 \
    --minibatch-size 512 \
    --lr 3e-4 \
    --lr-min 1e-5 \
    --gamma 0.997 \
    --gae-lambda 0.95 \
    --clip-eps 0.2 \
    --vf-coef 0.5 \
    --ent-coef-start 0.005 \
    --ent-coef-end 0.001 \
    --max-grad-norm 0.5 \
    --target-kl 0.015 \
    --threshold 18.0 \
    --window 20 \
    --replay-frac 0.3 \
    --video-interval 25_000 \
    --checkpoint-interval 250_000 \
    --keep-checkpoints 5

# Terminal 2 — Monitor (auto-detects latest W&B run)
uv run python monitor.py

# Quick test (debugging, not full training)
uv run python train.py \
    --total-steps 500_000 --num-envs 4 \
    --video-interval 25_000 --checkpoint-interval 100_000

# Resume from checkpoint
uv run python train.py --resume checkpoints/ppo_step02500000_lvl05.pt \
    --total-steps 10_000_000 --num-envs 8 --subproc --compile \
    --video-interval 25_000 --keep-checkpoints 5

# Offline (no internet)
uv run python train.py --wandb-offline [other flags]
```

After training starts a W&B URL is printed in the terminal. Open it in a
browser to watch every metric in real time. If you ran with `--wandb-offline`,
sync later with `wandb sync wandb/`.

### Test a Checkpoint Locally

```bash
# Random policy (sanity check — renders one episode to MP4)
uv run python test_video.py

# Trained model
uv run python test_video.py --checkpoint checkpoints/ppo_step00250000_lvl01.pt --track 1

# Hard track
uv run python test_video.py --checkpoint checkpoints/ppo_step05000000_lvl08.pt --track 17
```

### Load a Checkpoint for Inference

```python
import torch
from train import PPOActorCritic

ckpt  = torch.load("checkpoints/ppo_final.pt", map_location="cpu")
model = PPOActorCritic()
model.load_state_dict(ckpt["model"])
model.eval()
```

---

## What the Agent Is Optimising

The reward function has six components that directly map to the training objectives:

| # | Term | When | Value | Objective |
|---|------|------|-------|-----------|
| 1 | On-track speed | Every step on road | `max(0, speed/max_speed) * 0.20` | Must move forward to earn reward (prevents zero-speed plateau) |
| 1b | Off-track | Every step off road | `-0.1` | Penalize leaving road |
| 3 | Crash event | On-track to off-track transition | `-1.0` | Fewer crashes |
| 4 | **Waypoint progress** | Every step, on-track only | `+advance * (15.0 / n_waypoints)` | **Dense directional signal toward lap completion** |
| 5 | Lap completion | Gate crossed with speed > 0.3 | `+50 * time_ratio * dist_ratio` | Faster time, shorter distance |
| 6 | Out of bounds | Terminal | `-100` | Stay on screen |

### Waypoint Progress (dense lap signal)

The track centerline is defined by waypoints forming a closed polygon. Each step, the nearest waypoint is found. Advancing forward around the track earns `+5.0 / n_waypoints`; going backward earns the same as a penalty. Only counted while on-track (prevents off-road shortcuts).

- Full lap of forward progress = +15.0 (consistent across all tracks, comparable to speed reward)
- Track 1 (81 waypoints): +0.185 per waypoint advance
- Track 13 (8 waypoints): +1.875 per waypoint advance

This solves the sparse reward problem — without it, the only lap signal is the +50 completion bonus, giving the agent no gradient toward moving *around* the track.

### Lap Completion

```
time_ratio = clamp(par_time_steps / actual_lap_steps,  0.5, 2.0)
dist_ratio = clamp(optimal_dist   / actual_lap_dist,   0.5, 1.0)
lap_bonus  = 50 * time_ratio * dist_ratio
```

- `par_time_steps`: expected lap time at 70% max speed
- `optimal_dist`: track centerline perimeter (theoretical shortest on-track path)
- Faster than par: `time_ratio > 1` (up to 2x bonus for fast laps)
- `dist_ratio` capped at 1.0: no bonus for corner cutting (off-track distance excluded)
- Worst case: `50 * 0.5 * 0.5 = 12.5` just for finishing

### How Objectives Emerge

| Goal | How it's achieved |
|------|------------------|
| Fewer crashes | `-1.0` crash penalty + `all(crashes==0)` advancement gate |
| Shorter distance | `dist_ratio = optimal/actual` — tighter racing line scores higher |
| Faster lap time | `time_ratio = par/actual` — beating par rewarded up to 2x |
| Lap completion | Waypoint progress gives dense gradient; +50 bonus on completion |

Complexity scales the curriculum threshold only, NOT the reward — keeping value-function targets comparable when easy and hard tracks mix in the same rollout buffer.

### Expected Rewards Per Episode

At 70% max speed, 1 clean lap at par time (speed reward ≈ 0.14/step on-track):

| Track | Level | Complexity | Est. 1-lap | Eff. threshold | Passes? |
|-------|-------|-----------|----------:|---------------:|---------|
| Wide Oval | 1 | 1.00 | ~70 | 18 | yes |
| Standard Oval | 2 | 1.58 | ~63 | 28 | yes |
| Rounded Rectangle | 5 | 1.49 | ~66 | 27 | yes |
| Stadium Oval | 6 | 1.92 | ~61 | 35 | yes |
| Hairpin Track | 9 | 1.79 | ~64 | 32 | yes |
| Chicane Track | 10 | 1.92 | ~63 | 35 | yes |
| L-Shape Circuit | 13 | 2.13 | ~68 | 38 | yes |
| T-Notch Circuit | 14 | 2.64 | ~69 | 48 | yes |
| Two-Strait Bottleneck | 17 | 5.23 | ~67 | 94 | 2 laps needed |
| Bottleneck Circuit | 18 | 4.72 | ~66 | 85 | 2 laps needed |

Tracks 17-18 have very high complexity (narrow chokes). The agent needs 2+ clean laps to exceed the threshold, which rewards consistent driving.

Note: with the speed-only on-track reward (no free +0.05/step), there is no
incentive to stay still. The agent must drive forward to earn any reward.

---

## Curriculum

Training progresses through 10 tracks across 5 difficulty tiers:

```
Tier A — Easy ovals           tracks  1, 2     Wide Oval, Standard Oval
Tier B — Rectangular shapes   tracks  5, 6     Rounded Rect, Stadium Oval
Tier C — Hairpins & chicanes  tracks  9, 10    Hairpin, Chicane
Tier D — Complex polygons     tracks 13, 14    L-Shape, T-Notch
Tier E — Choke + catch-up     tracks 17, 18    Two-Strait Bottleneck, Bottleneck Circuit
```

**TRAIN** (10 tracks): 1,2, 5,6, 9,10, 13,14, 17,18 — curriculum progression.
**VAL** (5 tracks): 3, 7, 11, 15, 19 — gating after each advance, never trained on.
**TEST** (5 tracks): 4, 8, 12, 16, 20 — held out entirely, run once at the end.

### How Advancement Works

The agent spends 70% of episodes on the current *frontier* track and 30% on
randomly sampled *mastered* tracks (anti-forgetting replay). It advances when
**all three** conditions are met over the last `--window` (30) episodes:

1. **Every** episode completed at least 1 lap
2. **Every** episode had zero crashes
3. Mean reward exceeds `threshold * track.complexity`

```
effective_threshold = --threshold * track.complexity
```

Each advancement triggers a validation run on all 5 VAL tracks. Results appear
in W&B as `val/mean_reward` and a per-track table.

### Expected Progression Timeline

| Steps | Expected Level | Tracks Mastered |
|-------|---------------|----------------|
| 0 - 200K | 1 | Wide Oval |
| 200K - 500K | 2-3 | Standard Oval, Rounded Rectangle |
| 500K - 1.5M | 4-6 | Stadium Oval, Hairpin, Chicane |
| 1.5M - 3M | 7-8 | L-Shape, T-Notch |
| 3M - 6M | 9 | Two-Strait Bottleneck |
| 6M - 10M | 10 | Bottleneck Circuit |

If `curriculum/level` stalls for 200K+ steps, `run_training.py` automatically:
1. Loads the latest checkpoint and runs a greedy episode
2. Diagnoses the behavior (see Diagnosis Categories below)
3. Applies the appropriate fix and restarts
4. Escalates through 4 fix levels if stalls persist

See the failure modes section for manual intervention.

---

## Model Architecture

```
ImpalaCNN(image 3x64x64)
    -> 256-d feature map
MLP(7 scalars)
    -> 32-d scalar embedding
Concat -> 288-d feature vector
    |-> actor_mean   Linear(288, 2)   -> [accel, steer] mean
    |   actor_log_std  Parameter(2)   -> shared log std
    |   -> Normal distribution -> sample action
    |-> critic        Linear(288, 1)  -> scalar value estimate
```

- **Actions**: `[accel, steer]` continuous, clamped to `[-1, 1]`
- **Distribution**: Gaussian with learned mean, shared log-std parameter (clamped to [-3.0, -0.3], std in [0.05, 0.74])
- **Init**: orthogonal weights (gain=0.01 for actor, gain=1.0 for critic), log-std starts at -1.0 (std=0.37)

### Observation

| Component | Shape | Description |
|-----------|-------|-------------|
| `image` | `(3, 64, 64)` | Headlight cone crop, RGB, normalised to `[0, 1]` |
| `scalars` | `(7,)` | `[angular_velocity, speed, ray_left, ray_front_left, ray_front, ray_front_right, ray_right]` |

---

## Hyperparameter Tuning Rationale

### PPO Core

| Parameter | Value | Why |
|-----------|-------|-----|
| `total-steps` | 10,000,000 | 10 tracks need ~1M/track on average |
| `num-envs` | 8 | Saturates CPU cores; effective batch = 4096*8 = 32768 |
| `subproc` | on | Parallel env stepping for GPU utilisation |
| `rollout-steps` | 4096 | Covers 2+ full laps per env; avoids truncated-return bias |
| `ppo-epochs` | 4 | Standard; target-kl guards overtraining |
| `minibatch-size` | 512 | 32768/512 = 64 updates/epoch, stable gradients |

### Learning Rate and Entropy

| Parameter | Value | Why |
|-----------|-------|-----|
| `lr` | 3e-4 | Adam default, proven for PPO |
| `lr-min` | 1e-5 | Gentle final LR for fine-tuning hard tracks |
| `ent-coef-start` | 0.005 | Low entropy; log-std clamp [-3, -0.3] prevents max-entropy plateau, so less entropy pressure is needed |
| `ent-coef-end` | 0.001 | Minimal exploration for late hard tracks; policy tightens for precise racing lines |
| `target-kl` | 0.015 | Tighter than default (0.02) to prevent collapse on curriculum transitions |

### Discount and Advantage

| Parameter | Value | Why |
|-----------|-------|-----|
| `gamma` | 0.997 | Higher than default (0.99); values lap bonus ~500 steps away at 0.997^500 = 0.22 vs 0.99^500 = 0.007. Stronger incentive to complete laps fast |
| `gae-lambda` | 0.95 | Standard bias/variance sweet spot |

### Policy Update

| Parameter | Value | Why |
|-----------|-------|-----|
| `clip-eps` | 0.2 | Standard PPO clipping |
| `vf-coef` | 0.5 | Standard value loss weight |
| `max-grad-norm` | 0.5 | Standard gradient clipping |

### Curriculum

| Parameter | Value | Why |
|-----------|-------|-----|
| `threshold` | 18.0 | Lower gate so the agent advances quickly, spending more budget on hard tracks where distance/time optimization matters |
| `window` | 20 | Fast advancement decisions; still requires 20 consecutive clean-lap, zero-crash episodes |
| `replay-frac` | 0.3 | 30% anti-forgetting replay from mastered tracks |

---

## Monitoring — 50k Step Health Checks

**`run_training.py` handles monitoring automatically.** If running `train.py`
manually, run `monitor.py` alongside it.

The monitor polls W&B every 60 seconds, prints a live metrics row, and fires a
PASS/FAIL report at every 50k boundary. If FAIL, it prints the exact resume
command with fixes.

### Thresholds

| Step | Min Reward | Min On-Track % | Min Explained Var | Max Grad Norm |
|------|-----------|----------------|-------------------|---------------|
| 50K | -500 | 60 | 0.50 | 30 |
| 100K | -200 | 75 | 0.70 | 15 |
| 200K | 0 | 85 | 0.85 | 10 |
| 500K | 40 | 93 | 0.90 | 8 |
| 1M | 80 | 95 | 0.93 | 6 |
| 2M | 130 | 96 | 0.95 | 5 |
| 3M | 170 | 96 | 0.95 | 4 |
| 5M | 200 | 97 | 0.96 | 4 |
| 10M | 200 | 97 | 0.96 | 4 |

### What a Healthy Report Looks Like

```
======================================================================
  PASS  250k check at step 252,416
  step= 252,416  reward=  35.4  on_track= 92.0%  ev= 0.890  kl= 0.0032
======================================================================
```

### What a Failing Report Looks Like

```
======================================================================
  FAIL  200k check at step 204,312
  FAIL  episode/reward -975.4 < 0
  WARN  grad_norm 91.7 > 10

  Fix — kill train.py then run:
    uv run python train.py --resume checkpoints/... --ent-coef-start 0.02 ...
======================================================================
```

Kill train.py, run the fix command exactly as printed, restart monitor.

### EV (Explained Variance) — Warn vs Fail

- **WARN** (first low check): watch next check, no action (critic lags during fast improvement)
- **FAIL** (second consecutive low check): resume with `--vf-coef 1.0`

---

## Sanity Checks — First 5 Minutes

Open W&B and verify these within the first 1-2 PPO updates:

| Metric | Expected at step ~16K | Red flag |
|--------|----------------------|----------|
| `ppo/policy_loss` | Finite number | **NaN** |
| `ppo/approx_kl` | 0.001 - 0.015 | NaN or always exactly 0.015 |
| `ppo/entropy` | ~2.5 - 2.8 | NaN or 0 |
| `ppo/early_stopped` | 0 most of the time | **always 1** |
| `system/steps_per_sec` | > 50 | < 10 |

If `ppo/policy_loss` is NaN and `ppo/early_stopped` is always 1: policy is
frozen (KL fires before any gradient step). Kill, fix, restart from scratch.

---

## Inference Videos (Every 25K Steps)

Videos are logged to W&B under `inference/track_NN_Name`. Check these to confirm:

- Agent is moving forward around the track (not circling or stuck)
- Staying on road (not cutting corners through grass)
- Completing laps (reaching start/finish line)
- Taking efficient racing lines (hugging the inside on turns)

Videos render all mastered + frontier tracks at each interval.

---

## Automated Diagnosis Categories

When `run_training.py` detects a stall, it loads the checkpoint, runs a greedy
episode, and classifies the behavior:

| Diagnosis | Condition | Automated Fix |
|-----------|-----------|---------------|
| `NOT_STEERING` | Off-track > 70% | Reduce initial speed/accel bias; increase on-track reward in `rl_splits.py` |
| `NOT_PROGRESSING` | On-track ok but < 20 waypoints | Increase `PROGRESS_SCALE` in `rl_splits.py` |
| `ALMOST_LAPPING` | > 20 waypoints but no lap | Continue; optionally lower `--threshold` |
| `LAPPING_WITH_CRASHES` | Laps > 0 but crashes > 0 | Continue; crashes decrease naturally |
| `MASTERING` | Clean laps, good reward | No fix; check if threshold too high |

The script then applies escalating hyperparameter fixes (more exploration,
lower threshold, more replay). If all fixes are exhausted, training continues
and the diagnosis is logged for manual review.

---

## Plateau Diagnosis and Fixes

A plateau is when `episode/reward` and `curriculum/rolling_mean` stop improving
for more than 100K steps.

### Pattern 1 — Maximum-Entropy Plateau (largely prevented by log-std clamp)

**Symptoms:**
- `ppo/entropy` stays at ~2.84-2.90 (theoretical max for 2D Gaussian)
- `ppo/grad_norm` collapses to 0.3-0.5
- `episode/reward` flat at 0.3-0.7 (tiny speed rewards, never lap bonus)
- `episode/laps` = 0 throughout

**What was happening:** The entropy bonus gradient on `log_std` is constant (+1 per dim),
which overwhelmed weak early policy gradients. `log_std` drifted from -1.0 (std=0.37) up
to ~0.0 (std=1.0), making actions uniformly random.

**Prevention:** `actor_log_std` is now clamped to `[-3.0, -0.3]` (std in [0.05, 0.74])
in the forward pass, preventing entropy from pushing std above 0.74. This keeps the
policy exploratory but learnable. Combined with lower `ent-coef-start` (0.005), the
max-entropy plateau should not occur.

**If it still happens** (entropy stuck above 2.5 after 100K steps):
```bash
uv run python train.py --resume <latest_checkpoint> \
    --num-envs 8 --compile --subproc \
    --rollout-steps 4096 \
    --ent-coef-start 0.003 --ent-coef-end 0.001 \
    --threshold 15 --window 20 \
    --video-interval 25_000 --keep-checkpoints 5
```

| Change | Why |
|--------|-----|
| `--rollout-steps 4096` | Full lap sequences visible to PPO |
| `--ent-coef-start 0.003` | Reduce entropy pressure further; log-std clamp provides sufficient exploration |
| `--threshold 15` | Lower gate so agent escapes Track 1 sooner |

### Pattern 2 — Late-Stage Policy Collapse

**Symptoms (sudden):**
- `ppo/early_stopped` flips to 1 permanently
- `ppo/grad_norm` explodes from ~0.5 to 100+
- `episode/reward` crashes to -1000

**Fix:** Resume from checkpoint BEFORE the collapse:
```bash
uv run python train.py --resume <checkpoint_before_collapse> \
    --target-kl 0.01 --max-grad-norm 0.5 --lr 1e-4 \
    --video-interval 25_000 --keep-checkpoints 5
```

### Pattern 3 — Curriculum Stuck After Advance

**Symptoms:** `curriculum/level` advances once then stalls for 200K+ steps.

**Fix (in order):**
1. `--replay-frac 0.4` — more anti-forgetting replay
2. `--threshold` x 0.8 — lower the bar for the new track
3. `--ent-coef-start 0.03` — more exploration on the harder track

---

## Failure Mode Quick Reference

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Agent stays still | Zero-speed local optimum (prevented by speed-only reward) | Should not happen; if it does, check that on-track reward has no base constant |
| Agent drives but never completes a lap | Sparse reward | Increase `--rollout-steps` to 4096; increase `--ent-coef-start` to 0.05 |
| Completes laps but crashes too often | Risk-seeking | Check `episode/crashes`; if > 5, raise `--ent-coef-start 0.03` |
| Reward high but no advancement | Strict AND gate | Check that ALL 30 episodes have laps>=1 AND crashes==0 |
| KL spikes after level change | Normal; persistent = problem | Lower `--lr` to 1e-4 |
| Forgets earlier tracks | Catastrophic forgetting | Increase `--replay-frac` to 0.4 |
| `ppo/explained_variance` negative | Value fn diverged | Reduce `--lr` 10x, resume from last checkpoint |
| SPS doesn't scale with more envs | Env-stepping bottleneck | Expected; `--num-envs 8` is the practical ceiling on CPU |
| Training slow (< 30 sps) | Display driver active | Confirm `SDL_VIDEODRIVER=dummy` (set automatically by train.py) |

---

## W&B Dashboard Setup

Recommended panels:

**Training Health:**
- `ppo/approx_kl` — should stay below 0.015
- `ppo/explained_variance` — should trend toward 0.9+
- `ppo/entropy` — should decrease slowly from ~2.8 toward ~2.0

**Objectives:**
- `episode/reward` — noisy upward trend
- `episode/crashes` — decreasing toward 0
- `episode/laps` — increasing toward laps_target
- `episode/on_track_pct` — above 95%

**Curriculum:**
- `curriculum/track_level` — staircase shape
- `curriculum/rolling_mean` vs `curriculum/threshold` on same chart

**Inference:**
- `inference/*` — video panels, one per track, updated every 25K steps

**Validation (after each advance):**
- `val/mean_reward`, `val/completion_rate`

---

## Checkpointing

- Saves every `--checkpoint-interval` steps (default 250K) to `checkpoints/ppo_step<NNNNNNNN>_lvl<LL>.pt`
- **Keeps only the last 5 checkpoints** (`--keep-checkpoints 5`); older ones auto-deleted
- Final model always saved as `checkpoints/ppo_final.pt`
- Contains: model weights, optimizer state, curriculum state, reward window, W&B run ID

Resume restores everything:
```bash
uv run python train.py --resume checkpoints/ppo_step01000000_lvl04.pt
```
Charts continue on the same W&B run. `--total-steps` budget is respected
relative to the original count. You can change `--num-envs` or `--compile` on
resume without affecting the model.

---

## All Training Arguments

```
uv run python train.py --help
```

| Argument | Default | What it controls |
|----------|---------|-----------------|
| `--total-steps` | `5_000_000` | Total environment steps |
| `--rollout-steps` | `2048` | Steps per env per PPO update |
| `--num-envs` | `4` | Parallel environments |
| `--ppo-epochs` | `4` | Update passes per rollout |
| `--minibatch-size` | `1024` | Minibatch size per gradient step |
| `--lr` | `3e-4` | Initial learning rate (Adam) |
| `--lr-min` | `1e-5` | Final LR after linear decay |
| `--gamma` | `0.99` | Discount factor |
| `--gae-lambda` | `0.95` | GAE smoothing parameter |
| `--clip-eps` | `0.2` | PPO clipping range |
| `--vf-coef` | `0.5` | Value loss weight |
| `--ent-coef-start` | `0.01` | Entropy coefficient at step 0 |
| `--ent-coef-end` | `0.001` | Entropy coefficient at final step |
| `--max-grad-norm` | `0.5` | Gradient clipping norm |
| `--target-kl` | `0.02` | KL early-stop threshold |
| `--threshold` | `30.0` | Base curriculum advancement threshold |
| `--window` | `50` | Rolling window for advancement |
| `--replay-frac` | `0.3` | Anti-forgetting replay fraction |
| `--val-episodes` | `10` | Episodes per VAL track after advance |
| `--video-interval` | `25_000` | Steps between inference video logs |
| `--checkpoint-interval` | `500_000` | Steps between checkpoint saves |
| `--keep-checkpoints` | `5` | Keep only last N checkpoints (0 = all) |
| `--checkpoint-dir` | `checkpoints` | Directory for saved models |
| `--resume` | `None` | Path to checkpoint to resume from |
| `--compile` | `False` | Enable `torch.compile` |
| `--subproc` | `False` | Subprocess-parallel env stepping |
| `--seed` | `42` | RNG seed |
| `--device` | auto | `cuda` if available, else `cpu` |
| `--wandb-project` | `curriculum-car-racer` | W&B project name |
| `--wandb-run-name` | `None` | W&B run name |
| `--wandb-offline` | `False` | Disable W&B network calls |

Note: `run_training.py` overrides several defaults (gamma=0.997, ent-coef-start=0.005,
rollout-steps=4096, threshold=18, window=20) for optimal full-curriculum performance.
These values are tuned to work with the log-std clamp in the model.
