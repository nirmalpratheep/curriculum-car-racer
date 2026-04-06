# Training Change Log

Each entry records what changed, why, and what was observed before/after.

---

## v1 — Original (baseline)

**Command:**
```
--total-steps 10000000 --num-envs 8 --subproc --compile
--rollout-steps 4096 --ppo-epochs 4 --minibatch-size 512
--lr 3e-4 --lr-min 1e-5 --gamma 0.997 --gae-lambda 0.95
--clip-eps 0.2 --vf-coef 0.5
--ent-coef-start 0.005 --ent-coef-end 0.001
--max-grad-norm 0.5 --target-kl 0.015
--threshold 80.0 --window 20 --replay-frac 0.3
--video-interval 25_000 --checkpoint-interval 250_000 --keep-checkpoints 5
```

**Reward (rl_splits.py):** Original — `diff * progress_per_wp - 0.05 + accel * 0.05 - |steer| * 0.02`

**Observed:**
- Reward stuck at 0.1–0.6 for 500k steps, level never advanced past 0
- `on_track=100%` but zero laps completed
- Agent learned to crawl on-track without completing a lap
- `ent-coef=0.005` too low: policy collapsed to safe/slow local optimum
- `threshold=80.0` + `window=20` requires 20 consecutive clean laps → unreachable without first fixing exploration

---

## v2 — More entropy, lower threshold

**Changes vs v1:**
- `--ent-coef-start 0.005 → 0.02` — force more exploration
- `--ent-coef-end 0.001 → 0.005` — maintain exploration throughout
- `--threshold 80.0 → 30.0` — more achievable (1 lap ≈ 115 reward)
- `--window 20 → 10` — need 10 consecutive clean laps not 20
- `--rollout-steps 4096 → 2048` — more frequent policy updates
- `--num-envs 8 → 32` (user change)

**Reward:** Unchanged from v1

**Observed:**
- Agent now U-turns on the wide oval and drives the WRONG WAY around the track
- Going backward: positive speed (not physical reverse) but negative waypoint diff
- `episode/reward: -153`, `rolling_mean: -144`, still 0 laps
- `ppo/explained_variance: 0.04` — value function broken (returns too noisy from crashes)
- `ppo/grad_norm: 11.5` clipped to 0.5 (22× reduction) — learning paralysed
- Higher entropy caused U-turn exploration without enough penalty to deter it

---

## v3 — Penalise wrong-way driving (BROKEN attempt)

**Changes vs v2 (rl_splits.py):**
- `diff * progress_per_wp` → `max(0, diff) * progress_per_wp` — clamp backward progress to 0
- Added backward SPEED penalty: `if speed < 0: reward -= |speed|`
- Throttle bonus: `accel * 0.05` → `max(0, accel) * 0.05`

**Command:**
- `--vf-coef 0.5 → 1.0` — prioritise value learning (EV was 0.04)
- `--max-grad-norm 0.5 → 1.0` — reduce gradient suppression
- `--ent-coef-start 0.02 → 0.01` — slightly less noise

**Observed:**
- `episode/reward: -271` (worse!), `rolling_mean: -237`
- `episode/length: 1592` (longer, but still 0 laps)
- `ppo/grad_norm: 64.5` — even larger gradients after reward change
- `ppo/explained_variance: 0.23` — slightly better
- **Bug**: `max(0, diff)` removed the gradient signal for wrong-way driving.
  Going backward now looks identical to standing still (both give 0 waypoint reward).
  Also, the backward SPEED penalty targeted physical reverse (speed < 0) but the
  real problem is wrong-way driving at POSITIVE speed (car turned 180° on oval).

---

## v4 — Fix wrong-way penalty + add speed bonus (current)

**Changes vs v3 (rl_splits.py):**
- Restore negative diff signal: remove `max(0, diff)` clamp
- Wrong-way driving: `diff * progress_per_wp * 3` when diff < 0 (triple penalty)
- Remove backward SPEED penalty (was targeting wrong problem)
- Add forward SPEED bonus: `+max(0, speed) / max_speed * 0.05`
  → full-speed forward = +0.05/step extra incentive to maintain throttle
- Throttle bonus still `max(0, accel) * 0.05` (positive accel only)

**Command:**
- `--max-grad-norm 1.0 → 5.0` — allow value fn to update with new reward targets
- `--vf-coef 1.0` — keep (EV was 0.23, still needs emphasis)
- `--ent-coef-start 0.01` — keep

**Expected behaviour:**
- Wrong-way driving is now clearly the worst option (3× negative waypoint)
- Standing still: just time penalty (-0.05/step)
- Forward slow: small positive
- Forward fast: waypoint + speed bonus = strongest incentive

---

## v5 — Reference-based redesign: always forward + GPS (current)

**Core insight from reference repo (nirmalpratheep/RL-CarNavigationAgent):**
- Car always moves forward at fixed speed — agent cannot stop
- Agent only controls steering
- Waypoint direction (GPS) is explicitly in the observation
- Simple binary reward: right direction = positive, wrong direction = negative

**Code changes:**

`game/rl_splits.py`:
- Crash penalty: -100 → -10
- Out-of-bounds: -100 → -10
- Episode termination: removed max_steps and laps_target conditions — only crash/OOB ends episode
- Reward: replaced all per-step bonuses with: +1 (diff>0), -1 (diff<0), 0 (diff=0)
- `_obs()`: added wp_sin, wp_cos (GPS direction to next waypoint relative to heading)

`env/models.py`:
- Added wp_sin, wp_cos fields to RaceObservation (scalars 7→9)

`env/environment.py`:
- _to_obs(): pass obs[7], obs[8] as wp_sin, wp_cos

`env/encoder.py`:
- nn.Linear(7, ...) → nn.Linear(9, ...) in scalar MLP

`env/subproc_vec_env.py`:
- Updated _StepResult comment: 7→9 scalars

`train.py`:
- buf_scalars_np: shape (T,N,7) → (T,N,9)
- SubprocVecEnv: max_steps=3000,laps_target=3 → max_steps=500_000,laps_target=9999
- Rollout step dispatch: accel fixed at 1.0; policy only controls steer dimension

**Command:**
- gamma: 0.997 → 0.99 (shorter episodes, faster credit assignment)
