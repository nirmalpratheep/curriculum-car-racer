"""
inference/infer_videos.py — Run a trained checkpoint over tracks and save videos.

Usage
─────
  uv run python inference/infer_videos.py --checkpoint checkpoints/ppo_torchrl_final.pt
  uv run python inference/infer_videos.py --checkpoint checkpoints/... --tracks train
  uv run python inference/infer_videos.py --checkpoint checkpoints/... --tracks all --video-dir my_videos
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
import torch

from tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type

from env import DriveAction
from env.environment import RaceEnvironment
from game.rl_splits import TRAIN, VAL, TEST, ALL_ORDERED, difficulty_of
from training.train_torchrl import build_policy_and_value


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint file")
    p.add_argument("--tracks",     default="all", choices=["all", "train", "val", "test"],
                   help="Which track split to run")
    p.add_argument("--video-dir",  default="inference_videos")
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-steps",  type=int, default=3000)
    p.add_argument("--frame-skip", type=int, default=2,  help="Render every N steps")
    p.add_argument("--fps",        type=int, default=20)
    return p.parse_args()


def _game_frame(race_env) -> np.ndarray:
    import pygame
    from game.oval_racer import draw_car, draw_headlights

    ce   = race_env._env
    surf = ce.track.surface.copy()
    draw_headlights(surf, ce._x, ce._y, ce._angle)
    draw_car(surf, ce._x, ce._y, ce._angle)
    small = pygame.transform.scale(surf, (450, 300))
    return pygame.surfarray.array3d(small).transpose(1, 0, 2).copy()


@torch.no_grad()
def run_track(policy_module, track, device, max_steps, frame_skip):
    track.build()
    env     = RaceEnvironment(track, max_steps=max_steps, laps_target=1, use_image=True)
    raw_obs = env.reset()
    frames  = [_game_frame(env)]
    step    = 0
    total_r = 0.0

    while not raw_obs.done:
        img = (torch.from_numpy(raw_obs.image.copy())
               .float().div(255.0).permute(2, 0, 1).unsqueeze(0).to(device))
        scalars = torch.tensor(raw_obs.scalars, dtype=torch.float32,
                               device=device).unsqueeze(0)

        td = TensorDict({"image": img, "scalars": scalars}, batch_size=[1])
        with set_exploration_type(ExplorationType.MEAN):
            td = policy_module(td)
        action = td.get("action")[0].clamp(-1.0, 1.0).cpu().numpy()

        raw_obs = env.step(DriveAction(accel=float(action[0]), steer=float(action[1])))
        total_r += float(raw_obs.reward)
        step    += 1
        if step % frame_skip == 0:
            frames.append(_game_frame(env))

    ce = env._env
    return {
        "frames":       np.stack(frames, axis=0),
        "total_reward": total_r,
        "steps":        step,
        "laps":         ce._laps,
        "crashes":      ce._crash_count,
        "on_track_pct": 100.0 * ce._ep_on_track / max(step, 1) if hasattr(ce, "_ep_on_track") else float("nan"),
    }


def main():
    args   = parse_args()
    device = torch.device(args.device)

    split_map = {"all": ALL_ORDERED, "train": TRAIN, "val": VAL, "test": TEST}
    tracks = split_map[args.tracks]

    print(f"\nLoading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    step = ckpt.get("step", 0)
    lvl  = ckpt.get("curriculum_level", "?")
    print(f"  step={step:,}  curriculum_level={lvl}")

    policy_module, _, _ = build_policy_and_value(device)
    state_dict = ckpt["policy"]
    # torch.compile prefixes keys with "_orig_mod." — strip it for inference
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}
    policy_module.load_state_dict(state_dict)
    policy_module.eval()

    os.makedirs(args.video_dir, exist_ok=True)

    import imageio.v3 as iio
    from game.rl_splits import _ensure_pygame
    _ensure_pygame()

    rows = []
    print(f"\nRunning {len(tracks)} track(s) → {args.video_dir}/\n")

    for track in tracks:
        print(f"  track {track.level:02d}  {track.name:<24}", end="", flush=True)
        result = run_track(policy_module, track, device, args.max_steps, args.frame_skip)

        slug     = track.name.replace(" ", "_")
        filename = f"track{track.level:02d}_{slug}.mp4"
        out_path = os.path.join(args.video_dir, filename)
        iio.imwrite(out_path, result["frames"], fps=args.fps, codec="libx264", plugin="pyav")

        rows.append({
            "level":   track.level,
            "name":    track.name,
            "tier":    difficulty_of(track),
            "reward":  result["total_reward"],
            "laps":    result["laps"],
            "crashes": result["crashes"],
            "steps":   result["steps"],
            "file":    filename,
        })
        print(f"  laps={result['laps']}  crashes={result['crashes']}"
              f"  reward={result['total_reward']:+.1f}  steps={result['steps']}"
              f"  → {filename}")

    # Summary table
    print(f"\n{'='*78}")
    print(f"  {'Lvl':<4} {'Name':<24} {'Tier':<16} {'Reward':>8} {'Laps':>5} {'Crash':>6} {'Steps':>6}")
    print(f"  {'-'*74}")
    for r in rows:
        print(f"  {r['level']:<4} {r['name']:<24} {r['tier']:<16}"
              f"  {r['reward']:>8.1f} {r['laps']:>5} {r['crashes']:>6} {r['steps']:>6}")
    print(f"{'='*78}")
    completed = sum(1 for r in rows if r["laps"] >= 1)
    print(f"  Completed: {completed}/{len(rows)} tracks  "
          f"({100*completed/max(len(rows),1):.0f}%)\n")


if __name__ == "__main__":
    main()
