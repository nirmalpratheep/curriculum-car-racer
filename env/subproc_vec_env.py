"""
SubprocVecEnv — N environments running in parallel worker subprocesses.

Each worker owns one RaceEnvironment and runs it in isolation.
The main process sends actions to all workers simultaneously, then
collects results — replacing the sequential env-stepping loop with
a single parallel scatter/gather.

Protocol (over multiprocessing.Pipe):
    main → worker : (_CMD_STEP,  (accel, steer))
    main → worker : (_CMD_RESET, track_level: int)
    main → worker : (_CMD_CLOSE, None)
    worker → main : _StepResult namedtuple

On Linux (fork) workers start instantly by inheriting the parent's
memory. SDL_VIDEODRIVER=dummy is set before any pygame import so
every worker gets a headless pygame context.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
from typing import List, NamedTuple, Optional

import numpy as np

# ── IPC command tokens ────────────────────────────────────────────────────────
_CMD_STEP  = 0
_CMD_RESET = 1
_CMD_CLOSE = 2


class _StepResult(NamedTuple):
    """Compact observation sent from worker to main process each step."""
    image:           np.ndarray        # (64, 64, 3) uint8
    scalars:         np.ndarray        # (9,) float32: speed, ang_vel, 5 rays, wp_sin, wp_cos
    reward:          float
    done:            bool
    metadata:        dict


# ── Worker entry point ────────────────────────────────────────────────────────

def _worker_fn(conn: mp.connection.Connection, max_steps: int, laps_target: int) -> None:
    """
    Runs inside a subprocess.  Owns one RaceEnvironment.

    Receives commands from the main process via `conn` and sends
    _StepResult objects back.  The worker's pygame is headless
    (SDL_VIDEODRIVER=dummy) and completely independent of the
    main process.
    """
    # Must be set before any pygame/game import
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    # Lazy imports — these happen inside the subprocess so each worker
    # gets its own pygame state and pygame Surface objects.
    from game.tracks import TRACKS
    from env.environment import RaceEnvironment
    from env.models import DriveAction

    tracks_by_level = {t.level: t for t in TRACKS}
    env: Optional[RaceEnvironment] = None

    try:
        while True:
            cmd, data = conn.recv()

            if cmd == _CMD_RESET:
                track_level: int = data
                track = tracks_by_level[track_level]
                track.build()          # builds pygame.Surface inside this worker
                env = RaceEnvironment(track, max_steps, laps_target, use_image=True)
                obs = env.reset()
                conn.send(_make_result(obs))

            elif cmd == _CMD_STEP:
                accel, steer = data
                obs = env.step(DriveAction(accel=accel, steer=steer))
                conn.send(_make_result(obs))

            elif cmd == _CMD_CLOSE:
                break

    except (EOFError, BrokenPipeError, KeyboardInterrupt):
        pass
    finally:
        conn.close()


def _make_result(obs) -> _StepResult:
    img = obs.image  # (64, 64, 3) uint8 numpy array
    scalars = np.array(obs.scalars, dtype=np.float32)
    return _StepResult(
        image=img,
        scalars=scalars,
        reward=obs.reward,
        done=obs.done,
        metadata=obs.metadata if obs.metadata else {},
    )


# ── Main-process interface ────────────────────────────────────────────────────

class SubprocVecEnv:
    """
    N environments stepping in parallel subprocesses.

    Replaces the sequential ``for n in range(N): envs[n].step(...)`` loop
    in the rollout collector with a scatter/gather over worker pipes.

    Parameters
    ----------
    n_envs      : number of worker subprocesses to launch
    max_steps   : max steps per episode (passed to RaceEnvironment)
    laps_target : laps per episode (passed to RaceEnvironment)
    """

    def __init__(self, n_envs: int, max_steps: int = 3000, laps_target: int = 3):
        self.n_envs = n_envs

        # fork is the fastest start method on Linux (workers inherit parent memory).
        # Use spawn on Windows/macOS where fork is unavailable or unsafe.
        start_method = "fork" if sys.platform.startswith("linux") else "spawn"
        ctx = mp.get_context(start_method)

        self._remotes: List[mp.connection.Connection] = []
        work_remotes: List[mp.connection.Connection] = []
        for _ in range(n_envs):
            main_end, work_end = ctx.Pipe(duplex=True)
            self._remotes.append(main_end)
            work_remotes.append(work_end)

        self._procs: List[mp.Process] = []
        for wr in work_remotes:
            p = ctx.Process(
                target=_worker_fn,
                args=(wr, max_steps, laps_target),
                daemon=True,
            )
            p.start()
            self._procs.append(p)
            wr.close()   # worker-end not needed in main process

        print(f"[SubprocVecEnv] {n_envs} worker processes started "
              f"(start_method={start_method!r})")

    # ── Bulk reset ────────────────────────────────────────────────────────────

    def reset(self, track_levels: List[int]) -> List[_StepResult]:
        """
        Reset all N envs simultaneously.

        Parameters
        ----------
        track_levels : list of int, length n_envs — one track level per worker
        """
        for remote, level in zip(self._remotes, track_levels):
            remote.send((_CMD_RESET, level))
        return [r.recv() for r in self._remotes]

    # ── Single-env reset (for episode end during rollout) ─────────────────────

    def reset_one(self, n: int, track_level: int) -> _StepResult:
        """Reset worker n on track_level and return its first observation."""
        self._remotes[n].send((_CMD_RESET, track_level))
        return self._remotes[n].recv()

    # ── Parallel step ─────────────────────────────────────────────────────────

    def step_async(self, actions: List[tuple]) -> None:
        """
        Broadcast actions to all workers (non-blocking).

        actions : list of (accel, steer) float tuples, length n_envs
        """
        for remote, (accel, steer) in zip(self._remotes, actions):
            remote.send((_CMD_STEP, (float(accel), float(steer))))

    def step_wait(self) -> List[_StepResult]:
        """Collect one _StepResult from every worker."""
        return [r.recv() for r in self._remotes]

    def step(self, actions: List[tuple]) -> List[_StepResult]:
        """Send actions to all workers and wait for all results."""
        self.step_async(actions)
        return self.step_wait()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        for remote in self._remotes:
            try:
                remote.send((_CMD_CLOSE, None))
            except Exception:
                pass
        for p in self._procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        for p in self._procs:
            p.join(timeout=2)
