"""
FastAPI entry point for serving RaceEnvironment over HTTP/WebSocket.

Usage:
    uvicorn env.server.app:app --host 0.0.0.0 --port 8000

Or via OpenEnv CLI:
    openenv serve env.server.app:app
"""

import os
import sys

# Headless pygame — must come before any game/env import
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from openenv.core.env_server import create_app

from env.environment import RaceEnvironment
from env.models import DriveAction, RaceObservation
from game.rl_splits import TRAIN, _ensure_pygame

# Initialise pygame in headless mode
_ensure_pygame()

# Build track 0 (simplest) as default for the remote server
track = TRAIN[0]
track.build()

env = RaceEnvironment(track, max_steps=3000, laps_target=1, use_image=True)
app = create_app(env, DriveAction, RaceObservation)
