"""
OpenEnv client for the car racing environment.

Async usage:
    async with RaceEnvClient(base_url="http://localhost:8000") as client:
        result = await client.reset()
        result = await client.step(DriveAction(accel=1.0, steer=0.0))

Sync usage:
    with RaceEnvClient(base_url="http://localhost:8000").sync() as client:
        result = client.reset()
        result = client.step(DriveAction(accel=1.0, steer=0.0))
"""

from typing import Any, Dict

from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult

from .models import DriveAction, RaceObservation, RaceState


class RaceEnvClient(EnvClient[DriveAction, RaceObservation, RaceState]):
    """Client for a running RaceEnvironment server.

    Communicates via WebSocket with a FastAPI server hosting the
    RaceEnvironment. Handles serialization of DriveAction and
    deserialization of RaceObservation/RaceState.
    """

    def _step_payload(self, action: DriveAction) -> Dict[str, Any]:
        """Convert DriveAction to JSON dict for the server."""
        return action.model_dump()

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[RaceObservation]:
        """Parse server response into StepResult[RaceObservation]."""
        obs_data = payload.get("observation", payload)
        return StepResult(
            observation=RaceObservation(**obs_data),
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> RaceState:
        """Parse server state response into RaceState."""
        return RaceState(**payload)
