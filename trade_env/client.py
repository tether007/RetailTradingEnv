from typing import Dict
from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State
from .models import TradeAction, TradeObservation


class TradeEnv(EnvClient[TradeAction, TradeObservation, State]):
    """
    Client for RetailTraderBehaviorCoach environment.
    
    Example:
        >>> with TradeEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     result = client.step(TradeAction(action=0))
    """

    def _step_payload(self, action: TradeAction) -> Dict:
        return {"action": action.action}

    def _parse_result(self, payload: Dict) -> StepResult[TradeObservation]:
        obs_data = payload.get("next_state", {})
        observation = TradeObservation(
            timestep=obs_data.get("timestep", 0),
            price=obs_data.get("price", 100.0),
            position=obs_data.get("position", 0),
            loss_streak=obs_data.get("loss_streak", 0),
            pnl=obs_data.get("pnl", 0.0),
            trader_action=payload.get("info", {}).get("trader_action", "HOLD"),
            behaviour=payload.get("info", {}).get("behaviour", "normal"),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("timestep", 0),
        )