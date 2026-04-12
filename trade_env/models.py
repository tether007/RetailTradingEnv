from openenv.core.env_server.types import Action, Observation
from pydantic import Field

class TradeAction(Action):
    action: int = Field(..., description="0=NO, 1=WARN, 2=REDUCE, 3=EXIT, 4=COOLDOWN")

class TradeObservation(Observation):
    timestep: int = Field(default=0)
    price: float = Field(default=100.0)
    position: int = Field(default=0)
    loss_streak: int = Field(default=0)
    pnl: float = Field(default=0.0)
    trader_action: str = Field(default="HOLD")
    behaviour: str = Field(default="normal")