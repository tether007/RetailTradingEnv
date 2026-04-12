""" What the environment returns as an observation
    - price_features(can be nos)
    - position
    - pnl
    - loss_streak
    - overtrade_score(ego)
"""


from pydantic import BaseModel, Field

class State(BaseModel):
    timestep: int
    price: float
    position: int
    loss_streak: int
    pnl: float
    overtrade_score: float = Field(default=0.0, description="ego/overtrading signal 0-1")
