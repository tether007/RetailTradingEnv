""" What is the step be taken from openEnv's step()
    - reward
    - next_state
    - done 
    - info 
"""

from typing import Dict, Any
from pydantic import BaseModel
from trade_env.schemas.state import State


class StepResponse(BaseModel):
    next_state: State      
    reward: float          
    done: bool         
    info: Dict[str, Any]   