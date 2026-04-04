""" Actions agent can take
    buy/sell( enum)
    
"""


from enum import Enum
from pydantic import BaseModel
class ActionType(Enum):
    NO = 0
    WARN = 1
    REDUCE = 2
    EXIT = 3
    COOLDOWN = 4

class Action(BaseModel):
    action : ActionType
