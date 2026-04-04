"""
CoachEnv: RL environment for training a behavioral trading coach.

Simulates a market + trader system where an agent learns when to intervene
IMP: agent is an intervener like a coach
(e.g., NO, warn, reduce size, exit) to improve trader behavior and outcomes.
Normalized btw 0-1
NO -> no_operation coach does nothing lets trader continue
warn -> light nudge (.2)
reduce size -> must reduce the volue (gets some pattern of loss)(.4)
exit -> to say when to exit so to stop further bleeding(0)
COOLDOWN -> a break due overwhelm (1.0)

Follows OpenAI's Gym-style API:
- reset(): initialize environment and return initial state
- step(action): apply intervention and return (next_state, reward, done, info)

State includes market, trader, and behavioral signals.
Reward balances profit, behavior correction, and intervention cost.

position attribute :
0  → no position (not in a trade)
1  → long (bought, expecting price to go up)
-1 → short (sold, expecting price to go down)
"""
import random
from enum import Enum
from trade_env.schemas.action import ActionType, Action


class Action(Enum):
    
    NO = 0
    WARN = 1
    REDUCE = 2
    EXIT = 3
    COOLDOWN = 4 #force stop for a tmframe 
    

class CoachEnv:
    
    def __init__(self):
        
        self.t = 0
        self.price = 100 # current market price of the stock trader is trying to trade etc.. for Now 100 demo
        self.pnl = 0
        self.loss_streak = 0
        self.pos = 0
    
    def reset(self):
        """ resets the env
            per se after an episode
        """
        self.t = 0
        self.price = 100
        self.pnl = 0
        self.loss_streak = 0
        self.pos = 0
        
        
        return self._get_state()
    
    def step(self, action: Action):
        """next_step,
           reward,
           info,
           basically apply an action and move forward
        Args:
            action (): task for the agent to take given the sensor inputs in the env present
        """
        action_type = action.action_type
        
        intr = 0
        if(action_type == ActionType.WARN):
            intr = .2
        elif action_type == ActionType.REDUCE_SIZE:
            intr = 0.4
        elif action_type == ActionType.EXIT_POSITION:
            self.pos = 0
        elif action_type == ActionType.COOLDOWN:
            intr = 1.0
            
        risk_prob = 0.5 + (0.1 * self.loss_streak)
        risk_prob = max(0, min(1, risk_prob - intr))
        
        #demo sim
        trader = "HOLD"
        
        if random.random() < risk_prob:
            trader = random.choice(["BUY", "SELL"])

        price_change = random.uniform(-2, 2)
        self.price += price_change

        if trader == "BUY":
            self.pos = 1
            self.entry_price = self.price
        elif trader == "SELL":
            self.pos = -1
            self.entry_price = self.price

        if self.pos == 1:
            step_pnl = self.price - self.entry_price
        elif self.pos == -1:
            step_pnl = self.entry_price - self.price
        else:
            step_pnl = 0

        self.pnl += step_pnl

        if step_pnl < 0:
            self.loss_streak += 1
        else:
            self.loss_streak = 0

        reward = step_pnl - (0.1 * intr)

        self.t += 1
        done = False

        if self.t >= 100:
            done = True

        if self.pnl < -50:
            done = True

        next_state = self._get_state()

        info = {
            "trader_action": trader,
            "price": self.price,
            "pnl": self.pnl
        }

        return next_state, reward, done, info
    
    def _get_state(self):
        return {
            "timestep": self.t,
            "price": self.price,
            "position": self.pos,
            "loss_streak": self.loss_streak,
            "pnl": self.pnl
        }