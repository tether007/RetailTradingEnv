# train.py
from trade_env.env.coach_env import CoachEnv
from trade_env.schemas.action import Action, ActionType
from trade_env.agent.ppo_agent import PPOAgent

env = CoachEnv()
agent = PPOAgent(state_dim=6, action_dim=5)

for episode in range(2000):
    state = env.reset()
    done = False

    while not done:
        action_idx = agent.select_action(state)
        action = Action(action=ActionType(action_idx))
        next_state, reward, done, info = env.step(action)
        agent.store_outcome(reward, done)
        state = next_state

    agent.update()
    print(f"Ep {episode} | PnL: {info['pnl']:.2f} | Action: {action_idx} | Trader: {info['trader_action']}")