from trade_env.env import coach_env

ce = coach_env.CoachEnv()

state = ce.reset()
print("Initial State:", state)

actions = [
    coach_env.Action.NO,
    coach_env.Action.WARN,
    coach_env.Action.REDUCE,
    coach_env.Action.EXIT,
    coach_env.Action.COOLDOWN
]

for i in range(10):
    action = actions[i % len(actions)]

    next_state, reward, done, info = ce.step(action)

    print(f"\nStep {i+1}")
    print("Action:", action)
    print("State:", next_state)
    print("Reward:", reward)
    print("Done:", done)

    if done:
        break