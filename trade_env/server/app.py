
"""
fast api endpoints which will be an HTTP server

"""

from fastapi import FastAPI
import uvicorn
from trade_env.env.coach_env import CoachEnv
from trade_env.schemas.action import Action
from trade_env.schemas.state import State
from trade_env.schemas.step_response import StepResponse

app = FastAPI()

env = CoachEnv()


@app.post("/reset",response_model=State)
def reset():
    state = env.reset()    
    return State(**state)   

@app.post("/step", response_model=StepResponse)
def step(action: Action):
    next_state, reward, done, info = env.step(action)

    return StepResponse(
        next_state=State(**next_state),
        reward=reward,
        done=done,
        info=info
    )

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()

    