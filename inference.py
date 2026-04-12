"""
inference.py - root directory
"""
import asyncio
import os
from typing import List, Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

from trade_env.env.coach_env import CoachEnv
from trade_env.schemas.action import Action, ActionType


API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.2-3B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

TASK_NAME   = "trader-coach"
BENCHMARK   = "coach-env"
MAX_STEPS   = 20
SUCCESS_SCORE_THRESHOLD = 0.1


def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step, action, reward, done, error=None):
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)


def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def get_llm_action(client: OpenAI, state: dict, step: int) -> int:
    prompt = f"""You are a trading behavior coach. Given trader state:
- timestep: {state['timestep']}
- price: {state['price']:.2f}
- position: {state['position']}
- loss_streak: {state['loss_streak']}
- pnl: {state['pnl']:.2f}

Choose intervention (reply with single integer only):
0=NO, 1=WARN, 2=REDUCE, 3=EXIT, 4=COOLDOWN"""

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0.0,
        )
        raw = (completion.choices[0].message.content or "").strip()
        action = int(raw)
        if action not in range(5):
            action = 0
        return action
    except Exception:
        # rule-based fallback with normalized values
        loss = state["loss_streak"]   # 0.0 to 1.0
        pnl  = state["pnl"]          # -1.0 to 1.0

        if loss >= 0.2:   return 4   # COOLDOWN
        if loss >= 0.1:   return 3   # EXIT  
        if pnl  < -0.3:   return 2   # REDUCE
        if loss >  0.0:   return 1   # WARN
        return 0                     # NO


def main():
    client = OpenAI(
        api_key=HF_TOKEN,
        base_url=API_BASE_URL
    )

    env = CoachEnv()
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(TASK_NAME, BENCHMARK, MODEL_NAME)

    try:
        state = env.reset()

        for step in range(1, MAX_STEPS + 1):
            action_idx = get_llm_action(client, state, step)
            action = Action(action=ActionType(action_idx))

            next_state, reward, done, info = env.step(action)

            log_step(step, ActionType(action_idx).name, reward, done)

            rewards.append(reward)
            steps_taken = step
            state = next_state

            if done:
                break

        score = sum(rewards) / MAX_STEPS
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        log_step(steps_taken + 1, "NO", 0.0, True, error=str(e))
        success = False
        score = 0.0
        rewards = rewards or [0.0]

    finally:
        log_end(success, steps_taken, score, rewards)


if __name__ == "__main__":
    main()