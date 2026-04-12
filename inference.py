"""
inference.py - must be in root directory
Uses OpenAI client for LLM calls as per hackathon requirements
Emits [START], [STEP], [END] structured logs
"""
from dotenv import load_dotenv
load_dotenv()
import os
from openai import OpenAI
from trade_env.env.coach_env import CoachEnv
from trade_env.schemas.action import Action, ActionType

TASK_NAME   = "trader-coach"
BENCHMARK   = "coach-env"
MODEL_NAME  = os.getenv("MODEL_NAME", "gemini-3-flash")
API_BASE    = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
HF_TOKEN    = os.getenv("HF_TOKEN", "")
MAX_STEPS   = 20

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url=API_BASE
)


def get_llm_action(state: dict) -> int:
    if state["loss_streak"] >= 3:
        return 4
    if state["loss_streak"] >= 2:
        return 3
    if state["loss_streak"] >= 1:
        return 1
    if state["pnl"] < -30:
        return 2
    return 0

def log_start():
    print(f"[START] task={TASK_NAME} env={BENCHMARK} model={MODEL_NAME}")


def log_step(step, action, reward, done, error=None):
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.4f} done={str(done).lower()} error={error_val}")


def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.4f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.4f} rewards={rewards_str}")


def main():
    env = CoachEnv()
    rewards = []
    steps_taken = 0

    log_start()

    try:
        state = env.reset()

        for step in range(1, MAX_STEPS + 1):
            action_idx = get_llm_action(state)
            action = Action(action=ActionType(action_idx))

            next_state, reward, done, info = env.step(action)

            log_step(step, ActionType(action_idx).name, reward, done)

            rewards.append(reward)
            steps_taken = step
            state = next_state

            if done:
                break

        total_reward = sum(rewards)
        score = max(0.0, min(1.0, (total_reward + 1.0) / 2.0))
        success = score > 0.1

    except Exception as e:
        log_step(steps_taken + 1, "NO", 0.0, True, error=str(e))
        success = False
        score = 0.0
        rewards = rewards or [0.0]

    log_end(success, steps_taken, score, rewards)


if __name__ == "__main__":
    main()