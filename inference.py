"""
inference.py - root directory
3 tasks:
1. revenge-trade-detection  — catch loss_streak >= 2
2. panic-sell-prevention    — catch deep pnl < -0.3
3. overconfidence-correction — catch win streak + overtrading
"""
import os
from typing import List
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

from trade_env.env.coach_env import CoachEnv
from trade_env.schemas.action import Action, ActionType

API_BASE_URL     = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME       = os.getenv("MODEL_NAME", "meta-llama/Llama-3.2-3B-Instruct")
HF_TOKEN         = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

BENCHMARK        = "coach-env"
MAX_STEPS        = 20
SUCCESS_SCORE_THRESHOLD = 0.1

TASKS = {
    "revenge-trade-detection": {
        "desc": "Detect and intervene on revenge trading after loss streaks",
        "trigger": lambda s: s["loss_streak"] >= 0.2,
        "correct_actions": [3, 4],  # EXIT or COOLDOWN
    },
    "panic-sell-prevention": {
        "desc": "Prevent panic selling during drawdowns",
        "trigger": lambda s: s["pnl"] < -0.3,
        "correct_actions": [2, 3],  # REDUCE or EXIT
    },
    "overconfidence-correction": {
        "desc": "Correct overconfident trading after wins",
        "trigger": lambda s: s["overtrade_score"] >= 0.7 and s["pnl"] > 0.1,
        "correct_actions": [1, 2],  # WARN or REDUCE
    },
}


def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step, action, reward, done, error=None):
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)


def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def get_llm_action(client, state, task_name):
    prompt = (
        f"You are a trading behavior coach. Task: {task_name}.\n"
        f"Trader state: loss_streak={state['loss_streak']:.2f}, "
        f"pnl={state['pnl']:.2f}, overtrade_score={state['overtrade_score']:.2f}.\n"
        f"Reply with single digit only. 0=ignore 1=warn 2=reduce 3=exit 4=cooldown"
    )
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=3,
            temperature=0.0,
        )
        raw = (completion.choices[0].message.content or "").strip()[0]
        action = int(raw)
        if action not in range(5):
            raise ValueError
        return action
    except:
        pass

    # rule-based fallback
    loss = state["loss_streak"]
    pnl  = state["pnl"]
    over = state["overtrade_score"]

    if task_name == "revenge-trade-detection":
        if loss >= 0.2: return 4
        if loss >= 0.1: return 3
        if loss >  0.0: return 1
        return 0

    if task_name == "panic-sell-prevention":
        if pnl < -0.3: return 3
        if pnl < -0.1: return 2
        return 0

    if task_name == "overconfidence-correction":
        if over >= 0.7: return 2
        if over >= 0.5: return 1
        return 0

    return 0


def run_task(client, task_name: str) -> float:
    task = TASKS[task_name]
    env = CoachEnv()
    rewards: List[float] = []
    steps_taken = 0
    correct_interventions = 0
    total_triggers = 0

    log_start(task_name, BENCHMARK, MODEL_NAME)

    try:
        state = env.reset()

        for step in range(1, MAX_STEPS + 1):
            action_idx = get_llm_action(client, state, task_name)
            action = Action(action=ActionType(action_idx))

            next_state, reward, done, info = env.step(action)

            # grade: did agent pick correct action when trigger fired?
            if task["trigger"](state):
                total_triggers += 1
                if action_idx in task["correct_actions"]:
                    correct_interventions += 1
                    reward = abs(reward) + 0.1  # bonus for correct intervention

            log_step(step, ActionType(action_idx).name, reward, done)

            rewards.append(reward)
            steps_taken = step
            state = next_state

            if done:
                break

        # score = intervention accuracy when triggers fired
        if total_triggers > 0:
            score = correct_interventions / total_triggers
        else:
            score = sum(r for r in rewards if r > 0)
            score = min(1.0, score / 0.5)

        score = min(1.0, max(0.0, score))
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        log_step(steps_taken + 1, "NO", 0.0, True, error=str(e))
        success = False
        score = 0.0
        rewards = rewards or [0.0]

    finally:
        log_end(success, steps_taken, score, rewards)

    return score


def main():
    client = OpenAI(
        api_key=HF_TOKEN,
        base_url=API_BASE_URL
    )

    all_scores = []
    for task_name in TASKS:
        score = run_task(client, task_name)
        all_scores.append(score)

    avg = sum(all_scores) / len(all_scores)
    print(f"[SUMMARY] tasks={len(all_scores)} avg_score={avg:.3f}", flush=True)


if __name__ == "__main__":
    main()