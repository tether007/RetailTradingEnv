"""Trade Env Server Environment - wraps CoachEnv for OpenEnv server."""

from trade_env.env.coach_env import CoachEnv

env = CoachEnv()

def get_env() -> CoachEnv:
    return env