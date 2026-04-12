---
title: coach-env
emoji: ❕
colorFrom: red
colorTo: green
sdk: docker
pinned: false
---

# Retail Trader Behavior Coach

RL-powered trading behavior coach. Detects revenge trading, panic selling, and overconfidence — intervenes before the account blows up.

## Endpoints
- `POST /reset` — reset environment
- `POST /step` — take intervention action
- `GET /health` — health check

## Actions
- `0` = NO (do nothing)
- `1` = WARN (light nudge)
- `2` = REDUCE (reduce size)
- `3` = EXIT (exit position)
- `4` = COOLDOWN (force break)