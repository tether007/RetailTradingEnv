"""Proximal policy Optimization(PPO)"""
from trade_env.schemas.action import Action
from trade_env.schemas.state import State
from trade_env.schemas.step_response import StepResponse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class ActorCritic(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.actor = nn.Linear(64, action_dim)
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        x = self.shared(x)
        return self.actor(x), self.critic(x)


class PPOAgent(nn.Module):
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2):
        super().__init__()
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.model = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self._clear_memory()

    def _clear_memory(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def _state_to_tensor(self, state):
        return torch.tensor(list(state.values()), dtype=torch.float32)

    def select_action(self, state):
        state_t = self._state_to_tensor(state)
        with torch.no_grad():
            logits, value = self.model(state_t)
        
        dist = Categorical(logits=logits)
        action = dist.sample()

        self.states.append(state_t)
        self.actions.append(action)
        self.log_probs.append(dist.log_prob(action))
        self.values.append(value.squeeze())

        return action.item()

    def store_outcome(self, reward, done):
        self.rewards.append(reward)
        self.dones.append(done)

    def _compute_returns(self):
        returns = []
        G = 0
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                G = 0
            G = reward + self.gamma * G
            returns.insert(0, G)
        return torch.tensor(returns, dtype=torch.float32)

    def update(self, epochs=4):
        returns = self._compute_returns()

        # detach everything collected during rollout
        states = torch.stack(self.states).detach()
        actions = torch.stack(self.actions).detach()
        log_probs_old = torch.stack(self.log_probs).detach()
        values_old = torch.stack(self.values).detach()

        advantages = returns - values_old
        # normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(epochs):
            logits, new_values = self.model(states)
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions)

            ratio = torch.exp(new_log_probs - log_probs_old)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(new_values.squeeze(), returns)
            entropy_bonus = dist.entropy().mean()

            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy_bonus

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()

        self._clear_memory()

if __name__ == "__main__":
    agent = PPOAgent(state_dim=6, action_dim=5)
    print("PPOAgent instantiated successfully.")