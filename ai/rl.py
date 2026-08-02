"""PyTorch RL agents for Brick Blast experiments.

These implementations keep the training logic lightweight and testable while
remaining compatible with the repository's GPU-first tensor workflows.
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from env.tensor_env import TensorBrickBlastEnv


class DQNAgent(nn.Module):
    """Simple DQN with a target network for stable value learning."""

    def __init__(self, input_dim=21, action_dim=18, hidden_dim=64, gamma=0.99, device="cpu"):
        super().__init__()
        self.input_dim = int(input_dim)
        self.action_dim = int(action_dim)
        self.gamma = float(gamma)
        self.device = torch.device(device)

        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.action_dim),
        ).to(self.device)
        self.target_net = copy.deepcopy(self.net).to(self.device)
        self.target_net.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)

    def forward(self, obs):
        obs = obs.to(self.device)
        return self.net(obs)

    def act(self, obs, epsilon=0.0):
        obs = obs.to(self.device)
        if torch.rand(1).item() < epsilon:
            return torch.randint(0, self.action_dim, (obs.shape[0],), device=self.device)
        q_values = self(obs)
        return q_values.argmax(dim=-1)

    def update(self, obs, actions, next_obs, rewards, dones):
        obs = obs.to(self.device)
        actions = actions.to(self.device).long()
        next_obs = next_obs.to(self.device)
        rewards = rewards.to(self.device).float()
        dones = dones.to(self.device).bool()

        current_q = self(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(next_obs).max(dim=1).values
            target_q = rewards + (1.0 - dones.float()) * self.gamma * next_q

        loss = F.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._soft_update_target()
        return loss.detach()

    @torch.no_grad()
    def _soft_update_target(self, tau=0.005):
        for target_param, param in zip(self.target_net.parameters(), self.net.parameters()):
            target_param.data.mul_(1.0 - tau).add_(param.data, alpha=tau)


class PPOAgent(nn.Module):
    """Minimal PPO-style policy-gradient agent for tensorized environments."""

    def __init__(self, input_dim=21, action_dim=18, hidden_dim=64, device="cpu"):
        super().__init__()
        self.input_dim = int(input_dim)
        self.action_dim = int(action_dim)
        self.device = torch.device(device)

        self.actor = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, self.action_dim),
        ).to(self.device)

        self.critic = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        ).to(self.device)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

    def forward(self, obs):
        obs = obs.to(self.device)
        return self.actor(obs)

    def evaluate(self, obs):
        obs = obs.to(self.device)
        logits = self.actor(obs)
        log_probs = F.log_softmax(logits, dim=-1)
        values = self.critic(obs).squeeze(-1)
        return logits, log_probs, values

    def update(self, obs, actions, old_log_probs, returns, advantages):
        obs = obs.to(self.device)
        actions = actions.to(self.device).long()
        old_log_probs = old_log_probs.to(self.device).float()
        returns = returns.to(self.device).float()
        advantages = advantages.to(self.device).float()

        _, log_probs, values = self.evaluate(obs)
        action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        ratio = torch.exp(action_log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 0.8, 1.2)
        actor_loss = -(torch.minimum(ratio * advantages, clipped_ratio * advantages)).mean()

        critic_loss = F.mse_loss(values, returns)
        total_loss = actor_loss + 0.5 * critic_loss

        self.actor_opt.zero_grad()
        self.critic_opt.zero_grad()
        total_loss.backward()
        self.actor_opt.step()
        self.critic_opt.step()
        return total_loss.detach()


class TensorPolicyTrainer:
    """Collect trajectories from the batched tensor env and train RL plugins."""

    def __init__(self, agent, batch_size=64, max_turns=100, device="cpu", seed=None):
        self.agent = agent
        self.batch_size = int(batch_size)
        self.max_turns = int(max_turns)
        self.device = torch.device(device)
        self.env = TensorBrickBlastEnv(batch_size=self.batch_size, max_balls=30, device=self.device, seed=seed)
        self.seed = seed

    @staticmethod
    def _grid_to_obs(grids, globals_arr):
        batch_size = grids.shape[0]
        flat_grid = grids.reshape(batch_size, -1)
        return torch.cat([flat_grid, globals_arr], dim=-1)

    def _action_to_angle(self, actions):
        action_values = actions.to(self.device).long()
        angle_table = torch.linspace(3.0, 177.0, self.agent.action_dim, device=self.device)
        return angle_table[action_values]

    def collect_trajectories(self, rollout_steps=1, epsilon=0.1):
        obs_list = []
        action_list = []
        reward_list = []
        next_obs_list = []
        done_list = []

        for _ in range(int(rollout_steps)):
            if self.env.terminated.all():
                self.env.reset(seed=self.seed)

            grids, globals_arr = self.env.get_grid_observation()
            obs = self._grid_to_obs(grids, globals_arr)
            actions = self.agent.act(obs, epsilon=epsilon)
            angles = self._action_to_angle(actions)

            (next_grids, next_globals), rewards, terminated, info = self.env.step(angles)
            next_obs = self._grid_to_obs(next_grids, next_globals)

            obs_list.append(obs)
            action_list.append(actions)
            reward_list.append(rewards)
            next_obs_list.append(next_obs)
            done_list.append(terminated)

        return {
            "obs": torch.stack(obs_list, dim=0),
            "actions": torch.stack(action_list, dim=0),
            "rewards": torch.stack(reward_list, dim=0),
            "next_obs": torch.stack(next_obs_list, dim=0),
            "dones": torch.stack(done_list, dim=0),
        }

    def train_step(self, rollout_steps=1, epsilon=0.1):
        traj = self.collect_trajectories(rollout_steps=rollout_steps, epsilon=epsilon)

        flat_obs = traj["obs"].reshape(-1, self.agent.input_dim)
        flat_next_obs = traj["next_obs"].reshape(-1, self.agent.input_dim)
        flat_actions = traj["actions"].reshape(-1)
        flat_rewards = traj["rewards"].reshape(-1)
        flat_dones = traj["dones"].reshape(-1)

        if isinstance(self.agent, DQNAgent):
            return self.agent.update(flat_obs, flat_actions, flat_next_obs, flat_rewards, flat_dones)

        if isinstance(self.agent, PPOAgent):
            logits, log_probs, values = self.agent.evaluate(flat_obs)
            returns = flat_rewards.clone()
            advantages = flat_rewards.clone()
            return self.agent.update(
                flat_obs,
                flat_actions,
                log_probs.gather(1, flat_actions.unsqueeze(1)).squeeze(1).detach(),
                returns,
                advantages,
            )

        raise TypeError(f"Unsupported agent type: {type(self.agent)}")
