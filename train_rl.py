#!/usr/bin/env python3
"""Minimal RL trainer for Brick Blast using the tensorized environment.

This script exercises the repo's PPO/DQN trainer harness and collects real
rollout trajectories from the batched tensor environment.
"""

import argparse
import time

import torch

from ai.rl import DQNAgent, PPOAgent, TensorPolicyTrainer


def build_agent(agent_type, input_dim, action_dim, device):
    if agent_type == "dqn":
        return DQNAgent(input_dim=input_dim, action_dim=action_dim, device=device)
    if agent_type == "ppo":
        return PPOAgent(input_dim=input_dim, action_dim=action_dim, device=device)
    raise ValueError(f"Unsupported agent type: {agent_type}")


def main():
    parser = argparse.ArgumentParser(description="Brick Blast RL Trainer")
    parser.add_argument("--agent", choices=["dqn", "ppo"], default="dqn")
    parser.add_argument("--episodes", type=int, default=50, help="Number of rollout epochs to run")
    parser.add_argument("--rollout-steps", type=int, default=8, help="Rollout steps per training update")
    parser.add_argument("--batch-size", type=int, default=32, help="Tensor env batch size")
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--save-path", type=str, default="saved_models/best_rl_model.pt")
    args = parser.parse_args()

    input_dim = 2 * 10 * 8 + 2  # flattened grid channels + global state
    action_dim = 18

    agent = build_agent(args.agent, input_dim=input_dim, action_dim=action_dim, device=args.device)
    trainer = TensorPolicyTrainer(
        agent=agent,
        batch_size=args.batch_size,
        max_turns=args.max_turns,
        device=args.device,
        seed=args.seed,
    )

    print(f"=== Brick Blast RL Training ===")
    print(f"Agent: {args.agent.upper()} | Device: {args.device.upper()} | Episodes: {args.episodes}")

    start = time.time()
    for episode in range(1, args.episodes + 1):
        loss = trainer.train_step(rollout_steps=args.rollout_steps, epsilon=args.epsilon)
        print(f"[Ep {episode:03d}] loss={float(loss):.4f}")

    runtime = time.time() - start
    torch.save(agent.state_dict(), args.save_path)
    print(f"Training complete in {runtime:.2f}s | saved model to {args.save_path}")


if __name__ == "__main__":
    main()
