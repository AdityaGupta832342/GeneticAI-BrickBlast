#!/usr/bin/env python3
"""
Watch a trained AI model play Brick Blast!
Usage:
    python watch_ai.py
    python watch_ai.py --model saved_models/best_rl_model.pt
"""
import argparse
import os
import sys
import torch
from brickblast.pygame_compat import PYGAME_AVAILABLE

if not PYGAME_AVAILABLE:
    print("ERROR: pygame is not installed. Watch mode requires pygame.")
    print("Install it with:  pip install pygame")
    sys.exit(1)

from ai.genome import Genome
from ai.rl import DQNAgent, PPOAgent
from env.brickblast_env import BrickBlastEnv
from brickblast.constants import WIDTH, HEIGHT, FPS
from brickblast.game import BrickBlastGame
from brickblast.pygame_compat import pygame


class TensorRLPolicy:
    """Visual-play adapter for tensor-trained discrete RL agents."""

    def __init__(self, model_path, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        state_dict = torch.load(model_path, map_location=self.device)
        self.agent = self._build_agent(state_dict)
        self.agent.load_state_dict(state_dict)
        self.agent.eval()
        self.angle_table = torch.linspace(3.0, 177.0, self.agent.action_dim, device=self.device)

    def _build_agent(self, state_dict):
        if any(key.startswith("actor.") for key in state_dict):
            input_dim = state_dict["actor.0.weight"].shape[1]
            action_dim = state_dict["actor.4.weight"].shape[0]
            return PPOAgent(input_dim=input_dim, action_dim=action_dim, device=self.device)

        input_dim = state_dict["net.0.weight"].shape[1]
        action_dim = state_dict["net.4.weight"].shape[0]
        return DQNAgent(input_dim=input_dim, action_dim=action_dim, device=self.device)

    @staticmethod
    def _grid_to_obs(grid, globals_arr):
        grid_t = torch.as_tensor(grid, dtype=torch.float32).reshape(1, -1)
        globals_t = torch.as_tensor(globals_arr, dtype=torch.float32).reshape(1, -1)
        return torch.cat([grid_t, globals_t], dim=-1)

    @torch.no_grad()
    def select_action(self, env):
        grid, globals_arr = env.get_grid_observation()
        obs = self._grid_to_obs(grid, globals_arr).to(self.device)

        if isinstance(self.agent, DQNAgent):
            action_idx = int(self.agent.act(obs, epsilon=0.0).item())
        else:
            logits = self.agent(obs)
            action_idx = int(logits.argmax(dim=-1).item())

        return float(self.angle_table[action_idx].item())


def load_policy(model_path):
    if model_path.endswith(".pt"):
        policy = TensorRLPolicy(model_path)
        print(f"Loaded tensor RL model from '{model_path}' ({policy.agent.__class__.__name__})")
        return policy

    genome = Genome.load(model_path)
    print(
        f"Loaded {genome.model_type.upper()} GA model from '{model_path}' "
        f"(Fitness: {genome.fitness:.1f}, Turns: {genome.turns_survived})"
    )
    return genome


def main():
    parser = argparse.ArgumentParser(description="Watch Brick Blast AI Play")
    parser.add_argument(
        "--model",
        type=str,
        default="saved_models/best_tensor_model.json",
        help="Path to a saved JSON GA model or .pt tensor RL model",
    )
    parser.add_argument("--fast", action="store_true", help="Enable fast-forward by default")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found. Train an AI first.")
        return

    policy = load_policy(args.model)

    if getattr(pygame, "init", None):
        pygame.init()

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Brick Blast - Watch AI Play")
    clock = pygame.time.Clock()

    game = BrickBlastGame(headless=False)
    game.fast_forward = args.fast
    env = BrickBlastEnv()
    env.game = game  # Share the visual game instance

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        game.fast_forward = bool(keys[pygame.K_f]) or args.fast

        if game.state == "aiming" and not game.board.game_over:
            # Let AI choose the best angle
            best_angle = policy.select_action(env)
            game.set_aim(best_angle)
            game.start_shot()
            print(f"[Turn {game.board.turn}] AI Aimed at {best_angle:.1f}° | Score: {game.board.score}")

        game.step_simulation()
        game.render(screen)
        pygame.display.flip()
        clock.tick(FPS)

    if getattr(pygame, "quit", None):
        pygame.quit()


if __name__ == "__main__":
    main()
