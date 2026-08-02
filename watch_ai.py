#!/usr/bin/env python3
"""
Watch a trained Genetic AI model play Brick Blast!
Usage:
    python watch_ai.py --model saved_models/best_model.json
"""
import argparse
import time
import os
import sys
from brickblast.pygame_compat import PYGAME_AVAILABLE

if not PYGAME_AVAILABLE:
    print("ERROR: pygame is not installed. Watch mode requires pygame.")
    print("Install it with:  pip install pygame")
    sys.exit(1)

from ai.genome import Genome
from env.brickblast_env import BrickBlastEnv
from brickblast.constants import WIDTH, HEIGHT, FPS
from brickblast.game import BrickBlastGame
from brickblast.pygame_compat import pygame


def main():
    parser = argparse.ArgumentParser(description="Watch Brick Blast Genetic AI Play")
    parser.add_argument("--model", type=str, default="saved_models/best_model.json", help="Path to saved JSON model")
    parser.add_argument("--fast", action="store_true", help="Enable fast-forward by default")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found. Train an AI first with train_genetic.py!")
        return

    genome = Genome.load(args.model)
    print(f"Loaded best AI model from '{args.model}' (Fitness: {genome.fitness:.1f}, Turns: {genome.turns_survived})")

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
            best_angle = genome.select_action(env)
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
