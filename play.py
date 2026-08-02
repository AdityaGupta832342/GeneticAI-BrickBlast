#!/usr/bin/env python3
"""
Play Brick Blast interactively in Pygame!
Controls:
- Mouse Move: Aim trajectory
- Left Click / SPACE: Shoot balls
- 'F' Key: Toggle Fast-Forward (3x speed)
- 'R' Key: Recall all balls immediately to start position
"""
import math
import sys
from brickblast.pygame_compat import PYGAME_AVAILABLE

if not PYGAME_AVAILABLE:
    print("ERROR: pygame is not installed. The interactive play mode requires pygame.")
    print("Install it with:  pip install pygame")
    sys.exit(1)

from brickblast.constants import WIDTH, HEIGHT, BOTTOM_MARGIN, FPS
from brickblast.game import BrickBlastGame
from brickblast.pygame_compat import pygame


def main():
    if getattr(pygame, "init", None):
        pygame.init()

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Brick Blast - Pygame & Genetic AI")
    clock = pygame.time.Clock()

    game = BrickBlastGame(headless=False)
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if game.state == "aiming":
                    game.start_shot()

        # Keyboard controls
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE] and game.state == "aiming":
            game.start_shot()
        if keys[pygame.K_r] and game.state in ("shooting", "simulating"):
            game.recall_balls()
        game.fast_forward = bool(keys[pygame.K_f])

        # Mouse aiming calculation
        if game.state == "aiming" and not game.board.game_over:
            mx, my = pygame.mouse.get_pos()
            dx = mx - game.board.launch_x
            dy = (HEIGHT - BOTTOM_MARGIN - 7) - my
            if dy > 5:
                angle_rad = math.atan2(dy, dx)
                angle_deg = math.degrees(angle_rad)
                game.set_aim(angle_deg)

        # Step simulation and render
        game.step_simulation()
        game.render(screen)
        pygame.display.flip()
        clock.tick(FPS)

    if getattr(pygame, "quit", None):
        pygame.quit()


if __name__ == "__main__":
    main()
