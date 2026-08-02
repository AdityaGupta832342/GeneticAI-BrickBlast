#!/usr/bin/env python3
"""
Render a high-resolution sample frame of the Brick Blast game to demonstrate
the 8x10 board, numbered bricks, and Redirect/Multiplier/Laser powerups.
"""
import os
import sys
from brickblast.constants import WIDTH, HEIGHT, TOP_MARGIN, BOTTOM_MARGIN
from brickblast.game import BrickBlastGame
from brickblast.brick import Brick
from brickblast.powerups import RedirectPowerup, MultiplierPowerup, LaserPowerup
from brickblast.pygame_compat import pygame, Rect, Surface


def render_preview(filepath):
    game = BrickBlastGame(headless=False, seed=101)
    # Clear and set up a custom showcase board with all features
    board = game.board
    board.bricks.clear()
    board.powerups.clear()
    board.turn = 12
    board.score = 1480

    # Add diverse colored bricks
    board.bricks.append(Brick(0, 1, 40))   # Green
    board.bricks.append(Brick(2, 1, 75))   # Blue
    board.bricks.append(Brick(4, 1, 130))  # Yellow
    board.bricks.append(Brick(6, 1, 210))  # Red
    board.bricks.append(Brick(7, 1, 280))  # Purple

    board.bricks.append(Brick(1, 3, 60))
    board.bricks.append(Brick(3, 3, 115))
    board.bricks.append(Brick(5, 3, 160))

    board.bricks.append(Brick(2, 5, 90))
    board.bricks.append(Brick(4, 5, 45))
    board.bricks.append(Brick(6, 5, 140))

    # Add all powerup variants
    board.powerups.append(RedirectPowerup(1, 1, 45))
    board.powerups.append(RedirectPowerup(5, 1, 90))
    board.powerups.append(RedirectPowerup(0, 3, 135))
    board.powerups.append(MultiplierPowerup(4, 3))
    board.powerups.append(LaserPowerup(2, 2, "horizontal"))
    board.powerups.append(LaserPowerup(7, 3, "vertical"))
    board.powerups.append(LaserPowerup(3, 5, "both"))

    # Add a sample active laser beam animation to visualize white light hitting blocks
    board.laser_beams.append({
        "mode": "horizontal",
        "y": TOP_MARGIN + 2 * ((HEIGHT - TOP_MARGIN - BOTTOM_MARGIN) // 10) + ((HEIGHT - TOP_MARGIN - BOTTOM_MARGIN) // 20),
        "timer": 15,
        "max_time": 15
    })

    game.set_aim(68.0)

    surface = Surface((WIDTH, HEIGHT))
    game.render(surface)

    # Ensure directory exists and save PNG
    dirname = os.path.dirname(filepath)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    pygame.image.save(surface, filepath)
    print(f"Saved demo preview frame to: {filepath}")


if __name__ == "__main__":
    out_path = sys.argv[1] if len(sys.argv) > 1 else "game_preview.png"
    render_preview(out_path)
