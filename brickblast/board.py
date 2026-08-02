"""
Board class managing the 8x10 grid, brick and powerup spawning, and turn progression.
"""
import random
from brickblast.constants import (
    GRID_COLS,
    GRID_ROWS,
    WIDTH,
    HEIGHT,
    TOP_MARGIN,
    BOTTOM_MARGIN,
    CELL_HEIGHT,
    COLOR_BORDER,
    COLOR_GROUND,
    COLOR_TEXT_WHITE,
)
from brickblast.brick import Brick
from brickblast.powerups import RedirectPowerup, MultiplierPowerup, LaserPowerup
from brickblast.pygame_compat import pygame, Rect


class Board:
    def __init__(self, seed=None):
        if seed is not None:
            random.seed(seed)
        self.bricks = []
        self.powerups = []
        self.used_powerups = set()  # ids of powerups triggered this turn
        self.laser_beams = []  # active laser animation effects
        self.turn = 1
        self.score = 0
        self.game_over = False
        self.launch_x = float(WIDTH // 2)
        self.next_launch_x = None
        self.reset()

    def reset(self, seed=None):
        if seed is not None:
            random.seed(seed)
        self.bricks.clear()
        self.powerups.clear()
        self.used_powerups.clear()
        self.laser_beams.clear()
        self.turn = 1
        self.score = 0
        self.game_over = False
        self.launch_x = float(WIDTH // 2)
        self.next_launch_x = None

        # Spawn initial 3 rows
        for _ in range(3):
            self.step_turn(initial=True)
        self.turn = 1

    def spawn_row(self):
        """
        Spawn bricks and powerups in row 0.
        """
        base_hp = self.turn * 10
        cols = list(range(GRID_COLS))
        random.shuffle(cols)

        # Ensure at least 3 bricks per row
        num_bricks = random.randint(3, 5)
        num_powerups = random.randint(0, 2)

        for col in cols[:num_bricks]:
            hp_variation = random.randint(-max(1, base_hp // 4), max(1, base_hp // 4))
            hp = max(5, base_hp + hp_variation)
            self.bricks.append(Brick(col, 0, hp))

        for col in cols[num_bricks:num_bricks + num_powerups]:
            pu_type = random.choice(["redirect", "multiplier", "laser"])
            if pu_type == "redirect":
                angle = random.choice([45, 90, 135])
                self.powerups.append(RedirectPowerup(col, 0, angle))
            elif pu_type == "multiplier":
                self.powerups.append(MultiplierPowerup(col, 0))
            elif pu_type == "laser":
                mode = random.choice(["horizontal", "vertical", "both"])
                self.powerups.append(LaserPowerup(col, 0, mode))

    def step_turn(self, initial=False, layers=1):
        """
        Advance all items down 'layers' rows and spawn 'layers' new top rows.
        """
        for _ in range(layers):
            # Check game over before shifting
            for b in self.bricks:
                if b.row >= GRID_ROWS - 1 and not initial:
                    self.game_over = True
                    return

            for b in self.bricks:
                b.move_down()
            for pu in self.powerups:
                pu.move_down()

            # Check if any brick overflowed bottom row
            for b in self.bricks:
                if b.row >= GRID_ROWS:
                    self.game_over = True
                    return

            # Remove powerups that were triggered during the turn
            self.clean_used_powerups()

            self.spawn_row()

        if not initial:
            self.turn += 1
            if self.next_launch_x is not None:
                self.launch_x = self.next_launch_x
                self.next_launch_x = None

    def clean_used_powerups(self):
        """Remove powerups that were triggered during this turn."""
        if self.used_powerups:
            self.powerups = [pu for pu in self.powerups if id(pu) not in self.used_powerups]
            self.used_powerups.clear()

    def draw(self, surface, font):
        # Draw subtle grid background lines
        grid_color = (32, 45, 80)
        for c in range(GRID_COLS + 1):
            x = c * (WIDTH // GRID_COLS)
            pygame.draw.line(surface, grid_color, (x, TOP_MARGIN), (x, HEIGHT - BOTTOM_MARGIN), 1)
        for r in range(GRID_ROWS + 1):
            y = TOP_MARGIN + r * ((HEIGHT - TOP_MARGIN - BOTTOM_MARGIN) // GRID_ROWS)
            pygame.draw.line(surface, grid_color, (0, y), (WIDTH, y), 1)

        # Draw top playing field border glow
        pygame.draw.line(surface, (55, 80, 150), (0, TOP_MARGIN), (WIDTH, TOP_MARGIN), 2)

        # Draw bottom ground line and styled play zone
        ground_y = HEIGHT - BOTTOM_MARGIN
        pygame.draw.rect(surface, (15, 24, 52), Rect(0, ground_y, WIDTH, BOTTOM_MARGIN))
        pygame.draw.line(surface, (60, 95, 175), (0, ground_y), (WIDTH, ground_y), 2)
        pygame.draw.line(surface, (30, 48, 95), (0, ground_y + 2), (WIDTH, ground_y + 2), 1)

        # Draw checkered Finish / Danger Line at row 9 (where bricks trigger Game Over)
        finish_y = TOP_MARGIN + (GRID_ROWS - 1) * CELL_HEIGHT
        pygame.draw.line(surface, (220, 50, 50), (0, finish_y), (WIDTH, finish_y), 2)
        square_size = 12
        for x in range(0, WIDTH, square_size * 2):
            pygame.draw.rect(surface, (255, 200, 40), Rect(x, finish_y + 2, square_size, 4))
            pygame.draw.rect(surface, (200, 40, 40), Rect(x + square_size, finish_y + 2, square_size, 4))

        # Draw all bricks and powerups
        for b in self.bricks:
            b.draw(surface, font)
        for pu in self.powerups:
            pu.draw(surface, font)

        # Draw animated laser beams of white light hitting all blocks
        if hasattr(self, 'laser_beams') and self.laser_beams:
            active_beams = []
            for beam in self.laser_beams:
                timer = beam.get("timer", 15)
                max_time = beam.get("max_time", 15)
                if timer > 0:
                    alpha_ratio = timer / max(1, max_time)
                    w_outer = max(6, int(18 * alpha_ratio))
                    w_mid = max(4, int(10 * alpha_ratio))
                    w_core = max(2, int(5 * alpha_ratio))
                    mode = beam.get("mode", "horizontal")
                    if mode in ("horizontal", "both") and "y" in beam:
                        y = beam["y"]
                        pygame.draw.line(surface, (100, 180, 255), (0, y), (WIDTH, y), w_outer)
                        pygame.draw.line(surface, (200, 240, 255), (0, y), (WIDTH, y), w_mid)
                        pygame.draw.line(surface, (255, 255, 255), (0, y), (WIDTH, y), w_core)
                        # Visualize white light hitting all blocks along the row
                        for c in range(GRID_COLS):
                            cx = c * (WIDTH // GRID_COLS) + (WIDTH // (2 * GRID_COLS))
                            pygame.draw.circle(surface, (255, 255, 255), (cx, y), max(4, int(9 * alpha_ratio)))
                            pygame.draw.circle(surface, (180, 225, 255), (cx, y), max(7, int(15 * alpha_ratio)), 2)
                    if mode in ("vertical", "both") and "x" in beam:
                        x = beam["x"]
                        pygame.draw.line(surface, (100, 180, 255), (x, TOP_MARGIN), (x, HEIGHT - BOTTOM_MARGIN), w_outer)
                        pygame.draw.line(surface, (200, 240, 255), (x, TOP_MARGIN), (x, HEIGHT - BOTTOM_MARGIN), w_mid)
                        pygame.draw.line(surface, (255, 255, 255), (x, TOP_MARGIN), (x, HEIGHT - BOTTOM_MARGIN), w_core)
                        # Visualize white light hitting all blocks along the column
                        for r in range(GRID_ROWS):
                            cy = TOP_MARGIN + r * ((HEIGHT - TOP_MARGIN - BOTTOM_MARGIN) // GRID_ROWS) + ((HEIGHT - TOP_MARGIN - BOTTOM_MARGIN) // (2 * GRID_ROWS))
                            pygame.draw.circle(surface, (255, 255, 255), (x, cy), max(4, int(9 * alpha_ratio)))
                            pygame.draw.circle(surface, (180, 225, 255), (x, cy), max(7, int(15 * alpha_ratio)), 2)
                    beam["timer"] = timer - 1
                    if beam["timer"] > 0:
                        active_beams.append(beam)
            self.laser_beams = active_beams
