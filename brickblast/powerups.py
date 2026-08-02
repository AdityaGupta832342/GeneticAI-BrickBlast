"""
Powerup classes for Brick Blast:
- RedirectPowerup (45 deg, 90 deg, 145 deg)
- MultiplierPowerup (Redirect and balls * 3)
- LaserPowerup (horizontal, vertical, or cross)
"""
import math
from brickblast.constants import (
    CELL_WIDTH,
    CELL_HEIGHT,
    BRICK_WIDTH,
    BRICK_HEIGHT,
    TOP_MARGIN,
    COLOR_REDIRECT,
    COLOR_MULTIPLIER,
    COLOR_LASER,
    COLOR_TEXT_WHITE,
    BALL_SPEED,
)
from brickblast.pygame_compat import pygame, Rect


def _angle_to_vel(angle_deg, speed=BALL_SPEED):
    """
    Convert angle in degrees (0 = right, 90 = up, 180 = left) to velocity vector (vx, vy).
    Pygame y-axis points down, so upwards is negative vy.
    """
    rad = math.radians(angle_deg)
    vx = math.cos(rad) * speed
    vy = -math.sin(rad) * speed
    return vx, vy


def _draw_arrow(surface, color, start, end, width=2, arrow_size=4):
    """Draw a line with an arrowhead from start to end."""
    pygame.draw.line(surface, color, start, end, width)
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    angle = math.atan2(dy, dx)
    w1_x = end[0] - arrow_size * math.cos(angle - math.pi / 6)
    w1_y = end[1] - arrow_size * math.sin(angle - math.pi / 6)
    w2_x = end[0] - arrow_size * math.cos(angle + math.pi / 6)
    w2_y = end[1] - arrow_size * math.sin(angle + math.pi / 6)
    pygame.draw.polygon(surface, color, [end, (w1_x, w1_y), (w2_x, w2_y)])


class BasePowerup:
    def __init__(self, col, row):
        self.col = int(col)
        self.row = int(row)
        self.scored = False  # True after first ball triggers it (score awarded once)
        self.update_rect()

    def update_rect(self):
        x = self.col * CELL_WIDTH + 1
        y = TOP_MARGIN + self.row * CELL_HEIGHT + 1
        self.rect = Rect(x, y, BRICK_WIDTH, BRICK_HEIGHT)

    def trigger(self, board):
        pass

    def draw(self, surface, font):
        pass

    def move_down(self):
        self.row += 1
        self.update_rect()


class RedirectPowerup(BasePowerup):
    def __init__(self, col, row, angle_deg=None):
        super().__init__(col, row)
        self.angles = [45, 90, 135]
        if angle_deg in self.angles:
            self.angle_idx = self.angles.index(angle_deg)
        else:
            self.angle_idx = 0
        self.type = "redirect"

    @property
    def angle_deg(self):
        return self.angles[self.angle_idx]

    @angle_deg.setter
    def angle_deg(self, value):
        if value in self.angles:
            self.angle_idx = self.angles.index(value)

    def apply(self, ball):
        """
        Redirect ball to current angle_deg at constant BALL_SPEED,
        then alternate to the next angle in (45°, 90°, 135°).
        Changes velocity in-place and nudges ball in the new direction.
        """
        current_angle = self.angle_deg
        vx, vy = _angle_to_vel(current_angle, BALL_SPEED)
        # Originate ball from the exact center of the circle
        ball.x = float(self.rect.centerx)
        ball.y = float(self.rect.centery)
        ball.vx = vx
        ball.vy = vy
        # Nudge ball one step in new direction to clear powerup zone
        ball.x += vx * 0.5
        ball.y += vy * 0.5
        # Advance to next angle in alternating sequence
        self.angle_idx = (self.angle_idx + 1) % len(self.angles)

    def draw(self, surface, font):
        cx, cy = self.rect.centerx, self.rect.centery
        # Outer glowing orbit ring
        pygame.draw.circle(surface, (15, 30, 22), (cx, cy + 2), 17)  # Shadow
        pygame.draw.circle(surface, (46, 204, 113), (cx, cy), 17, 2)
        # Inner emerald badge background
        pygame.draw.circle(surface, (18, 65, 48), (cx, cy), 14)
        # Directional arrow icon originating from the center of the circle
        rad = math.radians(self.angle_deg)
        tip_x = cx + int(math.cos(rad) * 13)
        tip_y = cy - int(math.sin(rad) * 13)
        _draw_arrow(surface, (255, 255, 255), (cx, cy), (tip_x, tip_y), width=2, arrow_size=5)
        # Center white ball drawn over arrow origin
        pygame.draw.circle(surface, (255, 255, 255), (cx, cy), 3)
        # Angle text label below badge
        label = f"{self.angle_deg}°"
        text_shadow = font.render(label, True, (10, 20, 30))
        text_surf = font.render(label, True, (255, 255, 255))
        surface.blit(text_shadow, (cx - text_surf.get_width() // 2 + 1, cy + 12 + 1))
        surface.blit(text_surf, (cx - text_surf.get_width() // 2, cy + 12))


class MultiplierPowerup(BasePowerup):
    def __init__(self, col, row):
        super().__init__(col, row)
        self.type = "multiplier"

    def apply(self, ball, new_balls_list):
        """
        Redirect ball and spawn 2 additional active balls so 3 total balls
        travel outwards at 45°, 90°, and 135°.
        """
        from brickblast.ball import Ball
        angles = [45, 90, 135]
        # Originate all 3 balls from the exact center of the circle
        spawn_x = float(self.rect.centerx)
        spawn_y = float(self.rect.centery)

        # Assign first angle to the incoming ball
        vx0, vy0 = _angle_to_vel(angles[0], BALL_SPEED)
        ball.x = spawn_x + vx0 * 0.5
        ball.y = spawn_y + vy0 * 0.5
        ball.vx = vx0
        ball.vy = vy0

        # Spawn 2 new balls for remaining angles (90 and 135)
        for ang in angles[1:]:
            vx, vy = _angle_to_vel(ang, BALL_SPEED)
            new_ball = Ball(spawn_x + vx * 0.5, spawn_y + vy * 0.5)
            new_ball.active = True
            new_ball.vx = vx
            new_ball.vy = vy
            # Inherit parent's triggered set so spawned balls don't
            # re-trigger the same multiplier (prevents infinite chain)
            new_ball.triggered_powerups = set(ball.triggered_powerups)
            new_balls_list.append(new_ball)

    def draw(self, surface, font):
        cx, cy = self.rect.centerx, self.rect.centery
        # Outer glowing blue/indigo orbit ring (matching real game icon)
        pygame.draw.circle(surface, (15, 25, 45), (cx, cy + 2), 17)  # Shadow
        pygame.draw.circle(surface, (80, 160, 255), (cx, cy), 17, 2)
        # Inner rich indigo background
        pygame.draw.circle(surface, (25, 55, 125), (cx, cy), 14)
        # Three directional arrows (45°, 90°, 135°) originating from center of circle
        for ang in [45, 90, 135]:
            rad = math.radians(ang)
            tip_x = cx + int(math.cos(rad) * 13)
            tip_y = cy - int(math.sin(rad) * 13)
            _draw_arrow(surface, (255, 230, 100), (cx, cy), (tip_x, tip_y), width=2, arrow_size=4)
        # Center white ball drawn over arrow origin
        pygame.draw.circle(surface, (255, 255, 255), (cx, cy), 3)
        # Label x3
        label = "x3"
        text_shadow = font.render(label, True, (10, 20, 30))
        text_surf = font.render(label, True, (255, 255, 255))
        surface.blit(text_shadow, (cx - text_surf.get_width() // 2 + 1, cy + 12 + 1))
        surface.blit(text_surf, (cx - text_surf.get_width() // 2, cy + 12))


class LaserPowerup(BasePowerup):
    def __init__(self, col, row, mode="horizontal"):
        super().__init__(col, row)
        # Modes: 'horizontal', 'vertical', 'both'
        if mode not in ("horizontal", "vertical", "both"):
            mode = "horizontal"
        self.mode = mode
        self.type = "laser"

    def trigger(self, board):
        """
        Shoots laser across row/column, dealing 1 damage to all bricks in line.
        """
        if not hasattr(board, "laser_beams"):
            board.laser_beams = []
        if self.mode in ("horizontal", "both"):
            board.laser_beams.append({"mode": "horizontal", "y": self.rect.centery, "timer": 15, "max_time": 15})
        if self.mode in ("vertical", "both"):
            board.laser_beams.append({"mode": "vertical", "x": self.rect.centerx, "timer": 15, "max_time": 15})

        destroyed_bricks = []
        for brick in list(board.bricks):
            hit_by_laser = False
            if self.mode in ("horizontal", "both") and brick.row == self.row:
                hit_by_laser = True
            if self.mode in ("vertical", "both") and brick.col == self.col:
                hit_by_laser = True

            if hit_by_laser:
                if brick.hit(damage=1):
                    destroyed_bricks.append(brick)

        for b in destroyed_bricks:
            if b in board.bricks:
                board.bricks.remove(b)
                board.score += 10
        return len(destroyed_bricks)

    def draw(self, surface, font):
        cx, cy = self.rect.centerx, self.rect.centery
        # Outer glowing crimson target ring
        pygame.draw.circle(surface, (30, 15, 20), (cx, cy + 2), 17)  # Shadow
        pygame.draw.circle(surface, (240, 60, 85), (cx, cy), 17, 2)
        # Inner ruby badge background
        pygame.draw.circle(surface, (90, 20, 30), (cx, cy), 14)
        # Center white ball
        pygame.draw.circle(surface, (255, 255, 255), (cx, cy), 3)
        # Laser beams / target reticle arrows
        if self.mode in ("horizontal", "both"):
            _draw_arrow(surface, (255, 120, 140), (cx, cy), (cx - 12, cy), width=2, arrow_size=4)
            _draw_arrow(surface, (255, 120, 140), (cx, cy), (cx + 12, cy), width=2, arrow_size=4)
        if self.mode in ("vertical", "both"):
            _draw_arrow(surface, (255, 120, 140), (cx, cy), (cx, cy - 12), width=2, arrow_size=4)
            _draw_arrow(surface, (255, 120, 140), (cx, cy), (cx, cy + 12), width=2, arrow_size=4)
