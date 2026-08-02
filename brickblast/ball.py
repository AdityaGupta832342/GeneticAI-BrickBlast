"""
Ball class for Brick Blast:
Handles movement, wall/brick bouncing with proper penetration resolution,
and powerup tile triggers.
"""
import math
from brickblast.constants import (
    WIDTH,
    HEIGHT,
    TOP_MARGIN,
    BOTTOM_MARGIN,
    BALL_RADIUS,
    BALL_SPEED,
    COLOR_BALL,
)
from brickblast.pygame_compat import pygame, Rect


class Ball:
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)
        self.vx = 0.0
        self.vy = 0.0
        self.radius = BALL_RADIUS
        self.active = False
        self.landed = False
        self.triggered_powerups = set()  # track powerup ids already triggered by THIS ball

    def get_rect(self):
        return Rect(
            int(self.x - self.radius),
            int(self.y - self.radius),
            self.radius * 2,
            self.radius * 2,
        )

    def update(self, board, new_balls_list):
        if not self.active or self.landed:
            return

        # Perform 3 sub-steps per frame to prevent high-speed tunneling through blocks
        sub_steps = 3
        sub_vx = self.vx / sub_steps
        sub_vy = self.vy / sub_steps

        for _ in range(sub_steps):
            if not self.active or self.landed:
                break

            # Advance position by sub-step
            self.x += sub_vx
            self.y += sub_vy

            # Wall collisions — correct position AND velocity
            if self.x - self.radius <= 0:
                self.x = self.radius + 0.5
                self.vx = abs(self.vx)
                sub_vx = abs(sub_vx)
            elif self.x + self.radius >= WIDTH:
                self.x = WIDTH - self.radius - 0.5
                self.vx = -abs(self.vx)
                sub_vx = -abs(sub_vx)

            if self.y - self.radius <= TOP_MARGIN:
                self.y = TOP_MARGIN + self.radius + 0.5
                self.vy = abs(self.vy)
                sub_vy = abs(sub_vy)

            # Bottom ground check
            ground_y = HEIGHT - BOTTOM_MARGIN
            if self.y + self.radius >= ground_y:
                self.y = ground_y - self.radius
                self.active = False
                self.landed = True
                if board.next_launch_x is None:
                    board.next_launch_x = self.x
                return

            # --- Brick collisions with proper MTV ejection ---
            # Resolve up to 3 overlapping bricks per frame to prevent
            # the ball from getting stuck between adjacent blocks.
            for _ in range(3):
                brick, push_x, push_y, axis = self._find_brick_collision(board.bricks)
                if brick is None:
                    break

                # Eject ball completely outside the brick
                self.x += push_x
                self.y += push_y
                # Reflect the correct velocity component
                if axis == "x":
                    self.vx = -self.vx
                    sub_vx = -sub_vx
                elif axis == "y":
                    self.vy = -self.vy
                    sub_vy = -sub_vy
                elif isinstance(axis, tuple) and axis[0] == "corner":
                    _, nx, ny = axis
                    dot = self.vx * nx + self.vy * ny
                    if dot < 0:
                        speed = math.hypot(self.vx, self.vy)
                        self.vx = self.vx - 2 * dot * nx
                        self.vy = self.vy - 2 * dot * ny
                        new_speed = math.hypot(self.vx, self.vy)
                        if new_speed > 0:
                            self.vx = (self.vx / new_speed) * speed
                            self.vy = (self.vy / new_speed) * speed
                        sub_vx = self.vx / sub_steps
                        sub_vy = self.vy / sub_steps

                # Apply damage
                board.score += 1
                destroyed = brick.hit(damage=1)
                if destroyed:
                    if brick in board.bricks:
                        board.bricks.remove(brick)
                        board.score += 5
                # Note: Do not break here when a brick is destroyed!
                # If the ball simultaneously overlapped two bricks at a seam,
                # we must continue the loop to resolve the second brick overlap.

            # --- Powerup collisions ---
            self._check_powerup_collisions(board, new_balls_list)

    def _check_powerup_collisions(self, board, new_balls_list):
        """Check and trigger any powerups the ball intersects."""
        # Powerups persist for the entire turn — every ball gets the effect.
        # Each ball tracks which powerups it already triggered to avoid
        # re-triggering the same one every frame.
        for pu in board.powerups:
            pu_id = id(pu)
            if pu_id in self.triggered_powerups:
                continue
            if self._overlaps_rect(pu.rect):
                self.triggered_powerups.add(pu_id)
                board.used_powerups.add(pu_id)  # mark for end-of-turn removal

                if not pu.scored:
                    board.score += 5
                    pu.scored = True

                if pu.type == "redirect":
                    pu.apply(self)
                elif pu.type == "multiplier":
                    pu.apply(self, new_balls_list)
                elif pu.type == "laser":
                    pu.trigger(board)
                break

    def _overlaps_rect(self, rect):
        """Float-precision circle vs AABB overlap test."""
        # Find closest point on rect to ball center
        closest_x = max(float(rect.x), min(self.x, float(rect.x + rect.w)))
        closest_y = max(float(rect.y), min(self.y, float(rect.y + rect.h)))
        dx = self.x - closest_x
        dy = self.y - closest_y
        return (dx * dx + dy * dy) < (self.radius * self.radius)

    def _find_brick_collision(self, bricks):
        """
        Find the brick the ball overlaps with the LEAST penetration
        and compute the minimum translation vector (MTV) to eject it.
        Returns (brick, push_x, push_y, axis) or (None, 0, 0, None).
        """
        best_brick = None
        best_pen = float("inf")
        best_push = (0.0, 0.0)
        best_axis = None

        bl = self.x - self.radius
        br = self.x + self.radius
        bt = self.y - self.radius
        bb = self.y + self.radius

        for brick in bricks:
            r = brick.rect
            rl, rt = float(r.x), float(r.y)
            rr, rb = rl + float(r.w), rt + float(r.h)

            # Quick AABB reject
            if br <= rl or bl >= rr or bb <= rt or bt >= rb:
                continue

            # Penetration depth from each side
            pen_l = br - rl   # ball entered from left side of brick
            pen_r = rr - bl   # ball entered from right side
            pen_t = bb - rt   # ball entered from top
            pen_b = rb - bt   # ball entered from bottom

            min_pen = min(pen_l, pen_r, pen_t, pen_b)

            if min_pen < best_pen:
                best_pen = min_pen
                best_brick = brick
                is_corner_hit = (self.x < rl or self.x > rr) and (self.y < rt or self.y > rb)
                if is_corner_hit:
                    cx = rl if (self.x < rl) else rr
                    cy = rt if (self.y < rt) else rb
                    dx = self.x - cx
                    dy = self.y - cy
                    dist = math.hypot(dx, dy)
                    if dist > 1e-6:
                        nx = dx / dist
                        ny = dy / dist
                    else:
                        nx, ny = 0.7071, 0.7071
                    px = -(pen_l + 0.5) if (self.x < rl) else (pen_r + 0.5)
                    py = -(pen_t + 0.5) if (self.y < rt) else (pen_b + 0.5)
                    best_push = (px, py)
                    best_axis = ("corner", nx, ny)
                elif min_pen == pen_l:
                    best_push = (-(pen_l + 0.5), 0.0)
                    best_axis = "x"
                elif min_pen == pen_r:
                    best_push = (pen_r + 0.5, 0.0)
                    best_axis = "x"
                elif min_pen == pen_t:
                    best_push = (0.0, -(pen_t + 0.5))
                    best_axis = "y"
                else:
                    best_push = (0.0, pen_b + 0.5)
                    best_axis = "y"

        if best_brick is None:
            return None, 0.0, 0.0, None
        return best_brick, best_push[0], best_push[1], best_axis

    def draw(self, surface):
        if not self.active:
            return
        ix, iy = int(self.x), int(self.y)
        # Outer golden rim / shadow
        pygame.draw.circle(surface, (200, 150, 0), (ix, iy), self.radius)
        # Main golden sphere
        pygame.draw.circle(surface, COLOR_BALL, (ix, iy), max(1, self.radius - 1))
        # Specular white shine on top-left for 3D sphere look
        if self.radius > 3:
            pygame.draw.circle(surface, (255, 255, 220), (ix - 2, iy - 2), 2)
