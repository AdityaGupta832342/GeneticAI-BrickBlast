"""
BrickBlastGame class managing the turn loop, aiming, ball launching, and rendering.
Supports both interactive GUI mode and fast headless RL simulation mode.
"""
import math
from brickblast.constants import (
    WIDTH,
    HEIGHT,
    TOP_MARGIN,
    BOTTOM_MARGIN,
    INITIAL_BALLS,
    BALL_SPEED,
    LAUNCH_INTERVAL,
    COLOR_BG,
    COLOR_TEXT_WHITE,
    COLOR_AIM_LINE,
    COLOR_BALL,
)
from brickblast.board import Board
from brickblast.ball import Ball
from brickblast.powerups import _angle_to_vel
from brickblast.pygame_compat import pygame, Rect, PYGAME_AVAILABLE


class BrickBlastGame:
    def __init__(self, headless=False, seed=None):
        self.headless = headless
        self.board = Board(seed=seed)
        self.state = "aiming"
        self.aim_angle = 90.0  # degrees (10 to 170)
        self.active_balls = []
        self.balls_left_to_launch = INITIAL_BALLS
        self.launch_timer = 0
        self.total_balls_capacity = INITIAL_BALLS
        self.fast_forward = False
        self.frenzy_cooldown = 0
        self.frenzy_active = False

        self.font_large = None
        self.font_small = None
        if not self.headless:
            pygame.font.init()
            self.font_large = pygame.font.SysFont("Arial", 28, bold=True)
            self.font_small = pygame.font.SysFont("Arial", 18, bold=True)

    def reset(self, seed=None):
        self.board.reset(seed=seed)
        self.state = "aiming"
        self.aim_angle = 90.0
        self.active_balls.clear()
        self.balls_left_to_launch = INITIAL_BALLS
        self.total_balls_capacity = INITIAL_BALLS
        self.fast_forward = False
        self.frenzy_cooldown = 0
        self.frenzy_active = False

    def set_aim(self, angle_deg):
        """
        Set aiming angle between 3° and 177°.
        """
        self.aim_angle = max(3.0, min(177.0, float(angle_deg)))

    def start_shot(self):
        """
        Trigger ball launch from aiming state.
        """
        if self.state == "aiming" and not self.board.game_over:
            self.state = "shooting"
            self.balls_left_to_launch = self.total_balls_capacity
            self.launch_timer = 0
            self.active_balls.clear()

    def recall_balls(self):
        """
        Recall all active balls immediately to the ground (ends turn).
        """
        if self.state in ("shooting", "simulating"):
            self.active_balls.clear()
            self.balls_left_to_launch = 0
            self._end_turn()

    def step_simulation(self):
        """
        Advance one frame of physics simulation.
        """
        if self.board.game_over:
            self.state = "game_over"
            return

        if self.state == "shooting":
            self.launch_timer += 1
            if self.launch_timer >= LAUNCH_INTERVAL:
                self.launch_timer = 0
                if self.balls_left_to_launch > 0:
                    ball = Ball(self.board.launch_x, HEIGHT - BOTTOM_MARGIN - 7)
                    ball.active = True
                    vx, vy = _angle_to_vel(self.aim_angle, BALL_SPEED)
                    ball.vx = vx
                    ball.vy = vy
                    self.active_balls.append(ball)
                    self.balls_left_to_launch -= 1
                else:
                    self.state = "simulating"

        if self.state in ("shooting", "simulating"):
            new_balls = []
            steps = 3 if self.fast_forward else 1
            for _ in range(steps):
                for ball in self.active_balls:
                    ball.update(self.board, new_balls)
                # Cap active balls to prevent runaway multiplier chains
                if len(self.active_balls) + len(new_balls) > 500:
                    break

            # Append balls spawned by Multiplier powerup
            self.active_balls.extend(new_balls)
            if len(self.active_balls) > 500:
                self.active_balls = self.active_balls[:500]

            # Remove landed balls
            self.active_balls = [b for b in self.active_balls if b.active and not b.landed]

            # Check turn completion
            if self.state == "simulating" and len(self.active_balls) == 0:
                self._end_turn()

    def _end_turn(self):
        # Trigger 3-layer / 3x-ball surge when very few blocks remain (<= 3)
        # and cooldown is 0 so it does not trigger too frequently (5-turn cooldown)
        very_few_blocks = (len(self.board.bricks) <= 3)
        if very_few_blocks and self.frenzy_cooldown == 0:
            self.board.step_turn(layers=3)
            self.total_balls_capacity = INITIAL_BALLS * 3
            self.frenzy_cooldown = 5
            self.frenzy_active = True
        else:
            self.board.step_turn(layers=1)
            self.total_balls_capacity = INITIAL_BALLS
            self.frenzy_active = False
            if self.frenzy_cooldown > 0:
                self.frenzy_cooldown -= 1

        if self.board.game_over:
            self.state = "game_over"
        else:
            self.state = "aiming"

    def play_turn_headless(self, angle_deg, max_steps=10000):
        """
        Execute an entire turn at angle_deg without rendering.
        Returns (score_gained, game_over).
        """
        start_score = self.board.score
        self.set_aim(angle_deg)
        self.start_shot()

        steps = 0
        while self.state in ("shooting", "simulating") and steps < max_steps:
            # Launch all remaining balls immediately to save frames
            while self.balls_left_to_launch > 0:
                ball = Ball(self.board.launch_x, HEIGHT - BOTTOM_MARGIN - 7)
                ball.active = True
                vx, vy = _angle_to_vel(self.aim_angle, BALL_SPEED)
                ball.vx = vx
                ball.vy = vy
                self.active_balls.append(ball)
                self.balls_left_to_launch -= 1
            self.state = "simulating"

            new_balls = []
            for ball in self.active_balls:
                ball.update(self.board, new_balls)
            self.active_balls.extend(new_balls)
            # Cap active balls to prevent runaway multiplier chains
            if len(self.active_balls) > 500:
                self.active_balls = self.active_balls[:500]
            self.active_balls = [b for b in self.active_balls if b.active and not b.landed]
            steps += 1
            if len(self.active_balls) == 0:
                break

        if len(self.active_balls) == 0:
            self._end_turn()
        else:
            self.recall_balls()

        score_gained = self.board.score - start_score
        return score_gained, self.board.game_over

    def render(self, surface):
        surface.fill(COLOR_BG)

        # Draw styled top header bar
        pygame.draw.rect(surface, (20, 30, 62), Rect(0, 0, WIDTH, TOP_MARGIN - 2))
        pygame.draw.line(surface, (50, 75, 140), (0, TOP_MARGIN - 2), (WIDTH, TOP_MARGIN - 2), 2)

        # Draw Turn title with drop shadow
        label = f"Turn {self.board.turn}"
        t_shadow = self.font_large.render(label, True, (10, 15, 35))
        t_surf = self.font_large.render(label, True, COLOR_TEXT_WHITE)
        surface.blit(t_shadow, (WIDTH // 2 - t_surf.get_width() // 2 + 2, 22))
        surface.blit(t_surf, (WIDTH // 2 - t_surf.get_width() // 2, 20))

        # Draw Score in golden yellow with drop shadow
        s_label = f"Score: {self.board.score}"
        s_shadow = self.font_small.render(s_label, True, (10, 15, 35))
        s_surf = self.font_small.render(s_label, True, (255, 215, 50))
        surface.blit(s_shadow, (21, 81))
        surface.blit(s_surf, (20, 80))

        # Draw Balls count with drop shadow
        if getattr(self, "frenzy_active", False):
            b_label = f"Balls: x{self.total_balls_capacity} (3X SURGE)"
            b_color = (255, 140, 50)
        else:
            b_label = f"Balls: x{self.total_balls_capacity}"
            b_color = COLOR_TEXT_WHITE
        b_shadow = self.font_small.render(b_label, True, (10, 15, 35))
        b_surf = self.font_small.render(b_label, True, b_color)
        surface.blit(b_shadow, (WIDTH - b_surf.get_width() - 20 + 1, 81))
        surface.blit(b_surf, (WIDTH - b_surf.get_width() - 20, 80))

        # --- Draw Stage Progress Bar toward Finish Line ---
        stage = (self.board.score // 500) + 1
        score_in_stage = self.board.score % 500
        progress_pct = score_in_stage / 500.0
        bar_rect = Rect(30, 52, WIDTH - 60, 16)
        pygame.draw.rect(surface, (12, 18, 40), bar_rect, border_radius=8)
        pygame.draw.rect(surface, (50, 75, 130), bar_rect, width=1, border_radius=8)
        fill_w = int((WIDTH - 60) * progress_pct)
        if fill_w > 0:
            pygame.draw.rect(surface, (255, 195, 40), Rect(30, 52, fill_w, 16), border_radius=8)

        prog_label = f"Stage {stage} Progress  —  {score_in_stage}/500 to Finish Line"
        p_shadow = self.font_small.render(prog_label, True, (10, 15, 35))
        p_surf = self.font_small.render(prog_label, True, COLOR_TEXT_WHITE)
        surface.blit(p_shadow, (WIDTH // 2 - p_surf.get_width() // 2 + 1, 52 + 1))
        surface.blit(p_surf, (WIDTH // 2 - p_surf.get_width() // 2, 52))

        # Checkered Finish Line Flag badge at right end of progress bar
        flag_x = WIDTH - 26
        flag_y = 54
        for r in range(2):
            for c in range(2):
                col = (255, 255, 255) if (r + c) % 2 == 0 else (20, 20, 20)
                pygame.draw.rect(surface, col, Rect(flag_x + c * 6, flag_y + r * 6, 6, 6))

        # Draw 8x10 board grid and items
        self.board.draw(surface, self.font_small)

        # Draw aiming trajectory line and highlight point of first collision
        if self.state == "aiming" and not self.board.game_over:
            self._draw_aim_line(surface)

        # Draw active balls
        for b in self.active_balls:
            b.draw(surface)

        # Draw launch origin marker
        pygame.draw.circle(
            surface,
            COLOR_BALL,
            (int(self.board.launch_x), HEIGHT - BOTTOM_MARGIN),
            8,
        )

        # Draw Game Over overlay if terminated
        if self.state == "game_over":
            overlay = pygame.Surface((WIDTH, 120))
            overlay.fill((180, 20, 40))
            go_text = self.font_large.render("GAME OVER", True, COLOR_TEXT_WHITE)
            surface.blit(overlay, (0, HEIGHT // 2 - 60))
            surface.blit(go_text, (WIDTH // 2 - go_text.get_width() // 2, HEIGHT // 2 - 20))

    def _ray_intersect_aabb(self, x, y, dx, dy, rx, ry, rw, rh):
        """Slab method for 2D ray vs AABB. Returns distance t_min or None."""
        t_min = 0.0
        t_max = float("inf")

        # X axis
        if abs(dx) < 1e-9:
            if x < rx or x > rx + rw:
                return None
        else:
            t1 = (rx - x) / dx
            t2 = (rx + rw - x) / dx
            if t1 > t2:
                t1, t2 = t2, t1
            t_min = max(t_min, t1)
            t_max = min(t_max, t2)

        # Y axis
        if abs(dy) < 1e-9:
            if y < ry or y > ry + rh:
                return None
        else:
            t1 = (ry - y) / dy
            t2 = (ry + rh - y) / dy
            if t1 > t2:
                t1, t2 = t2, t1
            t_min = max(t_min, t1)
            t_max = min(t_max, t2)

        if t_min <= t_max and t_min > 1e-4:
            return t_min
        return None

    def _draw_aim_line(self, surface):
        """
        Raytrace the aiming trajectory, reflecting off side walls,
        and draw glowing highlights at wall bounces and the point of first collision.
        """
        x = float(self.board.launch_x)
        y = float(HEIGHT - BOTTOM_MARGIN - 7)
        rad = math.radians(self.aim_angle)
        dx = math.cos(rad)
        dy = -math.sin(rad)  # Upwards in pygame is negative y

        points = [(x, y)]
        bounce_contacts = []
        hit_item = None
        final_contact = None
        max_bounces = 1

        for _ in range(max_bounces + 1):
            best_t = float("inf")
            hit_type = None
            hit_obj = None

            # 1. Check side walls
            if dx < -1e-9:
                t_left = (7.0 - x) / dx
                if 1e-4 < t_left < best_t:
                    best_t = t_left
                    hit_type = "left_wall"
            if dx > 1e-9:
                t_right = (WIDTH - 7.0 - x) / dx
                if 1e-4 < t_right < best_t:
                    best_t = t_right
                    hit_type = "right_wall"

            # 2. Check top wall
            if dy < -1e-9:
                t_top = (TOP_MARGIN + 7.0 - y) / dy
                if 1e-4 < t_top < best_t:
                    best_t = t_top
                    hit_type = "top_wall"

            # 3. Check Bricks & Powerups (expand AABB by ball radius 7)
            for item in self.board.bricks + self.board.powerups:
                r = item.rect
                rx = float(r.x) - 7.0
                ry = float(r.y) - 7.0
                rw = float(r.w) + 14.0
                rh = float(r.h) + 14.0
                t_item = self._ray_intersect_aabb(x, y, dx, dy, rx, ry, rw, rh)
                if t_item is not None and 1e-4 < t_item < best_t:
                    best_t = t_item
                    hit_type = "item"
                    hit_obj = item

            if best_t == float("inf"):
                points.append((x + dx * 400, y + dy * 400))
                break

            hit_x = x + best_t * dx
            hit_y = y + best_t * dy

            if hit_type == "item":
                r = hit_obj.rect
                cx = max(float(r.x), min(hit_x, float(r.x + r.w)))
                cy = max(float(r.y), min(hit_y, float(r.y + r.h)))
                points.append((hit_x, hit_y))
                final_contact = (cx, cy)
                hit_item = hit_obj
                break
            elif hit_type == "top_wall":
                points.append((hit_x, hit_y))
                final_contact = (hit_x, float(TOP_MARGIN))
                break
            elif hit_type in ("left_wall", "right_wall"):
                cx = 0.0 if hit_type == "left_wall" else float(WIDTH)
                bounce_contacts.append((cx, hit_y))
                points.append((hit_x, hit_y))
                x, y = hit_x, hit_y
                dx = -dx

        # --- Draw Trajectory Line Segments ---
        for i in range(len(points) - 1):
            p1 = (int(points[i][0]), int(points[i][1]))
            p2 = (int(points[i + 1][0]), int(points[i + 1][1]))
            pygame.draw.line(surface, (200, 170, 30), p1, p2, 4)
            pygame.draw.line(surface, (255, 235, 120), p1, p2, 2)

        # --- Draw Glowing Highlights at Wall Bounce Points ---
        for bx, by in bounce_contacts:
            pt = (int(bx), int(by))
            pygame.draw.circle(surface, (255, 200, 50), pt, 8, 2)
            pygame.draw.circle(surface, (255, 255, 180), pt, 4)

        # --- Highlight Hit Brick / Powerup Box ---
        if hit_item is not None:
            pygame.draw.rect(surface, (255, 225, 80), hit_item.rect, 3, border_radius=6)
            pygame.draw.rect(surface, (255, 255, 255), hit_item.rect, 1, border_radius=6)

        # --- Highlight the Point of First Collision ---
        if final_contact is not None:
            cx, cy = int(final_contact[0]), int(final_contact[1])
            pygame.draw.circle(surface, (255, 80, 180), (cx, cy), 12, 2)
            pygame.draw.circle(surface, (255, 215, 40), (cx, cy), 7)
            pygame.draw.circle(surface, (255, 255, 255), (cx, cy), 4)
