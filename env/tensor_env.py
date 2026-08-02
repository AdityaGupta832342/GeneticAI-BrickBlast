"""
GPU-Native Vectorized Tensor Environment for Brick Blast (TensorBrickBlastEnv).
Simulates B parallel games simultaneously inside PyTorch CUDA tensors without CPU Python loops.
"""
import math
import torch
import numpy as np
from brickblast.constants import (
    GRID_COLS,
    GRID_ROWS,
    WIDTH,
    HEIGHT,
    TOP_MARGIN,
    BOTTOM_MARGIN,
    BALL_RADIUS,
    BALL_SPEED,
    CELL_WIDTH,
    CELL_HEIGHT,
)


class TensorBrickBlastEnv:
    def __init__(self, batch_size=200, max_balls=30, device=None, seed=None):
        self.batch_size = batch_size
        self.max_balls = max_balls
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")

        if seed is not None:
            torch.manual_seed(seed)

        self.grid_hp = torch.zeros((batch_size, GRID_ROWS, GRID_COLS), dtype=torch.float32, device=self.device)
        self.grid_powerup = torch.zeros((batch_size, GRID_ROWS, GRID_COLS), dtype=torch.long, device=self.device)
        self.launch_x = torch.full((batch_size,), WIDTH / 2.0, dtype=torch.float32, device=self.device)
        self.turn = torch.ones((batch_size,), dtype=torch.long, device=self.device)
        self.terminated = torch.zeros((batch_size,), dtype=torch.bool, device=self.device)
        self.total_reward = torch.zeros((batch_size,), dtype=torch.float32, device=self.device)

        self.reset()

    def _spawn_top_row(self, mask=None):
        """
        Spawn new bricks in row 0 for active games.
        """
        if mask is None:
            mask = torch.ones(self.batch_size, dtype=torch.bool, device=self.device)

        num_active = int(mask.sum().item())
        if num_active == 0:
            return

        # Probability of spawning a brick in each cell: ~65%
        spawn_mask = torch.rand((num_active, GRID_COLS), device=self.device) < 0.65
        # Ensure at least 1 brick spawns per row
        empty_rows = ~spawn_mask.any(dim=1)
        if empty_rows.any():
            rand_cols = torch.randint(0, GRID_COLS, (int(empty_rows.sum().item()),), device=self.device)
            spawn_mask[empty_rows, rand_cols] = True

        # HP scales with turn number
        turns_active = self.turn[mask].float().unsqueeze(-1)
        base_hp = torch.randint(1, 3, (num_active, GRID_COLS), device=self.device, dtype=torch.float32) + turns_active * 0.5
        new_hp = torch.where(spawn_mask, base_hp, torch.zeros_like(base_hp))

        # Powerups (~10% redirect=1, ~10% multiplier=2, ~5% laser=3)
        pu_rand = torch.rand((num_active, GRID_COLS), device=self.device)
        new_pu = torch.zeros((num_active, GRID_COLS), dtype=torch.long, device=self.device)
        new_pu = torch.where(spawn_mask & (pu_rand < 0.10), torch.full_like(new_pu, 1), new_pu)
        new_pu = torch.where(spawn_mask & (pu_rand >= 0.10) & (pu_rand < 0.20), torch.full_like(new_pu, 2), new_pu)
        new_pu = torch.where(spawn_mask & (pu_rand >= 0.20) & (pu_rand < 0.25), torch.full_like(new_pu, 3), new_pu)

        self.grid_hp[mask, 0, :] = new_hp
        self.grid_powerup[mask, 0, :] = new_pu

    def reset(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        self.grid_hp.zero_()
        self.grid_powerup.zero_()
        self.launch_x.fill_(WIDTH / 2.0)
        self.turn.fill_(1)
        self.terminated.zero_()
        self.total_reward.zero_()
        self._spawn_top_row()
        return self.get_grid_observation()

    def get_grid_observation(self):
        """
        Returns (grids, globals):
        grids: (B, 2, 10, 8) tensor [hp_norm, pu_norm]
        globals: (B, 2) tensor [launch_x_norm, turn_norm]
        """
        hp_norm = (self.grid_hp / 20.0).clamp(0.0, 10.0)
        pu_norm = self.grid_powerup.float() / 3.0
        grids = torch.stack([hp_norm, pu_norm], dim=1)

        launch_norm = (self.launch_x / float(WIDTH)).clamp(0.0, 1.0)
        turn_norm = (self.turn.float() / 100.0).clamp(0.0, 1.0)
        globals_arr = torch.stack([launch_norm, turn_norm], dim=1)
        return grids, globals_arr

    @torch.no_grad()
    def step(self, angles_deg):
        """
        Simulate 1 turn across all B games simultaneously on CUDA Tensors.
        angles_deg: (B,) tensor in [3.0, 177.0]
        Returns: (grids, globals), rewards, terminated, info_dict
        """
        angles_rad = angles_deg.to(self.device).clamp(3.0, 177.0) * (math.pi / 180.0)
        cos_a = torch.cos(angles_rad)
        sin_a = -torch.abs(torch.sin(angles_rad))  # negative vy travels UP towards bricks

        # Active mask for non-terminated games
        active_games = ~self.terminated
        if not active_games.any():
            grids, globals_arr = self.get_grid_observation()
            return (grids, globals_arr), torch.zeros_like(self.total_reward), self.terminated, {"turn": self.turn.clone()}

        B = self.batch_size
        M = self.max_balls

        # Initialize ball positions and velocities: (B, M)
        # Add slight spatial stagger along launch vector to simulate stream of balls
        stagger = torch.linspace(0.0, 15.0, M, device=self.device).unsqueeze(0)  # (1, M)
        ball_x = self.launch_x.unsqueeze(1) - cos_a.unsqueeze(1) * stagger
        ball_y = torch.full((B, M), float(HEIGHT - BOTTOM_MARGIN - BALL_RADIUS), device=self.device) - sin_a.unsqueeze(1) * stagger

        ball_vx = (cos_a * BALL_SPEED).unsqueeze(1).expand(B, M).clone()
        ball_vy = (sin_a * BALL_SPEED).unsqueeze(1).expand(B, M).clone()

        ball_active = active_games.unsqueeze(1).expand(B, M).clone()

        step_reward = torch.zeros((B,), dtype=torch.float32, device=self.device)
        ground_y = float(HEIGHT - BOTTOM_MARGIN)

        # Vectorized physics loop (max 100 sub-steps per turn)
        for _ in range(100):
            if not ball_active.any():
                break

            ball_x = torch.where(ball_active, ball_x + ball_vx, ball_x)
            ball_y = torch.where(ball_active, ball_y + ball_vy, ball_y)

            # Left wall bounce
            hit_left = ball_active & (ball_x <= BALL_RADIUS)
            ball_x = torch.where(hit_left, torch.full_like(ball_x, BALL_RADIUS + 0.5), ball_x)
            ball_vx = torch.where(hit_left, torch.abs(ball_vx), ball_vx)

            # Right wall bounce
            hit_right = ball_active & (ball_x >= WIDTH - BALL_RADIUS)
            ball_x = torch.where(hit_right, torch.full_like(ball_x, WIDTH - BALL_RADIUS - 0.5), ball_x)
            ball_vx = torch.where(hit_right, -torch.abs(ball_vx), ball_vx)

            # Top wall bounce
            hit_top = ball_active & (ball_y <= TOP_MARGIN + BALL_RADIUS)
            ball_y = torch.where(hit_top, torch.full_like(ball_y, TOP_MARGIN + BALL_RADIUS + 0.5), ball_y)
            ball_vy = torch.where(hit_top, torch.abs(ball_vy), ball_vy)

            # Brick grid collisions
            col_idx = ((ball_x) / CELL_WIDTH).long().clamp(0, GRID_COLS - 1)
            row_idx = ((ball_y - TOP_MARGIN) / CELL_HEIGHT).long().clamp(0, GRID_ROWS - 1)

            in_grid = ball_active & (ball_y >= TOP_MARGIN) & (ball_y <= ground_y)

            # Gather HP of brick cell currently at each ball's location: (B, M)
            b_indices = torch.arange(B, device=self.device).unsqueeze(1).expand(B, M)
            cell_hp = self.grid_hp[b_indices, row_idx, col_idx]
            hit_brick = in_grid & (cell_hp > 0)

            if hit_brick.any():
                # Check powerups on hit bricks
                cell_pu = self.grid_powerup[b_indices, row_idx, col_idx]
                is_laser = hit_brick & (cell_pu == 3)
                is_mult = hit_brick & (cell_pu == 2)
                is_redir = hit_brick & (cell_pu == 1)

                # Damage dealt: 1 default, 3 if multiplier
                damage = torch.where(is_mult, torch.full_like(cell_hp, 3.0), torch.ones_like(cell_hp))
                damage = torch.where(hit_brick, damage, torch.zeros_like(damage))

                # Scatter subtract damage back to grid_hp
                # Iterate columns to avoid scatter race conflicts across multiple balls in same cell
                for m in range(M):
                    hm = hit_brick[:, m]
                    if hm.any():
                        b_true = torch.nonzero(hm, as_tuple=True)[0]
                        r = row_idx[b_true, m]
                        c = col_idx[b_true, m]
                        dmg = damage[b_true, m]
                        old_val = self.grid_hp[b_true, r, c]
                        actual_dmg = torch.minimum(old_val, dmg)
                        self.grid_hp[b_true, r, c] -= actual_dmg
                        step_reward[b_true] += actual_dmg

                        # If laser powerup triggered, clear entire column
                        laser_mask = self.grid_powerup[b_true, r, c] == 3
                        if laser_mask.any():
                            l_idx = b_true[laser_mask]
                            l_cols = c[laser_mask]
                            for idx_l, col_l in zip(l_idx, l_cols):
                                col_hp_sum = self.grid_hp[idx_l, :, col_l].sum()
                                step_reward[idx_l] += col_hp_sum
                                self.grid_hp[idx_l, :, col_l] = 0.0
                                self.grid_powerup[idx_l, :, col_l] = 0

                # Reflect ball vy on brick hit
                ball_vy = torch.where(hit_brick, -ball_vy, ball_vy)
                # If redirect powerup, randomize vx direction
                rand_sign = torch.where(torch.rand_like(ball_vx) < 0.5, 1.0, -1.0)
                ball_vx = torch.where(is_redir, torch.abs(ball_vx) * rand_sign, ball_vx)

            # Ground landing check
            landed = ball_active & (ball_y >= ground_y)
            ball_active = ball_active & (~landed)

        # End of turn processing for active games
        # 1. Check game over condition: any remaining brick in row 9 (index GRID_ROWS-1)
        row9_occupied = (self.grid_hp[:, GRID_ROWS - 1, :] > 0).any(dim=1)
        new_terminations = active_games & row9_occupied
        self.terminated = self.terminated | new_terminations

        # 2. Advance surviving games
        surviving = active_games & (~self.terminated)
        if surviving.any():
            self.turn[surviving] += 1
            # Shift board down 1 row
            self.grid_hp[surviving, 1:, :] = self.grid_hp[surviving, :-1, :].clone()
            self.grid_powerup[surviving, 1:, :] = self.grid_powerup[surviving, :-1, :].clone()
            self.grid_hp[surviving, 0, :] = 0.0
            self.grid_powerup[surviving, 0, :] = 0
            self._spawn_top_row(mask=surviving)

        self.total_reward += step_reward
        grids, globals_arr = self.get_grid_observation()
        return (grids, globals_arr), step_reward, self.terminated, {"turn": self.turn.clone()}
