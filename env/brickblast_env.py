"""
Gym/Farama-compatible Environment Wrapper for Brick Blast.
Provides normalized observations, reward calculation, and raycasting evaluation features.
"""
import math
import numpy as np
from brickblast.constants import WIDTH, HEIGHT, GRID_COLS, GRID_ROWS, TOP_MARGIN, BOTTOM_MARGIN
from brickblast.game import BrickBlastGame


class BrickBlastEnv:
    def __init__(self, seed=None):
        self.game = BrickBlastGame(headless=True, seed=seed)
        self.action_space_low = 3.0
        self.action_space_high = 177.0
        self.max_turns = 100

    def reset(self, seed=None):
        self.game.reset(seed=seed)
        return self.get_observation(), {"turn": self.game.board.turn, "score": self.game.board.score}

    def step(self, action):
        """
        Execute one turn.
        action: Aiming angle in degrees [3.0, 177.0], or continuous [-1.0, 1.0] which is mapped to [3.0, 177.0].
        """
        val = float(action)
        if -1.0 <= val <= 1.0 and val < 3.0:
            angle_deg = 90.0 + val * 87.0
        else:
            angle_deg = max(3.0, min(177.0, val))

        score_gained, game_over = self.game.play_turn_headless(angle_deg)

        # Reward shaping
        reward = float(score_gained) + 5.0  # survival bonus per turn
        if game_over:
            reward -= 100.0

        terminated = game_over
        truncated = (self.game.board.turn >= self.max_turns)
        obs = self.get_observation()
        info = {
            "turn": self.game.board.turn,
            "score": self.game.board.score,
            "game_over": game_over,
        }
        return obs, reward, terminated, truncated, info

    def get_observation(self):
        """
        Returns a 21-dimensional normalized float feature vector representing board state.
        """
        b = self.game.board

        # 1. Normalized launch X (1)
        launch_x_norm = b.launch_x / float(WIDTH)

        # 2. Normalized turn (1)
        turn_norm = min(1.0, b.turn / 100.0)

        # 3. Lowest row per column (8)
        col_lowest = [0.0] * GRID_COLS
        # 4. Total HP per column (8)
        col_hp = [0.0] * GRID_COLS

        for brick in b.bricks:
            c = brick.col
            if 0 <= c < GRID_COLS:
                col_lowest[c] = max(col_lowest[c], (brick.row + 1.0) / float(GRID_ROWS))
                col_hp[c] += brick.hp / 1000.0

        # 5. Powerup counts by type (3)
        pu_redirect = 0.0
        pu_multiplier = 0.0
        pu_laser = 0.0
        for pu in b.powerups:
            if pu.type == "redirect":
                pu_redirect += 1.0
            elif pu.type == "multiplier":
                pu_multiplier += 1.0
            elif pu.type == "laser":
                pu_laser += 1.0

        obs = [
            launch_x_norm,
            turn_norm,
            *col_lowest,
            *col_hp,
            min(1.0, pu_redirect / 5.0),
            min(1.0, pu_multiplier / 5.0),
            min(1.0, pu_laser / 5.0),
        ]
        return np.array(obs, dtype=np.float32)

    def get_grid_observation(self):
        """
        Returns a 2D multi-channel grid observation for CNN models:
        - grid: shape (2, 10, 8) -> channel 0 = normalized brick HP, channel 1 = powerup type
        - globals: shape (2,) -> [normalized_launch_x, normalized_turn]
        """
        b = self.game.board
        grid = np.zeros((2, GRID_ROWS, GRID_COLS), dtype=np.float32)

        for brick in b.bricks:
            if 0 <= brick.row < GRID_ROWS and 0 <= brick.col < GRID_COLS:
                grid[0, brick.row, brick.col] = min(1.0, brick.hp / 20.0)

        for pu in b.powerups:
            if 0 <= pu.row < GRID_ROWS and 0 <= pu.col < GRID_COLS:
                val = 0.33
                if pu.type == "multiplier":
                    val = 0.66
                elif pu.type == "laser":
                    val = 1.0
                grid[1, pu.row, pu.col] = val

        globals_arr = np.array([b.launch_x / float(WIDTH), min(1.0, b.turn / 100.0)], dtype=np.float32)
        return grid, globals_arr

    def get_action_eval_features(self, candidate_angle_deg):
        """
        Returns a 24-dimensional feature vector combining the 21 board features
        with 3 candidate action features (angle, raycast distance, first hit type).
        Used by the GA neural network to score and choose the best shot angle.
        """
        base_obs = self.get_observation()

        ang_norm = (candidate_angle_deg - 90.0) / 80.0
        dist_norm, hit_type = self._raycast(candidate_angle_deg)

        action_feats = np.array([ang_norm, dist_norm, hit_type], dtype=np.float32)
        return np.concatenate([base_obs, action_feats])

    def _raycast(self, angle_deg):
        """
        Simulate a ray from launch point at angle_deg to find distance and type of first hit.
        hit_type: 0.0 = wall/ceiling, 1.0 = brick, 2.0 = redirect, 3.0 = multiplier, 4.0 = laser.
        """
        start_x = self.game.board.launch_x
        start_y = float(HEIGHT - BOTTOM_MARGIN - 7)
        rad = math.radians(angle_deg)
        dx = math.cos(rad) * 4.0
        dy = -math.sin(rad) * 4.0

        x, y = start_x, start_y
        for step in range(200):
            x += dx
            y += dy
            if x <= 0 or x >= WIDTH or y <= TOP_MARGIN:
                return (step * 4.0) / 800.0, 0.0
            for pu in self.game.board.powerups:
                if pu.rect.collidepoint(int(x), int(y)):
                    t_map = {"redirect": 2.0, "multiplier": 3.0, "laser": 4.0}
                    return (step * 4.0) / 800.0, t_map.get(pu.type, 2.0)
            for brick in self.game.board.bricks:
                if brick.rect.collidepoint(int(x), int(y)):
                    return (step * 4.0) / 800.0, 1.0

        return 1.0, 0.0
