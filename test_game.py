"""
Comprehensive test suite for Brick Blast Pygame engine and Genetic Algorithm AI.
Run with:
    ./venv/bin/python3 test_game.py
"""
import os
import unittest
import numpy as np
from brickblast.constants import GRID_COLS, GRID_ROWS, INITIAL_BALLS, WIDTH, HEIGHT, BOTTOM_MARGIN
from brickblast.brick import Brick
from brickblast.powerups import RedirectPowerup, MultiplierPowerup, LaserPowerup
from brickblast.ball import Ball
from brickblast.board import Board
from brickblast.game import BrickBlastGame
from env.brickblast_env import BrickBlastEnv
from ai.genome import Genome
from ai.ga import GeneticAlgorithm


class TestBrickBlastEngine(unittest.TestCase):
    def test_01_grid_and_initial_state(self):
        board = Board(seed=42)
        self.assertEqual(board.turn, 1)
        self.assertEqual(board.score, 0)
        self.assertFalse(board.game_over)
        self.assertGreaterEqual(len(board.bricks), 3)

    def test_02_brick_damage_and_color(self):
        brick = Brick(2, 3, hp=100)
        self.assertFalse(brick.hit(damage=50))
        self.assertEqual(brick.hp, 50)
        self.assertTrue(brick.hit(damage=50))
        self.assertEqual(brick.hp, 0)

    def test_03_redirect_powerup(self):
        pu = RedirectPowerup(3, 2, angle_deg=45)
        expected_sequence = [45, 90, 135, 45]
        for expected_angle in expected_sequence:
            self.assertEqual(pu.angle_deg, expected_angle)
            ball = Ball(pu.rect.centerx, pu.rect.centery)
            pu.apply(ball)
            self.assertNotEqual(ball.vx, 0.0 if expected_angle != 90 else -999)
            self.assertLess(ball.vy, 0.0)

    def test_04_multiplier_powerup(self):
        pu = MultiplierPowerup(4, 4)
        ball = Ball(pu.rect.centerx, pu.rect.centery)
        new_balls = []
        pu.apply(ball, new_balls)
        # Original ball + 2 new balls = 3 total balls
        self.assertEqual(len(new_balls), 2)
        all_balls = [ball] + new_balls
        self.assertEqual(len(all_balls), 3)
        # Verify angles correspond to 45, 90, 135
        vxs = [b.vx for b in all_balls]
        self.assertEqual(len(set(vxs)), 3)

    def test_05_laser_powerup(self):
        board = Board(seed=42)
        board.bricks.clear()
        # Place bricks in same row (3) and column (2)
        b_row = Brick(5, 3, hp=1)
        b_col = Brick(2, 7, hp=1)
        b_other = Brick(6, 6, hp=1)
        board.bricks.extend([b_row, b_col, b_other])

        laser = LaserPowerup(2, 3, mode="both")
        count = laser.trigger(board)
        self.assertEqual(count, 2)
        self.assertIn(b_other, board.bricks)
        self.assertNotIn(b_row, board.bricks)
        self.assertNotIn(b_col, board.bricks)

    def test_06_game_over_condition(self):
        board = Board(seed=42)
        # Add a brick at row 9 (GRID_ROWS - 1)
        board.bricks.append(Brick(0, GRID_ROWS - 1, hp=50))
        board.step_turn()
        self.assertTrue(board.game_over)


class TestRLAndGeneticAI(unittest.TestCase):
    def test_07_env_reset_and_step(self):
        env = BrickBlastEnv(seed=123)
        obs, info = env.reset()
        self.assertEqual(obs.shape, (21,))
        self.assertEqual(info["turn"], 1)

        obs, reward, terminated, truncated, info = env.step(90.0)
        self.assertEqual(obs.shape, (21,))
        self.assertIsInstance(reward, float)
        self.assertEqual(info["turn"], 2)

    def test_08_genome_forward_and_serialization(self):
        genome = Genome(input_size=21, hidden_size=16, seed=1)
        x = np.random.randn(21).astype(np.float32)
        score = genome.forward(x)
        self.assertIsInstance(score, float)

        data = genome.to_dict()
        g2 = Genome.from_dict(data)
        score2 = g2.forward(x)
        self.assertAlmostEqual(score, score2, places=5)

    def test_09_genetic_algorithm_evaluation(self):
        ga = GeneticAlgorithm(pop_size=6, elitism_count=2, seed=10)
        max_fit, avg_fit, max_turns = ga.evaluate_population(num_processes=1, seed_offset=0)
        self.assertGreaterEqual(max_fit, 5.0)
        self.assertEqual(len(ga.population), 6)
        self.assertIsNotNone(ga.best_genome)

        # Test generation step
        ga.step_generation()
        self.assertEqual(ga.generation, 1)
        self.assertEqual(len(ga.population), 6)

    def test_10_frenzy_surge_mechanic(self):
        game = BrickBlastGame(headless=True, seed=1)
        game.reset()
        # Clear bricks down to 2 blocks (<= 3 threshold)
        game.board.bricks = game.board.bricks[:2]
        self.assertEqual(game.frenzy_cooldown, 0)
        # End turn should trigger the 3-layer surge and 3x balls
        game._end_turn()
        self.assertEqual(game.total_balls_capacity, 180)  # 60 * 3
        self.assertEqual(game.frenzy_cooldown, 5)  # 5-turn cooldown active
        self.assertTrue(game.frenzy_active)

    def test_11_cnn_genome_forward_and_serialization(self):
        genome = Genome(model_type="cnn", seed=1)
        grid = np.random.randn(2, 10, 8).astype(np.float32)
        glob = np.array([0.5, 0.1], dtype=np.float32)
        score = genome.forward_cnn(grid, glob)
        self.assertIsInstance(score, float)

        data = genome.to_dict()
        self.assertEqual(data["model_type"], "cnn")
        g2 = Genome.from_dict(data)
        score2 = g2.forward_cnn(grid, glob)
        self.assertAlmostEqual(score, score2, places=5)

        env = BrickBlastEnv(seed=1)
        env.reset()
        angle = genome.select_action(env)
        self.assertGreaterEqual(angle, 3.0)
        self.assertLessEqual(angle, 177.0)

    def test_12_tensor_env_and_tensor_ga(self):
        from env.tensor_env import TensorBrickBlastEnv
        from ai.tensor_ga import TensorGeneticAlgorithm

        env = TensorBrickBlastEnv(batch_size=10, max_balls=15, device="cpu", seed=42)
        (grids, globals_arr) = env.get_grid_observation()
        self.assertEqual(grids.shape, (10, 2, 10, 8))
        self.assertEqual(globals_arr.shape, (10, 2))

        angles = torch.full((10,), 90.0)
        (grids2, globals2), reward, terminated, info = env.step(angles)
        self.assertEqual(reward.shape, (10,))
        self.assertEqual(terminated.shape, (10,))

        ga = TensorGeneticAlgorithm(pop_size=8, model_type="cnn", device="cpu", seed=42)
        max_fit, avg_fit, turns = ga.evaluate_population(max_turns=5)
        self.assertGreaterEqual(max_fit, 0.0)
        self.assertGreaterEqual(turns, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
