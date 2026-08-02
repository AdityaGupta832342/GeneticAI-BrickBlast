"""
Batched GPU-Native Genetic Algorithm (TensorGeneticAlgorithm).
Evaluates an entire population of B Genomes simultaneously on B parallel TensorBrickBlastEnv games
using PyTorch CUDA tensor operations and batched forward inference.
"""
import os
import time
import random
import torch
import numpy as np
from ai.genome import Genome
from env.tensor_env import TensorBrickBlastEnv


class TensorGeneticAlgorithm:
    def __init__(
        self,
        pop_size=200,
        mutation_rate=0.15,
        mutation_scale=0.25,
        elitism_count=4,
        model_type="cnn",
        device=None,
        seed=None,
    ):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        self.model_type = model_type
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")

        self.elitism_count = max(2, min(pop_size // 5, elitism_count))

        self.population = [
            Genome(model_type=self.model_type, device=self.device)
            for _ in range(pop_size)
        ]
        self.generation = 0
        self.best_genome = None
        self.fitness_history = []  # list of (gen, max_fit, avg_fit, max_turns)

    @torch.no_grad()
    def _predict_actions_batched(self, grids, globals_arr):
        """
        Predict continuous aiming angles in [3.0, 177.0] for all B genomes in parallel.
        grids: (B, 2, 10, 8)
        globals_arr: (B, 2)
        Returns tensor of shape (B,)
        """
        angles = []
        for idx, genome in enumerate(self.population):
            grid_i = grids[idx : idx + 1]
            glob_i = globals_arr[idx : idx + 1]
            if self.model_type == "cnn":
                val = genome.forward_cnn_tensor(grid_i, glob_i).view(1)
            else:
                val = genome.forward_tensor(glob_i).view(1)
            angles.append(val)
        vals_t = torch.cat(angles, dim=0)
        angles_deg = 90.0 + vals_t * 87.0
        return angles_deg.clamp(3.0, 177.0)

    def evaluate_population(self, max_turns=50, seed_offset=0):
        """
        Run B parallel games on CUDA for up to max_turns.
        Returns (max_fit, avg_fit, max_turns_survived).
        """
        env = TensorBrickBlastEnv(
            batch_size=self.pop_size, max_balls=30, device=self.device, seed=seed_offset
        )
        (grids, globals_arr) = env.get_grid_observation()

        for turn in range(max_turns):
            if env.terminated.all():
                break
            angles_deg = self._predict_actions_batched(grids, globals_arr)
            (grids, globals_arr), rewards, terminated, info = env.step(angles_deg)

        # Assign fitnesses
        fits = env.total_reward.cpu().numpy()
        turns = env.turn.cpu().numpy()
        for idx, g in enumerate(self.population):
            g.fitness = float(fits[idx])
            g.turns_survived = int(turns[idx])

        max_idx = int(np.argmax(fits))
        self.best_genome = self.population[max_idx]
        max_fit = float(fits[max_idx])
        avg_fit = float(np.mean(fits))
        max_turns_survived = int(np.max(turns))

        self.fitness_history.append((self.generation, max_fit, avg_fit, max_turns_survived))
        return max_fit, avg_fit, max_turns_survived

    def _tournament_select(self, k=3):
        candidates = random.sample(self.population, k)
        return max(candidates, key=lambda g: g.fitness)

    def step_generation(self):
        """
        Advance population to next generation using elitism, crossover, and mutation.
        """
        new_pop = []
        for i in range(self.elitism_count):
            elite = Genome.from_dict(self.population[i].to_dict(), device=self.device)
            new_pop.append(elite)

        while len(new_pop) < self.pop_size:
            p1 = self._tournament_select()
            p2 = self._tournament_select()
            child = p1.crossover(p2)
            child.mutate(self.mutation_rate, self.mutation_scale)
            new_pop.append(child)

        self.population = new_pop
        self.generation += 1

    def save_best(self, filepath="saved_models/best_model.json"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        if self.best_genome:
            self.best_genome.save(filepath)
