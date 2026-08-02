"""
Genetic Algorithm engine for evolving Brick Blast AI Genomes.
Uses Python multiprocessing to evaluate generation fitness across CPU cores in parallel.
"""
import os
import random
import multiprocessing as mp
import numpy as np
import torch
from ai.genome import Genome
from env.brickblast_env import BrickBlastEnv


def _evaluate_single_genome(args):
    """
    Worker function for parallel evaluation of a single genome dictionary.
    Returns (fitness, turns_survived).
    """
    torch.set_num_threads(1)  # Prevent CPU thread oversubscription in worker processes
    if len(args) == 3:
        genome_dict, seed, device = args
    else:
        genome_dict, seed = args
        device = "cpu"
    genome = Genome.from_dict(genome_dict, device=device)
    env = BrickBlastEnv(seed=seed)

    obs, _ = env.reset(seed=seed)
    total_fitness = 0.0
    terminated = False
    truncated = False

    while not terminated and not truncated:
        angle = genome.select_action(env)
        obs, reward, terminated, truncated, info = env.step(angle)
        total_fitness += reward

    return float(total_fitness), int(info["turn"])


class GeneticAlgorithm:
    def __init__(self, pop_size=40, mutation_rate=0.15, mutation_scale=0.25, elitism_count=4, model_type="mlp", device="cpu", seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        self.model_type = model_type
        self.device = device
        self.elitism_count = max(2, min(pop_size // 5, elitism_count))

        self.population = [Genome(model_type=self.model_type, device=self.device) for _ in range(pop_size)]
        self.generation = 0
        self.best_genome = None
        self.fitness_history = []  # list of (gen, max_fit, avg_fit, max_turns)

    def evaluate_population(self, num_processes=None, seed_offset=0):
        """
        Evaluate all genomes in population in parallel.
        """
        if num_processes is None:
            num_processes = max(1, mp.cpu_count() - 1)

        args_list = [(g.to_dict(), seed_offset, self.device) for idx, g in enumerate(self.population)]

        if num_processes > 1:
            try:
                with mp.Pool(processes=num_processes) as pool:
                    results = pool.map(_evaluate_single_genome, args_list)
            except Exception:
                # Fallback to sequential if multiprocessing fails in restricted environment
                results = [_evaluate_single_genome(arg) for arg in args_list]
        else:
            results = [_evaluate_single_genome(arg) for arg in args_list]

        for g, (fit, turns) in zip(self.population, results):
            g.fitness = fit
            g.turns_survived = turns

        # Sort descending by fitness
        self.population.sort(key=lambda g: g.fitness, reverse=True)
        self.best_genome = self.population[0]

        max_fit = self.best_genome.fitness
        avg_fit = sum(g.fitness for g in self.population) / len(self.population)
        max_turns = max(g.turns_survived for g in self.population)

        self.fitness_history.append((self.generation, max_fit, avg_fit, max_turns))
        return max_fit, avg_fit, max_turns

    def _tournament_select(self, k=3):
        candidates = random.sample(self.population, k)
        return max(candidates, key=lambda g: g.fitness)

    def step_generation(self):
        """
        Advance population to next generation using elitism, crossover, and mutation.
        """
        new_pop = []
        # Elitism: preserve top performers unconditionally
        for i in range(self.elitism_count):
            elite = Genome.from_dict(self.population[i].to_dict())
            new_pop.append(elite)

        # Generate offspring
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
