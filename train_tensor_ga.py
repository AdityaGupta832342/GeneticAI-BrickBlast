#!/usr/bin/env python3
"""
GPU-Native Batched Training Script for Brick Blast AI (train_tensor_ga.py).
Uses TensorBrickBlastEnv and TensorGeneticAlgorithm to simulate hundreds of games in parallel
on CUDA Tensors, eliminating CPU-bound PyGame loops and achieving high GPU utilization.
"""
import time
import argparse
import torch
import numpy as np
from ai.tensor_ga import TensorGeneticAlgorithm


def main():
    parser = argparse.ArgumentParser(description="GPU-Native Batched Brick Blast GA Training")
    parser.add_argument("--generations", type=int, default=1000, help="Number of generations to evolve")
    parser.add_argument("--pop-size", type=int, default=200, help="Population size per generation")
    parser.add_argument("--max-turns", type=int, default=100, help="Max turns per game per generation")
    parser.add_argument("--save-path", type=str, default="saved_models/best_tensor_model.json", help="Path to save best genome")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--model-type", type=str, default="cnn", choices=["mlp", "cnn"], help="Architecture type (mlp or cnn)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], help="PyTorch execution device")
    parser.add_argument("--mutation-rate", type=float, default=0.15, help="Per-parameter mutation probability")
    parser.add_argument("--mutation-scale", type=float, default=0.25, help="Gaussian mutation stddev")
    parser.add_argument("--weight-limit", type=float, default=3.0, help="Absolute parameter clamp applied after mutation")
    parser.add_argument("--ricochet-bonus-scale", type=float, default=0.05, help="Reward per active post-hit physics tick")
    parser.add_argument("--brick-destroy-bonus", type=float, default=2.0, help="Reward for each brick destroyed")
    parser.add_argument("--brick-hit-bonus", type=float, default=0.10, help="Reward for each ball-brick collision")
    parser.add_argument("--survival-bonus", type=float, default=3.0, help="Reward for completing a turn without game over")
    parser.add_argument("--game-over-penalty", type=float, default=100.0, help="Penalty when a game terminates")
    parser.add_argument("--danger-row-penalty", type=float, default=2.0, help="Penalty per brick in the bottom two rows after turn resolution")
    args = parser.parse_args()

    device_str = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    print(f"=== GPU-Native Tensor GA Brick Blast Training ===")
    print(f"Model Type: {args.model_type.upper()} | Device: {device_str.upper()} | Generations: {args.generations} | Pop Size: {args.pop_size} | Max Turns: {args.max_turns}")
    print(f"Mutation: rate={args.mutation_rate} scale={args.mutation_scale} weight_limit={args.weight_limit}")
    print(
        "Reward: "
        f"damage=1.0 destroy={args.brick_destroy_bonus} hit={args.brick_hit_bonus} "
        f"ricochet={args.ricochet_bonus_scale} survive={args.survival_bonus} "
        f"game_over=-{args.game_over_penalty} danger_row=-{args.danger_row_penalty}"
    )

    ga = TensorGeneticAlgorithm(
        pop_size=args.pop_size,
        mutation_rate=args.mutation_rate,
        mutation_scale=args.mutation_scale,
        weight_limit=args.weight_limit,
        model_type=args.model_type,
        device=device_str,
        seed=args.seed,
        ricochet_bonus_scale=args.ricochet_bonus_scale,
        brick_destroy_bonus=args.brick_destroy_bonus,
        brick_hit_bonus=args.brick_hit_bonus,
        survival_bonus=args.survival_bonus,
        game_over_penalty=args.game_over_penalty,
        danger_row_penalty=args.danger_row_penalty,
    )

    start_time = time.time()
    for gen in range(1, args.generations + 1):
        gen_start = time.time()
        max_fit, avg_fit, max_turns = ga.evaluate_population(
            max_turns=args.max_turns,
            seed_offset=gen * 1000,
        )
        duration = time.time() - gen_start

        print(f"[Gen {gen:04d}/{args.generations}] Max Fit: {max_fit:7.1f} | Best Ever: {ga.best_fitness:7.1f} | Avg Fit: {avg_fit:7.1f} | Max Turns: {max_turns:2d} | Time: {duration:5.2f}s")

        if gen % 10 == 0 or gen == args.generations:
            ga.save_best(args.save_path)
            print(f"  -> Saved best model to {args.save_path}")

        ga.step_generation()

    total_time = time.time() - start_time
    print(f"\nTraining Complete in {total_time:.2f}s! Best model saved to {args.save_path}")


if __name__ == "__main__":
    main()
