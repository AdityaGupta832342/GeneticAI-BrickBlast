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
    args = parser.parse_args()

    device_str = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    print(f"=== GPU-Native Tensor GA Brick Blast Training ===")
    print(f"Model Type: {args.model_type.upper()} | Device: {device_str.upper()} | Generations: {args.generations} | Pop Size: {args.pop_size} | Max Turns: {args.max_turns}")

    ga = TensorGeneticAlgorithm(
        pop_size=args.pop_size,
        model_type=args.model_type,
        device=device_str,
        seed=args.seed,
    )

    start_time = time.time()
    for gen in range(1, args.generations + 1):
        gen_start = time.time()
        max_fit, avg_fit, max_turns = ga.evaluate_population(
            max_turns=args.max_turns,
            seed_offset=gen * 1000,
        )
        duration = time.time() - gen_start

        print(f"[Gen {gen:04d}/{args.generations}] Max Fit: {max_fit:7.1f} | Avg Fit: {avg_fit:7.1f} | Max Turns: {max_turns:2d} | Time: {duration:5.2f}s")

        if gen % 10 == 0 or gen == args.generations:
            ga.save_best(args.save_path)
            print(f"  -> Saved best model to {args.save_path}")

        ga.step_generation()

    total_time = time.time() - start_time
    print(f"\nTraining Complete in {total_time:.2f}s! Best model saved to {args.save_path}")


if __name__ == "__main__":
    main()
