#!/usr/bin/env python3
"""
Train a Genetic Algorithm AI model on the Brick Blast environment.
Usage:
    python train_genetic.py --generations 15 --pop-size 20 --processes 4
"""
import argparse
import os
import sys
import time
from ai.ga import GeneticAlgorithm
from PIL import Image, ImageDraw, ImageFont


def _plot_fitness_history_pil(history, filename="fitness_history.png"):
    """
    Draw a fitness vs. generation chart using Pillow so it runs without external plotting dependencies.
    """
    w, h = 600, 400
    img = Image.new("RGB", (w, h), (26, 38, 78))
    draw = ImageDraw.Draw(img)

    margin_x = 60
    margin_y = 50
    plot_w = w - 2 * margin_x
    plot_h = h - 2 * margin_y

    # Draw axes
    draw.line([(margin_x, h - margin_y), (w - margin_x, h - margin_y)], fill=(255, 255, 255), width=2)
    draw.line([(margin_x, margin_y), (margin_x, h - margin_y)], fill=(255, 255, 255), width=2)

    if not history:
        img.save(filename)
        return

    max_val = max(max(r[1] for r in history), 10.0)
    num_gens = max(len(history) - 1, 1)

    def coords(gen_idx, val):
        x = margin_x + int((gen_idx / num_gens) * plot_w)
        y = (h - margin_y) - int((val / max_val) * plot_h)
        return x, y

    # Plot Max Fitness (Green)
    pts_max = [coords(idx, row[1]) for idx, row in enumerate(history)]
    if len(pts_max) > 1:
        draw.line(pts_max, fill=(77, 192, 77), width=3)

    # Plot Avg Fitness (Yellow)
    pts_avg = [coords(idx, row[2]) for idx, row in enumerate(history)]
    if len(pts_avg) > 1:
        draw.line(pts_avg, fill=(240, 192, 32), width=2)

    try:
        font = ImageFont.load_default()
        draw.text((margin_x + 10, margin_y + 10), "Max Fitness (Green) / Avg Fitness (Yellow)", fill=(255, 255, 255), font=font)
    except Exception:
        pass

    img.save(filename)


def main():
    parser = argparse.ArgumentParser(description="Train Brick Blast Genetic AI")
    parser.add_argument("--generations", type=int, default=1000, help="Number of generations to evolve")
    parser.add_argument("--pop-size", type=int, default=200, help="Population size per generation")
    parser.add_argument("--processes", type=int, default=8, help="CPU processes for evaluation")
    parser.add_argument("--save-path", type=str, default="saved_models/best_model.json", help="Path to save best genome")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--model-type", type=str, default="mlp", choices=["mlp", "cnn"], help="Architecture type (mlp or cnn)")
    args = parser.parse_args()

    print(f"=== Brick Blast GA AI Training ===")
    print(f"Model Type: {args.model_type.upper()} | Generations: {args.generations} | Pop Size: {args.pop_size} | Processes: {args.processes}")

    ga = GeneticAlgorithm(pop_size=args.pop_size, model_type=args.model_type, seed=args.seed)
    start_time = time.time()

    for gen in range(args.generations):
        gen_start = time.time()
        max_fit, avg_fit, max_turns = ga.evaluate_population(
            num_processes=args.processes,
            seed_offset=gen * 1000,
        )
        duration = time.time() - gen_start
        print(f"[Gen {gen+1:02d}/{args.generations:02d}] "
              f"Max Fit: {max_fit:7.1f} | Avg Fit: {avg_fit:7.1f} | "
              f"Max Turns: {max_turns:2d} | Time: {duration:4.2f}s")

        ga.save_best(args.save_path)
        if gen < args.generations - 1:
            ga.step_generation()

    total_time = time.time() - start_time
    print(f"\nTraining Complete in {total_time:.2f}s!")
    print(f"Best AI Model saved to: {args.save_path}")

    _plot_fitness_history_pil(ga.fitness_history, "fitness_history.png")
    print("Fitness progression chart saved to: fitness_history.png")


if __name__ == "__main__":
    main()
