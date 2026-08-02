#!/usr/bin/env python3
"""
Generate a sample fitness progression chart illustrating Genetic Algorithm evolution
in Brick Blast over 15 generations.
"""
from PIL import Image, ImageDraw, ImageFont


def plot_chart(filename="fitness_history.png"):
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

    # Simulated realistic evolution curve over 15 generations
    history_max = [1200, 2500, 4100, 6800, 9200, 11500, 14200, 16297, 18500, 21000, 24500, 26800, 29000, 31500, 34200]
    history_avg = [600, 1100, 1900, 3200, 4500, 5900, 7800, 9500, 11200, 13100, 14900, 16500, 18200, 20100, 22000]

    max_val = 35000.0
    num_gens = len(history_max) - 1

    def coords(gen_idx, val):
        x = margin_x + int((gen_idx / num_gens) * plot_w)
        y = (h - margin_y) - int((val / max_val) * plot_h)
        return x, y

    pts_max = [coords(idx, val) for idx, val in enumerate(history_max)]
    pts_avg = [coords(idx, val) for idx, val in enumerate(history_avg)]

    draw.line(pts_max, fill=(77, 192, 77), width=3)
    draw.line(pts_avg, fill=(240, 192, 32), width=2)

    try:
        font = ImageFont.load_default()
        draw.text((margin_x + 10, margin_y + 10), "Max Fitness (Green) / Avg Fitness (Yellow)", fill=(255, 255, 255), font=font)
    except Exception:
        pass

    img.save(filename)
    print(f"Saved chart to: {filename}")


if __name__ == "__main__":
    plot_chart()
