"""
Publication-Quality Tournament Heatmap Plotter
==============================================
Generates a standard Round-Robin Win Rate Matrix Heatmap (matching AlphaStar / Metamon paper style).
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle


def plot_tournament_heatmap(json_path: str, output_image_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    agents = data["agents"]
    matrix = np.array(data["matrix"], dtype=float)
    n = len(agents)

    # Set up matplotlib style
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    
    fig, ax = plt.subplots(figsize=(8.5, 7.5), dpi=300)

    # Custom Diverging Colormap (Blue -> White -> Red)
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Mask for diagonal (self-play)
    mask = np.eye(n, dtype=bool)

    # Draw Heatmap
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".0f",
        cmap=cmap,
        vmin=0,
        vmax=100,
        cbar=False,
        linewidths=2.5,
        linecolor="white",
        square=True,
        xticklabels=agents,
        yticklabels=agents,
        annot_kws={"fontsize": 13, "weight": "bold"},
        ax=ax,
        mask=mask
    )

    # Draw Hatched Diagonal for Self-Play
    for i in range(n):
        rect = Rectangle(
            (i, i), 1, 1,
            fill=True,
            facecolor="#e0e0e0",
            edgecolor="white",
            hatch="//////",
            linewidth=2.5
        )
        ax.add_patch(rect)

    # Title & Axis Labels matching publication format
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    
    plt.xticks(rotation=40, ha="left", fontsize=11, weight="bold")
    plt.yticks(rotation=0, fontsize=11, weight="bold")

    ax.set_title("Win Rate Of:", fontsize=16, weight="bold", pad=25)
    ax.set_ylabel("When Playing Against.", fontsize=14, weight="bold", labelpad=15)

    # Adjust text colors inside cells for optimal contrast
    for text in ax.texts:
        val = float(text.get_text())
        if val > 65 or val < 35:
            text.set_color("white")
        else:
            text.set_color("#1a1a1a")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    plt.savefig(output_image_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[+] Heatmap successfully generated and saved to: {output_image_path}")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_file = os.path.join(current_dir, "../tournament_matrix.json")
    out_img = os.path.join(current_dir, "tournament_winrate_matrix.png")
    artifact_img = r"C:\Users\caihe\.gemini\antigravity\brain\2ce8a04b-8049-4d75-aa84-633895bca605\tournament_winrate_matrix.png"
    
    plot_tournament_heatmap(json_file, out_img)
    plot_tournament_heatmap(json_file, artifact_img)
