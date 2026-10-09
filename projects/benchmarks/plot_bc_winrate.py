"""
Visualizador Gráfico de Win Rate e Performance do Modelo Behavioral Cloning (BC)
================================================================================
Gera gráficos detalhados do benchmark de 100 partidas contra os 5 bots da comunidade
e métricas de validação do modelo supervisionado.
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np

def generate_bc_benchmark_plot(json_path="projects/benchmarks/benchmark_results_100.json", output_png="projects/benchmarks/bc_winrate_benchmark.png"):
    if not os.path.exists(json_path):
        print(f"[-] Arquivo de benchmark não encontrado: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    bots = list(data.keys())
    win_rates = [data[b]["win_rate"] for b in bots]
    wins = [data[b]["wins"] for b in bots]
    losses = [data[b]["losses"] for b in bots]
    ties = [data[b].get("ties", 0) for b in bots]
    avg_turns = [data[b]["avg_turns"] for b in bots]

    # Ordenar por win rate decrescente
    sorted_indices = np.argsort(win_rates)[::-1]
    bots = [bots[i] for i in sorted_indices]
    win_rates = [win_rates[i] for i in sorted_indices]
    wins = [wins[i] for i in sorted_indices]
    losses = [losses[i] for i in sorted_indices]
    ties = [ties[i] for i in sorted_indices]
    avg_turns = [avg_turns[i] for i in sorted_indices]

    overall_wr = (sum(wins) / sum(wins + losses + ties)) * 100.0

    # Configuração de Estilo Moderno Dark
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [1.8, 1.2]})
    fig.patch.set_facecolor('#0d1117')
    ax1.set_facecolor('#161b22')
    ax2.set_facecolor('#161b22')

    # Subplot 1: Win Rate Bar Chart
    colors = ['#2ea043' if wr >= 50 else '#388bfd' if wr >= 25 else '#f85149' for wr in win_rates]
    bars = ax1.bar(bots, win_rates, color=colors, width=0.55, edgecolor='#30363d', linewidth=1.2, zorder=3)
    ax1.axhline(50, color='#8b949e', linestyle='--', linewidth=1.2, alpha=0.7, label='50% Paridade')
    ax1.axhline(overall_wr, color='#e3b341', linestyle='-', linewidth=2.0, alpha=0.9, label=f'Média Geral ({overall_wr:.1f}%)')

    for bar, wr, w, l in zip(bars, win_rates, wins, losses):
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 1.8, f"{wr:.1f}%\n({w}V-{l}D)", 
                 ha='center', va='bottom', fontsize=11, fontweight='bold', color='#f0f6fc')

    ax1.set_title("Amnesia-AI (Modelo BC) — Win Rate vs Bots da Comunidade (100 Partidas)", 
                  fontsize=13, fontweight='bold', color='#f0f6fc', pad=15)
    ax1.set_ylabel("Win Rate (%)", fontsize=11, color='#c9d1d9')
    ax1.set_ylim(0, 85)
    ax1.grid(axis='y', linestyle=':', alpha=0.25, color='#8b949e')
    ax1.tick_params(axis='x', labelsize=10, colors='#f0f6fc')
    ax1.tick_params(axis='y', labelsize=10, colors='#c9d1d9')
    ax1.legend(loc='upper right', framealpha=0.3, facecolor='#21262d', edgecolor='#30363d', fontsize=10)

    # Subplot 2: Partidas & Duração Média (Turnos)
    y_pos = np.arange(len(bots))
    hbars = ax2.barh(y_pos, avg_turns, color='#a371f7', height=0.5, edgecolor='#30363d', zorder=3)
    for hbar, turns in zip(hbars, avg_turns):
        xval = hbar.get_width()
        ax2.text(xval + 0.8, hbar.get_y() + hbar.get_height()/2.0, f"{turns:.1f} turns", 
                 ha='left', va='center', fontsize=10, color='#f0f6fc', fontweight='bold')

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(bots, fontsize=10, color='#f0f6fc')
    ax2.invert_yaxis()
    ax2.set_xlim(0, 45)
    ax2.set_title("Duração Média das Batalhas", fontsize=13, fontweight='bold', color='#f0f6fc', pad=15)
    ax2.set_xlabel("Média de Turnos por Partida", fontsize=11, color='#c9d1d9')
    ax2.grid(axis='x', linestyle=':', alpha=0.25, color='#8b949e')
    ax2.tick_params(axis='x', labelsize=10, colors='#c9d1d9')
    ax2.tick_params(axis='y', labelsize=10, colors='#f0f6fc')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    plt.savefig(output_png, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[+] Gráfico salvo com sucesso em: {output_png}")

if __name__ == "__main__":
    generate_bc_benchmark_plot()
