"""Module C Benchmark: MiniMax vs Alpha-Beta vs Alpha-Beta+IDS across depths 2-6.
Saves nodes-explored comparison chart to dock_search_chart.png.
"""

import os
import sys
import matplotlib
matplotlib.use('Agg')  # Headless rendering
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from module_c_game.game_engine import DockGameState
from module_c_game.minimax import GameSolver


def benchmark_search_algorithms():
    print("=" * 70)
    print(" MODULE C BENCHMARK: MINIMAX vs ALPHA-BETA vs IDS (DEPTHS 2 to 6)")
    print("=" * 70)

    initial_state = DockGameState()
    solver = GameSolver()

    depths = list(range(2, 7))
    minimax_nodes = []
    alphabeta_nodes = []
    ids_nodes = []

    print(f"\n{'Depth':<6} | {'MiniMax Nodes':<14} | {'Alpha-Beta Nodes':<18} | {'AB+IDS Nodes':<14} | {'Pruning Reduction':<18}")
    print("-" * 75)

    for d in depths:
        # 1. Plain MiniMax
        solver.nodes_visited = 0
        solver.minimax(initial_state, d, True)
        mm_count = solver.nodes_visited
        minimax_nodes.append(mm_count)

        # 2. Alpha-Beta Pruning
        solver.nodes_visited = 0
        solver.alphabeta(initial_state, d, -float('inf'), float('inf'), True)
        ab_count = solver.nodes_visited
        alphabeta_nodes.append(ab_count)

        # 3. Alpha-Beta + IDS with Move Ordering
        ids_res = solver.iterative_deepening(initial_state, d)
        ids_count = ids_res["total_nodes"]
        ids_nodes.append(ids_count)

        reduction_pct = (1.0 - (ab_count / mm_count)) * 100.0 if mm_count > 0 else 0.0
        print(f"{d:<6} | {mm_count:<14} | {ab_count:<18} | {ids_count:<14} | {reduction_pct:>6.1f}% reduction")

    # Generate matplotlib chart
    plt.figure(figsize=(9, 5))
    plt.plot(depths, minimax_nodes, 'ro-', linewidth=2, label='Plain MiniMax')
    plt.plot(depths, alphabeta_nodes, 'bs-', linewidth=2, label='Alpha-Beta Pruning')
    plt.plot(depths, ids_nodes, 'g^-', linewidth=2, label='Alpha-Beta + IDS + Move Ordering')

    plt.yscale('log')
    plt.xlabel('Search Depth Limit', fontsize=12)
    plt.ylabel('Nodes Explored (Log Scale)', fontsize=12)
    plt.title('Dock Contention Game Tree: Search Nodes Explored vs Depth', fontsize=14, fontweight='bold')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=11)
    plt.tight_layout()

    out_path = os.path.join(os.path.dirname(__file__), "dock_search_chart.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\n[+] Chart successfully generated and saved to: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    benchmark_search_algorithms()
