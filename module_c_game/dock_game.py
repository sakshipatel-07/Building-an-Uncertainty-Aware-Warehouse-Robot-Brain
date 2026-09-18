"""Module C — Game Playing: Dock Contention
Deliverable: python dock_game.py --depth 5

Simulates two autonomous mobile robots competing for the last charging dock.
Features:
- Alpha-Beta Pruning with logged node counts
- Iterative Deepening with move ordering
- Modes: AI vs AI, AI vs Random, AI vs Human
"""

import argparse
import sys
import os
import random
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from module_c_game.game_engine import DockGameState, DOCK_POS
from module_c_game.minimax import GameSolver


def play_dock_game(search_depth: int = 5, mode: str = "ai-vs-ai"):
    print("=" * 70)
    print(" MODULE C: DOCK CONTENTION ADVERSARIAL GAME")
    print(f" Mode: {mode.upper()} | Search Depth: {search_depth}")
    print(f" Target Charging Dock: {DOCK_POS} marked as [D]")
    print(" Robot 1 (MAX / AI): R1 starts at (0, 0)")
    print(" Robot 2 (MIN / Opponent): R2 starts at (3, 3)")
    print(" Obstacles: ### at (1, 0) and (2, 3)")
    print("=" * 70)

    state = DockGameState()
    solver = GameSolver()

    step_num = 0
    print(state.render())

    while True:
        term, winner = state.is_terminal()
        if term:
            print("=" * 70)
            if winner == 1:
                print(">>> GAME OVER: Robot 1 (AI) REACHED THE DOCK AND WON! <<<")
            elif winner == 2:
                print(">>> GAME OVER: Robot 2 (Opponent) REACHED THE DOCK AND WON! <<<")
            else:
                print(">>> GAME OVER: REACHED MAX MOVES — DRAW! <<<")
            print("=" * 70)
            return winner

        step_num += 1
        current_turn = state.turn
        actor_name = "Robot 1 (AI)" if current_turn == 1 else "Robot 2 (Opponent)"

        if current_turn == 1:
            # Robot 1: AI using Iterative Deepening + Alpha-Beta
            t0 = time.perf_counter()
            res = solver.iterative_deepening(state, max_depth=search_depth)
            dt = (time.perf_counter() - t0) * 1000.0
            chosen_move = res["best_move"]
            print(f"[Step {step_num:02d}] {actor_name} chooses {chosen_move} "
                  f"(Eval: {res['score']:.1f}, Nodes: {res['total_nodes']}, Time: {dt:.1f}ms)")
        else:
            # Robot 2
            if mode == "ai-vs-ai":
                t0 = time.perf_counter()
                res = solver.iterative_deepening(state, max_depth=search_depth)
                dt = (time.perf_counter() - t0) * 1000.0
                chosen_move = res["best_move"]
                print(f"[Step {step_num:02d}] {actor_name} chooses {chosen_move} "
                      f"(Eval: {res['score']:.1f}, Nodes: {res['total_nodes']}, Time: {dt:.1f}ms)")
            elif mode == "ai-vs-random":
                legal = state.get_legal_actions(2)
                chosen_move = random.choice(legal)
                print(f"[Step {step_num:02d}] {actor_name} chooses random action: {chosen_move}")
            else:
                legal = state.get_legal_actions(2)
                print(f"Available legal actions: {legal}")
                chosen_move = input("Enter action: ").strip().upper()
                if chosen_move not in legal:
                    chosen_move = legal[0]

        state = state.apply_action(chosen_move)
        print(state.render())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Warehouse Dock Contention Game")
    parser.add_argument("--depth", type=int, default=5, help="Search depth limit (default: 5)")
    parser.add_argument("--mode", type=str, default="ai-vs-ai",
                        choices=["ai-vs-ai", "ai-vs-random"],
                        help="Game mode: ai-vs-ai or ai-vs-random")
    args = parser.parse_args()

    play_dock_game(search_depth=args.depth, mode=args.mode)
