"""MiniMax, Alpha-Beta Pruning, and Iterative Deepening Search with Move-Ordering.
"""

import time
from typing import Tuple, Optional, List, Dict, Any
from module_c_game.game_engine import DockGameState, DOCK_POS, ACTIONS


class GameSolver:
    def __init__(self):
        self.nodes_visited = 0

    # --------------------------------------------------------------------------
    # 1. PLAIN MINIMAX
    # --------------------------------------------------------------------------
    def minimax(self, state: DockGameState, depth: int, is_max: bool) -> Tuple[float, Optional[str]]:
        self.nodes_visited += 1
        term, winner = state.is_terminal()
        if depth == 0 or term:
            return state.evaluate(), None

        legal_moves = state.get_legal_actions(state.turn)
        best_move = legal_moves[0]

        if is_max:
            best_val = -float('inf')
            for move in legal_moves:
                nxt = state.apply_action(move)
                val, _ = self.minimax(nxt, depth - 1, False)
                if val > best_val:
                    best_val = val
                    best_move = move
            return best_val, best_move
        else:
            best_val = float('inf')
            for move in legal_moves:
                nxt = state.apply_action(move)
                val, _ = self.minimax(nxt, depth - 1, True)
                if val < best_val:
                    best_val = val
                    best_move = move
            return best_val, best_move

    # --------------------------------------------------------------------------
    # 2. ALPHA-BETA PRUNING
    # --------------------------------------------------------------------------
    def alphabeta(
        self,
        state: DockGameState,
        depth: int,
        alpha: float,
        beta: float,
        is_max: bool,
        ordered_moves: Optional[List[str]] = None
    ) -> Tuple[float, Optional[str]]:
        self.nodes_visited += 1
        term, winner = state.is_terminal()
        if depth == 0 or term:
            return state.evaluate(), None

        moves = ordered_moves if ordered_moves is not None else state.get_legal_actions(state.turn)
        best_move = moves[0]

        if is_max:
            best_val = -float('inf')
            for move in moves:
                nxt = state.apply_action(move)
                val, _ = self.alphabeta(nxt, depth - 1, alpha, beta, False)
                if val > best_val:
                    best_val = val
                    best_move = move
                alpha = max(alpha, best_val)
                if beta <= alpha:
                    break  # Beta cutoff (pruning)
            return best_val, best_move
        else:
            best_val = float('inf')
            for move in moves:
                nxt = state.apply_action(move)
                val, _ = self.alphabeta(nxt, depth - 1, alpha, beta, True)
                if val < best_val:
                    best_val = val
                    best_move = move
                beta = min(beta, best_val)
                if beta <= alpha:
                    break  # Alpha cutoff (pruning)
            return best_val, best_move

    # --------------------------------------------------------------------------
    # 3. ITERATIVE DEEPENING WITH MOVE-ORDERING HEURISTIC
    # --------------------------------------------------------------------------
    def order_moves(self, state: DockGameState, moves: List[str], player: int, best_previous_move: Optional[str] = None) -> List[str]:
        """Move-ordering heuristic:
        1. Prioritize best move from previous shallower search (principal variation)
        2. Order by immediate proximity to dock
        """
        def move_priority(m: str) -> float:
            if m == best_previous_move:
                return 10000.0  # Search PV move first!
            dr, dc = ACTIONS[m]
            pos = state.r1_pos if player == 1 else state.r2_pos
            nxt_pos = (pos[0] + dr, pos[1] + dc)
            dist_to_dock = abs(nxt_pos[0] - DOCK_POS[0]) + abs(nxt_pos[1] - DOCK_POS[1])
            # For MAX, smaller dist is better -> higher priority
            return -dist_to_dock if player == 1 else dist_to_dock

        return sorted(moves, key=move_priority, reverse=True)

    def iterative_deepening(self, state: DockGameState, max_depth: int) -> Dict[str, Any]:
        total_nodes = 0
        best_overall_move = None
        best_overall_val = 0.0
        depth_stats = []

        is_max = (state.turn == 1)

        t_start = time.perf_counter()
        for d in range(1, max_depth + 1):
            self.nodes_visited = 0
            legal_moves = state.get_legal_actions(state.turn)
            ordered = self.order_moves(state, legal_moves, state.turn, best_overall_move)

            val, move = self.alphabeta(state, d, -float('inf'), float('inf'), is_max, ordered_moves=ordered)
            t_depth = time.perf_counter() - t_start
            total_nodes += self.nodes_visited
            best_overall_move = move
            best_overall_val = val

            depth_stats.append({
                "depth": d,
                "nodes": self.nodes_visited,
                "cumulative_nodes": total_nodes,
                "best_move": move,
                "score": val,
                "elapsed_sec": t_depth
            })

        return {
            "best_move": best_overall_move,
            "score": best_overall_val,
            "total_nodes": total_nodes,
            "depth_stats": depth_stats
        }
