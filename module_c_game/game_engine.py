"""Warehouse Dock Contention Game Engine.

A 4x4 warehouse grid where Robot 1 (MAX, AI) and Robot 2 (MIN, Opponent)
compete for the last available charging dock.
Branching factor >= 3 (Up, Down, Left, Right, Wait, minus walls).
"""

from typing import List, Tuple, Optional, Set
import copy


# Grid dimensions
GRID_SIZE = 4
DOCK_POS = (1, 2)
OBSTACLES = {(1, 0), (2, 3)}

# Actions: dx, dy
ACTIONS = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1),
    "WAIT": (0, 0)
}


class DockGameState:
    def __init__(
        self,
        r1_pos: Tuple[int, int] = (0, 0),
        r2_pos: Tuple[int, int] = (3, 3),
        turn: int = 1,  # 1 for MAX (Robot 1), 2 for MIN (Robot 2)
        steps: int = 0
    ):
        self.r1_pos = r1_pos
        self.r2_pos = r2_pos
        self.turn = turn
        self.steps = steps

    def is_valid_pos(self, pos: Tuple[int, int]) -> bool:
        r, c = pos
        return 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE and pos not in OBSTACLES

    def get_legal_actions(self, player: int) -> List[str]:
        pos = self.r1_pos if player == 1 else self.r2_pos
        other_pos = self.r2_pos if player == 1 else self.r1_pos
        legal = []
        for act_name, (dr, dc) in ACTIONS.items():
            new_pos = (pos[0] + dr, pos[1] + dc)
            # Valid cell and cannot move onto current position of other robot
            if self.is_valid_pos(new_pos) and new_pos != other_pos:
                legal.append(act_name)
        # Always at least WAIT if boxed in
        if not legal:
            legal.append("WAIT")
        return legal

    def apply_action(self, action: str) -> "DockGameState":
        dr, dc = ACTIONS[action]
        if self.turn == 1:
            new_r1 = (self.r1_pos[0] + dr, self.r1_pos[1] + dc)
            return DockGameState(new_r1, self.r2_pos, turn=2, steps=self.steps + 1)
        else:
            new_r2 = (self.r2_pos[0] + dr, self.r2_pos[1] + dc)
            return DockGameState(self.r1_pos, new_r2, turn=1, steps=self.steps + 1)

    def is_terminal(self) -> Tuple[bool, Optional[int]]:
        """Returns (is_terminal, winner). Winner: 1 (R1 wins), 2 (R2 wins), 0 (Draw), None (Not terminal)."""
        if self.r1_pos == DOCK_POS:
            return True, 1
        if self.r2_pos == DOCK_POS:
            return True, 2
        if self.steps >= 40:
            return True, 0
        return False, None

    def evaluate(self) -> float:
        """Heuristic evaluation from MAX (Robot 1)'s perspective."""
        term, winner = self.is_terminal()
        if term:
            if winner == 1:
                return 10000.0 - self.steps * 10
            elif winner == 2:
                return -10000.0 + self.steps * 10
            return 0.0

        # Manhattan distances to dock
        d1 = abs(self.r1_pos[0] - DOCK_POS[0]) + abs(self.r1_pos[1] - DOCK_POS[1])
        d2 = abs(self.r2_pos[0] - DOCK_POS[0]) + abs(self.r2_pos[1] - DOCK_POS[1])

        # Mobility (number of free moves)
        mob1 = len(self.get_legal_actions(1))
        mob2 = len(self.get_legal_actions(2))

        # Score: Closer is better (hence negative d1), opponent further is better (+d2)
        score = (d2 - d1) * 30.0 + (mob1 - mob2) * 5.0
        return score

    def render(self) -> str:
        rows = []
        for r in range(GRID_SIZE):
            line = []
            for c in range(GRID_SIZE):
                pos = (r, c)
                if pos == self.r1_pos and pos == DOCK_POS:
                    line.append("R1*")
                elif pos == self.r2_pos and pos == DOCK_POS:
                    line.append("R2*")
                elif pos == self.r1_pos:
                    line.append("R1 ")
                elif pos == self.r2_pos:
                    line.append("R2 ")
                elif pos == DOCK_POS:
                    line.append("[D]")
                elif pos in OBSTACLES:
                    line.append("###")
                else:
                    line.append(" . ")
            rows.append(" | ".join(line))
        return "\n" + "\n---+---+---+---\n".join(rows) + "\n"
