"""Goal Stack Planner (Classic STRIPS-style) for Blocks World (Boxes A-G).

Handles predicates:
- ON(x, y)
- ONTABLE(x)
- CLEAR(x)
- HOLDING(x)
- ARMEMPTY
"""

import copy
from typing import List, Set, Tuple, Dict, Any, Optional


class Action:
    def __init__(self, name: str, params: List[str], preconds: Set[str], add_list: Set[str], del_list: Set[str]):
        self.name = name
        self.params = params
        self.preconds = preconds
        self.add_list = add_list
        self.del_list = del_list

    def __repr__(self):
        return f"{self.name}({', '.join(self.params)})"


def parse_predicate(pred_str: str) -> Tuple[str, List[str]]:
    pred_str = pred_str.strip()
    if '(' in pred_str and pred_str.endswith(')'):
        name = pred_str[:pred_str.index('(')].strip()
        args = [a.strip() for a in pred_str[pred_str.index('(')+1:-1].split(',') if a.strip()]
        return name, args
    return pred_str, []


def make_action(name: str, *args: str) -> Action:
    if name == "STACK":
        x, y = args
        return Action(
            "STACK", [x, y],
            preconds={f"HOLDING({x})", f"CLEAR({y})"},
            add_list={f"ON({x},{y})", f"CLEAR({x})", "ARMEMPTY"},
            del_list={f"HOLDING({x})", f"CLEAR({y})"}
        )
    elif name == "UNSTACK":
        x, y = args
        return Action(
            "UNSTACK", [x, y],
            preconds={f"ON({x},{y})", f"CLEAR({x})", "ARMEMPTY"},
            add_list={f"HOLDING({x})", f"CLEAR({y})"},
            del_list={f"ON({x},{y})", f"CLEAR({x})", "ARMEMPTY"}
        )
    elif name == "PICKUP":
        (x,) = args
        return Action(
            "PICKUP", [x],
            preconds={f"ONTABLE({x})", f"CLEAR({x})", "ARMEMPTY"},
            add_list={f"HOLDING({x})"},
            del_list={f"ONTABLE({x})", f"CLEAR({x})", "ARMEMPTY"}
        )
    elif name == "PUTDOWN":
        (x,) = args
        return Action(
            "PUTDOWN", [x],
            preconds={f"HOLDING({x})"},
            add_list={f"ONTABLE({x})", f"CLEAR({x})", "ARMEMPTY"},
            del_list={f"HOLDING({x})"}
        )
    raise ValueError(f"Unknown action {name}")


class GoalStackPlanner:
    def __init__(self, start_state: Set[str], goal_state: Set[str]):
        self.state = set(start_state)
        self.goal_state = set(goal_state)
        self.plan: List[str] = []

    def get_supporting_actions(self, goal: str) -> List[Action]:
        name, args = parse_predicate(goal)
        actions = []
        if name == "ON":
            x, y = args
            actions.append(make_action("STACK", x, y))
        elif name == "ONTABLE":
            (x,) = args
            actions.append(make_action("PUTDOWN", x))
        elif name == "CLEAR":
            (x,) = args
            # Box x is clear if whatever is on x is unstacked
            # Look in current state for what is ON(z, x)
            for fact in self.state:
                p_name, p_args = parse_predicate(fact)
                if p_name == "ON" and len(p_args) == 2 and p_args[1] == x:
                    actions.append(make_action("UNSTACK", p_args[0], x))
            # Also can unstack any possible box if not in state
            for b in ["A", "B", "C", "D", "E", "F", "G"]:
                if b != x:
                    actions.append(make_action("UNSTACK", b, x))
        elif name == "HOLDING":
            (x,) = args
            # Can pickup from table or unstack from some block
            actions.append(make_action("PICKUP", x))
            for b in ["A", "B", "C", "D", "E", "F", "G"]:
                if b != x:
                    actions.append(make_action("UNSTACK", x, b))
        elif name == "ARMEMPTY":
            for b in ["A", "B", "C", "D", "E", "F", "G"]:
                actions.append(make_action("PUTDOWN", b))
                for c in ["A", "B", "C", "D", "E", "F", "G"]:
                    if c != b:
                        actions.append(make_action("STACK", b, c))
        return actions

    def solve(self, max_steps: int = 150) -> List[str]:
        stack = []
        # Push composite goal
        stack.append(frozenset(self.goal_state))
        for g in sorted(list(self.goal_state)):
            stack.append(g)

        steps = 0
        while stack and steps < max_steps:
            steps += 1
            top = stack.pop()

            if isinstance(top, frozenset):
                # Composite goal
                if not top.issubset(self.state):
                    # Push composite again, then individual goals
                    unsatisfied = [g for g in top if g not in self.state]
                    stack.append(top)
                    for g in unsatisfied:
                        stack.append(g)
            elif isinstance(top, Action):
                # Apply action to state
                self.state = (self.state - top.del_list) | top.add_list
                self.plan.append(str(top))
            elif isinstance(top, str):
                # Single predicate goal
                if top in self.state:
                    continue
                # Choose supporting action
                actions = self.get_supporting_actions(top)
                best_act = None
                best_score = float('inf')

                for act in actions:
                    # Prefer actions whose preconditions are already mostly met
                    unmet = len([p for p in act.preconds if p not in self.state])
                    if unmet < best_score:
                        best_score = unmet
                        best_act = act

                if best_act:
                    stack.append(best_act)
                    # Push action's preconds as composite, then individual
                    stack.append(frozenset(best_act.preconds))
                    for p in sorted(list(best_act.preconds)):
                        stack.append(p)

        return self.plan
