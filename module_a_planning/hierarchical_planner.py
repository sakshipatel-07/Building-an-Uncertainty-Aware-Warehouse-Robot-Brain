"""Hierarchical Planning (HTN-style) for Warehouse Pallet Loading.

High-level task:
    LoadPallet(P, Stacks) -> expands into Subtasks:
        PreparePallet(P)
        BuildStack(TargetStack) -> expands into Primitive Actions:
            PICKUP / UNSTACK -> STACK / PUTDOWN
"""

from typing import List, Dict, Any, Set


class HierarchicalPlanner:
    def __init__(self, current_state: Set[str]):
        self.state = set(current_state)
        self.trace = []

    def load_pallet(self, pallet_id: str, desired_stacks: List[List[str]]) -> Dict[str, Any]:
        """High-level operator: LoadPallet(P).
        desired_stacks: list of stacks from bottom to top, e.g. [['A', 'B', 'C'], ['D', 'E']]
        """
        self.trace.append(f"[HTN] Expanding High-Level Task: LoadPallet({pallet_id}) with stacks {desired_stacks}")
        primitive_plan = []

        for stack_idx, stack in enumerate(desired_stacks):
            base = stack[0]
            self.trace.append(f"  [HTN Decomposition] Subtask: PlaceBaseBox({base}, Pallet={pallet_id}, Slot={stack_idx})")
            # If base box is not on table/pallet, unstack or move it
            if f"ONTABLE({base})" not in self.state:
                # Find what it is on
                primitive_plan.append(f"PICKUP({base})")
                primitive_plan.append(f"PUTDOWN({base})")

            # Stack subsequent boxes
            for i in range(1, len(stack)):
                prev = stack[i - 1]
                curr = stack[i]
                self.trace.append(f"  [HTN Decomposition] Subtask: StackBox({curr} on top of {prev})")
                primitive_plan.append(f"PICKUP({curr})")
                primitive_plan.append(f"STACK({curr}, {prev})")

        return {
            "high_level_task": f"LoadPallet({pallet_id})",
            "decomposition_trace": self.trace,
            "primitive_plan": primitive_plan
        }
