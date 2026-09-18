"""Reactive Execution Layer for Warehouse Robot Pallet Stacking.

Monitors environmental state before and after each action execution.
If a disturbance event occurs (e.g., box knocked over mid-plan),
the reactive layer detects broken preconditions or displaced items and
injects localized repair actions *within* the existing plan instead of
replanning the entire trajectory from scratch.
"""

from typing import List, Set, Dict, Any, Tuple, Callable, Optional
import copy
from module_a_planning.goal_stack_planner import make_action, parse_predicate, Action


class ReactiveExecutor:
    def __init__(self, initial_state: Set[str]):
        self.state = set(initial_state)
        self.intervention_logs = []
        self.execution_trace = []

    def execute_action(self, action_str: str) -> bool:
        """Applies primitive action to state if preconditions hold."""
        p_name, p_args = parse_predicate(action_str)
        act = make_action(p_name, *p_args)

        # Check preconditions
        unmet = [p for p in act.preconds if p not in self.state]
        if unmet:
            return False

        # Apply state transition
        self.state = (self.state - act.del_list) | act.add_list
        return True

    def run_plan_with_disturbance(
        self,
        plan: List[str],
        disturbance_step: Optional[int] = None,
        disturbance_fn: Optional[Callable[[Set[str]], Tuple[Set[str], str]]] = None
    ) -> Dict[str, Any]:
        """Executes plan step-by-step with real-time sensory validation and reactive intervention."""
        self.execution_trace = []
        self.intervention_logs = []
        step_idx = 0
        plan_queue = list(plan)

        while plan_queue:
            step_idx += 1
            action_str = plan_queue.pop(0)

            # Check if disturbance is scheduled at this step
            if disturbance_step is not None and step_idx == disturbance_step and disturbance_fn:
                old_state = set(self.state)
                self.state, dist_desc = disturbance_fn(self.state)
                log_entry = {
                    "step": step_idx,
                    "event": "DISTURBANCE_OCCURRED",
                    "description": dist_desc,
                    "affected_state_delta": list(old_state ^ self.state)
                }
                self.intervention_logs.append(log_entry)
                self.execution_trace.append(f"!!! [DISTURBANCE @ STEP {step_idx}] {dist_desc}")

            # Verify action precondition
            p_name, p_args = parse_predicate(action_str)
            act = make_action(p_name, *p_args)
            unmet = [p for p in act.preconds if p not in self.state]

            if unmet:
                # Reactive layer intervenes!
                intervene_msg = (
                    f"REACTIVE INTERVENTION @ Step {step_idx}: Action '{action_str}' failed precondition(s): {unmet}. "
                    f"Synthesizing local repair sequence without discarding global plan."
                )
                self.execution_trace.append(f"--> [REACTIVE INTERVENTION] Preconditions broken: {unmet}")

                repair_actions = self._synthesize_local_repair(unmet)
                repair_log = {
                    "step": step_idx,
                    "event": "REACTIVE_REPAIR",
                    "unmet_preconditions": unmet,
                    "injected_repair_actions": repair_actions,
                    "plan_preserved_remaining": list(plan_queue)
                }
                self.intervention_logs.append(repair_log)

                # Prepend repair actions and the failed action back onto queue
                plan_queue = repair_actions + [action_str] + plan_queue
                continue

            # Execute normal action
            success = self.execute_action(action_str)
            self.execution_trace.append(f"Executed: {action_str} -> State: {sorted(list(self.state))}")

        return {
            "success": True,
            "final_state": sorted(list(self.state)),
            "execution_trace": self.execution_trace,
            "intervention_logs": self.intervention_logs
        }

    def _synthesize_local_repair(self, unmet_preconds: List[str]) -> List[str]:
        """Synthesize minimal local repair actions to restore broken preconditions."""
        repairs = []
        for cond in unmet_preconds:
            p_name, p_args = parse_predicate(cond)
            if p_name == "CLEAR" and len(p_args) == 1:
                target = p_args[0]
                # Find what is on target
                for s in self.state:
                    fn, fa = parse_predicate(s)
                    if fn == "ON" and len(fa) == 2 and fa[1] == target:
                        blocker = fa[0]
                        repairs.extend([f"UNSTACK({blocker},{target})", f"PUTDOWN({blocker})"])
            elif p_name == "HOLDING" and len(p_args) == 1:
                box = p_args[0]
                if f"ONTABLE({box})" in self.state and f"CLEAR({box})" in self.state:
                    repairs.append(f"PICKUP({box})")
            elif p_name == "ARMEMPTY":
                for s in self.state:
                    fn, fa = parse_predicate(s)
                    if fn == "HOLDING" and len(fa) == 1:
                        repairs.append(f"PUTDOWN({fa[0]})")
        return repairs
