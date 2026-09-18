"""Module A — Planning: Stack the Pallet (Blocks World)
CLI: python plan.py start.json goal.json

Demonstrates:
1. Goal Stack Planning (arbitrary start to goal)
2. Nonlinear Planning (least commitment / constraint posting / parallel execution)
3. Hierarchical Planning (LoadPallet operator expansion)
4. Reactive Execution Layer (mid-plan disturbance recovery without full replan)
"""

import sys
import json
import os

# Ensure import path includes project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from module_a_planning.goal_stack_planner import GoalStackPlanner
from module_a_planning.nonlinear_planner import NonlinearPlanner
from module_a_planning.hierarchical_planner import HierarchicalPlanner
from module_a_planning.reactive_layer import ReactiveExecutor


def run_planning_pipeline(start_file: str, goal_file: str):
    print("=" * 70)
    print(" MODULE A: WAREHOUSE ROBOT PALLET PLANNER & REACTIVE CONTROLLER")
    print("=" * 70)

    # 1. Load Start and Goal states
    with open(start_file, "r") as f:
        start_data = json.load(f)
    with open(goal_file, "r") as f:
        goal_data = json.load(f)

    start_state = set(start_data.get("state", []))
    goal_state = set(goal_data.get("goal", []))

    print(f"\n[+] Start State ({len(start_state)} predicates):")
    print("    " + ", ".join(sorted(list(start_state))))
    print(f"\n[+] Goal State ({len(goal_state)} predicates):")
    print("    " + ", ".join(sorted(list(goal_state))))

    # 2. Goal Stack Planning
    print("\n" + "-" * 70)
    print("1. CLASSIC GOAL STACK PLANNING")
    print("-" * 70)
    gsp = GoalStackPlanner(start_state, goal_state)
    gsp_plan = gsp.solve()
    print(f"Goal Stack Plan ({len(gsp_plan)} actions):")
    for idx, act in enumerate(gsp_plan, 1):
        print(f"  {idx:02d}. {act}")

    # 3. Nonlinear Planning (Least-Commitment / Constraint Posting)
    print("\n" + "-" * 70)
    print("2. NONLINEAR (LEAST-COMMITMENT) PLANNER & PARALLEL STRATA")
    print("-" * 70)
    nlp = NonlinearPlanner(start_state, goal_state)
    nlp_res = nlp.solve()
    print(f"Nonlinear Plan Length: {nlp_res['total_steps']} steps")
    print("Parallel Executable Strata (Concurrent Robot Manipulators):")
    for lvl_idx, stage in enumerate(nlp_res['parallel_strata'], 1):
        print(f"  Stage {lvl_idx}: {stage}")
    print(f"\nAdvantage: The nonlinear planner generates partially ordered causal links,")
    print(f"allowing independent stacks (e.g. Tower A-B-C and Tower D-E) to execute concurrently.")

    # 4. Hierarchical Planning (HTN)
    print("\n" + "-" * 70)
    print("3. HIERARCHICAL PLANNING: LoadPallet(P1)")
    print("-" * 70)
    htn = HierarchicalPlanner(start_state)
    htn_res = htn.load_pallet("Pallet_Bay_1", [["A", "B", "C"], ["D", "E"]])
    for line in htn_res["decomposition_trace"]:
        print(line)
    print(f"Expanded into {len(htn_res['primitive_plan'])} primitive actions: {htn_res['primitive_plan']}")

    # 5. Reactive Execution Layer with Simulated Disturbance
    print("\n" + "-" * 70)
    print("4. REACTIVE LAYER: EXECUTION & MID-PLAN DISTURBANCE RECOVERY")
    print("-" * 70)

    # Disturbance simulation: At step 3, box D is knocked over by warehouse forklift vibration
    def disturbance_event(current_state):
        mod_state = set(current_state)
        # Forklift knocked over box B, causing CLEAR(A) to be false or B to fall to table
        mod_state.discard("CLEAR(A)")
        mod_state.add("ON(B,A)")
        return mod_state, "VIBRATION HAZARD: Box B accidentally slipped onto Box A, blocking Box A!"

    executor = ReactiveExecutor(start_state)
    exec_result = executor.run_plan_with_disturbance(
        plan=gsp_plan,
        disturbance_step=3,
        disturbance_fn=disturbance_event
    )

    print("\nExecution Trace:")
    for trace_item in exec_result["execution_trace"]:
        print(f"  {trace_item}")

    print(f"\nReactive Interventions Logged ({len(exec_result['intervention_logs'])} events):")
    for log_item in exec_result["intervention_logs"]:
        print(f"  -> [Log Entry] Step {log_item['step']}: {log_item['event']}")
        if "description" in log_item:
            print(f"     Description: {log_item['description']}")
        if "injected_repair_actions" in log_item:
            print(f"     Injected Repairs: {log_item['injected_repair_actions']}")
            print(f"     Preserved Remaining Plan: {log_item['plan_preserved_remaining']}")

    print("\n" + "=" * 70)
    print(" MODULE A EXECUTION COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        start_path = sys.argv[1]
        goal_path = sys.argv[2]
    else:
        # Default fallback
        base_dir = os.path.dirname(__file__)
        start_path = os.path.join(base_dir, "start.json")
        goal_path = os.path.join(base_dir, "goal.json")

    run_planning_pipeline(start_path, goal_path)
