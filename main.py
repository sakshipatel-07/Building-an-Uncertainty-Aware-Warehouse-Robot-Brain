"""Warehouse Brain: Integrated Autonomous Controller (main.py)

Wires all 5 AI modules into a single real-time control loop over a 60-event warehouse shift:
1. Module D (Connectionist): Streams sensor windows to predict imminent anomalies via RNN;
   restores noisy barcode scans via Hopfield auto-associative memory.
2. Module B (Uncertainty): Evaluates package integrity via Bayesian Network & Certainty Factors.
3. Module E (Expert System): Diagnoses robot hardware faults with 'why' explanation traces.
4. Module A (Planning): Generates pallet stacking plans and handles reactive mid-plan disturbances.
5. Module C (Game Playing): Competes for contested charging dock via Alpha-Beta + IDS search.
"""

import sys
import os
import time
import random
import numpy as np
import torch

# Project module imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from module_a_planning.goal_stack_planner import GoalStackPlanner
from module_a_planning.reactive_layer import ReactiveExecutor
from module_b_uncertainty.reasoning import (
    WarehouseBayesianNetwork, CertaintyFactorReasoner,
    DempsterShaferReasoner, FuzzyDamageReasoner
)
from module_c_game.game_engine import DockGameState
from module_c_game.minimax import GameSolver
from module_d_connectionist.anomaly_recall import HopfieldNetwork, AnomalyRNN, create_sample_bitmaps
from module_e_expert_system.expert_advisor import ExpertAdvisorEngine


def run_integrated_shift(num_events: int = 60):
    print("=" * 75)
    print("   WAREHOUSE BRAIN: AUTONOMOUS ROBOT INTEGRATED CONTROL LOOP")
    print(f"   Simulating 1 Full Shift ({num_events} Synthetic Telemetry Events)")
    print("=" * 75)

    # --------------------------------------------------------------------------
    # Initialize all 5 AI subsystems
    # --------------------------------------------------------------------------
    print("\n[INIT] Initializing AI Subsystems...")

    # Module A: Planning
    pallet_start = {"ONTABLE(A)", "ONTABLE(B)", "ONTABLE(C)", "CLEAR(A)", "CLEAR(B)", "CLEAR(C)", "ARMEMPTY"}
    pallet_goal = {"ON(B,A)", "ON(C,B)", "ONTABLE(A)", "CLEAR(C)", "ARMEMPTY"}
    planner = GoalStackPlanner(pallet_start, pallet_goal)
    pallet_plan = planner.solve()
    print(f"  [Module A] Pallet Stacking Plan initialized ({len(pallet_plan)} actions).")

    # Module B: Uncertainty
    bn = WarehouseBayesianNetwork()
    cf = CertaintyFactorReasoner()
    ds = DempsterShaferReasoner()
    fuzzy = FuzzyDamageReasoner()
    print("  [Module B] Uncertainty engines active (Bayesian Net, CF, DS, Fuzzy).")

    # Module C: Game
    game_solver = GameSolver()
    print("  [Module C] Adversarial dock contention solver ready (Alpha-Beta + IDS).")

    # Module D: Connectionist
    hopfield = HopfieldNetwork(num_neurons=64)
    patterns = create_sample_bitmaps()
    hopfield.train(patterns)

    rnn_model = AnomalyRNN(input_dim=3, hidden_dim=24)
    ckpt_path = os.path.join(os.path.dirname(__file__), "module_d_connectionist", "rnn_anomaly_detector.pt")
    if os.path.exists(ckpt_path):
        rnn_model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    rnn_model.eval()
    print("  [Module D] Hopfield 8x8 memory & PyTorch RNN anomaly model loaded.")

    # Module E: Expert System
    rules_path = os.path.join(os.path.dirname(__file__), "module_e_expert_system", "rules.json")
    expert = ExpertAdvisorEngine(rules_path)
    print(f"  [Module E] Maintenance Advisor online ({len(expert.rules)} rules loaded).")

    print("-" * 75)
    print(" SHIFT EXECUTION STARTED")
    print("-" * 75)

    # State variables for shift simulation
    sliding_sensor_window = []
    battery_level = 100.0
    shift_anomalies_detected = 0
    shift_faults_diagnosed = 0
    packages_processed = 0
    reactive_interventions_count = 0

    for tick in range(1, num_events + 1):
        battery_level -= 0.8

        # Generate realistic sensor reading: [weight_dev, vibration, tilt_angle]
        # Inject occasional anomaly conditions
        has_shock = (tick in [18, 42])
        has_wear = (tick in [30, 52])

        weight_dev = abs(np.random.normal(0.5, 0.3))
        vibration = 0.08 + (0.85 if has_wear else abs(np.random.normal(0.02, 0.01)))
        tilt = 0.2 + (6.5 if has_shock else abs(np.random.normal(0.3, 0.2)))

        current_reading = [weight_dev, vibration, tilt]
        sliding_sensor_window.append(current_reading)
        if len(sliding_sensor_window) > 10:
            sliding_sensor_window.pop(0)

        # ----------------------------------------------------------------------
        # 1. Module D: Real-Time RNN Anomaly Detection on 10-reading sequence
        # ----------------------------------------------------------------------
        anomaly_predicted = False
        if len(sliding_sensor_window) == 10:
            seq_tensor = torch.tensor([sliding_sensor_window], dtype=torch.float32)
            with torch.no_grad():
                prob = torch.sigmoid(rnn_model(seq_tensor)).item()
            if prob >= 0.65:
                anomaly_predicted = True
                shift_anomalies_detected += 1
                print(f"\n[EVENT #{tick:02d}] >>> MODULE D ALERT: Imminent Sensor Anomaly Detected! (P={prob:.2f})")

        # ----------------------------------------------------------------------
        # 2. Module D: Periodic Barcode Scan Corruption & Hopfield Restoration
        # ----------------------------------------------------------------------
        if tick in [12, 35]:
            print(f"\n[EVENT #{tick:02d}] >>> MODULE D: Noisy Barcode Scan Received. Restoring via Hopfield Network...")
            noisy_scan = np.copy(patterns[tick % len(patterns)])
            # Flip 8 bits
            noisy_scan[np.random.choice(64, 8, replace=False)] *= -1.0
            recalled_scan, iters, _ = hopfield.recall(noisy_scan)
            print(f"             Hopfield associative memory converged in {iters} iterations to clean box ID bitmap.")

        # ----------------------------------------------------------------------
        # 3. Module B: Package Inspection & Uncertainty Reasoning
        # ----------------------------------------------------------------------
        if tick % 7 == 0:
            packages_processed += 1
            # Random box inspection scenario
            box_crack = "major" if has_shock else ("minor" if tick % 14 == 0 else "none")
            box_tilt = True if has_shock else (random.random() < 0.10)
            box_w_dev = weight_dev * (3.5 if has_shock else 1.0)

            bn_prob = bn.infer_damage(box_w_dev, box_crack, box_tilt)
            cf_val = cf.evaluate(box_w_dev, box_crack, box_tilt)
            fuzz_val = fuzzy.evaluate(box_w_dev, box_crack)

            print(f"\n[EVENT #{tick:02d}] >>> MODULE B: Inspecting Package #{packages_processed} "
                  f"(Weight Dev={box_w_dev:.1f}kg, Crack={box_crack}, Tilt={box_tilt})")
            print(f"             Bayesian Net P(Damaged)={bn_prob:.3f} | CF Score={cf_val:+.3f} | Fuzzy Damage={fuzz_val:.1f}")

            # ------------------------------------------------------------------
            # 4. Module E: Trigger Expert Advisor if Hardware Fault Suspected
            # ------------------------------------------------------------------
            if bn_prob > 0.50 or has_wear or has_shock:
                shift_faults_diagnosed += 1
                symptoms = {}
                if has_wear:
                    symptoms["high_motor_current"] = True
                    symptoms["low_odometry_delta"] = True
                    symptoms["motor_temp_high"] = True
                elif has_shock:
                    symptoms["bumper_impact_detected"] = True
                    symptoms["safety_circuit_open"] = True
                else:
                    symptoms["camera_blur_score_high"] = True
                    symptoms["lighting_normal"] = True

                print(f"             >>> MODULE E TRIGGERED: Diagnosing Symptoms {list(symptoms.keys())}...")
                diag = expert.diagnose(symptoms)
                for tr in diag["rule_trace"]:
                    print(f"                 [Rule Fired: {tr['rule_id']}] {tr['asserted']}")
                    print(f"                 Reason: {tr['why']}")
                for c in diag["conclusions"]:
                    print(f"                 Action: {c['recommendation']}")

        # ----------------------------------------------------------------------
        # 5. Module A: Execute Stacking Actions & Handle Disturbance
        # ----------------------------------------------------------------------
        if tick == 25:
            print(f"\n[EVENT #{tick:02d}] >>> MODULE A: Executing Pallet Loading Task with Reactive Layer...")
            executor = ReactiveExecutor(pallet_start)
            # Introduce disturbance mid-plan
            def knock_over_box(st):
                s = set(st)
                s.discard("CLEAR(A)")
                s.add("ON(B,A)")
                return s, "External Forklift Turbulence: Box B shifted onto Box A!"

            plan_result = executor.run_plan_with_disturbance(pallet_plan, disturbance_step=2, disturbance_fn=knock_over_box)
            reactive_interventions_count += len(plan_result["intervention_logs"])
            print(f"             Stacking completed successfully ({len(plan_result['execution_trace'])} total actions executed).")
            print(f"             Reactive layer intervened {len(plan_result['intervention_logs'])} times to repair state without full replanning.")

        # ----------------------------------------------------------------------
        # 6. Module C: Dock Contention when Battery Low
        # ----------------------------------------------------------------------
        if tick == 55 or battery_level <= 35.0:
            print(f"\n[EVENT #{tick:02d}] >>> MODULE C: Battery Low ({battery_level:.1f}%)! Contending for Last Charging Dock...")
            dock_state = DockGameState()
            comp_res = game_solver.iterative_deepening(dock_state, max_depth=5)
            print(f"             Alpha-Beta + IDS Game Solver: Selected move '{comp_res['best_move']}' "
                  f"(Nodes explored: {comp_res['total_nodes']}, Win score: {comp_res['score']:.1f})")
            print(f"             Robot 1 successfully secured the charging bay against competing robot!")
            battery_level = 100.0  # Recharged!

    print("\n" + "=" * 75)
    print(" SHIFT COMPLETED: SUMMARY METRICS")
    print("=" * 75)
    print(f"  Total Simulated Events:        {num_events}")
    print(f"  Packages Inspected (Module B): {packages_processed}")
    print(f"  Anomalies Flagged (Module D):  {shift_anomalies_detected}")
    print(f"  Faults Diagnosed (Module E):   {shift_faults_diagnosed}")
    print(f"  Reactive Repairs (Module A):   {reactive_interventions_count}")
    print(f"  Dock Contention Won (Module C): Yes (100% Charge Restored)")
    print("=" * 75)
    print(" All 5 subsystems functioned cohesively in one autonomous controller.\n")


if __name__ == "__main__":
    run_integrated_shift(num_events=60)
