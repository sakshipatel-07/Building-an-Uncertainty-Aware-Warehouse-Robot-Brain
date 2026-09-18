# Uncertainty-Aware Warehouse Robot Brain (warehouse-brain)

An integrated, multi-paradigm autonomous robot control system combining classical symbolic planning, multi-model uncertainty reasoning, adversarial game tree search, connectionist deep learning, and rule-based expert diagnosis.

---

## Quick Start & Execution Guide

Ensure Python 3.10+ is installed along with required packages:
```bash
pip install numpy scipy scikit-learn matplotlib pgmpy scikit-fuzzy torch
```

To run the entire system or individual modules from CMD:

```bash
# Run Integrated Full Shift Simulation (All 5 Modules)
python main.py

# Module A: Planning & Reactive Execution
python module_a_planning/plan.py module_a_planning/start.json module_a_planning/goal.json

# Module B: Multi-Paradigm Uncertainty Reasoning
python module_b_uncertainty/reasoning.py

# Module C: Adversarial Dock Contention Game & Benchmark Chart
python module_c_game/dock_game.py --depth 5 --mode ai-vs-ai
python module_c_game/generate_chart.py

# Module D: Connectionist Hopfield Recall & RNN Anomaly Prediction
python module_d_connectionist/anomaly_recall.py

# Module E: Maintenance Advisor Expert System
python module_e_expert_system/expert_advisor.py --symptoms high_motor_current low_odometry_delta motor_temp_high
python module_e_expert_system/expert_advisor.py --list-rules
```

---

## Module Demonstrations (Max 5 Sentences Each)

### Module A — Planning: Stack the Pallet (Blocks World)
Module A implements classical Goal Stack Planning, nonlinear Partial Order Planning (POP) via constraint posting, hierarchical task decomposition (`LoadPallet`), and a reactive execution layer. The nonlinear planner posts causal links and resolves threats dynamically, enabling parallel multi-arm execution and avoiding unnecessary unstacking steps seen in linear planners. Hierarchical planning decomposes macro-tasks into verified primitive actions before dispatch. When simulated external vibration knocks a box over mid-execution, the reactive layer senses broken preconditions and synthesizes minimal local repairs without replanning the entire global plan. This demonstrates how symbolic constraint-based reasoning ensures plan validity while reactive recovery guarantees execution robustness in physical environments.

### Module B — Reasoning Under Uncertainty: "Is This Box Damaged?"
Module B compares six distinct uncertainty paradigms—Nonmonotonic Default Logic, Bayes' Theorem, Certainty Factors (MYCIN), Bayesian Networks (`pgmpy`), Dempster-Shafer Theory, and Fuzzy Logic (`scikit-fuzzy`)—on the same ten sensor test cases. Nonmonotonic logic demonstrates default-assumption retraction upon receiving new facts, contrasting sharply with classical monotonic logic. The Bayesian Network models causal dependencies and sensor noise, distinguishing true damage from sensor false positives through exact inference. Dempster-Shafer represents epistemic ignorance through explicit belief-plausibility intervals, while fuzzy logic handles graded linguistic continuous boundaries. The comparative table reveals that probabilistic models maintain conservative posteriors during isolated false alarms where crisp default logic overreacts.

### Module C — Game Playing: Dock Contention
Module C models two autonomous mobile robots competing for the warehouse's sole remaining charging bay as a 4x4 adversarial grid game with a branching factor $\ge 3$. The engine benchmarks Plain MiniMax against Alpha-Beta Pruning and Iterative Deepening Search (IDS) with move ordering across search depths 2 through 6. Logging node counts empirically demonstrates an exponential search-space reduction, achieving a 79.2% node reduction at depth 6 through alpha-beta cutoffs. Move-ordering heuristics prioritize principal variation paths and Manhattan proximity to the dock, enabling deeper horizon lookahead within strict real-time decision windows. The playable script demonstrates optimal adversarial maneuvering, successfully blocking opponent paths and claiming the charger.

### Module D — Connectionist Models: Anomaly Recall
Module D couples a from-scratch auto-associative Hopfield network (Hebbian learning on 8x8 bipolar bitmaps) with a PyTorch LSTM sequence classifier predicting imminent sensor anomalies. The Hopfield network operates as content-addressable memory, reconstructing corrupted barcode scans with up to 15.6% bit-flip noise by descending its Lyapunov energy landscape into stored attractor basins. In contrast, the recurrent network processes sliding temporal windows of multidimensional telemetry, capturing subtle preceding drifts and surges to achieve 100% classification accuracy on a held-out test set. As highlighted in the code comparison table, Hopfield nets excel at static holographic pattern completion without temporal awareness, whereas RNNs learn continuous distributed temporal representations necessary for predictive maintenance.

### Module E — Expert System: The Maintenance Advisor
Module E provides a forward-chaining expert diagnostic engine equipped with 18 domain rules diagnosing robotic hardware faults including wheel slipping, encoder drift, battery degradation, and thermal runaway. The engine incorporates an Explanation Facility that prints the exact firing sequence, premises satisfied, and causal rationale behind every asserted diagnosis. A dedicated Knowledge Acquisition Interface allows technicians to append new rules directly via CLI or JSON without modifying source code, which the engine immediately hot-reloads. Intermediate assertions propagate forward to detect complex compound hazards, such as wheel slipping escalating into severe drivetrain motor thermal runaway. This bridges symbolic explainability with practical warehouse floor maintenance operations.

---

## Complete Project Structure
```
warehouse-brain/
│
├── main.py                             # Integrated autonomous control loop (all 5 modules)
├── README.md                           # Documentation & module summaries
├── run_all.bat                         # Automated batch script to test all modules
│
├── module_a_planning/                  # Module A: Planning & Reactive Layer
│   ├── plan.py                         # CLI: python plan.py start.json goal.json
│   ├── goal_stack_planner.py           # Classic STRIPS goal stack planner
│   ├── nonlinear_planner.py            # Partial order planner with constraint posting
│   ├── hierarchical_planner.py         # HTN LoadPallet operator decomposition
│   ├── reactive_layer.py               # Mid-plan disturbance detection & local repair
│   ├── start.json                      # Pallet initial state predicates
│   └── goal.json                       # Pallet target configuration
│
├── module_b_uncertainty/               # Module B: Uncertainty Reasoning
│   └── reasoning.py                    # All 6 paradigms + 10-reading comparison table
│
├── module_c_game/                      # Module C: Adversarial Dock Contention
│   ├── dock_game.py                    # Playable CLI: python dock_game.py --depth 5
│   ├── game_engine.py                  # 4x4 warehouse grid world state representation
│   ├── minimax.py                      # MiniMax, Alpha-Beta, & Iterative Deepening
│   ├── generate_chart.py               # Benchmark script across depths 2-6
│   └── dock_search_chart.png           # Nodes-explored comparison chart
│
├── module_d_connectionist/             # Module D: Connectionist Models
│   ├── anomaly_recall.py               # From-scratch Hopfield + PyTorch RNN
│   └── rnn_anomaly_detector.pt         # Saved trained PyTorch model weights
│
└── module_e_expert_system/             # Module E: Expert Diagnostic Advisor
    ├── expert_advisor.py               # CLI: python expert_advisor.py --symptoms ...
    └── rules.json                      # 18 domain diagnostic rules & causal explanations
```
