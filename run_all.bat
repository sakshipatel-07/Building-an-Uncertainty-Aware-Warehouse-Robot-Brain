@echo off
echo ======================================================================
echo    RUNNING ALL 5 WAREHOUSE-BRAIN MODULES + INTEGRATED CONTROLLER
echo ======================================================================
echo.

echo [1/6] Running Module A: Planning & Reactive Recovery...
python module_a_planning\plan.py module_a_planning\start.json module_a_planning\goal.json
echo.
pause

echo [2/6] Running Module B: Multi-Paradigm Uncertainty Reasoning...
python module_b_uncertainty\reasoning.py
echo.
pause

echo [3/6] Running Module C: Adversarial Dock Contention Game...
python module_c_game\dock_game.py --depth 5 --mode ai-vs-ai
python module_c_game\generate_chart.py
echo.
pause

echo [4/6] Running Module D: Hopfield Network & RNN Sequence Classifier...
python module_d_connectionist\anomaly_recall.py
echo.
pause

echo [5/6] Running Module E: Expert Diagnostic Advisor...
python module_e_expert_system\expert_advisor.py --symptoms high_motor_current low_odometry_delta motor_temp_high
echo.
pause

echo [6/6] Running Integrated Real-World Shift Simulation (main.py)...
python main.py
echo.

echo ======================================================================
echo    ALL MODULES EXECUTED SUCCESSFULLY!
echo ======================================================================
pause
