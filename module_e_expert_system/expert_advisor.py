"""Module E — Expert System: The Maintenance Advisor
Deliverable: python expert_advisor.py --symptoms <flags...>

Features:
1. Forward-chaining rule engine with ~18 domain diagnostic rules
2. Explanation Facility: prints fired rule trace, premise facts, and causal justifications
3. Knowledge Acquisition Interface: technician can add/append new rules to rules.json without code modification
"""

import sys
import os
import json
import argparse
from typing import Dict, Any, List, Set, Tuple


class ExpertAdvisorEngine:
    def __init__(self, rules_file: str):
        self.rules_file = rules_file
        self.rules = []
        self.load_rules()

    def load_rules(self):
        """Loads or reloads rules from rules.json."""
        if not os.path.exists(self.rules_file):
            raise FileNotFoundError(f"Rules file not found at: {self.rules_file}")
        with open(self.rules_file, "r", encoding="utf-8") as f:
            self.rules = json.load(f)

    def save_new_rule(self, rule_id: str, name: str, conditions: Dict[str, bool], conclusion: str, recommendation: str, explanation: str):
        """Knowledge Acquisition Interface: appends new rule to JSON and reloads."""
        new_entry = {
            "rule_id": rule_id,
            "name": name,
            "conditions": conditions,
            "conclusion": conclusion,
            "recommendation": recommendation,
            "explanation": explanation
        }
        self.rules.append(new_entry)
        with open(self.rules_file, "w", encoding="utf-8") as f:
            json.dump(self.rules, f, indent=2)
        self.load_rules()
        print(f"[+] Knowledge Base Updated: Rule {rule_id} ('{name}') appended successfully.")

    def diagnose(self, input_facts: Dict[str, bool]) -> Dict[str, Any]:
        """Forward-chaining inference engine with comprehensive rule explanation trace."""
        working_memory = dict(input_facts)
        fired_rules = []
        conclusions = []
        rule_trace = []

        changed = True
        step = 0

        while changed:
            changed = False
            step += 1
            for r in self.rules:
                rid = r["rule_id"]
                if rid in fired_rules:
                    continue

                # Check if all conditions are satisfied in working memory
                match = True
                matched_premises = {}
                for cond_key, expected_val in r["conditions"].items():
                    if working_memory.get(cond_key) != expected_val:
                        match = False
                        break
                    matched_premises[cond_key] = expected_val

                if match:
                    fired_rules.append(rid)
                    conclusion = r["conclusion"]
                    # Assert conclusion into working memory to allow forward chaining
                    working_memory[conclusion] = True
                    conclusions.append({
                        "rule_id": rid,
                        "rule_name": r["name"],
                        "diagnosis": conclusion,
                        "recommendation": r.get("recommendation", ""),
                        "explanation": r["explanation"],
                        "premises": matched_premises
                    })
                    rule_trace.append({
                        "step": step,
                        "rule_id": rid,
                        "rule_name": r["name"],
                        "premises": matched_premises,
                        "asserted": conclusion,
                        "why": r["explanation"]
                    })
                    changed = True

        return {
            "input_symptoms": input_facts,
            "conclusions": conclusions,
            "rule_trace": rule_trace,
            "fired_count": len(fired_rules)
        }


def format_diagnosis_report(diag_result: Dict[str, Any]) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append(" MODULE E: ROBOT MAINTENANCE ADVISOR (EXPERT SYSTEM)")
    lines.append("=" * 70)
    lines.append(f"\n[+] Input Symptoms ({len(diag_result['input_symptoms'])} flags asserted):")
    for k, v in diag_result["input_symptoms"].items():
        lines.append(f"    - {k}: {v}")

    lines.append(f"\n[+] Inference Engine Fired {diag_result['fired_count']} Rule(s).")
    lines.append("\n" + "-" * 70)
    lines.append(" EXPLANATION FACILITY: DETAILED 'WHY' RULE TRACE")
    lines.append("-" * 70)

    if not diag_result["rule_trace"]:
        lines.append("  No fault rules fired. All telemetry within nominal parameters.")
    else:
        for idx, tr in enumerate(diag_result["rule_trace"], 1):
            lines.append(f"\nStep {tr['step']} | Rule Fired: [{tr['rule_id']}] {tr['rule_name']}")
            lines.append(f"  Matched Premises: {tr['premises']}")
            lines.append(f"  Asserted Fact:    {tr['asserted']}")
            lines.append(f"  Causal Reason:    {tr['why']}")

    lines.append("\n" + "-" * 70)
    lines.append(" FINAL DIAGNOSES & MAINTENANCE ACTIONS")
    lines.append("-" * 70)

    if not diag_result["conclusions"]:
        lines.append("  Status: SYSTEM HEALTHY - NO INTERVENTION REQUIRED")
    else:
        for idx, c in enumerate(diag_result["conclusions"], 1):
            lines.append(f"  {idx}. {c['diagnosis']}")
            lines.append(f"     Action: {c['recommendation']}")

    lines.append("=" * 70)
    return "\n".join(lines)


def interactive_add_rule(engine: ExpertAdvisorEngine):
    print("\n--- KNOWLEDGE ACQUISITION INTERFACE ---")
    rule_id = input("Enter Rule ID (e.g., R19): ").strip()
    name = input("Enter Rule Name: ").strip()
    print("Enter conditions as key=value pairs separated by commas (e.g., motor_temp_high=True, coolant_leak=True):")
    cond_str = input("> ").strip()
    conditions = {}
    for part in cond_str.split(","):
        if "=" in part:
            k, v = part.split("=")
            conditions[k.strip()] = (v.strip().lower() in ["true", "1", "yes"])

    conclusion = input("Enter Conclusion / Fault diagnosis: ").strip()
    recommendation = input("Enter Technician Recommendation: ").strip()
    explanation = input("Enter 'Why' Causal Explanation: ").strip()

    engine.save_new_rule(rule_id, name, conditions, conclusion, recommendation, explanation)


def main():
    parser = argparse.ArgumentParser(description="Warehouse Robot Maintenance Expert Advisor")
    parser.add_argument("--symptoms", nargs="*", default=None,
                        help="List of symptom flags (e.g. --symptoms high_motor_current low_odometry_delta)")
    parser.add_argument("--add-rule", action="store_true",
                        help="Launch interactive knowledge acquisition interface to add a new rule")
    parser.add_argument("--list-rules", action="store_true",
                        help="List all rules in the knowledge base")
    args = parser.parse_args()

    base_dir = os.path.dirname(__file__)
    rules_file = os.path.join(base_dir, "rules.json")
    engine = ExpertAdvisorEngine(rules_file)

    if args.list_rules:
        print(f"\nKnowledge Base: {len(engine.rules)} Rules Loaded from {rules_file}:")
        for r in engine.rules:
            print(f"  [{r['rule_id']}] {r['name']}: {r['conditions']} -> {r['conclusion']}")
        return

    if args.add_rule:
        interactive_add_rule(engine)
        return

    # Determine symptoms
    if args.symptoms:
        symptoms = {sym: True for sym in args.symptoms}
    else:
        # Default test scenario: High motor current + low odometry delta + motor_temp_high
        # Demonstrates forward-chaining to both R01 and the compound R17 rule!
        symptoms = {
            "high_motor_current": True,
            "low_odometry_delta": True,
            "motor_temp_high": True
        }

    diag = engine.diagnose(symptoms)
    print(format_diagnosis_report(diag))


if __name__ == "__main__":
    main()
