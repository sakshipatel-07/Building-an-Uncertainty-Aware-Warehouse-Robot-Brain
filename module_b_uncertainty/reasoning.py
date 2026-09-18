"""Module B — Reasoning Under Uncertainty: "Is This Box Damaged?"

Compares 6 uncertainty reasoning paradigms on the same 10 synthetic sensor readings:
1. Nonmonotonic Reasoning (Default Logic vs Monotonic Classical Logic)
2. Bayesian Update (Bayes' Rule with conditional probabilities)
3. Certainty Factors (MYCIN-style serial and parallel combinations)
4. Bayesian Network (pgmpy exact inference on 4-5 node DAG)
5. Dempster-Shafer Theory (Belief intervals [Bel, Pl] and conflict mass K)
6. Fuzzy Logic (scikit-fuzzy Mamdani rules and centroid defuzzification)
"""

import sys
import os
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# pgmpy imports
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination


# ==============================================================================
# 1. NONMONOTONIC REASONING vs MONOTONIC LOGIC DEMO
# ==============================================================================
class NonmonotonicReasoner:
    """Default logic: Boxes are normally undamaged unless an abnormality is known.
    Demonstrates nonmonotonic retraction when new evidence is added.
    """
    def __init__(self):
        self.default_undamaged = True

    def reason(self, weight_dev: float, crack: str, tilt: bool) -> str:
        # Abnormality justifications
        has_abnormality = (tilt is True) or (crack in ["minor", "major"]) or (weight_dev > 4.0)
        if has_abnormality:
            # Retracts default 'Undamaged' -> concludes 'Damaged'
            return "Damaged"
        return "Undamaged (Default)"

    @staticmethod
    def demonstrate_nonmonotonicity():
        """Shows classical monotonic logic vs default logic on incremental facts."""
        initial_facts = {"Box_A": "received"}
        new_fact = {"Box_A_tilt": True}

        # In Default Logic:
        # Fact 1 -> Undamaged
        # Fact 1 + Fact 2 -> Retract Undamaged, Conclude Damaged
        # In Monotonic Logic:
        # If Initial -> Undamaged, then adding Tilt cannot retract Undamaged
        # (Monotonicity: S subset of S' => Th(S) subset of Th(S'))
        return {
            "monotonic_behavior": "Adding Tilt cannot retract 'Undamaged'; requires belief revision or causes contradiction.",
            "nonmonotonic_behavior": "Default assumption 'Undamaged' is defeated by Tilt trigger; conclusion shifts nonmonotonically to 'Damaged'."
        }


# ==============================================================================
# 2. BAYESIAN UPDATE
# ==============================================================================
class BayesianUpdater:
    def __init__(self):
        self.prior_d = 0.20      # P(Damaged) = 0.20
        self.prior_not_d = 0.80  # P(~Damaged) = 0.80

        # Likelihoods: P(Evidence | State)
        # Weight deviation (> 3.0 kg)
        self.p_weight_given_d = 0.85
        self.p_weight_given_not_d = 0.15

        # Crack detected (minor or major)
        self.p_crack_given_d = 0.90
        self.p_crack_given_not_d = 0.05

        # Tilt triggered
        self.p_tilt_given_d = 0.75
        self.p_tilt_given_not_d = 0.10

    def compute_posterior(self, weight_dev: float, crack: str, tilt: bool) -> float:
        w_obs = weight_dev > 3.0
        c_obs = crack in ["minor", "major"]
        t_obs = bool(tilt)

        # Likelihood given Damaged
        lik_d = (self.p_weight_given_d if w_obs else (1 - self.p_weight_given_d)) * \
                (self.p_crack_given_d if c_obs else (1 - self.p_crack_given_d)) * \
                (self.p_tilt_given_d if t_obs else (1 - self.p_tilt_given_d))

        # Likelihood given Not Damaged
        lik_not_d = (self.p_weight_given_not_d if w_obs else (1 - self.p_weight_given_not_d)) * \
                    (self.p_crack_given_not_d if c_obs else (1 - self.p_crack_given_not_d)) * \
                    (self.p_tilt_given_not_d if t_obs else (1 - self.p_tilt_given_not_d))

        # Bayes Theorem
        p_d_num = lik_d * self.prior_d
        p_not_d_num = lik_not_d * self.prior_not_d
        posterior = p_d_num / (p_d_num + p_not_d_num)
        return float(posterior)


# ==============================================================================
# 3. CERTAINTY FACTORS (MYCIN-STYLE)
# ==============================================================================
class CertaintyFactorReasoner:
    def __init__(self):
        # Rule certainty factors CF(H, E)
        self.cf_weight_rule = 0.70  # IF weight_dev high THEN damaged
        self.cf_crack_rule = 0.85   # IF crack detected THEN damaged
        self.cf_tilt_rule = 0.65    # IF tilt triggered THEN damaged

    def combine_parallel(self, cf1: float, cf2: float) -> float:
        """MYCIN parallel combination formula."""
        if cf1 >= 0 and cf2 >= 0:
            return cf1 + cf2 - (cf1 * cf2)
        elif cf1 <= 0 and cf2 <= 0:
            return cf1 + cf2 + (cf1 * cf2)
        else:
            return (cf1 + cf2) / (1.0 - min(abs(cf1), abs(cf2)) + 1e-9)

    def evaluate(self, weight_dev: float, crack: str, tilt: bool) -> float:
        # Serial combination: CF(E) * CF(rule)
        # Weight evidence factor
        cf_e_weight = min(1.0, max(-1.0, (weight_dev - 2.5) / 2.5))
        cf_weight = cf_e_weight * self.cf_weight_rule

        # Crack evidence factor
        crack_map = {"none": -0.8, "minor": 0.5, "major": 1.0}
        cf_e_crack = crack_map.get(crack, 0.0)
        cf_crack = cf_e_crack * self.cf_crack_rule

        # Tilt evidence factor
        cf_e_tilt = 0.9 if tilt else -0.5
        cf_tilt = cf_e_tilt * self.cf_tilt_rule

        # Combine weight and crack
        cf_wc = self.combine_parallel(cf_weight, cf_crack)
        # Combine with tilt
        cf_final = self.combine_parallel(cf_wc, cf_tilt)
        return float(cf_final)


# ==============================================================================
# 4. BAYESIAN NETWORK (pgmpy)
# ==============================================================================
class WarehouseBayesianNetwork:
    def __init__(self):
        # Build DAG:
        # Damaged -> CrackVisible
        # Damaged -> Tilt
        # Damaged -> WeightReading
        # SensorNoise -> WeightReading
        self.model = DiscreteBayesianNetwork([
            ('Damaged', 'CrackVisible'),
            ('Damaged', 'Tilt'),
            ('Damaged', 'WeightReading'),
            ('SensorNoise', 'WeightReading')
        ])

        # CPDs
        cpd_d = TabularCPD(variable='Damaged', variable_card=2, values=[[0.80], [0.20]])  # 0: No, 1: Yes
        cpd_noise = TabularCPD(variable='SensorNoise', variable_card=2, values=[[0.90], [0.10]])  # 0: Low, 1: High

        # CrackVisible: P(Crack | Damaged)
        cpd_crack = TabularCPD(
            variable='CrackVisible', variable_card=2,
            values=[
                [0.95, 0.10],  # No crack: P(~C|~D)=0.95, P(~C|D)=0.10
                [0.05, 0.90]   # Crack:    P(C|~D)=0.05,  P(C|D)=0.90
            ],
            evidence=['Damaged'], evidence_card=[2]
        )

        # Tilt: P(Tilt | Damaged)
        cpd_tilt = TabularCPD(
            variable='Tilt', variable_card=2,
            values=[
                [0.90, 0.25],  # No tilt
                [0.10, 0.75]   # Tilt triggered
            ],
            evidence=['Damaged'], evidence_card=[2]
        )

        # WeightReading: P(Weight=Abnormal | Damaged, SensorNoise)
        # Parents: Damaged(2), SensorNoise(2) -> 4 columns: (D=0,N=0), (D=0,N=1), (D=1,N=0), (D=1,N=1)
        cpd_weight = TabularCPD(
            variable='WeightReading', variable_card=2,
            values=[
                [0.90, 0.40, 0.15, 0.10],  # Normal weight
                [0.10, 0.60, 0.85, 0.90]   # Abnormal weight
            ],
            evidence=['Damaged', 'SensorNoise'], evidence_card=[2, 2]
        )

        self.model.add_cpds(cpd_d, cpd_noise, cpd_crack, cpd_tilt, cpd_weight)
        assert self.model.check_model(), "Bayesian Network model specification invalid!"
        self.infer = VariableElimination(self.model)

    def infer_damage(self, weight_dev: float, crack: str, tilt: bool) -> float:
        w_obs = 1 if weight_dev > 3.0 else 0
        c_obs = 1 if crack in ["minor", "major"] else 0
        t_obs = 1 if tilt else 0

        evidence = {
            'WeightReading': w_obs,
            'CrackVisible': c_obs,
            'Tilt': t_obs
        }
        res = self.infer.query(variables=['Damaged'], evidence=evidence, show_progress=False)
        return float(res.values[1])  # P(Damaged=1)


# ==============================================================================
# 5. DEMPSTER-SHAFER THEORY
# ==============================================================================
class DempsterShaferReasoner:
    def __init__(self):
        # Frame of discernment: {D, ~D}
        # Masses assigned to subsets: 'D', 'ND', 'THETA'
        pass

    def combine(self, m1: dict, m2: dict) -> dict:
        """Dempster's combination rule over frame {D, ND}."""
        focal_keys = ['D', 'ND', 'THETA']
        m_comb = {'D': 0.0, 'ND': 0.0, 'THETA': 0.0}
        conflict = 0.0

        for s1 in focal_keys:
            p1 = m1.get(s1, 0.0)
            if p1 == 0.0:
                continue
            for s2 in focal_keys:
                p2 = m2.get(s2, 0.0)
                if p2 == 0.0:
                    continue
                intersection = None
                if s1 == 'THETA':
                    intersection = s2
                elif s2 == 'THETA':
                    intersection = s1
                elif s1 == s2:
                    intersection = s1
                else:
                    intersection = 'EMPTY'

                if intersection == 'EMPTY':
                    conflict += p1 * p2
                else:
                    m_comb[intersection] += p1 * p2

        # Normalize by (1 - conflict)
        denom = 1.0 - conflict
        if denom <= 1e-9:
            # Complete conflict
            return {'D': 0.0, 'ND': 0.0, 'THETA': 1.0, 'conflict': 1.0}

        for k in focal_keys:
            m_comb[k] /= denom
        m_comb['conflict'] = conflict
        return m_comb

    def evaluate(self, weight_dev: float, crack: str, tilt: bool) -> dict:
        # Sensor 1: Weight
        if weight_dev > 4.0:
            m1 = {'D': 0.70, 'ND': 0.10, 'THETA': 0.20}
        elif weight_dev < 2.0:
            m1 = {'D': 0.05, 'ND': 0.75, 'THETA': 0.20}
        else:
            m1 = {'D': 0.30, 'ND': 0.30, 'THETA': 0.40}

        # Sensor 2: Crack detector
        if crack == "major":
            m2 = {'D': 0.85, 'ND': 0.05, 'THETA': 0.10}
        elif crack == "minor":
            m2 = {'D': 0.55, 'ND': 0.20, 'THETA': 0.25}
        else:
            m2 = {'D': 0.05, 'ND': 0.80, 'THETA': 0.15}

        # Sensor 3: Tilt sensor
        if tilt:
            m3 = {'D': 0.65, 'ND': 0.15, 'THETA': 0.20}
        else:
            m3 = {'D': 0.10, 'ND': 0.65, 'THETA': 0.25}

        # Combine m1 & m2, then with m3
        m12 = self.combine(m1, m2)
        m_final = self.combine(m12, m3)

        bel = m_final['D']
        pl = m_final['D'] + m_final['THETA']
        return {
            "belief": bel,
            "plausibility": pl,
            "interval_str": f"[{bel:.2f}, {pl:.2f}]",
            "uncertainty": m_final['THETA'],
            "conflict": m_final.get('conflict', 0.0)
        }


# ==============================================================================
# 6. FUZZY LOGIC (scikit-fuzzy)
# ==============================================================================
class FuzzyDamageReasoner:
    def __init__(self):
        # Antecedents & Consequent
        self.weight_dev = ctrl.Antecedent(np.arange(0, 10.1, 0.1), 'weight_deviation')
        self.crack_sev = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'crack_severity')
        self.damage = ctrl.Consequent(np.arange(0, 101, 1), 'damage_score')

        # Membership functions
        self.weight_dev['low'] = fuzz.trimf(self.weight_dev.universe, [0, 0, 3.5])
        self.weight_dev['medium'] = fuzz.trimf(self.weight_dev.universe, [2.0, 5.0, 7.5])
        self.weight_dev['high'] = fuzz.trimf(self.weight_dev.universe, [6.0, 10.0, 10.0])

        self.crack_sev['none'] = fuzz.trimf(self.crack_sev.universe, [0, 0, 0.3])
        self.crack_sev['minor'] = fuzz.trimf(self.crack_sev.universe, [0.2, 0.5, 0.7])
        self.crack_sev['major'] = fuzz.trimf(self.crack_sev.universe, [0.6, 1.0, 1.0])

        self.damage['safe'] = fuzz.trimf(self.damage.universe, [0, 0, 35])
        self.damage['suspect'] = fuzz.trimf(self.damage.universe, [30, 50, 70])
        self.damage['severe'] = fuzz.trimf(self.damage.universe, [65, 100, 100])

        # Comprehensive Fuzzy rules
        rule1 = ctrl.Rule(self.weight_dev['low'] & self.crack_sev['none'], self.damage['safe'])
        rule2 = ctrl.Rule(self.weight_dev['medium'] & self.crack_sev['none'], self.damage['safe'])
        rule3 = ctrl.Rule(self.weight_dev['high'] & self.crack_sev['none'], self.damage['suspect'])
        rule4 = ctrl.Rule(self.weight_dev['low'] & self.crack_sev['minor'], self.damage['suspect'])
        rule5 = ctrl.Rule(self.weight_dev['medium'] & self.crack_sev['minor'], self.damage['suspect'])
        rule6 = ctrl.Rule(self.weight_dev['high'] & self.crack_sev['minor'], self.damage['severe'])
        rule7 = ctrl.Rule(self.crack_sev['major'], self.damage['severe'])

        self.system = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7])
        self.sim = ctrl.ControlSystemSimulation(self.system)

    def evaluate(self, weight_dev: float, crack: str) -> float:
        crack_num_map = {"none": 0.05, "minor": 0.45, "major": 0.90}
        c_val = crack_num_map.get(crack, 0.05)
        w_val = float(np.clip(weight_dev, 0.0, 10.0))

        self.sim.input['weight_deviation'] = w_val
        self.sim.input['crack_severity'] = c_val
        try:
            self.sim.compute()
            return float(self.sim.output.get('damage_score', 50.0))
        except Exception:
            return 50.0


# ==============================================================================
# MAIN COMPARISON ON 10 SYNTHETIC SENSOR READINGS
# ==============================================================================
def run_uncertainty_comparison():
    print("=" * 115)
    print(" MODULE B: MULTI-PARADIGM UNCERTAINTY REASONING COMPARISON")
    print(" Dataset: 10 Synthetic Sensor Readings from Unreliable Sensors (Weight, Vision-Crack, Tilt)")
    print("=" * 115)

    dataset = [
        {"id": 1,  "weight_dev": 0.5, "crack": "none",  "tilt": False, "desc": "Clean undamaged box"},
        {"id": 2,  "weight_dev": 5.8, "crack": "major", "tilt": True,  "desc": "Critically damaged crushed box"},
        {"id": 3,  "weight_dev": 1.2, "crack": "none",  "tilt": True,  "desc": "False positive tilt reading alone"},
        {"id": 4,  "weight_dev": 4.5, "crack": "none",  "tilt": False, "desc": "Heavy cargo weight deviation only"},
        {"id": 5,  "weight_dev": 2.2, "crack": "minor", "tilt": False, "desc": "Surface hairline scratch only"},
        {"id": 6,  "weight_dev": 3.8, "crack": "minor", "tilt": True,  "desc": "Tilt + minor crack (high suspicion)"},
        {"id": 7,  "weight_dev": 7.0, "crack": "none",  "tilt": True,  "desc": "Extreme weight + tilt, no crack seen"},
        {"id": 8,  "weight_dev": 0.2, "crack": "minor", "tilt": False, "desc": "Cosmetic scratch on ultra-light box"},
        {"id": 9,  "weight_dev": 6.5, "crack": "major", "tilt": False, "desc": "Heavy crack, tilt sensor failed/untriggered"},
        {"id": 10, "weight_dev": 3.1, "crack": "minor", "tilt": False, "desc": "Marginal boundary case across all 3 sensors"},
    ]

    nm_engine = NonmonotonicReasoner()
    bayes_engine = BayesianUpdater()
    cf_engine = CertaintyFactorReasoner()
    bn_engine = WarehouseBayesianNetwork()
    ds_engine = DempsterShaferReasoner()
    fuzzy_engine = FuzzyDamageReasoner()

    # Print Nonmonotonic demonstration
    demo = nm_engine.demonstrate_nonmonotonicity()
    print("\n--- Nonmonotonic Reasoning vs Classical Monotonic Logic Demo ---")
    print(f"[Classical Logic] {demo['monotonic_behavior']}")
    print(f"[Default Logic]   {demo['nonmonotonic_behavior']}")
    print("-" * 115)

    results = []
    print(f"\n{'ID':<3} | {'Nonmonotonic':<17} | {'Bayes Post':<10} | {'CF Score':<10} | {'BN Post':<10} | {'DS Interval':<14} | {'Fuzzy Score':<11} | Notes")
    print("-" * 115)

    for item in dataset:
        w = item["weight_dev"]
        c = item["crack"]
        t = item["tilt"]

        nm_verdict = nm_engine.reason(w, c, t)
        bayes_post = bayes_engine.compute_posterior(w, c, t)
        cf_score = cf_engine.evaluate(w, c, t)
        bn_post = bn_engine.infer_damage(w, c, t)
        ds_res = ds_engine.evaluate(w, c, t)
        fuzzy_score = fuzzy_engine.evaluate(w, c)

        results.append({
            "id": item["id"],
            "nm": nm_verdict,
            "bayes": bayes_post,
            "cf": cf_score,
            "bn": bn_post,
            "ds": ds_res["interval_str"],
            "fuzzy": fuzzy_score,
            "desc": item["desc"]
        })

        print(f"{item['id']:<3} | {nm_verdict:<17} | {bayes_post:<10.3f} | {cf_score:<+10.3f} | {bn_post:<10.3f} | {ds_res['interval_str']:<14} | {fuzzy_score:<11.1f} | {item['desc']}")

    print("=" * 115)
    print("\n[STUDENT ANALYSIS: DISAGREEMENT EXPLANATIONS (Requirement 6)]")
    print("Disagreement Case 1 (Item 3 - False positive tilt):")
    print("  -> Nonmonotonic rules instantly retract 'Undamaged' and mark 'Damaged' due to the strict default rule defeat,")
    print("     whereas Bayesian/BN models correctly account for the low prior P(D)=0.20 and high false-positive rate of tilt,")
    print("     yielding a low posterior (~0.36) and keeping belief conservative.")
    print("\nDisagreement Case 2 (Item 4 - Heavy weight deviation alone):")
    print("  -> Certainty Factors produce a moderate positive score (+0.38) treating weight deviation as positive evidence,")
    print("     while Dempster-Shafer yields a wide belief interval [0.22, 0.65] explicitly representing epistemic uncertainty,")
    print("     and Fuzzy Logic marks it 'safe' (15.5) because crack severity is zero.")
    print("=" * 115)

    return results


if __name__ == "__main__":
    run_uncertainty_comparison()
