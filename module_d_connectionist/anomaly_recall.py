"""Module D - Connectionist Models: Anomaly Recall
Deliverable: From-scratch Hopfield implementation + PyTorch RNN sequence classifier.

1. Hopfield Network (from scratch): Stores 5 8x8 binary patterns (64-bit).
   Corrupts pattern by 10-15% bit-flips and demonstrates associative recall convergence.
2. PyTorch RNN: Classifies 10-step sensor sequences to predict imminent anomalies.
   Evaluated on a held-out test set.
3. Comparison table comparing Hopfield, RNN, and Symbolic AI.
"""

import sys
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)


# ==============================================================================
# 1. HOPFIELD NETWORK (FROM SCRATCH - NO NEURAL LIBRARY)
# ==============================================================================
class HopfieldNetwork:
    """Discrete Auto-associative Hopfield Network with Hebbian Learning."""
    def __init__(self, num_neurons: int = 64):
        self.num_neurons = num_neurons
        self.weights = np.zeros((num_neurons, num_neurons), dtype=np.float64)

    def train(self, patterns: np.ndarray):
        """Hebbian learning rule: W = (1/N) * sum(x * x^T) with zero diagonal."""
        num_patterns = len(patterns)
        self.weights = np.zeros((self.num_neurons, self.num_neurons), dtype=np.float64)
        for p in patterns:
            p_col = p.reshape(-1, 1)
            self.weights += np.dot(p_col, p_col.T)
        self.weights /= num_patterns
        np.fill_diagonal(self.weights, 0.0)

    def compute_energy(self, state: np.ndarray) -> float:
        """Lyapunov Energy: E = -0.5 * s^T * W * s"""
        return -0.5 * float(np.dot(state.T, np.dot(self.weights, state)))

    def recall(self, corrupted_pattern: np.ndarray, max_iters: int = 20) -> tuple:
        """Asynchronous/Synchronous update until stable attractor state."""
        state = np.copy(corrupted_pattern).astype(np.float64)
        energy_history = [self.compute_energy(state)]

        for it in range(max_iters):
            # Update state: s_new = sgn(W * s)
            activations = np.dot(self.weights, state)
            new_state = np.where(activations >= 0, 1.0, -1.0)

            energy = self.compute_energy(new_state)
            energy_history.append(energy)

            if np.array_equal(new_state, state):
                return new_state, it + 1, energy_history
            state = new_state

        return state, max_iters, energy_history


def create_sample_bitmaps() -> np.ndarray:
    """Create 5 orthogonal/distinct 8x8 label bitmaps (+1: black, -1: white)."""
    # 1. Cross pattern
    cross = np.full((8, 8), -1.0)
    cross[3:5, :] = 1.0
    cross[:, 3:5] = 1.0

    # 2. Border / Box Outline
    border = np.full((8, 8), -1.0)
    border[0, :] = 1.0; border[7, :] = 1.0
    border[:, 0] = 1.0; border[:, 7] = 1.0

    # 3. Diagonal stripes
    diagonal = np.full((8, 8), -1.0)
    for i in range(8):
        diagonal[i, i] = 1.0
        if i + 1 < 8: diagonal[i, i + 1] = 1.0

    # 4. H-pattern
    h_pat = np.full((8, 8), -1.0)
    h_pat[:, 1:3] = 1.0
    h_pat[:, 5:7] = 1.0
    h_pat[3:5, :] = 1.0

    # 5. Checker corners
    corners = np.full((8, 8), -1.0)
    corners[0:3, 0:3] = 1.0
    corners[5:8, 5:8] = 1.0
    corners[0:3, 5:8] = 1.0
    corners[5:8, 0:3] = 1.0

    patterns = np.array([
        cross.flatten(),
        border.flatten(),
        diagonal.flatten(),
        h_pat.flatten(),
        corners.flatten()
    ])
    return patterns


def render_bitmap(vec: np.ndarray) -> str:
    """Renders 64-dim bipolar vector as an 8x8 text grid (ASCII compatible)."""
    mat = vec.reshape(8, 8)
    lines = []
    for r in range(8):
        lines.append("  " + "".join(["##" if val > 0 else ".." for val in mat[r]]))
    return "\n".join(lines)


# ==============================================================================
# 2. PYTORCH RECURRENT NEURAL NETWORK (RNN / LSTM)
# ==============================================================================
class AnomalyRNN(nn.Module):
    """Sequence classifier: reads 10 sequential sensor readings -> predicts imminent anomaly."""
    def __init__(self, input_dim: int = 3, hidden_dim: int = 24, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        # x: [batch, seq_len=10, features=3]
        lstm_out, (hn, cn) = self.lstm(x)
        last_hidden = hn[-1]  # [batch, hidden_dim]
        logits = self.fc(last_hidden).squeeze(-1)
        return logits


def generate_synthetic_stream_data(num_samples: int = 1000):
    """Generates sequences of length 10 representing [weight, vibration, tilt_angle].
    Class 0: Normal sensor operation (stationary noise).
    Class 1: Anomaly imminent (monotonic drift or high-frequency vibration surge).
    """
    X = []
    y = []

    for _ in range(num_samples):
        is_anomaly = random.random() > 0.5
        seq = np.zeros((10, 3), dtype=np.float32)

        # Baseline noise
        seq[:, 0] = np.random.normal(5.0, 0.2, 10)  # Weight nominal 5kg
        seq[:, 1] = np.random.normal(0.1, 0.05, 10) # Vibration nominal
        seq[:, 2] = np.random.normal(0.0, 0.5, 10)  # Tilt nominal 0 deg

        if is_anomaly:
            anomaly_type = random.choice(["vibe_surge", "tilt_drift", "weight_drop"])
            if anomaly_type == "vibe_surge":
                # High vibration spike in last 4 steps
                seq[6:, 1] += np.linspace(0.4, 1.2, 4)
            elif anomaly_type == "tilt_drift":
                # Tilt angle drifting past threshold
                seq[5:, 2] += np.linspace(2.0, 8.0, 5)
            elif anomaly_type == "weight_drop":
                # Sudden weight decrease (box shifting off gripper)
                seq[7:, 0] -= np.linspace(1.0, 3.5, 3)
            y.append(1.0)
        else:
            y.append(0.0)

        X.append(seq)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ==============================================================================
# 3. COMPARISON TABLE: HOPFIELD vs RNN vs SYMBOLIC AI
# ==============================================================================
"""
---------------------------------------------------------------------------------------------
PRACTICAL COMPARISON: HOPFIELD NETWORK vs RECURRENT NEURAL NETWORK vs SYMBOLIC AI
---------------------------------------------------------------------------------------------
Dimension            | Hopfield Network (From Scratch)   | RNN / LSTM (Sequence Model)     | Symbolic AI (Module A & E)
---------------------+-----------------------------------+---------------------------------+---------------------------
Primary Purpose      | Auto-associative memory & content-| Temporal sequence classification| Explicit logical inference
                     | addressable pattern retrieval     | & forward trend forecasting     | & deterministic planning
Representation       | Distributed bipolar state (-1/+1) | Continuous real-valued latent   | Discrete predicates & logic
                     | stored as energy landscape basins | embeddings over time steps      | rules (ON, CLEAR, IF-THEN)
Temporal Dynamics    | Static fixed-point relaxation;    | Dynamic recurrence unfolding    | Discrete state transition
                     | no notion of historical sequence  | across time (t_0 -> t_T)        | graph / plan steps
Fault / Noise Handle | Reconstructs corrupted static     | Predicts future failures from   | Brittle to noise without
                     | bitmaps via energy minimization   | subtle preceding trends         | explicit uncertainty rules
Storage Capacity     | Theoretical limit ~ 0.14*N        | Scales with parameter weights;  | Unbounded rule/fact base;
                     | (~9 patterns for 64 neurons)      | handles long temporal horizons  | search space grows exponentially
Explainability       | Implicit energy Lyapunov function | Black-box hidden representations| Fully transparent trace
                     | (no symbolic explanation)         | (needs attention / saliency)    | ("Rule 14 fired because...")
---------------------------------------------------------------------------------------------
"""


def run_connectionist_module():
    print("=" * 70)
    print(" MODULE D: CONNECTIONIST MODELS - ASSOCIATIVE RECALL & ANOMALY PREDICTION")
    print("=" * 70)

    # 1. Hopfield Network Evaluation
    print("\n" + "-" * 70)
    print("1. FROM-SCRATCH HOPFIELD NETWORK (ASSOCIATIVE RECALL OF 8x8 LABELS)")
    print("-" * 70)

    patterns = create_sample_bitmaps()
    hopfield = HopfieldNetwork(num_neurons=64)
    hopfield.train(patterns)
    print(f"[+] Trained Hopfield Network with {len(patterns)} orthogonal 8x8 patterns (64 neurons).")

    # Corrupt pattern 0 (Cross pattern) by flipping ~15% bits (10 bits out of 64)
    target_idx = 0
    orig_pattern = patterns[target_idx]
    corrupted_pattern = np.copy(orig_pattern)
    flip_indices = np.random.choice(64, size=10, replace=False)  # 15.6% noise
    corrupted_pattern[flip_indices] *= -1.0

    print(f"\n[+] Injecting 15.6% Bit-Flip Noise (10 bits flipped out of 64)...")
    recalled_pattern, iters, energies = hopfield.recall(corrupted_pattern)

    bit_errors_corrupted = np.sum(corrupted_pattern != orig_pattern)
    bit_errors_recalled = np.sum(recalled_pattern != orig_pattern)

    print(f"[+] Hopfield Converged in {iters} iterations.")
    print(f"[+] Energy trajectory: {' -> '.join([f'{e:.1f}' for e in energies])}")
    print(f"[+] Initial Corrupted Bit Errors: {bit_errors_corrupted}/64")
    print(f"[+] Post-Recall Bit Errors:      {bit_errors_recalled}/64 (Recall Success: {bit_errors_recalled == 0})")

    print("\nVisual Demonstration:")
    print("ORIGINAL STORED PATTERN:")
    print(render_bitmap(orig_pattern))
    print("\nCORRUPTED INPUT PATTERN (15% NOISE):")
    print(render_bitmap(corrupted_pattern))
    print("\nASSOCIATIVELY RECALLED ATTRACTOR:")
    print(render_bitmap(recalled_pattern))

    # 2. RNN Anomaly Classifier Evaluation
    print("\n" + "-" * 70)
    print("2. PYTORCH RNN / LSTM SEQUENCE CLASSIFIER FOR IMMINENT ANOMALIES")
    print("-" * 70)

    X, y = generate_synthetic_stream_data(num_samples=1000)
    # 80/20 train/test split
    X_train, X_test = torch.tensor(X[:800]), torch.tensor(X[800:])
    y_train, y_test = torch.tensor(y[:800]), torch.tensor(y[800:])

    model = AnomalyRNN(input_dim=3, hidden_dim=24)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    print(f"[+] Training RNN on 800 sequences of length 10 (3 features: weight, vibration, tilt)...")
    model.train()
    for epoch in range(25):
        optimizer.zero_grad()
        logits = model(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1:02d}/25 - Loss: {loss.item():.4f}")

    # Evaluation on held-out test set
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test)
        test_preds = (torch.sigmoid(test_logits) >= 0.5).float()
        accuracy = (test_preds == y_test).float().mean().item() * 100.0

    print(f"\n[+] Held-Out Test Set Accuracy (200 sequences): {accuracy:.2f}%")

    # 3. Display Comparison Table
    print("\n" + "-" * 70)
    print("3. PARADIGM COMPARISON (HOPFIELD vs RNN vs SYMBOLIC AI)")
    print("-" * 70)
    print("See docstring in module_d_connectionist/anomaly_recall.py for comprehensive matrix.")
    print("Key insight: Hopfield net acts as a static content-addressable memory restoring degraded")
    print("snapshots, while the RNN captures dynamical temporal trends in sensor streams before")
    print("critical failure thresholds are crossed.")
    print("=" * 70)

    # Save model checkpoint for main integration
    ckpt_path = os.path.join(os.path.dirname(__file__), "rnn_anomaly_detector.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"[+] Saved trained model checkpoint to: {ckpt_path}\n")

    return accuracy


if __name__ == "__main__":
    run_connectionist_module()
