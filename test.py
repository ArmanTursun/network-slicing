# File: test_agent.py
# A single, self-contained script to test the agent's internal logic.

import numpy as np

# ===================================================================
# 1. SUPPORTING CLASSES
# ===================================================================

class SV:
    def __init__(self, dimension, budget):
        self.landmarks = np.zeros((budget, dimension), dtype=np.float32)
        self.counter = 0
        self.budget = budget
        self.coeff = np.zeros(budget, dtype=np.float32)
        self.is_full = False

    def add_support_vector(self, x, coeff_value):
        self.landmarks[self.counter, :] = x
        self.coeff[self.counter] = coeff_value
        if not self.is_full and self.counter == self.budget - 1:
            self.is_full = True
        self.counter = (self.counter + 1) % self.budget

class SimpleGaussianKernel:
    def __init__(self, gamma = 1.0):
        self.gamma = gamma

    def k_vector(self, x, landmarks):
        if landmarks.shape[0] == 0:
            return np.array([])
        dist_sq = np.sum((landmarks - x)**2, axis=1)
        return np.exp(-self.gamma * dist_sq)

# ===================================================================
# 2. THE LEARNING ALGORITHM CLASS
# ===================================================================

class KernelizedOnlineQuantileRegressor:
    def __init__(self, sv, kernel, quantile=0.95, learning_rate=0.01):
        self.sv = sv
        self.kernel = kernel
        self.quantile = quantile
        self.learning_rate = learning_rate

    def _get_prediction_and_uncertainty(self, x):
        if self.sv.is_full:
            num_active_svs = self.sv.budget
        else:
            num_active_svs = self.sv.counter
        if num_active_svs == 0: return 0.0, 1.0
        active_landmarks = self.sv.landmarks[:num_active_svs]
        active_coeffs = self.sv.coeff[:num_active_svs]
        k = self.kernel.k_vector(x, active_landmarks)
        prediction = k @ active_coeffs
        uncertainty = 1.0 - np.max(k) if k.size > 0 else 1.0
        return prediction, uncertainty

    def predict(self, x):
        prediction, _ = self._get_prediction_and_uncertainty(x)
        return prediction

    def predict_with_uncertainty(self, x):
        return self._get_prediction_and_uncertainty(x)

    def update(self, x, y_true):
        prediction = self.predict(x)
        error = y_true - prediction
        update_alpha = self.learning_rate * self.quantile if error > 0 else -self.learning_rate * (1 - self.quantile)
        self.sv.add_support_vector(x, update_alpha)

# ===================================================================
# 3. THE MAIN TEST HARNESS
# ===================================================================
if __name__ == '__main__':
    print("--- Running Standalone Agent Test ---")

    # 1. Create a dummy learner instance
    STATE_DIM = 5
    BUDGET = 10
    sv_store = SV(dimension=STATE_DIM + 1, budget=BUDGET)
    kernel = SimpleGaussianKernel(gamma=1.0)
    learner_algorithm = KernelizedOnlineQuantileRegressor(sv=sv_store, kernel=kernel)

    # 2. Simulate one step of the UPDATE process
    print("\n--- Testing the UPDATE method ---")
    dummy_x = np.random.rand(STATE_DIM + 1)
    dummy_y_true = 0.5
    print(f"Calling update() with a single float y_true: {dummy_y_true}")
    try:
        learner_algorithm.update(dummy_x, dummy_y_true)
        print("SUCCESS: update() method completed without error.")
    except Exception as e:
        print(f"FAILURE: update() method crashed. Error: {e}")

    # 3. Simulate one step of the PREDICTION process
    print("\n--- Testing the PREDICT methods ---")
    try:
        # Test the method for the 'update' process
        pred_val = learner_algorithm.predict(dummy_x)
        print(f"SUCCESS: predict() returned a single value of type: {type(pred_val)}")

        # Test the method for the 'select_action' (UCB) process
        pred_val_2, uncertainty_val = learner_algorithm.predict_with_uncertainty(dummy_x)
        print(f"SUCCESS: predict_with_uncertainty() returned two values of types: {type(pred_val_2)}, {type(uncertainty_val)}")
        
        print("\nCONCLUSION: The class logic is internally consistent.")

    except Exception as e:
        print(f"FAILURE: One of the predict methods crashed. Error: {e}")