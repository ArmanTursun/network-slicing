import numpy as np


# This SV class
# It's an efficient, circular buffer for storing support vectors and their coefficients.
class SV:
    def __init__(self, dimension, budget):
        self.landmarks = np.zeros((budget, dimension), dtype=np.float32)
        self.counter = 0
        self.budget = budget
        self.coeff = np.zeros((budget), dtype=np.float32)
        self.is_full = False # New flag to track if the buffer has wrapped around

    def add_support_vector(self, x, coeff_value):
        """
        Atomically adds a support vector and its coefficient at the current
        counter position, then increments the counter.
        """
        self.landmarks[self.counter, :] = x
        self.coeff[self.counter] = coeff_value
        
        # Check if we are about to wrap around, which means the buffer is now full
        if self.counter == self.budget - 1:
            self.is_full = True
        #print(self.counter)
        self.counter = (self.counter + 1) % self.budget

class SimpleGaussianKernel:
    # This is also from your code, acting as a helper.
    def __init__(self, gamma = 1.0):
        self.gamma = gamma

    def k_vector(self, x, landmarks):
        if landmarks.shape[0] == 0:
            return np.array([])
        dist_sq = np.sum((landmarks - x)**2, axis=1)
        return np.exp(-self.gamma * dist_sq)

# This is combined regressor class
class KernelizedOnlineQuantileRegressor:
    '''
    Merges the SV architecture with the Quantile Regression learning rule.
    '''
    def __init__(self, sv, kernel, quantile=0.95, learning_rate=0.01):
        self.sv = sv
        self.kernel = kernel
        self.quantile = quantile
        self.learning_rate = learning_rate

    def _get_prediction_and_uncertainty(self, x):
        """
        Makes a continuous prediction for the quantile.
        This is the same core logic as your code, but we return the raw score 'f'.
        """
        # Determine how many support vectors are active
        if self.sv.is_full:
            num_active_svs = self.sv.budget
        else:
            num_active_svs = self.sv.counter

        if num_active_svs == 0:
            return 0.0, 1.0
        
        # Get active support vectors and coefficients up to the current counter
        active_landmarks = self.sv.landmarks[:num_active_svs]
        active_coeffs = self.sv.coeff[:num_active_svs]

        k = self.kernel.k_vector(x, active_landmarks)
        
        # This is the predicted quantile value
        prediction = k @ active_coeffs

        # Uncertainty is high if the similarity to all known points is low.
        uncertainty = 1.0 - np.max(k) if k.size > 0 else 1.0

        return prediction, uncertainty

    def predict(self, x):
        """
        Public method for the UPDATE process. Returns ONLY a single float prediction.
        """
        prediction, _ = self._get_prediction_and_uncertainty(x)
        return prediction

    def predict_with_uncertainty(self, x):
        """
        Public method for the UCB CONTROLLER. Returns a tuple (prediction, uncertainty).
        """
        return self._get_prediction_and_uncertainty(x)
    
    def update(self, x, y_true):
        """
        This is the new learning rule based on pinball loss.
        """
        # Step 1: Make a prediction with the current model
        prediction = self.predict(x)
        error = y_true - prediction
        
        # Step 2: Determine the gradient update based on the pinball loss
        # This small value is the new coefficient for our new support vector.
        if error > 0:
            # We under-predicted, apply a large push upwards
            gradient_update = self.learning_rate * self.quantile
        else:
            # We over-predicted, apply a small push downwards
            gradient_update = -self.learning_rate * (1 - self.quantile)
            
        # Step 3: Add the new data point x as a support vector and set its coefficient
        self.sv.add_support_vector(x, gradient_update)
