#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ArmanTursun

Learner and QR_control

"""
import numpy as np
# This SV class
# It's an efficient, circular buffer for storing support vectors and their coefficients.
class SV:
    def __init__(self, dimension, budget):
        self.landmarks = np.zeros((budget, dimension), dtype=np.float32)
        self.counter = 0
        self.budget = budget
        self.coeff = np.zeros((budget), dtype=np.float32)
        self.outcomes = np.zeros(budget, dtype=np.float32)
        self.is_full = False # New flag to track if the buffer has wrapped around

    def add_support_vector(self, x, coeff_value, outcome_value):
        """
        Atomically adds a support vector and its coefficient at the current
        counter position, then increments the counter.
        """

        idx_to_write = self.counter
        
        if self.is_full:
            # If memory is full, find the index of the least influential landmark to replace
            idx_to_write = np.argmin(np.abs(self.coeff))
        else:
            # If memory is not full, check if we are about to fill it
            if self.counter == self.budget - 1:
                self.is_full = True
            self.counter += 1
        
        self.landmarks[idx_to_write, :] = x
        self.coeff[idx_to_write] = coeff_value
        self.outcomes[idx_to_write] = outcome_value
    
    def get_active_data(self):
        """ Returns all active data needed for prediction and fallbacks. """
        num_active = self.budget if self.is_full else self.counter
        return {
            "landmarks": self.landmarks[:num_active],
            "coeffs": self.coeff[:num_active],
            "outcomes": self.outcomes[:num_active]
        }

class SimpleGaussianKernel:
    def __init__(self, gamma = 1.0):
        self.gamma = gamma

    def k_vector(self, x, landmarks):
        if landmarks.shape[0] == 0:
            return np.array([])
        dist_sq = np.sum((landmarks - x)**2, axis=1)
        return np.exp(-self.gamma * dist_sq)
    
    def __call__(self, x, landmarks):
        return self.k_vector(x, landmarks)

class MaternKernel:
    """Matérn kernel supporting ν = 0.5, 1.5, 2.5."""
    def __init__(self, length_scale=1.0, nu=2.5):
        assert nu in [0.5, 1.5, 2.5], "Only ν = 0.5, 1.5, 2.5 are supported"
        self.length_scale = length_scale
        self.nu = nu

    def k_vector(self, x, landmarks):
        if landmarks.shape[0] == 0:
            return np.array([])

        # Euclidean distances
        dists = np.sqrt(np.sum((landmarks - x) ** 2, axis=1)) / self.length_scale

        if self.nu == 0.5:
            # Exponential kernel
            return np.exp(-dists)

        elif self.nu == 1.5:
            sqrt3_d = np.sqrt(3) * dists
            return (1 + sqrt3_d) * np.exp(-sqrt3_d)

        elif self.nu == 2.5:
            sqrt5_d = np.sqrt(5) * dists
            return (1 + sqrt5_d + (5 / 3) * dists**2) * np.exp(-sqrt5_d)

    def __call__(self, x, landmarks):
        return self.k_vector(x, landmarks)

# This is combined regressor class
class KernelizedOnlineQuantileRegressor:
    '''
    Merges the SV architecture with the Quantile Regression learning rule.
    '''
    def __init__(self, sv, kernel, quantile=0.95, learning_rate=0.01, gradient_penalty = 10.0):
        self.sv = sv
        self.kernel = kernel
        self.quantile = quantile
        self.learning_rate = learning_rate
        self.gradient_penalty = gradient_penalty

    def _get_prediction_and_uncertainty(self, x):
        # Determine how many support vectors are active

        active_data = self.sv.get_active_data()
        active_landmarks = active_data["landmarks"]
        active_coeffs = active_data["coeffs"]

        if active_landmarks.shape[0] == 0:
            return 0.0, 1.0

        k = self.kernel(x, active_landmarks)
        
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
    
    def update(self, x, y_true, sla_threshold): # , sla_threshold
        """
        This is the new learning rule based on pinball loss.
        """
        # Step 1: Make a prediction with the current model
        prediction = self.predict(x)
        error = y_true - prediction

        if error > 0:
            # Standard under-prediction update
            gradient_update = self.learning_rate * self.quantile
            #if y_true >= sla_threshold and prediction < sla_threshold:
            #    gradient_update *= (self.gradient_penalty / 5) 
        else:
            # Standard over-prediction update
            gradient_update = -self.learning_rate * (1 - self.quantile)
            if y_true < sla_threshold and prediction >= sla_threshold:
                gradient_update *= self.gradient_penalty
        # Step 3: Add the new data point x as a support vector and set its coefficient
        self.sv.add_support_vector(x, gradient_update, y_true)
    

    def add_penalty_update(self, x, penalty_coefficient=-0.05):
        """
        Adds a support vector with a SMALL, fixed negative coefficient to penalize
        an action that was proposed but deemed infeasible.
        """
        # The outcome is not real, so we store a placeholder
        placeholder_outcome = -1.0 
        self.sv.add_support_vector(x, penalty_coefficient, placeholder_outcome)
    
    def set_quantile(self, new_quantile):
        self.quantile = new_quantile