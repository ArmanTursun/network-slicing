# File: qr_util.py
# Contains the core learning algorithm (BQR with SGLD) and the PER buffer.

import numpy as np
import random

# ===================================================================
# 1. PRIORITIZED EXPERIENCE REPLAY (PER) IMPLEMENTATION
# ===================================================================

class SumTree:
    """Helper class for PER data structure."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.data_pointer = 0
        self.n_entries = 0

    def add(self, priority, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_idx, priority)
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, tree_idx, priority):
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        parent_idx = 0
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1
            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                if v <= self.tree[left_child_idx]:
                    parent_idx = left_child_idx
                else:
                    v -= self.tree[left_child_idx]
                    parent_idx = right_child_idx
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_priority(self):
        return self.tree[0]

class PrioritizedReplayBuffer:
    """A fixed-size replay buffer that samples experiences based on their priority."""
    def __init__(self, capacity, prob_alpha=0.6):
        self.tree = SumTree(capacity)
        self.prob_alpha = prob_alpha
        self.epsilon = 0.01

    def add(self, experience, error):
        priority = (np.abs(error) + self.epsilon) ** self.prob_alpha
        self.tree.add(priority, experience)

    def sample(self, batch_size):
        batch, idxs, priorities = [], [], []
        segment = self.tree.total_priority / batch_size
        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get_leaf(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)
        return batch, idxs, np.array(priorities)

    def update_priorities(self, tree_idxs, errors):
        priorities = (np.abs(errors) + self.epsilon) ** self.prob_alpha
        for i, p in zip(tree_idxs, priorities):
            self.tree.update(i, p)

    def __len__(self):
        return self.tree.n_entries

# ===================================================================
# 2. ADVANCED KERNEL AND BAYESIAN LEARNER
# ===================================================================

class MaternKernel:
    """An advanced kernel that makes more realistic smoothness assumptions."""
    def __init__(self, length_scale=1.0, nu=2.5):
        self.length_scale = length_scale
        self.nu = nu # nu=2.5 is a common choice, twice differentiable

    def k_vector(self, x, landmarks):
        if landmarks.shape[0] == 0:
            return np.array([])
        # Matérn formula for nu=2.5
        dist = np.sqrt(np.sum((landmarks - x)**2, axis=1)) / self.length_scale
        term1 = 1 + np.sqrt(5) * dist + 5/3 * dist**2
        term2 = np.exp(-np.sqrt(5) * dist)
        return term1 * term2

class BayesianQuantileRegressor_SGLD:
    """
    The advanced learning algorithm. Uses a kernel method with a Bayesian update rule (SGLD)
    and maintains its own memory of support vectors.
    """
    def __init__(self, input_dim, budget, kernel, quantile=0.95, learning_rate=0.01):
        self.budget = budget
        self.kernel = kernel
        self.quantile = quantile
        self.learning_rate = learning_rate
        
        # Internal memory for support vectors (replaces the separate SV class)
        self.landmarks = np.zeros((budget, input_dim), dtype=np.float32)
        self.alphas = np.zeros(budget, dtype=np.float32)
        self.counter = 0
        self.is_full = False

    def _add_sv(self, x, alpha):
        """
        self.landmarks[self.counter, :] = x
        self.alphas[self.counter] = alpha
        if not self.is_full and self.counter == self.budget - 1:
            self.is_full = True
        self.counter = (self.counter + 1) % self.budget
        """
        if not self.is_full:
            # Buffer is not full, just add to the next available slot
            self.landmarks[self.counter, :] = x
            self.alphas[self.counter] = alpha
            self.counter += 1
            if self.counter == self.budget:
                self.is_full = True
        else:
            # Buffer is full. Find the index of the least important support vector.
            # The importance is the absolute value of its alpha coefficient.
            idx_to_replace = np.argmin(np.abs(self.alphas))
            
            # Replace the least important memory with the new one.
            self.landmarks[idx_to_replace, :] = x
            self.alphas[idx_to_replace] = alpha

    def predict_with_uncertainty(self, x):
        num_svs = self.budget if self.is_full else self.counter
        if num_svs == 0:
            return 0.0, 1.0

        active_landmarks = self.landmarks[:num_svs]
        active_alphas = self.alphas[:num_svs]
        
        k = self.kernel.k_vector(x, active_landmarks)
        prediction = k @ active_alphas
        uncertainty = 1.0 - np.max(k) if k.size > 0 else 1.0
        return prediction, uncertainty

    def update(self, x, y_true):
        """
        Performs a single SGLD update step for one data point.
        Returns the error for PER.
        """
        prediction, _ = self.predict_with_uncertainty(x)
        error = y_true - prediction
        
        # Calculate the gradient of the pinball loss
        if error > 0:
            gradient = -self.quantile
        else:
            gradient = (1 - self.quantile)
        
        # SGLD step: gradient descent + Gaussian noise
        # The noise variance is proportional to the learning rate
        noise = np.random.normal(0, np.sqrt(2 * self.learning_rate))
        
        # The new alpha combines the gradient step and the noise
        update_alpha = -self.learning_rate * gradient + noise
        
        self._add_sv(x, update_alpha)
        return error