# File: bqr_control.py
# The main controller that manages the new Bayesian agents.

import numpy as np
import time
from bqr_util import PrioritizedReplayBuffer

class BQR_Learner:
    """Data container holding the smart agent and its SLA metadata."""
    def __init__(self, algorithm, indexes, initial_action, sla_threshold, constraint_type, kpi_key, kpi_index, buffer_capacity):
        self.algorithm = algorithm # Now an instance of BayesianQuantileRegressor_SGLD
        self.indexes = indexes
        self.initial_action = initial_action
        self.sla_threshold = sla_threshold
        self.constraint_type = constraint_type
        self.kpi_key = kpi_key
        self.kpi_index = kpi_index
        self.replay_buffer = PrioritizedReplayBuffer(capacity=buffer_capacity)

class BQR_Control:
    """The high-level controller that manages learning and decision-making."""
    def __init__(self, rng, learners, n_prbs, buffer_capacity, exploration_factor=1.0, resource_cost_factor=0.1):
        self.rng = rng
        self.learners = learners
        self.n_slices = len(learners)
        self.n_prbs = n_prbs
        self.action = np.array([h.initial_action for h in learners], dtype=np.int16)
        self.exploration_factor = exploration_factor
        self.resource_cost_factor = resource_cost_factor
        self.adjusted = 0

    def select_action(self, state):
        # This logic remains the same as our final "Safe but Optimistic" version
        action = np.zeros(self.n_slices, dtype=np.int16)
        for i, h in enumerate(self.learners):
            l1_state = state[h.indexes]
            safe_actions = []
            for a in range(self.n_prbs + 1):
                x = np.append(l1_state, a / self.n_prbs)
                pessimistic_prediction, _ = h.algorithm.predict_with_uncertainty(x)
                is_safe = False
                if h.constraint_type == 'upper' and pessimistic_prediction <= h.sla_threshold:
                    is_safe = True
                elif h.constraint_type == 'lower' and pessimistic_prediction >= h.sla_threshold:
                    is_safe = True
                if is_safe:
                    safe_actions.append(a)
            
            best_action_for_slice = 0
            best_final_score = -np.inf
            if not safe_actions:
                best_action_for_slice = self.rng.integers(0, self.n_prbs)
            else:
                for a in safe_actions:
                    x = np.append(l1_state, a / self.n_prbs)
                    prediction, uncertainty = h.algorithm.predict_with_uncertainty(x)
                    optimistic_score = prediction + self.exploration_factor * uncertainty
                    resource_penalty = self.resource_cost_factor * (a / self.n_prbs)
                    final_score = optimistic_score - resource_penalty
                    if final_score > best_final_score:
                        best_final_score = final_score
                        best_action_for_slice = a
            action[i] = best_action_for_slice
        
        assigned_prbs = action.sum()
        if assigned_prbs > self.n_prbs:
            action = self.adjust_action(action, assigned_prbs)
            self.adjusted = 1
        self.action = action
        return action

    def adjust_action(self, action, assigned_prbs):
        if assigned_prbs == 0: return np.zeros_like(action)
        relative_p = action / assigned_prbs
        new_action = np.floor(self.n_prbs * relative_p).astype(np.int16)
        remainder = self.n_prbs - new_action.sum()
        for i in range(int(remainder)):
            new_action[i % self.n_slices] += 1
        return new_action

    def train_agents(self, batch_size):
        for h in self.learners:
            # Skip training if the learner's buffer is not ready
            if len(h.replay_buffer) < batch_size:
                continue

            # Sample a batch from this learner's specific buffer
            experiences, idxs, _ = h.replay_buffer.sample(batch_size)
            
            errors = []
            # Train the learner on its batch of data
            for experience in experiences:
                # The experience is already learner-specific: (l1_state, l1_action, kpi_value)
                l1_state, l1_action, kpi_value = experience
                
                x = np.append(l1_state, l1_action / self.n_prbs)
                error = h.algorithm.update(x, kpi_value)
                errors.append(np.abs(error))
            
            # Update priorities in the learner's specific buffer
            h.replay_buffer.update_priorities(idxs, np.array(errors))

    def run(self, system, steps, train_batch_size=32):
        action = self.action
        reward_history = np.zeros(steps, dtype=np.float64)
        violation_history = np.zeros(steps, dtype=np.int16)
        adjusted_actions = np.zeros(steps, dtype=np.int16)
        resources_history = np.zeros(steps, dtype=np.int16)
        state, info = system.reset()
        start = time.perf_counter()
        for i in range(steps):
            
            new_state, reward, _, _, info = system.step(action)
            
            # Add the experience to the respective replay buffer for each learner.
            for j, h in enumerate(self.learners):
                # Extract learner-specific state and action
                l1_state = state[h.indexes]
                l1_action = self.action[j] # Use the action that was taken

                # Extract the relevant KPI for this learner from the 'info' dictionary
                kpi_values = info[h.kpi_key][h.kpi_index][h.kpi_index]['cbr_th']
                all_kpi_value = [v for v in kpi_values.values()]
                min_kpi_value = min(all_kpi_value) if all_kpi_value else 0

                # The experience tuple contains only what the learner needs
                experience = (l1_state, l1_action, min_kpi_value)
                
                # Calculate initial priority for the new experience.
                # Here, we use a simple heuristic based on the global reward signal.
                initial_error = np.abs(reward) if reward < 0 else 0.1 
                h.replay_buffer.add(experience, initial_error)
            
            # Trigger a training step
            self.train_agents(train_batch_size)
                
            end = time.perf_counter()   
            duration_ms = (end - start) * 1000
            print(f"Step: {i:>5}, Action = {action}, adjusted = {self.adjusted}, Reward = {reward:>6.2f}, Total Violations = {info.get('total_violations', 0):>3}, Duration = {duration_ms:>5.1f}ms")
            start = time.perf_counter()
            
            # The agent selects the next action based on the new state
            action = self.select_action(new_state)
            
            # Update the state for the next iteration
            state = new_state

            # Record metrics for this step
            reward_history[i] = reward
            violation_history[i] = info.get('total_violations', 0)
            resources_history[i] = action.sum()
            adjusted_actions[i] = self.adjusted
            
            
        # Print a summary of the run
        print('\n--- Run Summary ---')
        print(f'Mean allocated resources = {resources_history.mean():.2f}')
        print(f'Total SLA violations = {violation_history.sum()}')
        print(f'Percentage of adjusted actions = {adjusted_actions.mean() * 100:.2f}%')
        print('-------------------')

        # Compile the output dictionary
        output = {
            'reward': reward_history, 
            'resources': resources_history, 
            'adjusted': adjusted_actions,
            'violation': violation_history
        }

        return output