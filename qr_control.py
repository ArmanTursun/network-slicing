#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Your Name Here
"""
import numpy as np
import time


class QR_Learner:
    '''
    Auxiliary data class to hold the elements for a single SLA constraint learner.
    '''
    def __init__(self, algorithm, indexes, initial_action, sla_threshold, constraint_type, kpi_key, kpi_index):
        self.algorithm = algorithm
        self.indexes = indexes
        self.initial_action = initial_action
        self.sla_threshold = sla_threshold
        self.constraint_type = constraint_type # 'upper' (e.g., delay) or 'lower' (e.g., throughput)
        self.kpi_key = kpi_key # Key to find the KPI in the environment's info dictionary
        self.kpi_index = kpi_index

class QR_Control:
    '''
    QR_Control: An agent that uses Quantile Regression to make safe, lightweight decisions
    for resource allocation among RAN slices.
    '''
    def __init__(self, rng, learners, n_prbs, exploration_factor=1.5, resource_cost_factor = 0.5):
        self.rng = rng
        self.learners = learners
        self.n_slices = len(learners)
        self.n_prbs = n_prbs
        self.action = np.array([h.initial_action for h in learners], dtype=np.int16)
        self.adjusted = 0
        self.exploration_factor = exploration_factor
        self.resource_cost_factor = resource_cost_factor
    '''
    def select_action(self, state):
        #Implements the "Threshold Seeker" logic.
        #action = np.zeros(self.n_slices, dtype=np.int16)
        action = self.action
        for i, h in enumerate(self.learners):
            l1_state = state[h.indexes]
            #print('l1_state: ', l1_state)
            # Default to a safe, minimal action if no suitable action is found
            chosen_action = self.rng.integers(0, self.n_prbs) #self.n_prbs
            #chosen_action = 0

            # Perform a linear scan to find the cheapest action that meets the constraint.
            for a in range(self.n_prbs + 1):
                # Normalize action to be on a similar scale as state variables
                x = np.append(l1_state, a / self.n_prbs)
                
                # Get the predicted plausible worst-case outcome
                predicted_quantile = h.algorithm.predict(x)

                # Check if the constraint is met
                if h.constraint_type == 'upper':
                    if predicted_quantile <= h.sla_threshold:
                        chosen_action = a
                        break # Found the cheapest safe action
                elif h.constraint_type == 'lower':
                    if predicted_quantile >= h.sla_threshold:
                        chosen_action = a
                        break # Found the cheapest safe action
            
            action[i] = chosen_action

        #print(predicted_quantile)
        # Adjust actions if total allocation exceeds system capacity
        assigned_prbs = action.sum()
        if assigned_prbs > self.n_prbs:
            self.adjusted = 1
            action = self.adjust_action(action, assigned_prbs)
        else:
            self.adjusted = 0
        
        self.action = action
        return action, self.adjusted
    '''

    # --- FINAL, DEFINITIVE select_action method in QRF_Control class ---

    def select_action(self, state):
        """
        Implements a "Pessimistic Safety, Optimistic Performance" strategy.
        It first identifies all "safe" actions and then uses UCB to select
        the most promising one from that safe set.
        """
        # This example assumes a two-slice scenario where each slice has one
        # safety learner (e.g., for delay) and we want to maximize a general
        # reward function (like throughput, also modeled by a learner).
        # This requires a more structured 'learners' object.
        # For this example, we'll assume a simplified case where one learner
        # handles the hard safety constraint.

        action = np.zeros(self.n_slices, dtype=np.int16)
        action = self.action

        for i, h in enumerate(self.learners):
            l1_state = state[h.indexes]
            
            # --- Stage 1: Identify all "Plausibly Safe" Actions ---
            safe_actions = []
            for a in range(self.n_prbs + 1):
                x = np.append(l1_state, a / self.n_prbs)
                
                # Use the simple 'predict' method which gives the pessimistic quantile prediction
                pessimistic_prediction, uncertainty = h.algorithm.predict_with_uncertainty(x)
                #print(a, pessimistic_prediction, uncertainty, h.sla_threshold)

                is_safe = False
                if h.constraint_type == 'upper':
                    if pessimistic_prediction <= h.sla_threshold:
                        is_safe = True
                elif h.constraint_type == 'lower':
                    if pessimistic_prediction >= h.sla_threshold:
                        is_safe = True
                
                if is_safe:
                    safe_actions.append(a)

            # --- Stage 2: Find the Most Optimistic Action from the Safe Set ---
            best_action_for_slice = 0 # Default to a safe action if the safe_actions list is empty
            best_optimistic_score = -np.inf # We want to maximize our optimistic score
            #print(len(safe_actions))

            if not safe_actions:
                # If no action is deemed safe, default to the safest possible action
                best_action_for_slice = self.rng.integers(0, self.n_prbs) #self.n_prbs
            else:
                # Now, only search within the list of safe actions
                for a in safe_actions:
                    x = np.append(l1_state, a / self.n_prbs)
                    
                    # Get both prediction and uncertainty for the UCB calculation
                    prediction, uncertainty = h.algorithm.predict_with_uncertainty(x)
                    #print(a, prediction, uncertainty)

                    # The goal is always to explore where performance might be highest.
                    # Here, we assume higher quantile prediction is better.
                    # This could be adapted to use a separate performance learner.
                    optimistic_score = prediction + self.exploration_factor * uncertainty

                    resource_penalty = self.resource_cost_factor * (a / self.n_prbs)
                    final_score = optimistic_score - resource_penalty
                    
                    if final_score > best_optimistic_score:
                        best_optimistic_score = final_score
                        best_action_for_slice = a
            
            action[i] = best_action_for_slice

        # Adjust actions if total allocation exceeds system capacity
        assigned_prbs = action.sum()
        if assigned_prbs > self.n_prbs:
            self.adjusted = 1
            action = self.adjust_action(action, assigned_prbs)
        else:
            self.adjusted = 0
        
        self.action = action
        return action, self.adjusted
    
    def adjust_action(self, action, assigned_prbs):
        # A simple proportional adjustment
        if assigned_prbs == 0: return np.zeros_like(action) # Avoid division by zero
        relative_p = action / assigned_prbs
        new_action = np.floor(self.n_prbs * relative_p).astype(np.int16)
        
        # Distribute remainder due to flooring to ensure sum is exactly n_prbs
        remainder = self.n_prbs - new_action.sum()
        for i in range(remainder):
            new_action[i % self.n_slices] += 1
            
        return new_action

    def update_control(self, state, action, info):
        '''Updates each learner with the true KPI outcome from the last step.'''
        #print("\n--- INSIDE update_control ---")
        for i, h in enumerate(self.learners):
            #print(f"--- Updating Learner {i} for Slice {h.kpi_key.split('_')[1]} ---")
            l1_state = state[h.indexes]
            #print('Found state ', state)
            l1_action = action[i]
            #print('Action ', l1_action)
            
            # Get the true KPI value for this learner's constraint from the info dict.
            # This requires the environment to provide these specific keys.
            #print('Found info ', info)
            if h.kpi_key in info:
                #print(f"Found key '{h.kpi_key}'. Value: {info[h.kpi_key]}")
                #print(info)
                #                    l1_info     slice_info
                kpi_values = info[h.kpi_key][h.kpi_index][h.kpi_index]['cbr_th']
                all_kpi_value = []
                for ue in kpi_values.keys():
                    all_kpi_value.append(kpi_values[ue])
                #avg_kpi_value = sum(all_kpi_value) / len(all_kpi_value)
                min_kpi_value = min(all_kpi_value)
                #noise = np.random.uniform(-0.2, 0.2)
                #true_kpi_value = np.clip(true_kpi_value + noise, 0.0, 1.0)
                #print(f"Successfully accessed true_kpi_value: {true_kpi_value}")

                x = np.append(l1_state, l1_action / self.n_prbs)
                
                # Update the quantile regressor model with the true continuous value
                #prediction_before = h.algorithm.predict(x)
                h.algorithm.update(x, min_kpi_value)
                #prediction_after = h.algorithm.predict(x)
                
                #print(f"Prediction before update: {prediction_before:.4f}")
                #print(f"Prediction after update:  {prediction_after:.4f}")
                #print(f"Change in prediction: {prediction_after - prediction_before:.4f}")

 
            else:
                print(f"Warning: KPI key '{h.kpi_key}' not found in environment info dictionary. Learner {i} was not updated.")
        
        #print("--- FINISHED update_control ---\n")

    def run(self, system, steps, learning_time=-1):
        """
        Main loop to run the agent in a gym-like environment.

        Args:
            system: The gym environment.
            steps (int): The total number of steps to run.
            learning_time (int): The number of steps for which learning is active. -1 for continuous learning.

        Returns:
            A dictionary containing the history of various metrics.
        """
        action = self.action

        # Initialize arrays to store historical data
        reward_history = np.zeros(steps, dtype=np.float64)
        violation_history = np.zeros(steps, dtype=np.int16)
        adjusted_actions = np.zeros(steps, dtype=np.int16)
        resources_history = np.zeros(steps, dtype=np.int16)

        # Get initial state from the environment
        state, info = system.reset()

        # Set the learning cutoff point
        if learning_time == -1:
            learning_cutoff = steps
        else:
            learning_cutoff = learning_time
        start = time.perf_counter()
        for i in range(steps):
            
            
            # Agent takes an action in the environment
            new_state, reward, _, _, info = system.step(action)
            
            # The agent learns from the outcome
            if i < learning_cutoff:
                # The key change: passing the full 'info' dictionary
                self.update_control(state, action, info)
            
            end = time.perf_counter()   
            duration_ms = (end - start) * 1000
            print(f"Step: {i:>5}, Action = {action}, adjusted = {self.adjusted}, Reward = {reward:>6.2f}, Total Violations = {info.get('total_violations', 0):>3}, Duration = {duration_ms:>5.1f}ms")
            start = time.perf_counter()
            # The agent selects the next action based on the new state
            action, adjusted = self.select_action(new_state)
            
            # Update the state for the next iteration
            state = new_state

            # Record metrics for this step
            reward_history[i] = reward
            violation_history[i] = info.get('total_violations', 0)
            resources_history[i] = action.sum()
            adjusted_actions[i] = adjusted
            
            
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