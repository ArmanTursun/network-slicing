#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ArmanTursun

Learner and QR_control

"""
import numpy as np
import time
from sklearn.preprocessing import StandardScaler
from collections import deque

class QR_Learner:
    '''
    Auxiliary data class to hold the elements for a single SLA constraint learner.
    '''
    def __init__(self, algorithm, indexes, sla_threshold, constraint_type, kpi_key, kpi_index):
        self.algorithm = algorithm
        self.indexes = indexes
        self.sla_threshold = sla_threshold
        self.constraint_type = constraint_type # 'upper' (e.g., delay) or 'lower' (e.g., throughput)
        self.kpi_key = kpi_key # Key to find the KPI in the environment's info dictionary
        self.kpi_index = kpi_index

class QR_Control:
    '''
    QR_Control: An agent that uses Quantile Regression to make safe, lightweight decisions
    for resource allocation among RAN slices.
    '''
    def __init__(self, rng, learners, n_prbs, state_variables_embb, norms, exploration_factor=1.5, 
                 resource_cost_factor = 0.5, epsilon = 0.2, k = 100, adjustment_penalty = -0.05):
        self.rng = rng
        self.learners = learners
        self.n_slices = len(learners)
        self.n_prbs = n_prbs
        self.action = np.array([0 for h in learners], dtype=np.int16)
        self.adjusted = 0
        self.exploration_factor = exploration_factor
        self.resource_cost_factor = resource_cost_factor
        self.len_safe_set = [0 for i in range(self.n_slices)]
        self.state_variables_embb = state_variables_embb
        self.epsilon = epsilon
        self.k = k
        self.norms = norms
        self.current_step = 0
        self.adjustment_penalty = adjustment_penalty
        self.choices = [None for _ in self.learners]
        self.scalers = [StandardScaler() for _ in self.learners]
        self.margin = [self.rng.integers(2,  5) for _ in self.learners]

        self.violation_history = [deque(maxlen=100) for h in learners] # Track the last 100 outcomes
        self.target_sla_success_rate = [0.99 for h in learners] # The business goal (e.g., 99%)
        self.current_quantiles = [h.algorithm.quantile for h in learners]

    
    # --- FINAL, DEFINITIVE select_action method in QRF_Control class ---

    def softmax(self, x, temp=0.1):
        e_x = np.exp((x - np.max(x)) / temp)
        return e_x / e_x.sum()

    def select_action(self, enriched_state_dict):
        """
        Implements a "Pessimistic Safety, Optimistic Performance" strategy.
        It first identifies all "safe" actions and then uses UCB to select
        the most promising one from that safe set.
        """
        intended_action = np.zeros(self.n_slices, dtype=np.int16)
        intended_action_with_margin = np.zeros(self.n_slices, dtype=np.int16)
        final_uncertainties = np.zeros(self.n_slices, dtype=np.float64)
        temp_predictions = np.zeros(self.n_slices, dtype=np.float64)
        temp_thresholds = np.zeros(self.n_slices, dtype=np.float64)
        self.len_safe_set = [0, 0, 0]
        self.choices = [None for _ in self.learners]

        for i, h in enumerate(self.learners):
            l1_state = enriched_state_dict[i]
            
            # --- Stage 1: Identify all "Plausibly Safe" Actions ---
            ue_index = self.state_variables_embb.index('cbr_ue')
            if self.current_step != 0 and l1_state[ue_index] == 0:
                intended_action[i] = 0 #self.n_prbs
                margin_prbs = 2
                intended_action_with_margin[i] = intended_action[i] + margin_prbs
                final_uncertainties[i] = 0
                self.len_safe_set[i] = 0
                self.choices[i] = 'safe'
                continue
            
            all_actions = np.arange(self.n_prbs+1)
            random_scores = np.zeros_like(all_actions, dtype=np.float64)
            random_uncertainties = np.zeros_like(all_actions, dtype=np.float64)
            safe_actions = []
            best_prediction = -np.inf
            best_fallback_action = 0
            best_fallback_uncertainty = 1
            best_safe_score = -np.inf
            best_safe_action = 0
            best_safe_action_uncertainty = 1
            best_action_for_slice = 0
            best_action_uncertainty = 1
            best_action_with_margin_for_slice = 0
            random_scores[0] = -np.inf
            for a in range(1, self.n_prbs + 1):
                x = np.append(l1_state, a / self.n_prbs)
                prediction, uncertainty = h.algorithm.predict_with_uncertainty(x)
                random_scores[a] = prediction - uncertainty
                random_uncertainties[a] = uncertainty
                if prediction > best_prediction: #  - uncertainty
                    best_prediction = prediction #- uncertainty
                    best_fallback_action = a     
                    best_fallback_uncertainty = uncertainty
                    temp_predictions[i] = prediction
                    temp_thresholds[i] = uncertainty
                    
                is_safe = False
                if h.constraint_type == 'lower' and prediction >= h.sla_threshold:
                    is_safe = True
                elif h.constraint_type == 'upper' and prediction <= h.sla_threshold:
                    is_safe = True
                if is_safe:
                    safe_actions.append((a, uncertainty))
                    optimistic_score = prediction + self.exploration_factor * uncertainty
                    resource_penalty = self.resource_cost_factor * (a / self.n_prbs)
                    final_score = optimistic_score - resource_penalty
                    if final_score > best_safe_score:
                        best_safe_score = final_score
                        best_safe_action = a
                        best_safe_action_uncertainty = uncertainty
                    
            # --- Stage 2: Find the Most Optimistic Action from the Safe Set ---
            self.len_safe_set[i] = len(safe_actions)
            if not safe_actions:
                if h.algorithm.sv.counter == 0:# or self.rng.random() < self.epsilon:
                    best_action_for_slice = self.rng.integers(0, best_fallback_action + 1)
                    best_action_uncertainty = 1
                    margin_prbs = int(round(best_action_uncertainty * self.margin[i]))
                    self.choices[i] = 'random'
                elif self.rng.random() < self.epsilon:
                    probabilities = self.softmax(random_scores)
                    best_action_for_slice = self.rng.choice(all_actions, p=probabilities)
                    best_action_uncertainty = 1
                    margin_prbs = int(round(best_action_uncertainty * self.margin[i]))
                    self.choices[i] = 'random'
                else:
                    active_data = h.algorithm.sv.get_active_data()
                    all_landmarks = active_data["landmarks"]
                    landmark_states = all_landmarks[:, :-1]
                    landmark_actions = all_landmarks[:, -1]
                    landmark_outcomes = active_data["outcomes"]
                    
                    non_zero_action_indices = np.where(landmark_actions > 0)[0]

                    if len(non_zero_action_indices) > 0:
                        relevant_states = landmark_states[non_zero_action_indices]
                        relevant_actions = landmark_actions[non_zero_action_indices]
                        relevant_outcomes = landmark_outcomes[non_zero_action_indices]
                        
                        distances = np.linalg.norm(relevant_states - l1_state, axis=1)
                        sorted_indices = np.argsort(distances)
                        num_neighbors = min(self.k, h.algorithm.sv.counter)
                        k_nearest_indices = sorted_indices[:num_neighbors]
                        k_nearest_outcomes = relevant_outcomes[k_nearest_indices]
                        successful_mask = k_nearest_outcomes >= h.sla_threshold

                        if np.any(successful_mask):
                            successful_actions_normalized = relevant_actions[k_nearest_indices][successful_mask]
                            chosen_action_normalized = np.max(successful_actions_normalized)
                            best_action_for_slice = int(round(chosen_action_normalized * self.n_prbs))
                            x = np.append(l1_state, best_action_for_slice / self.n_prbs)
                            _, best_action_uncertainty = h.algorithm.predict_with_uncertainty(x)
                            margin_prbs = int(round(best_action_uncertainty * self.margin[i]))
                            self.choices[i] = 'knn'
                        else:
                            successful_actions_normalized = relevant_actions[k_nearest_indices]
                            chosen_action_normalized = np.max(successful_actions_normalized)
                            best_action_for_slice = int(round(chosen_action_normalized * self.n_prbs))
                            x = np.append(l1_state, best_action_for_slice / self.n_prbs)
                            _, best_action_uncertainty = h.algorithm.predict_with_uncertainty(x)
                            margin_prbs = int(round(best_action_uncertainty * self.margin[i]))
                            self.choices[i] = 'knn'
                    else:
                        probabilities = self.softmax(random_scores)
                        best_action_for_slice = self.rng.choice(all_actions, p=probabilities)
                        best_action_uncertainty = 1
                        margin_prbs = int(round(best_action_uncertainty * self.margin[i]))
                        self.choices[i] = 'random'
                        #print('fallback2')
            else:
                best_action_for_slice = best_safe_action
                best_action_uncertainty = best_safe_action_uncertainty
                margin_prbs = int(round(best_action_uncertainty * self.margin[i]))
                self.choices[i] = 'safe'
            best_action_with_margin_for_slice = best_action_for_slice + margin_prbs
            intended_action_with_margin[i] = best_action_with_margin_for_slice
            intended_action[i] = best_action_for_slice
            final_uncertainties[i] = best_action_uncertainty

        # Adjust actions if total allocation exceeds system capacity
        original_assigned_prbs = intended_action.sum()
        margined_assigned_prbs = intended_action_with_margin.sum()
        final_action = intended_action_with_margin
        if margined_assigned_prbs > self.n_prbs:
            if original_assigned_prbs > self.n_prbs:
                self.adjusted = 1
            else:
                self.adjusted = 0
            final_action = self.adjust_action(intended_action_with_margin, margined_assigned_prbs)
        else:
            self.adjusted = 0
        
        self.action = final_action
        #print(self.choices, temp_predictions, temp_thresholds) # 
        return final_action, intended_action, intended_action_with_margin, final_uncertainties
    
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

    def update_control(self, enriched_state_dict, action, new_state):
        for i, h in enumerate(self.learners):
            state_that_led_to_action = enriched_state_dict[i]
            # skip update when there is no UE
            ue_index = self.state_variables_embb.index('cbr_ue') if 'cbr_ue' in self.state_variables_embb else -1
            if ue_index != -1 and state_that_led_to_action[ue_index] == 0:
                continue

            l1_action = action[i]
            new_l1_state = new_state[h.indexes]
            #state_variables_embb = ['5th_cbr_th', 'cbr_prb' 
            #                        'cbr_queue', 'cbr_snr', 'cbr_ue'] 
            th_index = self.state_variables_embb.index('5th_cbr_th')              
            kpi_value = new_l1_state[th_index]
            x = np.append(state_that_led_to_action, l1_action / self.n_prbs)               
            # Update the quantile regressor model with the true continuous value
            h.algorithm.update(x, kpi_value, h.sla_threshold)
    
    def penalize_original_actions(self, enriched_state_dict, original_action, final_action):
        """
        Applies a DYNAMIC penalty update to any learner whose action was adjusted.
        The penalty is proportional to the size of the adjustment.
        """
        base_penalty_coeff = self.adjustment_penalty

        for i, h in enumerate(self.learners):
            if self.adjusted and original_action[i] > final_action[i]:
                adjustment_delta = original_action[i] #- final_action[i]
                dynamic_penalty = base_penalty_coeff * (adjustment_delta / self.n_prbs)

                l1_state = enriched_state_dict[i]
                original_x = np.append(l1_state, original_action[i] / self.n_prbs)
                h.algorithm.add_penalty_update(original_x, penalty_coefficient=dynamic_penalty)
    
    def get_global_state(self, info):
        # get global info
        global_state = np.zeros(2, dtype=np.float64)
        #global_state[0] = np.mean(info['n_prbs']) / self.n_prbs
        #global_state[1] = np.mean(info['ues']) / self.norms['cbr_ue']
        return global_state

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
        
        # Initialize arrays to store historical data
        reward_history = np.zeros(steps, dtype=np.float64)
        violation_history = np.zeros(steps, dtype=object)
        adjusted_actions = np.zeros(steps, dtype=np.int16)
        resources_history = np.zeros(steps, dtype=np.int16)
        ue_history = np.zeros(steps, dtype=object)
        safe_history = np.zeros(steps, dtype=object)
        quantile_history = np.zeros(steps, dtype=object)

        uncertainty_threshold = 0.90
        min_quantile = 0.01          # Floor for pessimism
        max_quantile = 0.05
        min_margin = 2
        max_margin = 10
        self.target_sla_success_rate = 0.99

        cum_violation = 0
        cum_adjusted = 0
        cum_reward = 0
        cum_empty_safe = 0

        # Get initial state from the environment
        state, info = system.reset()
        global_features_prev_step = self.get_global_state(info)
        
        # Set the learning cutoff point
        if learning_time == -1:
            learning_cutoff = steps
        else:
            learning_cutoff = learning_time
        
        action_choices = {0: {'knn': 0, 'safe': 0, 'random': 0, 'fallback': 0}, 1: {'knn': 0, 'safe': 0, 'random': 0, 'fallback': 0}, 2: {'knn': 0, 'safe': 0, 'random': 0, 'fallback': 0}}
        violation_per_choice = {0: {'knn': 0, 'safe': 0, 'random': 0, 'fallback': 0}, 1: {'knn': 0, 'safe': 0, 'random': 0, 'fallback': 0}, 2: {'knn': 0, 'safe': 0, 'random': 0, 'fallback': 0}}

        for i in range(steps):
            self.current_step = i
            start = time.perf_counter()

            state_for_decision = state
            global_features_for_decision = global_features_prev_step

            enriched_state_dict = {}
            raw_local_states_for_update = {} 

            for j, h in enumerate(self.learners):
                raw_local_state = state[h.indexes]
                enriched_state_dict[j] = np.append(raw_local_state, global_features_for_decision)

            final_action, original_action, intended_action, final_uncertainty = self.select_action(enriched_state_dict)
            new_state, reward, _, _, info = system.step(final_action)
            cur_violations = info.get('violations')

            # Update all learners with the new quantile for the next decision
            for j, h in enumerate(self.learners):
                self.violation_history[j].append(cur_violations[j])
                if len(self.violation_history[j]) == self.violation_history[j].maxlen and (i+ 1) % 1 == 0:
                    recent_success_rate = 1.0 - (sum(self.violation_history[j]) / self.violation_history[j].maxlen)
                    # PRIORITY 1: High Uncertainty Override
                    if final_uncertainty[j] > uncertainty_threshold:
                        self.current_quantiles[j] -= 0.01 
                    # PRIORITY 2: Poor Performance
                    #elif recent_success_rate < self.target_sla_success_rate:
                    elif info.get('violations')[j] > 0:
                        self.current_quantiles[j] -= 0.005
                        self.margin[j] += 2
                        #print(self.margin)
                    # PRIORITY 3: Excellent Performance
                    elif recent_success_rate >= 0.999: # Almost perfect
                        self.current_quantiles[j] += 0.005
                        self.margin[j] -= 1
                    # Enforce the bounds on the quantile
                    self.current_quantiles[j] = np.clip(self.current_quantiles[j], min_quantile, max_quantile)
                    self.margin[j] = np.clip(self.margin[j], min_margin, max_margin)
                h.algorithm.set_quantile(self.current_quantiles[j])
            if i < learning_cutoff:
                # The key change: passing the full 'info' dictionary
                self.update_control(enriched_state_dict, final_action, new_state)
                # Penalize if an adjustment occurred
                if self.adjusted == 1:
                    self.penalize_original_actions(enriched_state_dict, original_action, final_action)
            
            num_ue = info['ues']
            cum_violation += info.get('total_violations', 0)
            cum_adjusted += self.adjusted
            cum_reward += reward
            for slice in range(self.n_slices):
                if num_ue[slice] != 0 and self.len_safe_set[slice] == 0:
                    cum_empty_safe += 1
            
            for k, item in enumerate(self.choices):
                action_choices[k][item] += 1
                if info.get('violations')[k] > 0:
                    violation_per_choice[k][item] += 1

            end = time.perf_counter()   
            duration_ms = (end - start) * 1000
            action_str = ' '.join('{:>2}'.format(a) for a in final_action)
            safe_action_str = ' '.join('{:>2}'.format(a) for a in self.len_safe_set)
            margin_str = ' '.join('{:>2}'.format(a) for a in self.margin)
            ue_str = ' '.join('{:>1}'.format(a) for a in num_ue)
            choice_str = ' '.join('{:>8}'.format(a) for a in self.choices)
            quantile_str = ' '.join('{:>5.4}'.format(a) for a in self.current_quantiles)
            #print(f"Step: {i+1:>5}, UE: {ue_str}, Margins: {margin_str}, Choices: {choice_str}, SafeActions: {safe_action_str}, Action: {action_str}, adjusted = {self.adjusted:>1}, Reward = {reward:>6.2f}, Total Violations = {info.get('total_violations', 0):>3}, Duration = {duration_ms:>5.1f}ms")
            if (i+1) % 1000 == 0:
                print(f"Step: {i+1:>5}, UE: {ue_str}, Margins: {margin_str}, Quantiles: {quantile_str}, adjusted = {cum_adjusted:>3}, Reward = {cum_reward:>6.1f}, Total Violations = {cum_violation:>4}") # , Empty Safe Set: {cum_empty_safe:>4}
                #for slice in range(self.n_slices):
                #    print_str = "Slice " + str(slice) + ": "
                #    for key in action_choices[slice].keys():
                #        temp = round(violation_per_choice[slice][key] / action_choices[slice][key], 2) if action_choices[slice][key] > 0 else 0
                #        print_str += key + "(" + str(violation_per_choice[slice][key]) + "/" + str(action_choices[slice][key]) + "=" + str(temp) + "), "
                #    print(print_str)
                #print()
            start = time.perf_counter()
            
            # Record metrics for this step
            reward_history[i] = reward
            violation_history[i] = info.get('violations')
            resources_history[i] = final_action.sum()
            adjusted_actions[i] = self.adjusted
            ue_history[i] = num_ue
            safe_history[i] = self.len_safe_set
            quantile_history[i] = self.current_quantiles
            
            state = new_state
            global_features_prev_step = self.get_global_state(info)
           
        # Print a summary of the run
        print('\n--- Run Summary ---')
        print(f'Action Choices = {action_choices}')
        print(f'Violation per Choices = {violation_per_choice}')
        print(f'Mean allocated resources = {resources_history.mean():.2f}')
        print(f'Total SLA violations = {cum_violation}')
        print(f'Percentage of adjusted actions = {adjusted_actions.mean() * 100:.2f}%')
        print('-------------------')

        # Compile the output dictionary
        output = {
            'reward': reward_history, 
            'resources': resources_history, 
            'adjusted': adjusted_actions,
            'violation': violation_history,
            'ue': ue_history,
            'safe set': safe_history,
            'quantile': quantile_history
        }

        return output