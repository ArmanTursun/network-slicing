#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This class generates a wrapper for the slice environment with the OpenAI gym environment

@author: Arman

Classes:

ReportWrapper

"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from itertools import product
import time
from typing import List, Tuple
import random

class ReportWrapper(gym.Wrapper):
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    this environment holds the history of the env variables
    - self.violation_history
    - self.reward_history
    - self.action_history 
    done = True if the number of steps is reached
    """
    def __init__(self, env, actions = None, steps = 2000, t_actions = 1000, #safe_threshold = 0.0,
                 control_steps = 500, env_id = 1, extra_samples = 10, path = './logs/', verbose = False, config_dict = None):
        # Call the parent constructor, so we can access self.env later
        super(ReportWrapper, self).__init__(env)
        self.base_env = env.unwrapped
        self.is_sppo = config_dict.get("SPPO", False)
        self.n_slices = self.base_env.node_b.n_slices_l1
        self.n_prbs = self.base_env.node_b.n_prbs
        self.n_variables = self.base_env.node_b.get_n_variables()
        self.t_actions = t_actions
        #self.safe_threshold = safe_threshold
        #self.action_space = spaces.Box(low=0.0, high = 1.0,
        #                                shape=(self.n_slices,), dtype=np.float64)
        #self.actions = generate_random_actions(n_actions=self.t_actions, n_slices=self.n_slices, total_prbs=self.n_prbs)
        self.actions = actions
        if self.is_sppo:
            self.safe_init_actions = self.select_balanced_high_sum_actions(tolerance = 8, top_k=config_dict["gp_init_num_inducing_points"])
            #self.init_safe_states, self.init_safe_values = self.get_initial_safe_set_H0(self.safe_threshold)
            print("safe action length: ", len(self.safe_init_actions))
        #print("safe states length: ", len(self.init_safe_states))
        self.action_space = spaces.Discrete(self.t_actions, start=0)
        self.observation_space = spaces.Box(low=-1, high=1,
                                            shape=(self.n_variables,), dtype=np.float64)
        self.steps = steps
        self.step_counter = 0
        self.control_steps = control_steps
        self.env_id = env_id
        self.verbose = (verbose == 0)
        self.path = path
        self.file_path = '{}history_{}.npz'.format(path, env_id)
        self.extra_samples = extra_samples # for safety
        self.reset_history()

        print('t_actions = {}'.format(self.t_actions))
        print('n_prbs = {}'.format(self.n_prbs))
        print('n_slices = {}'.format(self.n_slices))
    
    def select_balanced_high_sum_actions(self, tolerance=5, top_k=10):
        balanced_actions = []

        for idx, a in enumerate(self.actions):
            if np.max(a) - np.min(a) <= tolerance:
                total = np.sum(a)
                balanced_actions.append((idx, a, total))

        # Sort by sum in descending order
        #balanced_actions.sort(key=lambda x: x[2], reverse=True)
        random.shuffle(balanced_actions)

        # Return top_k items: (original index, action)
        return [(idx, a) for idx, a, _ in balanced_actions[:top_k]]
    
    def reset_history(self):
        self.violation_history = np.zeros((self.steps), dtype = np.int16)
        self.reward_history = np.zeros((self.steps), dtype = np.float64)
        self.action_history = np.zeros((self.steps), dtype = np.int16)
  
    def reset(self, *, seed=None, options=None):
        """
        Reset the environment (but only when it is created)
        """
        #self.step_counter = 0
        obs, info  = self.env.reset(seed=seed, options=options)
        #obs, info  = self.reset_to_safe_set_H0()
        self.obs = obs
        if self.verbose:
            print('Environment {} RESET'.format(self.env_id))
        return self.obs, info

    def step(self, action):
        """
        :param action: ([float] or int) Action taken by the agent
        :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional informations
        """
        if not (0 <= action < self.t_actions):
            raise ValueError(f"Action index {action} out of bounds for map size {self.t_actions}")
        
        obs, reward, done, truncated, info = self.env.step(self.actions[action])

        if self.is_sppo:
            # Safety value m(s') for the new state (based on new allocations)
            m_s_prime = self._calculate_safety_value(obs)
            info['safety_value_m_s_prime'] =  m_s_prime 

        #for item in obs:
        #print(obs)

        # RL algorithms work better with normalized observations between -1 and 1
        obs = np.clip(obs,-0.5,1.5) 
        obs = obs - 0.5
        self.obs = obs

        # collect historical data
        violations = info['total_violations']

        if self.step_counter < self.steps:
            self.violation_history[self.step_counter] = violations
            self.reward_history[self.step_counter] = reward
            self.action_history[self.step_counter] = self.actions[action].sum()

        # increment counter
        self.step_counter += 1

        if self.step_counter % self.control_steps == 0:
            self.save_results()
        
        if self.verbose:
            #print('Environment {}: {}/{} steps, ues: {}, action: {}, reward: {}, violations: {}'
            #      .format(self.env_id, self.step_counter, self.steps, self.base_env.node_b.ues, self.actions[action], reward, info['total_violations']))
            action_str = ' '.join('{:<3}'.format(a) for a in self.actions[action])
            print('Environment {:<2}: {:>4}/{:<4} steps, UEs: {:>2}, Action: [{}], Reward: {:>4.1f}, Violations: {:<2}'
                    .format(self.env_id, self.step_counter, self.steps, self.base_env.node_b.ues, action_str, reward, info['total_violations']))
        #for key, value in info.items():
        #    print(f"{key}: {value}")
        # return obs, reward, done, info
        return self.obs.copy(), reward, done, truncated, info

    def _calculate_safety_value(self, info: np.ndarray) -> float:
        """ Calculates safety value m(s'). Positive if GBRs met, negative otherwise. """
        min_margin = []
        values_per_slice = self.n_variables // self.n_slices
        for i in range(self.n_slices):
            achieved = info[i * values_per_slice + 1]
            target = info[i * values_per_slice + 0]
            if target > 1e-5:
                margin = (achieved - target) / target
            elif achieved > 1e-5:
                margin = 1.0
            else:
                margin = 0.0
            min_margin.append(margin)
        # Safety is the minimum margin by which GBRs are met (or how badly the worst one is violated)
        # Normalize by GBR to make it relative? Or use raw PRB difference.
        # For example, min satisfaction margin.
        min_safe_margin = np.min(min_margin)
        
        # Scale it to be in a reasonable range, e.g. if GBRs are ~5-10 PRBs
        return float(np.clip(min_safe_margin , -1.0, 1.0)) # Example scaling to get values around -1 to 1

    def get_initial_safe_set_H0(self, safe_threshold) -> Tuple[List[np.ndarray], List[float]]:
        # Provide some known safe initial states and their safety values
        # Example: allocation meets GBRs, demands are low
        states = []
        safeties = []
        for item in self.safe_init_actions:
            idx, _ = item
            state_, _, _, _, info = self.env.step(self.actions[idx])
            safety = self._calculate_safety_value(state_)
            state = np.clip(state_,-0.5,1.5) 
            state = state - 0.5
            if safety >= safe_threshold:
                states.append(state)
                safeties.append(safety)
            #else: safety < 0:
            #    print(state_)

        # Ensure H0 states are different enough for GP
        print("Initialize safe states with length: ", len(states))
        return states, safeties
    
    def reset_to_safe_set_H0(self) -> Tuple[List[np.ndarray], List[float]]:
        # Provide some known safe initial states and their safety values
        # Example: allocation meets GBRs, demands are low
        state, info  = self.env.reset()
        id = np.random.randint(len(self.safe_init_actions))
        idx, _ = self.safe_init_actions[id]
        action = self.actions[idx]
        state, _, _, _, info = self.env.step(action)
        if self.verbose:
            print('Environment {} RESET to SAFE STATE'.format(self.env_id))
        return state, info
    
    def save_results(self):
        np.savez(self.file_path, violation = self.violation_history, 
                                reward = self.reward_history,
                                resources = self.action_history)
    
    def set_evaluation(self, eval_steps, new_path = None, change_name = False):
        self.step_counter = self.steps
        self.steps += eval_steps
        self.violation_history = np.pad(self.violation_history, [(0, eval_steps)])
        self.reward_history = np.pad(self.reward_history, [(0, eval_steps)])
        self.action_history = np.pad(self.action_history, [(0, eval_steps)])
        if new_path:
            self.path = new_path
        if change_name:
            self.file_path = '{}evaluation_{}.npz'.format(self.path, self.env_id)
    
    def close(self):
        print("Environment closed.")