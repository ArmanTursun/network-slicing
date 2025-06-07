#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Arman

This script evaluates the RL baselines algorithms provided by Stable Baselines 
(PPO1, PPO2, TRPO, SAC, A2C, TD3, DDPG) in 3 network-slicing scenarios. 

For each scenario, and each algorithm, the script launches 30 simulation runs.

Each run is divided into a learning phase of 40000 steps and an inference phase of 10000 steps.

The results of the K-th run of algorithm ALG on scenario N, are stored in:

./results/scenario_N/ALG/history_K.npz

"""

import os
from numpy.random import default_rng
from itertools import product
import concurrent.futures as cf
from scenario_creator import create_env
from wrapper import ReportWrapper
from stable_baselines3 import PPO, SAC, A2C, TD3, DDPG
from stable_baselines3.common.env_util import make_vec_env
#from tensorflow import set_random_seed
import torch
import numpy as np
from gen_actions import generate_combinations_sum_le, generate_random_actions
import random

RUNS = 5
PROCESSES = 16 # 30 if enough threads 
TRAIN_STEPS = 4096 #10240 # must be a multiple of 256  #39936
TOTAL_ACTIONS = 100
EVALUATION_STEPS = 0# 10500
CONTROL_STEPS = 60000 # 60000
PENALTY = 1
VERBOSE = True
SLOT_PER_STEP = 100

run_list = list(range(RUNS))
scenarios = [0] # ,1,2

algorithms = {
    #'SAC': SAC,
    'PPO': PPO
    #'A2C': A2C,
    #'TD3': TD3,
    #'DDPG': DDPG
}

deterministic = {
    #'SAC':True,
    'PPO':True
    #'A2C':False,
    #'TD3':False,
    #'DDPG':False
}

scenario_1 = {
    'n_prbs': 100,
    'n_embb': 3,
    'n_mmtc': 0
}

scenario_2 = {
    'n_prbs': 150,
    'n_embb': 3,
    'n_mmtc': 2
}

scenario_3 = {
    'n_prbs': 100,
    'n_embb': 1,
    'n_mmtc': 4
}

scenario_4 = {
    'n_prbs': 70,
    'n_embb': 1,
    'n_mmtc': 1
}

all_scenarios = [scenario_1, scenario_2, scenario_3, scenario_4]

ENV = scenario_1
MINPRB = 10
MULPRB = 2

class RLEvaluator():
    def __init__(self, scenario, algo_name, algorithm):
        self.scenario = scenario
        self.algo_name = algo_name
        self.algorithm = algorithm
        #self.actions = generate_random_actions(n_actions=TOTAL_ACTIONS, n_slices=ENV['n_embb']+ENV['n_mmtc'], total_prbs=ENV['n_prbs'])
        self.actions = generate_combinations_sum_le(n_slices=ENV['n_embb']+ENV['n_mmtc'], total_prbs=ENV['n_prbs'], min_prb_per_slice=MINPRB, multiple_of=MULPRB)
        random.shuffle(self.actions)
        self.t_actions = len(self.actions)
        self.path = './results/scenario_{}/{}/'.format(scenario, algo_name)
        if not os.path.isdir(self.path):
            try:
                os.makedirs(self.path)
            except OSError:
                print ("Creation of the directory %s failed" % self.path)
            else:
                print ("Successfully created the directory %s " % self.path)
        self.model_path = './trained_models/scenario_{}/{}/'.format(scenario, algo_name)
        if not os.path.isdir(self.model_path):
            try:
                os.makedirs(self.model_path)
            except OSError:
                print ("Creation of the directory %s failed" % self.model_path)
            else:
                print ("Successfully created the directory %s " % self.model_path)

    
    def evaluate(self, i):
        print('start evaluation of scenario {} run {} algorithm {}'.format(self.scenario, i, self.algo_name))
        rng = default_rng(seed = i) # environment seed
        #set_random_seed(i) # tensorflow seed
        torch.manual_seed(i)
        node_env = create_env(rng, all_scenarios = all_scenarios, n = self.scenario, slots_per_step = 100, penalty = PENALTY)
        print('environment created')
        node_env = ReportWrapper(node_env, self.actions, steps = TRAIN_STEPS, t_actions=self.t_actions, 
                            control_steps = CONTROL_STEPS, 
                            env_id = i, 
                            path = self.path,
                            verbose = VERBOSE)
        print('wrapped environment created')
        env = make_vec_env(lambda: node_env, n_envs=1)
        print('vectorised environment created')
        model = self.algorithm('MlpPolicy', env, learning_rate=0.0005, ent_coef=0.001, n_steps=8, batch_size=8, verbose=0)
        print('scenario {}: run {} of algorithm {} ... '.format(self.scenario, i, self.algo_name))
        model.learn(total_timesteps = TRAIN_STEPS)
        print('trainning done!')
        node_env.save_results()
        model_path = '{}{}_agent_{}'.format(self.model_path, self.algo_name, i)
        model.save(model_path)
        print('model saved')
        node_env.set_evaluation(EVALUATION_STEPS)
        obs = node_env.obs
        det = deterministic[self.algo_name]
        action, state = model.predict(obs, deterministic = det)
        for i in range(EVALUATION_STEPS):
            action, state = model.predict(obs, state = state, deterministic = det)
            obs, _, _, _, _ = node_env.step(action)
        print('evaluation done')
        node_env.save_results()
        print('results saved')


if __name__=='__main__':
    for scenario, (alg_name, alg) in product(scenarios, algorithms.items()):
        evaluator = RLEvaluator(scenario, alg_name, alg)
        # ################################################################
        # # use this code for sequential execution
        for run in run_list:
            evaluator.evaluate(run)
        # ################################################################

        # ################################################################
        # use this code for parallel execution
        #with cf.ProcessPoolExecutor(PROCESSES) as E:
            #results = E.map(evaluator.evaluate, run_list)
        # ################################################################
