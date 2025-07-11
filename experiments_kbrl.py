#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: juanjosealcaraz

This script evaluates the Kernel Model-Based RL (KBRL) algorithm in 3 network-slicing scenarios. 
For each scenario, and each delta, the script launches 30 simulation runs. Each run lasts 50000 steps.

The results of the K-th run of KBRL using accuracy factor (delta) 0.97, on scenario N, are stored in:

./results/scenario_N/KBRL_97/results_K.npz

"""

import os
from numpy import savez
from numpy.random import default_rng
from itertools import product
import concurrent.futures as cf
from scenario_creator import create_env
from kbrl_scenario_creator import create_kbrl_agent

scenarios = [0] # ,1,2
accuracy_list = [[0.97, 0.99]] # , [0.99, 0.999] 0.97, 

scenario_1 = { 'n_prbs': 100, 'n_embb': 3, 'n_mmtc': 0}
scenario_2 = { 'n_prbs': 150, 'n_embb': 3, 'n_mmtc': 2}
scenario_3 = { 'n_prbs': 100, 'n_embb': 1, 'n_mmtc': 4}
scenario_4 = { 'n_prbs': 70,  'n_embb': 1, 'n_mmtc': 1}
all_scenarios = [scenario_1, scenario_2, scenario_3, scenario_4]

RUNS = 30
PROCESSES = 10 # 30 if enough threads 
TRAIN_STEPS = 1 #10240 # must be a multiple of 256  #39936
CONTROL_STEPS = 60000 # 60000
PENALTY = 10
VERBOSE = True
SLOT_PER_STEP = 100
STEPS_PER_UPDATE = 50
EPOCH = 200
TRAIN_STEPS = STEPS_PER_UPDATE * EPOCH

run_list = list(range(RUNS))
name = 'KBRL'

class Evaluator():
    def __init__(self, scenario, a_range):
        self.scenario = scenario
        self.a_range = a_range
        a = int(a_range[0]*100)
        self.path = './results/scenario_{}/{}_{}/'.format(scenario, name, a)
        if not os.path.isdir(self.path):
            try:
                os.makedirs(self.path)
            except OSError:
                print('Creation of the directory {} failed'.format(self.path))
            else:
                print('Successfully created the directory {}'.format(self.path))
    
    def evaluate(self, i):
        rng = default_rng(seed = i)
        node_env = create_env(rng, all_scenarios = all_scenarios, n = self.scenario, slots_per_step = SLOT_PER_STEP, penalty = PENALTY)
        print('run {}: Environment created!'.format(i))
        kbrl_agent = create_kbrl_agent(rng, self.scenario, accuracy_range = self.a_range)
        print('run {}: KBRL agent created'.format(i))
        results = kbrl_agent.run(node_env, TRAIN_STEPS)
        print('run {}: KBRL agent trained'.format(i))
        file_path = '{}results_{}.npz'.format(self.path, i)
        savez(file_path, **results)
        print('run {}: Results saved!'.format(i))

if __name__=='__main__':
    for scenario, a_range in product(scenarios, accuracy_list):
        evaluator = Evaluator(scenario, a_range)
        # ################################################################
        # # use this code for sequential execution
        #for run in run_list:
        #    evaluator.evaluate(run)
        #     print('run {} finised!'.format(run))   
        # ################################################################

        # ################################################################
        # use this code for parallel execution
        with cf.ProcessPoolExecutor(PROCESSES) as E:
            results = E.map(evaluator.evaluate, run_list)
        # ################################################################
