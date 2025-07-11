#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: armantursun

Main

"""

# TODO
# test more advanced method when safe-set empty - self adjust margin/sec prb
# test with more UEs
# test with other params
# adjustment learning - mean field

import os
from numpy import savez
from numpy.random import default_rng
from itertools import product
import concurrent.futures as cf
from scenario_creator import create_env
from qr_scenario_creator import create_qr_agent
import time

scenarios = [0] # ,1,2
quantile_list = [0.05] # , 0.01, 0.1, 0.25   0.05, 0.1, 0.25, 0.5 # , 0.05, 0.1

# We define the SLA constraints for each slice type
# For eMBB, let's assume a max delay constraint
embb_sla = {'threshold': [0.5e6, 0.5e6, 1e6], 'quantile': quantile_list, 'type': 'lower', 'kpi_key': 'l1_info'} # Delay > 3e6

# For mMTC, let's assume a minimum success rate constraint
mmtc_sla = {'threshold': 0.99, 'quantile': quantile_list, 'type': 'lower', 'kpi_key': 'l1_info'} # Success Rate > 99%

scenario_1 = { 'n_prbs': 100, 'n_embb': 3, 'n_mmtc': 0}
scenario_2 = { 'n_prbs': 150, 'n_embb': 3, 'n_mmtc': 2}
scenario_3 = { 'n_prbs': 100, 'n_embb': 1, 'n_mmtc': 4}
scenario_4 = { 'n_prbs': 70,  'n_embb': 1, 'n_mmtc': 1}
all_scenarios = [scenario_1, scenario_2, scenario_3, scenario_4]

# --- Hyperparameters for the Learners ---
# These can be tuned based on experiments
QR_PARAMS = {
    'learning_rate': 0.1,
    'budget': 2000,              # Max number of support vectors to keep in memory
    'gamma': 10,                 # Gamma for the Gaussian Kernel
    'matern_length_scale': 0.1,  # for Matern kernel
    'matern_nu': 1.5,            # nu for Matern kernel, only 0.5, 1.5, and 2.5 available
    'exploration_factor': 0,
    'resource_cost_factor': 3.0,
    'gradient_penalty': 2.0,
    'epsilon': 0.1,
    'k_neighbors': 10,
    'adjustment_penalty': -0.01
}

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

matern_dict = {1.5: '15', 0.5: '05', 2.5: '25'}
explo_dict = {0: '0', 1: '1', 2: '2', 3: '3'}
cost_dict = {0: '0', 0.2: '02', 0.5: '05', 0.8: '08', 1: '1', 2: '2', 3: '3', 5: '5'}
gradient_dict = {1: '1', 2: '2', 5: '5', 10: '10'}
epsilon_dict = {0: '0', 0.05: '005', 0.1: '01', 0.2: '02', 0.3: '03', 0.5: '05', 1: '1'}
run_list = list(range(RUNS))
algo_name = 'QR'
name = algo_name + '_' + matern_dict[QR_PARAMS['matern_nu']]
name = name + '_' + explo_dict[QR_PARAMS['exploration_factor']]
name = name + '_' + cost_dict[QR_PARAMS['resource_cost_factor']]
name = name + '_' + gradient_dict[QR_PARAMS['gradient_penalty']]
name = name + '_' + epsilon_dict[QR_PARAMS['epsilon']]

class Evaluator():
    def __init__(self, scenario, quantile):
        self.scenario = scenario
        self.quantile = quantile
        a = int(quantile*100)
        self.a = a
        self.path = './results/scenario_{}/{}_cost{}/'.format(scenario, name, cost_dict[QR_PARAMS['resource_cost_factor']])
        if not os.path.isdir(self.path):
            try:
                os.makedirs(self.path)
            except OSError:
                print('Creation of the directory {} failed'.format(self.path))
            else:
                print('Successfully created the directory {}'.format(self.path))
    
    def evaluate(self, i):
        seed = int(time.time()+i)
        rng = default_rng(seed = seed)
        node_env = create_env(rng, all_scenarios = all_scenarios, n = self.scenario, slots_per_step = SLOT_PER_STEP, penalty = PENALTY, quantile = int(quantile*100))
        print('test {}, quantile {}, run {}: Environment created!'.format(name, self.a, i))
        qr_agent = create_qr_agent(rng, self.scenario, all_scenarios, quantile = self.quantile, 
                                     embb_sla = embb_sla, mmtc_sla = mmtc_sla, qr_params = QR_PARAMS, slots_per_step = SLOT_PER_STEP )
        print('test {}, quantile {}, run {}: QR agent created'.format(name, self.a, i))
        results = qr_agent.run(node_env, TRAIN_STEPS)
        print('test {}, quantile {}, run {}: QR agent trained'.format(name, self.a, i))
        file_path = '{}results_{}.npz'.format(self.path, i)
        savez(file_path, **results)
        print('test {}, quantile {}, run {}: Results saved!'.format(name, self.a, i))

if __name__=='__main__':
    for scenario, quantile in product(scenarios, quantile_list):
        evaluator = Evaluator(scenario, quantile)
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
