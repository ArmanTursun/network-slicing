#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Arman

This script evaluates the RL baselines algorithms PPO_mini and SPPO

"""

# TODO  
# 1. Different parameters: threshold, radius, beta, batch size, init_method, gp_iters
# 3. embb slice with different SLAs
# 4. Add other features to states (MCS, SNR, past throughput, previous PRB, maybe demand or SLA, optionally queue length, harq)

import os
from numpy.random import default_rng
from itertools import product
import concurrent.futures as cf
from scenario_creator import create_env
from wrapper_sppo import ReportWrapper
#from stable_baselines3 import PPO, SAC, A2C, TD3, DDPG
from PPO_Safe import SPPO
from stable_baselines3.common.env_util import make_vec_env
#from tensorflow import set_random_seed
import torch
import numpy as np
from gen_actions import generate_combinations_sum_le
import random


scenario_1 = { 'n_prbs': 80, 'n_embb': 3, 'n_mmtc': 0}
scenario_2 = { 'n_prbs': 150, 'n_embb': 3, 'n_mmtc': 2}
scenario_3 = { 'n_prbs': 100, 'n_embb': 1, 'n_mmtc': 4}
scenario_4 = { 'n_prbs': 70,  'n_embb': 1, 'n_mmtc': 1}
all_scenarios = [scenario_1, scenario_2, scenario_3, scenario_4]

RUNS = 3
PROCESSES = 8 # 30 if enough threads 
TRAIN_STEPS = 1 #10240 # must be a multiple of 256  #39936
CONTROL_STEPS = 60000 # 60000
PENALTY = 10
VERBOSE = True
SLOT_PER_STEP = 100
STEPS_PER_UPDATE = 50
EPOCH = 200
TRAIN_STEPS = STEPS_PER_UPDATE * EPOCH

run_list = list(range(RUNS))
scenarios = [0] # ,1,2
MINPRB = 2
MULPRB = 2

algorithms = {
    #'PPO': SPPO
    #'PPO_mini': PPO_mini
    #'SPPO': SPPO
    'PPO_term': SPPO
}
 
safety_threshold_hs = {0.1: '01'} # , 0.5: '05', -0.1: 'm01', -0.5: 'm05'
neighborhood_radius_vs = {0.1: '01'} # , 0.5: '05', 0.8: '08'
gp_noise_leves = {0.01: '001'} # , 0.05: '005', 0.1: '01'
beta_t_sqrt_vals = {1.0: '10'} # , 1.96:'196', 2.5: '25'

class RLEvaluator():
    def __init__(self, scenario, algo_name, algorithm, safety_params, radius_params, noise_params, beta_params):
        self.scenario = scenario
        self.algo_name = algo_name
        self.algorithm = algorithm

        safety, safety_value = safety_params
        radius, radius_value = radius_params
        noise, noise_value = noise_params
        beta, beta_value = beta_params

        self.config = {
        # Environment specific
            "SPPO": True if self.algo_name == 'SPPO' else False,
            "has_continuous_action_space": False, 
            "total_prbs_env": all_scenarios[self.scenario]['n_prbs'],
            "num_slices_env": all_scenarios[self.scenario]['n_embb']+all_scenarios[self.scenario]['n_mmtc'],
            "max_episode_length_env": 200, 
            # SPPO Safety parameters
            "safety_threshold_h": safety,                 # Tune
            "neighborhood_radius_v": radius,               # Tune
            "beta_t_sqrt_val": beta,                    ## Tune  1.0=68%, 1.645=90%, 1.96=95%, 2.576=99%
            "gp_length_scale": 0.7,       
            "gp_signal_variance": 1.0,    
            "gp_noise_level": noise,                     # How about tune noise level?
            "gp_num_inducing_points": 200,
            "gp_init_num_inducing_points": 200,
            "gp_lr": 0.01,
            "gp_iters": 10,                            
            "gp_init_iters": 500, 
            "inducing_points_init_method": "random_subset",    ## "random_subset" or "kmeans" or provide tensor
            "gp_training_batch_size": 1024,             
            # PPO specific params
            "action_std_init": 0.6, 
            "lr_actor": 0.0005,   
            "lr_critic": 0.001,      
            "gamma_discount": 0.99,           
            "K_epochs": 10,       
            "eps_clip": 0.2,                  
            # Training loop params
            "total_timesteps": TRAIN_STEPS, # epochs * steps_per_epoch
            "steps_per_epoch_for_buffer": STEPS_PER_UPDATE, # steps in each epoch
            "verbose_level": 0  # -1: GP, 0: wrapper, 1: sppo (epoch), 2: sppo (step)
        }

        self.actions = generate_combinations_sum_le(n_slices=all_scenarios[self.scenario]['n_embb']+all_scenarios[self.scenario]['n_mmtc'], 
                                                    total_prbs=all_scenarios[self.scenario]['n_prbs'], min_prb_per_slice=MINPRB, multiple_of=MULPRB)
        #random.shuffle(self.actions)
        self.t_actions = len(self.actions)

        foldername = algo_name + '_' + safety_value + '_' + radius_value + '_' + noise_value + '_' + beta_value
        self.path = './results/scenario_{}/{}/'.format(scenario, foldername)
        if not os.path.isdir(self.path):
            try:
                os.makedirs(self.path)
            except OSError:
                print ("Creation of the directory %s failed" % self.path)
            else:
                print ("Successfully created the directory %s " % self.path)
        self.model_path = './trained_models/scenario_{}/{}/'.format(scenario, foldername)
        if not os.path.isdir(self.model_path):
            try:
                os.makedirs(self.model_path)
            except OSError:
                print ("Creation of the directory %s failed" % self.model_path)
            else:
                print ("Successfully created the directory %s " % self.model_path)

    
    def evaluate(self, i):
        print('start evaluation of scenario {} algorithm {} with params: {}, {}, {}, {}, run {}'.format(self.scenario, self.algo_name, safety, radius, noise, beta, i))
        rng = default_rng(seed = i) # environment seed
        #set_random_seed(i) # tensorflow seed
        torch.manual_seed(i)
        node_env = create_env(rng, all_scenarios = all_scenarios, n = self.scenario, slots_per_step = SLOT_PER_STEP, penalty = PENALTY)
        print('environment created')
        node_env = ReportWrapper(node_env, self.actions, steps = TRAIN_STEPS, t_actions=self.t_actions, 
                            #safe_threshold = self.config["safety_threshold_h"],
                            control_steps = CONTROL_STEPS, 
                            env_id = i, 
                            path = self.path,
                            verbose = self.config["verbose_level"],
                            config_dict = self.config)
        print('wrapped environment created')
        env = make_vec_env(lambda: node_env, n_envs=1)
        print('vectorised environment created')
        print('scenario {}: run {} of algorithm {} ... '.format(self.scenario, i, self.algo_name))
        if self.algo_name == "SPPO" or self.algo_name == 'PPO' or self.algo_name == 'PPO_term':
            model = self.algorithm(
                env=env,
                lr_actor=self.config["lr_actor"],
                lr_critic=self.config["lr_critic"],
                gamma=self.config["gamma_discount"],
                K_epochs=self.config["K_epochs"],
                eps_clip=self.config["eps_clip"],
                has_continuous_action_space=self.config["has_continuous_action_space"],
                action_std_init=self.config["action_std_init"], # Not used by discrete
                # action_std_decay_rate, min_action_std, action_std_decay_freq not needed for discrete
                safety_threshold_h=self.config["safety_threshold_h"],
                neighborhood_radius_v=self.config["neighborhood_radius_v"],
                beta_t_sqrt_val=self.config["beta_t_sqrt_val"],
                gp_length_scale=self.config["gp_length_scale"],
                gp_signal_variance=self.config["gp_signal_variance"],
                gp_noise_level=self.config["gp_noise_level"],
                steps_per_epoch_for_buffer=self.config["steps_per_epoch_for_buffer"],
                verbose=self.config["verbose_level"],
                config_dict=self.config
            )
            model.learn(total_timesteps=self.config["total_timesteps"])
        else:
            model = self.algorithm(env, lr_actor = 0.0003, lr_critic = 0.0005, n_steps=self.config["steps_per_epoch_for_buffer"], 
                                   K_epochs = 10, eps_clip = 0.2, has_continuous_action_space = False, verbose=0)
            model.learn(total_timesteps = TRAIN_STEPS)
        
        print('trainning done!')
        node_env.save_results()
        model_path = '{}{}_agent_{}'.format(self.model_path, self.algo_name, i)
        model.save(model_path)
        print('model saved')


if __name__=='__main__':
    for scenario, (alg_name, alg) in product(scenarios, algorithms.items()):
        for (noise, noise_value) in gp_noise_leves.items():
            for (beta, beta_value) in beta_t_sqrt_vals.items():            
                for (safety, safety_value) in safety_threshold_hs.items():
                    for (radius, radius_value) in neighborhood_radius_vs.items():
                        if radius == 0.5 and beta ==1.96:
                            continue
                        safety_params = (safety, safety_value)
                        radius_params = (radius, radius_value)
                        noise_params = (noise, noise_value)
                        beta_params = (beta, beta_value)
                        evaluator = RLEvaluator(scenario, alg_name, alg, safety_params, radius_params, noise_params, beta_params)
                        # ################################################################
                        # # use this code for sequential execution
                        for run in run_list:
                            evaluator.evaluate(run)
                        # ################################################################

                        # ################################################################
                        # use this code for parallel execution
                        #with cf.ProcessPoolExecutor(PROCESSES) as E:
                        #    results = E.map(evaluator.evaluate, run_list)
                        # ################################################################
