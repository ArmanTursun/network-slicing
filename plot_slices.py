#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 5, June 2025

@author: Arman

"""
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import sys
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

import matplotlib.cm as cm


# trainning results
WINDOW = 100 #400
START = 0
END =  10000 # up to 39900  20000
SAFESET = True
#algo_names = ['A2C', 'PPO1', 'PPO2', 'TRPO', 'SAC', 'TD3', 'NAF', 'KBRL_97','KBRL_99']
#labels = ['A2C', 'PPO1', 'PPO2', 'TRPO', 'SAC', 'TD3', 'NAF', 'KBRL 0.97', 'KBRL 0.99']
algo_names = ['PPO'] # , 'PPO', 'SPPO'
labels = ['PPO'] # , 'PPO', 'SPPO'

# filename: threshold_radius_beta_batchsize_initmethod_gpiters

SPAN = END - START

prbs_values = [100, 150, 100, 70] #
scenarios = [0,1,2,3] # 

def movingaverage(values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma

def average_per_window(values, window):
    values = np.array(values)
    n = len(values)
    trimmed_length = (n // window) * window  # Drop extra elements
    reshaped = values[:trimmed_length].reshape(-1, window)
    return reshaped.mean(axis=1)

def sum_per_window(values, window):
    values = np.array(values)
    n = len(values)
    trimmed_length = (n // window) * window  # Drop extra elements
    reshaped = values[:trimmed_length].reshape(-1, window)
    return reshaped.sum(axis=1)

def get_folder_names(path):
    return [name for name in os.listdir(path)
            if os.path.isdir(os.path.join(path, name))]

safety_threshold_hs = {0.1: '01', 0.5: '05', -0.1: 'm01', -0.5: 'm05'} # 0.1: '01', 0.5: '05', -0.1: 'm01', -0.5: 'm05'
neighborhood_radius_vs = {0.1: '01', 0.5: '05', 0.8: '08'} # 0.1: '01', 0.5: '05', 0.8: '08'
gp_noise_leves = {0.01: '001'} # 0.01: '001', 0.05: '005', 0.1: '01'
beta_t_sqrt_vals = {1.0: '10'} # 1.0: '10', 1.96:'196', 2.5: '25'

def get_names():
    folder_names = []
    for (noise, noise_value) in gp_noise_leves.items():
            for (beta, beta_value) in beta_t_sqrt_vals.items():            
                for (safety, safety_value) in safety_threshold_hs.items():
                    for (radius, radius_value) in neighborhood_radius_vs.items():
                        foldername = 'SPPO' + '_' + safety_value + '_' + radius_value + '_' + noise_value + '_' + beta_value
                        folder_names.append(foldername)
    #folder_names.append('PPO')
    return folder_names

if __name__=='__main__':
    try:
        scenario = int(sys.argv[1])
    except IndexError:
        scenario = 0

    if scenario not in scenarios:
        scenario = 0

    dir_path = './results/scenario_{}/'.format(scenario)
    algo_names = get_folder_names(dir_path)
    algo_names.sort()
    #algo_names = get_names()
    #labels = algo_names
    labels = []
    for algo_name in algo_names:
        if algo_name == 'KBRL_97':
            labels.append('KBRL')
            continue
        labels.append(algo_name.split('_')[0] + '_' + algo_name.split('_')[-1])
    #print(algo_names)
    prbs = prbs_values[scenario]

    save_path = './results/scenario_{}/subplots_wkblr'.format(scenario)

    # Generate distinct colors from a colormap
    color_list = plt.cm.tab10.colors  # up to 10 distinct colors; you can also use tab20, Set3, etc.
    color_map = {algo: color_list[i % len(color_list)] for i, algo in enumerate(algo_names)}

    # subplot
    if SAFESET:
        fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(25, 11), constrained_layout=False)
        axs = axs.flatten()
    else:
        fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(25, 6), constrained_layout=False)
    fig.subplots_adjust(top=0.80)
    # iterate over algorithms
    for algo, label in zip(algo_names, labels):
        violations = np.empty([1])
        actions = np.empty([1])
        rewards = np.empty([1])
        regret = np.empty([1])
        safe_set = np.empty([1])
        data = False
        proposal = False
        path = './results/scenario_{}/{}/'.format(scenario, algo)
        runs = 0
        has_safe_set = False
        has_ues = False

        if not os.path.exists(path) or not os.path.isdir(path):
            continue

        # iterate over files
        for filename in os.listdir(path):
            if filename.endswith(".npz"):
                histories = np.load(path + filename, allow_pickle=True)

                _violations = histories['violation']
                _violations = np.array([np.sum(r) for r in _violations], dtype=np.int16)
                _resources = histories['resources']
                _rewards = histories['reward']
                if 'ue' in histories:
                    _ue = histories['ue']
                    _ue = np.array([np.sum(r) for r in _ue], dtype=np.int16)
                    has_ues = True
                if 'safe set' in histories:
                    _safe_set = histories['safe set']
                    _safe_set = np.array([np.mean(r) for r in _safe_set], dtype=np.int16)
                    has_safe_set = True
                if len(_violations) < END:
                    continue
                _violations = _violations[START:END]
                _resources = _resources[START:END]
                _rewards = _rewards[START:END]
                _ue = _ue[START:END]
                if 'safe set' in histories:
                    _safe_set = _safe_set[START:END]
                runs += 1
                # load data for each run
                if not data:
                    violations = movingaverage(_violations, WINDOW)
                    regret = movingaverage(_violations.cumsum(), WINDOW)
                    actions = movingaverage(_resources, WINDOW)
                    rewards = movingaverage(_rewards.cumsum(), WINDOW)
                    if has_ues:
                        ue = movingaverage(_ue, WINDOW)
                    if 'safe set' in histories:
                        safe_set = movingaverage(_safe_set, WINDOW)
                    if proposal:
                        accuracy = movingaverage(np.mean(histories['hits'], axis=0), WINDOW)
                    data = True
        
                else: # store the history of each run
                    violations = np.vstack((violations, movingaverage(_violations, WINDOW)))
                    regret = np.vstack((regret, movingaverage(_violations.cumsum(), WINDOW)))
                    actions = np.vstack((actions, movingaverage(_resources, WINDOW)))
                    rewards = np.vstack((rewards, movingaverage(_rewards.cumsum(), WINDOW)))
                    if has_ues:
                        ue = np.vstack((ue, movingaverage(_ue, WINDOW)))
                    if 'safe set' in histories:
                        safe_set = np.vstack((safe_set, movingaverage(_safe_set, WINDOW)))
                    if proposal:
                        accuracy = np.vstack((accuracy, movingaverage(np.mean(histories['hits'], axis=0), WINDOW)))
        
        print('Algorithm {}'.format(algo))
        
        # average over different runs

        actions_mean = np.mean(actions, axis=0)
        actions_std = np.std(actions, axis=0)
        actions_min = np.min(actions, axis=0)
        actions_max = np.max(actions, axis=0)
        action_idx = 0

        violations_mean = np.mean(violations, axis=0)
        violations_std = np.std(violations, axis=0)
        violations_min = np.min(violations, axis=0)
        violations_max = np.max(violations, axis=0)
        violations_idx = 3

        rewards_mean = np.mean(rewards, axis=0)
        rewards_std = np.std(rewards, axis=0)
        rewards_min = np.min(rewards, axis=0)
        rewards_max = np.max(rewards, axis=0)
        rewards_idx = 1

        ue_mean = np.mean(ue, axis=0)
        ue_std = np.std(ue, axis=0)
        ue_min = np.min(ue, axis=0)
        ue_max = np.max(ue, axis=0)
        ue_idx = 2

        regret_mean = np.mean(regret, axis=0)
        regret_std = np.std(regret, axis=0)
        regret_min = np.min(regret, axis=0)
        regret_max = np.min(regret, axis=0)
        regret_idx = 4

        if has_safe_set:
            safe_set_mean = np.mean(safe_set, axis=0)
            safe_set_std = np.std(safe_set, axis=0)
            safe_set_min = np.min(safe_set, axis=0)
            safe_set_max = np.max(safe_set, axis=0)
            safe_set_idx = 5

        if proposal:
            accuracy_mean = np.mean(accuracy, axis=0)
            accuracy_std = np.std(accuracy, axis=0)

        # plot results
        steps = np.arange(len(actions_mean[0:SPAN]))

        axs[action_idx].set_title('Resource allocation', fontsize=14)
        axs[action_idx].plot(steps, actions_mean[0:SPAN], label = label, linewidth = 2, color=color_map[algo])
        #axs[action_idx].fill_between(steps, actions_min, actions_max, alpha=0.3, label='_nolegend_', color=color_map[algo]) # , color = '#DDDDDD'
        axs[action_idx].fill_between(steps, actions_mean[0:SPAN] - 1.697 * actions_std[0:SPAN] / np.sqrt(runs), 
                        actions_mean[0:SPAN] + 1.697 * actions_std[0:SPAN] / np.sqrt(runs), color = color_map[algo],
                        alpha=0.3, label='_nolegend_')
        if algo == algo_names[-1]:
            axs[action_idx].set_ylim((0,prbs))
            axs[action_idx].set_yticks(np.arange(0, prbs+1, 10))
            axs[action_idx].set_xlabel('Step', fontsize=14)  # Add an x-label to the axes.
            axs[action_idx].set_ylabel('PRBs', fontsize=14)
            axs[action_idx].legend(loc='best', fontsize=14)
            axs[action_idx].grid()

        axs[violations_idx].set_title('SLA violations', fontsize=14)
        axs[violations_idx].plot(steps, violations_mean[0:SPAN], label = label, linewidth = 2, color=color_map[algo])
        #axs[violations_idx].fill_between(steps, violations_min, violations_max, alpha=0.3, label='_nolegend_', color=color_map[algo]) #, color = '#DDDDDD'
        axs[violations_idx].fill_between(steps, violations_mean[0:SPAN] - 1.697 * violations_std[0:SPAN] / np.sqrt(runs), 
                        violations_mean[0:SPAN] + 1.697 * violations_std[0:SPAN] / np.sqrt(runs), color = color_map[algo],
                        alpha=0.3, label='_nolegend_')
        if algo == algo_names[-1]:
            axs[violations_idx].set_xlabel('Step', fontsize=14)  # Add an x-label to the axes.
            axs[violations_idx].set_ylabel('SLA violations', fontsize=14)
            axs[violations_idx].set_ylim((0, 0.2))
            axs[violations_idx].set_yticks(np.arange(0, 0.21, 0.02))
            axs[violations_idx].legend(loc='best', fontsize=14)
            axs[violations_idx].grid()
        
        axs[rewards_idx].set_title('Rewards', fontsize=14)
        axs[rewards_idx].plot(steps, rewards_mean[0:SPAN], label = label, linewidth = 2, color=color_map[algo])
        #axs[rewards_idx].fill_between(steps, rewards_min, rewards_max, alpha=0.3, label='_nolegend_', color=color_map[algo]) # , color = '#DDDDDD'
        axs[rewards_idx].fill_between(steps, rewards_mean[0:SPAN] - 1.697 * rewards_std[0:SPAN] / np.sqrt(runs), 
                        rewards_mean[0:SPAN] + 1.697 * rewards_std[0:SPAN] / np.sqrt(runs), color = color_map[algo],
                        alpha=0.3, label='_nolegend_')
        if algo == algo_names[-1]:
            axs[rewards_idx].set_xlabel('Step', fontsize=14)  # Add an x-label to the axes.
            axs[rewards_idx].set_ylabel('Reward', fontsize=14)
            axs[rewards_idx].set_ylim((0,900000)) # 15000
            axs[rewards_idx].set_yticks(np.arange(0, 900001, 100000))
            axs[rewards_idx].legend(loc='best', fontsize=14)
            axs[rewards_idx].grid()

        axs[regret_idx].set_title('Cumulative SLA violations', fontsize=14)
        axs[regret_idx].plot(steps, regret_mean[0:SPAN], label = label, linewidth = 2, color=color_map[algo])
        #axs[regret_idx].fill_between(steps, regret_min, regret_max, alpha=0.3, label='_nolegend_', color=color_map[algo]) # , color = '#DDDDDD'
        axs[regret_idx].fill_between(steps, regret_mean[0:SPAN] - 1.697 * regret_std[0:SPAN] / np.sqrt(runs), 
                        regret_mean[0:SPAN] + 1.697 * regret_std[0:SPAN] / np.sqrt(runs), color = color_map[algo],
                        alpha=0.3, label='_nolegend_')
        if algo == algo_names[-1]:
            axs[regret_idx].set_xlabel('Step', fontsize=14)  # Add an x-label to the axes.
            axs[regret_idx].set_ylabel('cumulative SLA violations', fontsize=14)
            axs[regret_idx].set_ylim((0,200)) # 15000
            axs[regret_idx].set_yticks(np.arange(0, 201, 20))
            axs[regret_idx].legend(loc='best', fontsize=14)
            axs[regret_idx].grid() 

        if has_ues:
            axs[ue_idx].set_title('Number of UEs', fontsize=14)
            axs[ue_idx].plot(steps, ue_mean[0:SPAN], label = label, linewidth = 2, color=color_map[algo])
            #axs[ue_idx].fill_between(steps, regret_min, regret_max, alpha=0.3, label='_nolegend_', color=color_map[algo]) # , color = '#DDDDDD'
            axs[ue_idx].fill_between(steps, ue_mean[0:SPAN] - 1.697 * ue_std[0:SPAN] / np.sqrt(runs), 
                            ue_mean[0:SPAN] + 1.697 * ue_std[0:SPAN] / np.sqrt(runs), color = color_map[algo],
                            alpha=0.3, label='_nolegend_')
            if algo == algo_names[-1]:
                axs[ue_idx].set_xlabel('Step', fontsize=14)  # Add an x-label to the axes.
                axs[ue_idx].set_ylabel('number of UEs', fontsize=14)
                axs[ue_idx].set_ylim((0,10)) # 15000
                axs[ue_idx].set_yticks(np.arange(0, 11, 1))
                axs[ue_idx].legend(loc='best', fontsize=14)
                axs[ue_idx].grid()      

        if has_safe_set:
            axs[safe_set_idx].set_title('Safe Set', fontsize=14)
            axs[safe_set_idx].plot(steps, safe_set_mean[0:SPAN], label = label, linewidth = 2, color=color_map[algo])
            #axs[safe_set_idx].fill_between(steps, safe_set_min, safe_set_max, alpha=0.3, label='_nolegend_', color=color_map[algo]) # , color = '#DDDDDD'
            axs[safe_set_idx].fill_between(steps, safe_set_mean[0:SPAN] - 1.697 * safe_set_std[0:SPAN] / np.sqrt(runs), 
                            safe_set_mean[0:SPAN] + 1.697 * safe_set_std[0:SPAN] / np.sqrt(runs), color = color_map[algo],
                        alpha=0.3, label='_nolegend_')
            if algo == algo_names[-1]:
                axs[safe_set_idx].set_xlabel('Step', fontsize=14)  # Add an x-label to the axes.
                axs[safe_set_idx].set_ylabel('safe set', fontsize=14)
                axs[safe_set_idx].set_ylim((0,40)) # 15000
                axs[safe_set_idx].set_yticks(np.arange(0, 41, 5))
                axs[safe_set_idx].legend(loc='best', fontsize=14)
                axs[safe_set_idx].grid()   

                # Add zoom-in inset

                #axins = inset_axes(axs[4], width="40%", height="40%", loc='upper right', borderpad=2)
                #axins.plot(steps, safe_set_mean[0:SPAN], linewidth=2)
                #axins.fill_between(steps, safe_set_mean - safe_set_std, safe_set_mean + safe_set_std, alpha=0.3)
                #axins.set_xlim(0, 50)
                #axins.set_ylim(0.9, 1.1)
                #axins.set_xticks([0, 25, 50])
                #axins.set_yticks([0.9, 0.95, 1.1])
                #axins.tick_params(labelsize=8)
                #mark_inset(axs[4], axins, loc1=2, loc2=4, fc="none", ec="0.5")
        
        if algo == algo_names[-1]:
            # Create a single legend above all subplots
            ncol = len(labels)
            if len(labels) > 10:
                ncol = math.ceil(len(labels) / 3)
            elif len(labels) > 6:
                ncol = math.ceil(len(labels) / 2)
            
            fig.legend(labels, loc='upper center', ncol=ncol, bbox_to_anchor=(0.35, 1.0), frameon=True, fontsize=14)
            fig.tight_layout(rect=[0, 0, 1, 0.95])

            if START > 0:
                fig.savefig(save_path.format(scenario), format='png')
            else:
                # fig.savefig('./figures/subplots_{}.svg'.format(scenario), format='svg')
                fig.savefig(save_path.format(scenario), format='png')
            # fig.savefig('_subplots_' + scenario + '.svg', format='svg')       
