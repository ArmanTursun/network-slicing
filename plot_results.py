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

import matplotlib.pyplot as plt
import matplotlib.cm as cm


# trainning results
WINDOW = 100 #400
START = 0
END =  10000 # up to 39900  20000
RESETS = False
#algo_names = ['A2C', 'PPO1', 'PPO2', 'TRPO', 'SAC', 'TD3', 'NAF', 'KBRL_97','KBRL_99']
#labels = ['A2C', 'PPO1', 'PPO2', 'TRPO', 'SAC', 'TD3', 'NAF', 'KBRL 0.97', 'KBRL 0.99']
algo_names = ['PPO'] # , 'PPO', 'SPPO'
labels = ['PPO'] # , 'PPO', 'SPPO'

# filename: threshold_radius_beta_batchsize_initmethod_gpiters

SPAN = END - START

prbs_values = [80, 150, 100, 70] #
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
    #algo_names = get_names()
    labels = algo_names
    #print(algo_names)
    prbs = prbs_values[scenario]

    save_path = './figures/new/subplots_{}_wkblr'.format(scenario)

    # Generate distinct colors from a colormap
    color_list = plt.cm.tab10.colors  # up to 10 distinct colors; you can also use tab20, Set3, etc.
    color_map = {algo: color_list[i % len(color_list)] for i, algo in enumerate(algo_names)}

    # subplot
    if RESETS:
        fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(20, 5), constrained_layout=False)
    else:
        fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(20, 5), constrained_layout=False)
    fig.subplots_adjust(top=0.78)
    # iterate over algorithms
    for algo, label in zip(algo_names, labels):
        violations = np.empty([1])
        actions = np.empty([1])
        rewards = np.empty([1])
        regret = np.empty([1])
        resets = np.empty([1])
        data = False
        proposal = False
        path = './results/scenario_{}/{}/'.format(scenario, algo)
        runs = 0
        has_resets = False

        if not os.path.exists(path) or not os.path.isdir(path):
            continue

        # iterate over files
        for filename in os.listdir(path):
            if filename.endswith(".npz"):
                histories = np.load(path + filename)
                _violations = histories['violation']
                _resources = histories['resources']
                _rewards = histories['reward']
                if 'reset' in histories:
                    _resets = histories['reset']
                    has_resets = True
                if len(_violations) < END:
                    continue
                _violations = _violations[START:END]
                _resources = _resources[START:END]
                _rewards = _rewards[START:END]
                if 'reset' in histories:
                    _resets = _resets[START:END]
                runs += 1
                # load data for each run
                if not data:
                    violations = average_per_window(_violations, WINDOW)
                    regret = average_per_window(_violations.cumsum(), WINDOW)
                    actions = average_per_window(_resources, WINDOW)
                    rewards = average_per_window(_rewards, WINDOW)
                    if 'reset' in histories:
                        resets = average_per_window(_resets, WINDOW)
                    if proposal:
                        accuracy = average_per_window(np.mean(histories['hits'], axis=0), WINDOW)
                    data = True
                else: # store the history of each run
                    violations = np.vstack((violations, average_per_window(_violations, WINDOW)))
                    regret = np.vstack((regret, average_per_window(_violations.cumsum(), WINDOW)))
                    actions = np.vstack((actions, average_per_window(_resources, WINDOW)))
                    rewards = np.vstack((rewards, average_per_window(_rewards, WINDOW)))
                    if 'reset' in histories:
                        resets = np.vstack((resets, average_per_window(_resets, WINDOW)))
                    if proposal:
                        accuracy = np.vstack((accuracy, average_per_window(np.mean(histories['hits'], axis=0), WINDOW)))
        
        print('Algorithm {}'.format(algo))
        
        # average over different runs

        actions_mean = np.mean(actions, axis=0)
        actions_std = np.std(actions, axis=0)
        actions_min = np.min(actions, axis=0)
        actions_max = np.max(actions, axis=0)

        violations_mean = np.mean(violations, axis=0)
        violations_std = np.std(violations, axis=0)
        violations_min = np.min(violations, axis=0)
        violations_max = np.max(violations, axis=0)

        rewards_mean = np.mean(rewards, axis=0)
        rewards_std = np.std(rewards, axis=0)
        rewards_min = np.min(rewards, axis=0)
        rewards_max = np.max(rewards, axis=0)

        regret_mean = np.mean(regret, axis=0)
        regret_std = np.std(regret, axis=0)
        regret_min = np.min(regret, axis=0)
        regret_max = np.min(regret, axis=0)

        if has_resets:
            resets_mean = np.mean(resets, axis=0)
            resets_std = np.std(resets, axis=0)
            resets_min = np.min(resets, axis=0)
            resets_max = np.max(resets, axis=0)

        if proposal:
            accuracy_mean = np.mean(accuracy, axis=0)
            accuracy_std = np.std(accuracy, axis=0)

        # plot results
        steps = np.arange(len(actions_mean[0:SPAN]))

        axs[2].set_title('Resource allocation', fontsize=14)
        axs[2].plot(steps, actions_mean[0:SPAN], label = label, linewidth = 2, color=color_map[algo])
        #axs[2].fill_between(steps, actions_min, actions_max, alpha=0.3, label='_nolegend_', color=color_map[algo]) # , color = '#DDDDDD'
        axs[2].fill_between(steps, actions_mean[0:SPAN] - 1.697 * actions_std[0:SPAN] / np.sqrt(runs), 
                        actions_mean[0:SPAN] + 1.697 * actions_std[0:SPAN] / np.sqrt(runs), color = color_map[algo],
                        alpha=0.3, label='_nolegend_')
        if algo == algo_names[-1]:
            axs[2].set_ylim((0,prbs))
            axs[2].set_xlabel('Epoch', fontsize=14)  # Add an x-label to the axes.
            axs[2].set_ylabel('PRBs', fontsize=14)
            # axs[2].legend(loc='best')
            axs[2].grid()

        axs[0].set_title('SLA violations', fontsize=14)
        axs[0].plot(steps, violations_mean[0:SPAN], label = label, linewidth = 2, color=color_map[algo])
        #axs[0].fill_between(steps, violations_min, violations_max, alpha=0.3, label='_nolegend_', color=color_map[algo]) #, color = '#DDDDDD'
        axs[0].fill_between(steps, violations_mean[0:SPAN] - 1.697 * violations_std[0:SPAN] / np.sqrt(runs), 
                        violations_mean[0:SPAN] + 1.697 * violations_std[0:SPAN] / np.sqrt(runs), color = color_map[algo],
                        alpha=0.3, label='_nolegend_')
        if algo == algo_names[-1]:
            axs[0].set_xlabel('Epoch', fontsize=14)  # Add an x-label to the axes.
            axs[0].set_ylabel('SLA violations', fontsize=14)
            axs[3].set_ylim((0,1))
            #axs[0].legend(loc='best')
            axs[0].grid()
        
        axs[3].set_title('Rewards', fontsize=14)
        axs[3].plot(steps, rewards_mean[0:SPAN], label = label, linewidth = 2, color=color_map[algo])
        #axs[3].fill_between(steps, rewards_min, rewards_max, alpha=0.3, label='_nolegend_', color=color_map[algo]) # , color = '#DDDDDD'
        axs[3].fill_between(steps, rewards_mean[0:SPAN] - 1.697 * rewards_std[0:SPAN] / np.sqrt(runs), 
                        rewards_mean[0:SPAN] + 1.697 * rewards_std[0:SPAN] / np.sqrt(runs), color = color_map[algo],
                        alpha=0.3, label='_nolegend_')
        if algo == algo_names[-1]:
            axs[3].set_xlabel('Epoch', fontsize=14)  # Add an x-label to the axes.
            axs[3].set_ylabel('Reward', fontsize=14)
            axs[3].set_ylim((-50,80)) # 15000
            #axs[3].legend(loc='best')
            axs[3].grid()

        axs[1].set_title('Cumulative SLA violations', fontsize=14)
        axs[1].plot(steps, regret_mean[0:SPAN], label = label, linewidth = 2, color=color_map[algo])
        #axs[1].fill_between(steps, regret_min, regret_max, alpha=0.3, label='_nolegend_', color=color_map[algo]) # , color = '#DDDDDD'
        axs[1].fill_between(steps, regret_mean[0:SPAN] - 1.697 * regret_std[0:SPAN] / np.sqrt(runs), 
                        regret_mean[0:SPAN] + 1.697 * regret_std[0:SPAN] / np.sqrt(runs), color = color_map[algo],
                        alpha=0.3, label='_nolegend_')
        if algo == algo_names[-1]:
            axs[1].set_xlabel('Epoch', fontsize=14)  # Add an x-label to the axes.
            axs[1].set_ylabel('cumulative SLA violations', fontsize=14)
            axs[1].set_ylim((0,1000)) # 15000
            #axs[1].legend(loc='best')
            axs[1].grid()      

        if has_resets and resets.shape[0] > 0:
            axs[4].set_title('Resets', fontsize=14)
            axs[4].plot(steps, resets_mean[0:SPAN], label = label, linewidth = 2, color=color_map[algo])
            #axs[4].fill_between(steps, resets_min, resets_max, alpha=0.3, label='_nolegend_', color=color_map[algo]) # , color = '#DDDDDD'
            axs[1].fill_between(steps, regret_mean[0:SPAN] - 1.697 * regret_std[0:SPAN] / np.sqrt(runs), 
                            regret_mean[0:SPAN] + 1.697 * regret_std[0:SPAN] / np.sqrt(runs), color = color_map[algo],
                        alpha=0.3, label='_nolegend_')
            if algo == algo_names[-1]:
                axs[4].set_xlabel('Epoch', fontsize=14)  # Add an x-label to the axes.
                axs[4].set_ylabel('resets', fontsize=14)
                axs[4].set_ylim((0,1.2)) # 15000
                #axs[1].legend(loc='best')
                axs[4].grid()   

                # Add zoom-in inset

                #axins = inset_axes(axs[4], width="40%", height="40%", loc='upper right', borderpad=2)
                #axins.plot(steps, resets_mean[0:SPAN], linewidth=2)
                #axins.fill_between(steps, resets_mean - resets_std, resets_mean + resets_std, alpha=0.3)
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
            elif len(labels) > 5:
                ncol = math.ceil(len(labels) / 2)
            
            fig.legend(labels, loc='upper center', ncol=ncol, bbox_to_anchor=(0.5, 1.0), frameon=True, fontsize=14)
            fig.tight_layout(rect=[0, 0, 1, 0.8])

            if START > 0:
                fig.savefig(save_path.format(scenario), format='png')
            else:
                # fig.savefig('./figures/subplots_{}.svg'.format(scenario), format='svg')
                fig.savefig(save_path.format(scenario), format='png')
            # fig.savefig('_subplots_' + scenario + '.svg', format='svg')       
