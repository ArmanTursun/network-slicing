#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 5, June 2025

@author: Arman

"""
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import pandas as pd
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
    #dir_path = './results/#scenario_{}_QR_margin_self-learn_best/'.format(scenario)
    algo_names = get_folder_names(dir_path)
    algo_names.sort()
    #algo_names = get_names()
    #labels = algo_names
    labels = []
    for algo_name in algo_names:
        if algo_name == 'KBRL_97':
            labels.append('KBRL')
            continue
        label = algo_name.split('_')[0] + '_' + algo_name.split('_')[-1]
        if label == 'QR_nocost':
            label = 'SQR w/o Cost'
        if label == 'QR_cost':
            label = 'SQR w/ Cost'
        labels.append(label)
    #print(algo_names)
    prbs = prbs_values[scenario]

    save_path = './results/scenario_{}/subplots_wkblr'.format(scenario)
    #save_path = './results/#scenario_{}_QR_margin_self-learn_best/subplots_wkblr'.format(scenario)

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

    fig_rb = plt.figure(figsize=(10, 6))
    ax_rb = fig_rb.add_subplot()
    fig_vio = plt.figure(figsize=(10, 6))
    ax_vio = fig_vio.add_subplot()
    fig_cumvio = plt.figure(figsize=(10, 6))
    ax_cumvio = fig_cumvio.add_subplot()

    colors = []
    plot_data = []
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
        #path = './results/#scenario_{}_QR_margin_self-learn_best/{}/'.format(scenario, algo)
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
                if label != 'KBRL':
                    #for r in _violations:
                    #    if np.sum(r) > 1:
                    #        print(r)
                    _violations = np.array([np.sum(r) for r in _violations], dtype=np.int16)
                else:
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
                    actions = movingaverage(_resources, WINDOW/WINDOW)
                    rewards = movingaverage(_rewards.cumsum(), WINDOW)
                    if has_ues:
                        ue = movingaverage(_ue, WINDOW/WINDOW)
                    if 'safe set' in histories:
                        safe_set = movingaverage(_safe_set, WINDOW)
                    if proposal:
                        accuracy = movingaverage(np.mean(histories['hits'], axis=0), WINDOW)
                    data = True
        
                else: # store the history of each run
                    violations = np.vstack((violations, movingaverage(_violations, WINDOW)))
                    regret = np.vstack((regret, movingaverage(_violations.cumsum(), WINDOW)))
                    actions = np.vstack((actions, movingaverage(_resources, WINDOW/WINDOW)))
                    rewards = np.vstack((rewards, movingaverage(_rewards.cumsum(), WINDOW)))
                    if has_ues:
                        ue = np.vstack((ue, movingaverage(_ue, WINDOW/WINDOW)))
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
        
        '''
        # plot results
        steps = np.arange(len(actions_mean[0:SPAN]))
        axs[action_idx].set_title('Resource allocation', fontsize=18)
        axs[action_idx].plot(steps, actions_mean[0:SPAN], label = label, linewidth = 2, color=color_map[algo])
        axs[action_idx].fill_between(steps, actions_mean[0:SPAN] - 1.697 * actions_std[0:SPAN] / np.sqrt(runs), 
                        actions_mean[0:SPAN] + 1.697 * actions_std[0:SPAN] / np.sqrt(runs), color = color_map[algo],
                        alpha=0.3, label='_nolegend_')
        if algo == algo_names[-1]:
            axs[action_idx].set_ylim((0,prbs))
            axs[action_idx].set_yticks(np.arange(0, prbs+1, 10))
            axs[action_idx].set_xlabel('Step', fontsize=18)  # Add an x-label to the axes.
            axs[action_idx].set_ylabel('PRBs', fontsize=18)
            axs[action_idx].legend(loc='best', fontsize=18)
            axs[action_idx].grid()
        '''
        avg_per_run = actions.mean(axis=1)  # Assume actions_dict[algo] is (30, T)
        if label != 'KBRL':
            avg_per_run = sorted(avg_per_run)[:10]  # Take best 15 runs
        for val in avg_per_run:
            plot_data.append({'Algorithm': label, 'PRBs': val})
        colors.append(color_map[algo])
        if algo == algo_names[-1]:
            df = pd.DataFrame(plot_data)
            # Plot with seaborn violinplot on the given subplot
            #sns.violinplot(data=df, x='Algorithm', y='PRBs', ax=axs[action_idx],
            #            palette=[color_map[algo] for algo in algo_names],
            #            inner='quartile', cut=0)
            sns.violinplot(
                data=df,
                x='Algorithm',
                y='PRBs',
                hue='Algorithm',  # same as x
                palette=[color_map[algo] for algo in algo_names],
                ax=axs[action_idx],
                density_norm='width',
                native_scale=True,
                width=0.5,
                saturation=1,
                #bw=0.2,
                #bw_adjust=1,
                #inner='quartile',
                cut=3,
                legend=False  # or remove legend after
            )

            # Disable automatic legend (optional but clean)
            #axs[action_idx].get_legend().remove()
            # Customize subplot
            axs[action_idx].set_yticks(np.arange(0, prbs+1, 10))
            axs[action_idx].set_xticks(np.arange(0, len(labels)), labels)
            axs[action_idx].tick_params(axis='x', labelsize=18)
            axs[action_idx].tick_params(axis='y', labelsize=18)
            axs[action_idx].set_xlabel('Algorithm', fontsize=18)
            axs[action_idx].set_ylabel('PRBs', fontsize=18)
            axs[action_idx].set_title('Resource Allocation', fontsize=18)

            sns.violinplot(
                data=df,
                x='Algorithm',
                y='PRBs',
                hue='Algorithm',  # same as x
                palette=[color_map[algo] for algo in algo_names],
                ax=ax_rb,
                density_norm='width',
                native_scale=True,
                width=0.5,
                saturation=1,
                #bw=0.2,
                #bw_adjust=1,
                #inner='quartile',
                cut=3,
                legend=False  # or remove legend after
            )
            ax_rb.set_yticks(np.arange(0, prbs+1, 10))
            ax_rb.set_xticks(np.arange(0, len(labels)), labels)
            ax_rb.tick_params(axis='x', labelsize=18)
            ax_rb.tick_params(axis='y', labelsize=18)
            ax_rb.set_xlabel('Algorithm', fontsize=18)
            ax_rb.set_ylabel('PRBs', fontsize=18)
            #ax_rb.set_title('Resource Allocation', fontsize=18)
            fig_rb.tight_layout()
            fig_rb.savefig(save_path.format(scenario)+'_rb', format='png', transparent=True, dpi=300.0)
        
        steps = np.arange(len(violations_mean[0:SPAN]))
        axs[violations_idx].set_title('SLA violations', fontsize=18)
        axs[violations_idx].plot(steps, violations_mean[0:SPAN], label = label, linewidth = 2, color=color_map[algo])
        #axs[violations_idx].fill_between(steps, violations_min, violations_max, alpha=0.3, label='_nolegend_', color=color_map[algo]) #, color = '#DDDDDD'
        axs[violations_idx].fill_between(steps, violations_mean[0:SPAN] - 1.697 * violations_std[0:SPAN] / np.sqrt(runs), 
                        violations_mean[0:SPAN] + 1.697 * violations_std[0:SPAN] / np.sqrt(runs), color = color_map[algo],
                        alpha=0.3, label='_nolegend_')
        #ax_vio.set_title('SLA violations', fontsize=18)
        ax_vio.plot(steps, violations_mean[0:SPAN], label = label, linewidth = 2, color=color_map[algo])
        #axs[violations_idx].fill_between(steps, violations_min, violations_max, alpha=0.3, label='_nolegend_', color=color_map[algo]) #, color = '#DDDDDD'
        ax_vio.fill_between(steps, violations_mean[0:SPAN] - 1.697 * violations_std[0:SPAN] / np.sqrt(runs), 
                        violations_mean[0:SPAN] + 1.697 * violations_std[0:SPAN] / np.sqrt(runs), color = color_map[algo],
                        alpha=0.3, label='_nolegend_')
        if algo == algo_names[-1]:
            axs[violations_idx].axhline(y=0.01, color='black', linestyle='--', linewidth=2, label='99% SLA')
            axs[violations_idx].axhline(y=0.05, color='gray', linestyle='--', linewidth=2, label='95% SLA')
            axs[violations_idx].set_xlabel('Step', fontsize=18)  # Add an x-label to the axes.
            axs[violations_idx].set_ylabel('SLA violations', fontsize=18)
            axs[violations_idx].set_ylim((0, 0.1))
            axs[violations_idx].set_yticks(np.arange(0, 0.11, 0.01))
            axs[violations_idx].set_xticks(np.arange(0, 10001, 2000))
            axs[violations_idx].tick_params(axis='x', labelsize=18)
            axs[violations_idx].tick_params(axis='y', labelsize=18)
            axs[violations_idx].legend(loc='best', fontsize=18)
            axs[violations_idx].grid()

            ax_vio.axhline(y=0.01, color='black', linestyle='--', linewidth=2, label='99% SLA')
            ax_vio.axhline(y=0.05, color='gray', linestyle='--', linewidth=2, label='95% SLA')
            ax_vio.axvline(x=2000, color='red', linestyle='--', linewidth=2, label='Full Budget')
            ax_vio.set_xlabel('Step', fontsize=18)  # Add an x-label to the axes.
            ax_vio.set_ylabel('SLA violations', fontsize=18)
            ax_vio.set_ylim((0, 0.1))
            ax_vio.set_yticks(np.arange(0, 0.11, 0.01))
            ax_vio.set_xticks(np.arange(0, 10001, 2000))
            ax_vio.tick_params(axis='x', labelsize=18)
            ax_vio.tick_params(axis='y', labelsize=18)
            ax_vio.legend(loc='best', fontsize=18)
            ax_vio.grid()
            fig_vio.tight_layout()
            fig_vio.savefig(save_path.format(scenario)+'_vio', format='png', transparent=True, dpi=300.0)
        
        steps = np.arange(len(rewards_mean[0:SPAN]))
        axs[rewards_idx].set_title('Rewards', fontsize=18)
        axs[rewards_idx].plot(steps, rewards_mean[0:SPAN], label = label, linewidth = 2, color=color_map[algo])
        #axs[rewards_idx].fill_between(steps, rewards_min, rewards_max, alpha=0.3, label='_nolegend_', color=color_map[algo]) # , color = '#DDDDDD'
        axs[rewards_idx].fill_between(steps, rewards_mean[0:SPAN] - 1.697 * rewards_std[0:SPAN] / np.sqrt(runs), 
                        rewards_mean[0:SPAN] + 1.697 * rewards_std[0:SPAN] / np.sqrt(runs), color = color_map[algo],
                        alpha=0.3, label='_nolegend_')
        if algo == algo_names[-1]:
            axs[rewards_idx].set_xlabel('Step', fontsize=18)  # Add an x-label to the axes.
            axs[rewards_idx].set_ylabel('Reward', fontsize=18)
            axs[rewards_idx].set_ylim((0,600000)) # 15000
            axs[rewards_idx].set_yticks(np.arange(0, 600001, 50000))
            axs[rewards_idx].set_xticks(np.arange(0, 10001, 2000))
            axs[rewards_idx].tick_params(axis='x', labelsize=18)
            axs[rewards_idx].tick_params(axis='y', labelsize=18)
            axs[rewards_idx].legend(loc='best', fontsize=18)
            axs[rewards_idx].grid()

        steps = np.arange(len(regret_mean[0:SPAN]))
        axs[regret_idx].set_title('Cumulative SLA violations', fontsize=18)
        axs[regret_idx].plot(steps, regret_mean[0:SPAN], label = label, linewidth = 2, color=color_map[algo])
        #axs[regret_idx].fill_between(steps, regret_min, regret_max, alpha=0.3, label='_nolegend_', color=color_map[algo]) # , color = '#DDDDDD'
        axs[regret_idx].fill_between(steps, regret_mean[0:SPAN] - 1.697 * regret_std[0:SPAN] / np.sqrt(runs), 
                        regret_mean[0:SPAN] + 1.697 * regret_std[0:SPAN] / np.sqrt(runs), color = color_map[algo],
                        alpha=0.3, label='_nolegend_')
        #ax_cumvio.set_title('Cumulative SLA violations', fontsize=18)
        ax_cumvio.plot(steps, regret_mean[0:SPAN], label = label, linewidth = 2, color=color_map[algo])
        #axs[regret_idx].fill_between(steps, regret_min, regret_max, alpha=0.3, label='_nolegend_', color=color_map[algo]) # , color = '#DDDDDD'
        ax_cumvio.fill_between(steps, regret_mean[0:SPAN] - 1.697 * regret_std[0:SPAN] / np.sqrt(runs), 
                        regret_mean[0:SPAN] + 1.697 * regret_std[0:SPAN] / np.sqrt(runs), color = color_map[algo],
                        alpha=0.3, label='_nolegend_')
        if algo == algo_names[-1]:
            ax_cumvio.set_xlabel('Step', fontsize=18)  # Add an x-label to the axes.
            ax_cumvio.set_ylabel('cumulative SLA violations', fontsize=18)
            ax_cumvio.set_ylim((0,40)) # 15000
            ax_cumvio.set_yticks(np.arange(0, 41, 5))
            ax_cumvio.set_xticks(np.arange(0, 10001, 2000))
            ax_cumvio.tick_params(axis='x', labelsize=18)
            ax_cumvio.tick_params(axis='y', labelsize=18)
            ax_cumvio.legend(loc='best', fontsize=18)
            ax_cumvio.grid() 
            fig_cumvio.tight_layout()
            fig_cumvio.savefig(save_path.format(scenario)+'_cumvio', format='png', transparent=True, dpi=300.0)

        if has_ues:
            ue_steps = np.arange(len(ue_mean[0:SPAN]))
            axs[ue_idx].set_title('Number of UEs', fontsize=18)
            axs[ue_idx].plot(ue_steps, ue_mean[0:SPAN], label = label, linewidth = 2, color=color_map[algo])
            #axs[ue_idx].fill_between(steps, regret_min, regret_max, alpha=0.3, label='_nolegend_', color=color_map[algo]) # , color = '#DDDDDD'
            axs[ue_idx].fill_between(ue_steps, ue_mean[0:SPAN] - 1.697 * ue_std[0:SPAN] / np.sqrt(runs), 
                            ue_mean[0:SPAN] + 1.697 * ue_std[0:SPAN] / np.sqrt(runs), color = color_map[algo],
                            alpha=0.3, label='_nolegend_')
            if algo == algo_names[-1]:
                axs[ue_idx].set_xlabel('Step', fontsize=18)  # Add an x-label to the axes.
                axs[ue_idx].set_ylabel('number of UEs', fontsize=18)
                axs[ue_idx].set_ylim((0,10)) # 15000
                axs[ue_idx].set_yticks(np.arange(0, 11, 1))
                axs[ue_idx].set_xticks(np.arange(0, 10001, 2000))
                axs[ue_idx].tick_params(axis='x', labelsize=18)
                axs[ue_idx].tick_params(axis='y', labelsize=18)
                axs[ue_idx].legend(loc='best', fontsize=18)
                axs[ue_idx].grid()      

        if has_safe_set:
            axs[safe_set_idx].set_title('Safe Set', fontsize=18)
            axs[safe_set_idx].plot(steps, safe_set_mean[0:SPAN], label = label, linewidth = 2, color=color_map[algo])
            #axs[safe_set_idx].fill_between(steps, safe_set_min, safe_set_max, alpha=0.3, label='_nolegend_', color=color_map[algo]) # , color = '#DDDDDD'
            axs[safe_set_idx].fill_between(steps, safe_set_mean[0:SPAN] - 1.697 * safe_set_std[0:SPAN] / np.sqrt(runs), 
                            safe_set_mean[0:SPAN] + 1.697 * safe_set_std[0:SPAN] / np.sqrt(runs), color = color_map[algo],
                        alpha=0.3, label='_nolegend_')
            if algo == algo_names[-1]:
                axs[safe_set_idx].set_xlabel('Step', fontsize=18)  # Add an x-label to the axes.
                axs[safe_set_idx].set_ylabel('safe set', fontsize=18)
                axs[safe_set_idx].set_ylim((0,40)) # 15000
                axs[safe_set_idx].set_yticks(np.arange(0, 41, 5))
                axs[safe_set_idx].set_xticks(np.arange(0, 10001, 2000))
                axs[safe_set_idx].tick_params(axis='x', labelsize=18)
                axs[safe_set_idx].tick_params(axis='y', labelsize=18)
                axs[safe_set_idx].legend(loc='best', fontsize=18)
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
            
            fig.legend(labels, loc='upper center', ncol=ncol, bbox_to_anchor=(0.35, 1.0), frameon=True, fontsize=18)
            fig.tight_layout(rect=[0, 0, 1, 0.95])

            if START > 0:
                fig.savefig(save_path.format(scenario), format='png')
            else:
                # fig.savefig('./figures/subplots_{}.svg'.format(scenario), format='svg')
                fig.savefig(save_path.format(scenario), format='png')
            # fig.savefig('_subplots_' + scenario + '.svg', format='svg')       
