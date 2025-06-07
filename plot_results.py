#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 5, June 2025

@author: Arman

"""
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# trainning results
WINDOW = 50 #400
START = 0
END =  10000 # up to 39900  20000
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

if __name__=='__main__':
    try:
        scenario = int(sys.argv[1])
    except IndexError:
        scenario = 0

    if scenario not in scenarios:
        scenario = 0

    dir_path = './results/scenario_{}/'.format(scenario)
    algo_names = get_folder_names(dir_path)
    labels = algo_names
    print(algo_names)
    prbs = prbs_values[scenario]

    # subplot
    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(16, 4), constrained_layout=False)
    fig.subplots_adjust(top=0.78)
    # iterate over algorithms
    for algo, label in zip(algo_names, labels):
        violations = np.empty([1])
        actions = np.empty([1])
        rewards = np.empty([1])
        regret = np.empty([1])
        data = False
        proposal = False
        path = './results/scenario_{}/{}/'.format(scenario, algo)
        runs = 0

        # iterate over files
        for filename in os.listdir(path):
            if filename.endswith(".npz"):
                histories = np.load(path + filename)
                _violations = histories['violation']
                _resources = histories['resources']
                _rewards = histories['reward']
                if len(_violations) < END:
                    continue
                _violations = _violations[START:END]
                _resources = _resources[START:END]
                _rewards = _rewards[START:END]
                runs += 1
                # load data for each run
                if not data:
                    violations = average_per_window(_violations, WINDOW)
                    regret = average_per_window(_violations.cumsum(), WINDOW)
                    actions = average_per_window(_resources, WINDOW)
                    rewards = average_per_window(_rewards, WINDOW)
                    if proposal:
                        accuracy = average_per_window(np.mean(histories['hits'], axis=0), WINDOW)
                    data = True
                else: # store the history of each run
                    violations = np.vstack((violations, average_per_window(_violations, WINDOW)))
                    regret = np.vstack((regret, average_per_window(_violations.cumsum(), WINDOW)))
                    actions = np.vstack((actions, average_per_window(_resources, WINDOW)))
                    rewards = np.vstack((rewards, average_per_window(_rewards, WINDOW)))
                    if proposal:
                        accuracy = np.vstack((accuracy, average_per_window(np.mean(histories['hits'], axis=0), WINDOW)))
    
        print('Algorithm {}'.format(algo))
        
        # average over different runs

        actions_mean = np.mean(actions, axis=0)
        actions_std = np.std(actions, axis=0)

        violations_mean = np.mean(violations, axis=0)
        violations_std = np.std(violations, axis=0)

        rewards_mean = np.mean(rewards, axis=0)
        rewards_std = np.std(rewards, axis=0)

        regret_mean = np.mean(regret, axis=0)
        regret_std = np.std(regret, axis=0)

        if proposal:
            accuracy_mean = np.mean(accuracy, axis=0)
            accuracy_std = np.std(accuracy, axis=0)

        # plot results
        steps = np.arange(len(actions_mean[0:SPAN]))

        axs[2].set_title('Resource allocation', fontsize=14)
        axs[2].plot(steps, actions_mean[0:SPAN], label = label, linewidth = 2)
        axs[2].fill_between(steps, actions_mean - actions_std, actions_mean + actions_std, alpha=0.3, label='_nolegend_') # , color = '#DDDDDD'
        #axs[2].fill_between(steps, actions_mean[0:SPAN] - 1.697 * actions_std[0:SPAN] / np.sqrt(runs), 
        #                actions_mean[0:SPAN] + 1.697 * actions_std[0:SPAN] / np.sqrt(runs), color = '#DDDDDD')
        if algo == algo_names[-1]:
            axs[2].set_ylim((0,prbs))
            axs[2].set_xlabel('Epoch')  # Add an x-label to the axes.
            axs[2].set_ylabel('PRBs', fontsize=14)
            # axs[2].legend(loc='best')
            axs[2].grid()

        axs[0].set_title('SLA violations', fontsize=14)
        axs[0].plot(steps, violations_mean[0:SPAN], label = label, linewidth = 2)
        axs[0].fill_between(steps, violations_mean - violations_std, violations_mean + violations_std, alpha=0.3, label='_nolegend_') #, color = '#DDDDDD'
        #axs[0].fill_between(steps, violations_mean[0:SPAN] - 1.697 * violations_std[0:SPAN] / np.sqrt(runs), 
        #                violations_mean[0:SPAN] + 1.697 * violations_std[0:SPAN] / np.sqrt(runs), color = '#DDDDDD')
        if algo == algo_names[-1]:
            axs[0].set_xlabel('Epoch', fontsize=14)  # Add an x-label to the axes.
            axs[0].set_ylabel('SLA violations', fontsize=14)
            axs[3].set_ylim((0,3))
            #axs[0].legend(loc='best')
            axs[0].grid()
        
        axs[3].set_title('Rewards', fontsize=14)
        axs[3].plot(steps, rewards_mean[0:SPAN], label = label, linewidth = 2)
        axs[3].fill_between(steps, rewards_mean - rewards_std, rewards_mean + rewards_std, alpha=0.3, label='_nolegend_') # , color = '#DDDDDD'
        #axs[3].fill_between(steps, rewards_mean[0:SPAN] - 1.697 * rewards_std[0:SPAN] / np.sqrt(runs), 
        #                rewards_mean[0:SPAN] + 1.697 * rewards_std[0:SPAN] / np.sqrt(runs), color = '#DDDDDD')
        if algo == algo_names[-1]:
            axs[3].set_xlabel('Epoch', fontsize=14)  # Add an x-label to the axes.
            axs[3].set_ylabel('Reward', fontsize=14)
            axs[3].set_ylim((-10,50)) # 15000
            #axs[3].legend(loc='best')
            axs[3].grid()

        axs[1].set_title('Cumulative SLA violations', fontsize=14)
        axs[1].plot(steps, regret_mean[0:SPAN], label = label, linewidth = 2)
        axs[1].fill_between(steps, regret_mean - regret_std, regret_mean + regret_std, alpha=0.3, label='_nolegend_') # , color = '#DDDDDD'
        #axs[1].fill_between(steps, regret_mean[0:SPAN] - 1.697 * regret_std[0:SPAN] / np.sqrt(runs), 
        #                regret_mean[0:SPAN] + 1.697 * regret_std[0:SPAN] / np.sqrt(runs), color = '#DDDDDD')
        if algo == algo_names[-1]:
            axs[1].set_xlabel('Epoch', fontsize=14)  # Add an x-label to the axes.
            axs[1].set_ylabel('cumulative SLA violations', fontsize=14)
            axs[1].set_ylim((0,4000)) # 15000
            #axs[1].legend(loc='best')
            axs[1].grid()        
        
        if algo == algo_names[-1]:
            # Create a single legend above all subplots
            fig.legend(labels, loc='upper center', ncol=len(labels), bbox_to_anchor=(0.5, 1.0), frameon=True, fontsize=14)
            fig.tight_layout(rect=[0, 0, 1, 0.9])

            if START > 0:
                fig.savefig('./figures/_trained_subplots_{}.png'.format(scenario), format='png')
            else:
                # fig.savefig('./figures/subplots_{}.svg'.format(scenario), format='svg')
                fig.savefig('./figures/subplots_{}.png'.format(scenario), format='png')
            # fig.savefig('_subplots_' + scenario + '.svg', format='svg')       
