#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ArmanTursun

create_agent

"""

from bqr_control import BQR_Learner, BQR_Control
from bqr_util import BayesianQuantileRegressor_SGLD, MaternKernel

state_variables_embb = ['cbr_traffic', 'cbr_th', 'cbr_prb', 'cbr_queue', 'cbr_snr']
state_variables_mmtc = ['devices', 'avg_rep', 'delay']

# Initial random action range
embb_a = (2, 8)
mmtc_a = (2, 10)

# -------------------- create QR agent -------------------------

def create_bqr_agent(rng, n, scenarios, quantile, embb_sla, mmtc_sla, qr_params, slots_per_step):
    '''
    Returns a QR agent:
    - rng: for random number generation
    - n: selects the scenario (0, 1, 2)
    - state_variables_*: defines the state dimensions for each slice type
    - scenarios: list of scenario configurations
    '''
    time_per_step = slots_per_step * 1e-3

    sc = scenarios[n]
    n_prbs = sc['n_prbs']
    n_embb = sc['n_embb']
    n_mmtc = sc['n_mmtc']
    embb_dim = len(state_variables_embb)
    mmtc_dim = len(state_variables_mmtc)
    buffer_capacity = qr_params['budget'] * 10

    # -------------------- normalization constants ----------------------

    norm_const_embb = { # average in each step
        'cbr_traffic': 1.5e6 * time_per_step,
        'cbr_th': 1.5e6 * time_per_step,
        'cbr_prb': n_prbs * slots_per_step,
        'cbr_queue': 10e4 * slots_per_step,
        'cbr_snr': 35 * slots_per_step,
    }
    norm_const_mmtc = {
        'devices': 100 * slots_per_step,
        'avg_rep': 100 * slots_per_step,
        'delay': 100 * slots_per_step
    }

    learners = [] 
    i = 0
    index = 0
    # Create one learner instance per eMBB slice (with its delay SLA)
    for slice_idx in range(n_embb):
        input_dim = embb_dim + 1
        # Create the advanced Matérn kernel
        kernel = MaternKernel(length_scale=qr_params['gamma'], nu=2.5) # We can reuse 'gamma' as length_scale
        # Create the new Bayesian learning algorithm
        algorithm = BayesianQuantileRegressor_SGLD(
            input_dim=input_dim,
            budget=qr_params['budget'],
            kernel=kernel,
            quantile=quantile,
            learning_rate=qr_params['learning_rate']
        )        
        initial_action = rng.integers(embb_a[0], embb_a[1])       
        learner = BQR_Learner(
            algorithm=algorithm, 
            indexes=slice(i, i + embb_dim), 
            initial_action=initial_action,
            sla_threshold=embb_sla['threshold'] * time_per_step / norm_const_embb['cbr_th'], # Assumes normalization is handled elsewhere
            constraint_type=embb_sla['type'],
            kpi_key=embb_sla['kpi_key'],
            kpi_index=index,
            buffer_capacity=buffer_capacity
        )
        learners.append(learner)
        i += embb_dim
        index += 1

    # Create one learner instance per mMTC slice (with its success rate SLA)
    for slice_idx in range(n_mmtc):
        input_dim = mmtc_dim + 1
        # Create the advanced Matérn kernel
        kernel = MaternKernel(length_scale=qr_params['gamma'], nu=2.5) # We can reuse 'gamma' as length_scale
        # Create the new Bayesian learning algorithm
        algorithm = BayesianQuantileRegressor_SGLD(
            input_dim=input_dim,
            budget=qr_params['budget'],
            kernel=kernel,
            quantile=quantile,
            learning_rate=qr_params['learning_rate']
        )        
        initial_action = rng.integers(mmtc_a[0], mmtc_a[1])       
        learner = BQR_Learner(
            algorithm=algorithm, 
            indexes=slice(i, i + mmtc_dim), 
            initial_action=initial_action,
            sla_threshold=mmtc_sla['threshold'], # Assumes normalization is handled elsewhere
            constraint_type=mmtc_sla['type'],
            kpi_key=mmtc_sla['kpi_key'],
            kpi_index=index,
            buffer_capacity=buffer_capacity
        )
        learners.append(learner)
        i += mmtc_dim
        index += 1

    # The main controller wraps all the individual learners
    bqr_agent = BQR_Control(
        rng, 
        learners, 
        n_prbs,
        buffer_capacity=qr_params['budget'] * 10, # e.g., buffer is 10x the SV budget
        exploration_factor=qr_params['exploration_factor'], 
        resource_cost_factor=qr_params['resource_cost_factor']
    )

    return bqr_agent
