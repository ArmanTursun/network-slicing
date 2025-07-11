#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ArmanTursun

create_agent

"""

# ------------ QR Learner initialization values ------------------
from qr_control import QR_Learner, QR_Control
from qr_util import KernelizedOnlineQuantileRegressor, SV, SimpleGaussianKernel, MaternKernel
from qr_dqrrn import DeepQuantileRegressor 
from qr_esn import EchoStateNetworkRegressor

# Initial random action range
embb_a = (0, 1)
mmtc_a = (0, 1)

state_variables_embb = ['5th_cbr_th', 'cbr_queue', 'cbr_snr', 'cbr_ue'] #  , '50th_cbr_th', 'std_cbr_th', 'fair_cbr_prb', 'starve_cbr_prb' , 'cbr_prb'
state_variables_mmtc = ['devices', 'avg_rep', 'delay']

# -------------------- create QR agent -------------------------

def create_qr_agent(rng, n, scenarios, quantile, embb_sla, mmtc_sla, qr_params, slots_per_step):
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

    # -------------------- normalization constants ----------------------

    norm_const_embb = { # average in each step
        'cbr_traffic': [0.75e6 * time_per_step, 0.75e6 * time_per_step, 1.25e6 * time_per_step],
        'cbr_th': [0.75e6 * time_per_step, 0.75e6 * time_per_step, 1.25e6 * time_per_step],
        'cbr_prb': n_prbs * slots_per_step,
        'cbr_queue': 10e4 * slots_per_step,
        'cbr_snr': 35 * slots_per_step,
        'cbr_ue': 10
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
        # The learning algorithm for this specific SLA
        # Input dimension is state_dim + 1 (for the action)
        # 1. Create the dedicated memory store (SV) for this learner
        input_dim = embb_dim+1+2
        sv_store = SV(dimension=input_dim, budget=qr_params['budget'])

        # 2. Create the kernel object
        #kernel = SimpleGaussianKernel(gamma=qr_params['gamma'])
        kernel = MaternKernel(length_scale=qr_params['matern_length_scale'], nu=qr_params['matern_nu'])
        algorithm = KernelizedOnlineQuantileRegressor(sv=sv_store, kernel=kernel, quantile=quantile, 
                                                      learning_rate=qr_params['learning_rate'], 
                                                      gradient_penalty = qr_params['gradient_penalty'])
        # --- MODIFIED: Instantiate the new deep learner ---
        #algorithm = DeepQuantileRegressor(
        #        input_dim=input_dim,
        #        quantile=quantile, # Initial quantile
        #        learning_rate=qr_params['learning_rate'] # Use a smaller LR for NNs
        #)

        #algorithm = EchoStateNetworkRegressor(
        #    input_dim=input_dim,
        #    reservoir_size=qr_params.get('reservoir_size', 200),
        #    spectral_radius=qr_params.get('spectral_radius', 0.95),
        #    learning_rate=qr_params.get('esn_learning_rate', 0.8)
        #)

        # The learner holds the algorithm and the SLA definition
        learner = QR_Learner(
            algorithm=algorithm, 
            indexes=slice(i, i + embb_dim), 
            sla_threshold=embb_sla['threshold'][slice_idx] * time_per_step / norm_const_embb['cbr_th'][slice_idx],
            constraint_type=embb_sla['type'],
            kpi_key=embb_sla['kpi_key'], # Key to find the true KPI value from the env's info dict
            kpi_index = index
        )
        learners.append(learner)
        i += embb_dim
        index += 1

    # Create one learner instance per mMTC slice (with its success rate SLA)
    for slice_idx in range(n_mmtc):
        sv_store = SV(dimension=mmtc_dim+1+2, budget=qr_params['budget'])
        kernel = SimpleGaussianKernel(gamma=qr_params['gamma'])
        algorithm = KernelizedOnlineQuantileRegressor(sv=sv_store, kernel=kernel, quantile=quantile, 
                                                      learning_rate=qr_params['learning_rate'],
                                                      gradient_penalty = qr_params['gradient_penalty'])
        
        learner = QR_Learner(
            algorithm=algorithm, 
            indexes=slice(i, i + mmtc_dim), 
            sla_threshold=mmtc_sla['threshold'],
            constraint_type=mmtc_sla['type'],
            kpi_key=mmtc_sla['kpi_key'], # Assumes unique keys for KPIs
            kpi_index = index
        )
        learners.append(learner)
        i += mmtc_dim

    # The main controller wraps all the individual learners
    qr_agent = QR_Control(rng, learners, n_prbs, state_variables_embb, norms = norm_const_embb, 
                          exploration_factor=qr_params['exploration_factor'], resource_cost_factor=qr_params['resource_cost_factor'], 
                          epsilon = qr_params['epsilon'], k = qr_params['k_neighbors'],
                          adjustment_penalty = qr_params['adjustment_penalty'])

    return qr_agent