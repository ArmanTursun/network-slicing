#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Arman

create_env

"""

import gymnasium as gym
from itertools import count
from node_b import NodeB
from slice_l1 import SliceL1eMBB, SliceL1mMTC
from slice_ran_gbr import SliceRANmMTC, SliceRANeMBB
from schedulers import ProportionalFair
from channel_models import SINRSelectiveFading, MCSCodeset, SNRGenerator
from kbrl_control import KBRL_Control, Learner
from algorithms.kernel import GaussianKernel
from algorithms.projectron import SVvariable, Projectron

from qr_control import QR_Learner, QR_Control
from qr_util import KernelizedOnlineQuantileRegressor, SV, SimpleGaussianKernel

# ----------------- scenario parameters ------------------------

scenario_1 = { 'n_prbs': 80, 'n_embb': 3, 'n_mmtc': 0}
scenario_2 = { 'n_prbs': 150, 'n_embb': 3, 'n_mmtc': 2}
scenario_3 = { 'n_prbs': 100, 'n_embb': 1, 'n_mmtc': 4}
scenario_4 = { 'n_prbs': 70,  'n_embb': 1, 'n_mmtc': 1}
scenarios = [scenario_1, scenario_2, scenario_3, scenario_4]

# -------------------- eMBB parameters -------------------------

CBR_description = { # GBR traffic
#    'lambda': 1.0/60.0, # low traffic
    'lambda': 2.0/60.0, # UE arrivals: Poisson process with arrival rate = 2 users / min
    't_mean': 30.0, # UE connection time: Exponentially distributed with mean = 30 secs
    'bit_rate': 1.5e6 # Bit Rate 0.5MB/s
}

state_variables_embb = ['cbr_traffic', 'cbr_th', 'cbr_prb', 'cbr_queue', 'cbr_snr']

# -------------------- mMTC parameters -------------------------
# packet size 1000 bits
MTC_description = {
    'n_devices': 1000, # mmtc devices: 1000
    'repetition_set': [2,4,8,16,32,64,128],  # Packet repetitions
    'period_set': [1000, 50000, 10000, 15000, 20000, 25000, 50000, 100000] # transmission periods: seconds
}

state_variables_mmtc = ['devices', 'avg_rep', 'delay']

SLA_mmtc = {
    'delay': 300 # Maximum per user delay: 300ms
}

# -------------------- create environment -------------------------

def create_env(rng, all_scenarios = scenarios, n = 0, slots_per_step = 50, propagation_type = 'macro_cell_urban_2GHz', L1_level = True, penalty = 100):
    '''
    Returns slice ran environment:
    - rng: for random number generation
    - n: selects the scenario (0, 1, 2)
    '''
    time_per_step = slots_per_step * 1e-3

    sc = all_scenarios[n]
    n_prbs = sc['n_prbs']
    n_embb = sc['n_embb']
    n_mmtc = sc['n_mmtc']

    # -------------------- eMBB normalization constants ----------------------

    norm_const_embb = { # average in each step
        'cbr_traffic': 1.5e6 * time_per_step, # total traffic each step
        'cbr_th': 1.5e6 * time_per_step,
        'cbr_prb': n_prbs * slots_per_step, # total prb each step
        'cbr_queue': 100e4 * slots_per_step,
        'cbr_snr': 35 * slots_per_step,
    }

    # -------------------- mMTC normalization constants -----------------------

    norm_const_mmtc = {
        'devices': 100 * slots_per_step,
        'avg_rep': 100 * slots_per_step,
        'delay': 100 * slots_per_step
    }

    SLA_embb = { # overall
    'cbr_th': 1e6  * time_per_step / norm_const_embb['cbr_th'], # normed total throughput of slice each step?
    'cbr_prb': 20  * slots_per_step / norm_const_embb['cbr_prb'], # normed average 30  GBR authorized capacity 20 RBs/subframe
    'cbr_queue': 10e4 * slots_per_step / norm_const_embb['cbr_queue'], # normed average 5e4 Maximum average queue per GBR user: 100Kbit/UE
    'vbr_th': 10e4, # 10e6  # total throughput of slice ?
    'vbr_prb': 30, # 40 non-GBR QoS compliant capacity 30RBs/subframe
    'vbr_queue': 15e4 # Maximum average queue per non-GBR user: 150Kbit/UE
    }

    # ------------------- auxiliary functions -----------------------

    def new_slice_mmtc(id, rng):
        return SliceRANmMTC(rng, id, SLA_mmtc, MTC_description, state_variables_mmtc, norm_const_mmtc, slots_per_step)

    def new_slice_embb(id, rng, user_counter):
        return SliceRANeMBB(rng, user_counter, id, SLA_embb, CBR_description, state_variables_embb, norm_const_embb, slots_per_step)

    # ------------------- environment creation ------------------------

    snr_generator = SINRSelectiveFading(rng, propagation_type, n_prbs = n_prbs)

    mcs_codeset = MCSCodeset()

    scheduler = ProportionalFair(mcs_codeset)

    user_counter = count()

    slices_l1 = []

    if L1_level: # each slice has its own L1 resources
        index = 0
        for id in range(n_embb):
            slices_ran_embb = [new_slice_embb(id, rng, user_counter)]
            slice_l1_embb = SliceL1eMBB(rng, snr_generator, 20, slices_ran_embb, scheduler, l1sliceid = index)
            slices_l1.append(slice_l1_embb)
            index += 1

        for id in range(n_mmtc):
            slices_ran_mmtc = [new_slice_mmtc(id, rng)]
            slice_l1_mmtc = SliceL1mMTC(5, slices_ran_mmtc)
            slices_l1.append(slice_l1_mmtc)
            index += 1

    else: # slices are multiplexed in the L1 (the scheduler should handle ues from different slices) 

        slices_ran_embb = [new_slice_embb(id, rng, user_counter) for id in range(n_embb)]
        slice_l1_embb = SliceL1eMBB(rng, snr_generator, 20, slices_ran_embb, scheduler)
        slices_l1 = [slice_l1_embb]

        if n_mmtc > 0:
            slices_ran_mmtc = [new_slice_mmtc(id, rng) for id in range(n_mmtc)]
            slice_l1_mmtc = SliceL1mMTC(5, slices_ran_mmtc)
            slices_l1.append(slice_l1_mmtc)

    node = NodeB(slices_l1, slots_per_step, n_prbs) # create gNB

    node_env = gym.make('gym_ran_slice:RanSlice-v1', node_b = node, penalty = penalty)

    return node_env

# ------------ KBRL Learner initialization values ------------------

alfa = 0.05 # learning parameter

# initial offset and initial action are initialized at random
embb_sec = (2, 8)
embb_a = (2, 4)
mmtc_sec = (1, 4)
mmtc_a = (2, 10)

# -------------------- create KBRL agent -------------------------

def create_kbrl_agent(rng, n, accuracy_range = [0.99, 0.999]):
    '''
    Returns kbrl agent:
    - rng: for random number generation
    - n: selects the scenario (0, 1, 2)
    - accuracy_range: for the learner
    - budget: number of support vectors in memory
    '''
    sc = scenarios[n]
    n_prbs = sc['n_prbs']
    n_embb = sc['n_embb']
    n_mmtc = sc['n_mmtc']
    embb_dim = len(state_variables_embb)
    mmtc_dim = len(state_variables_mmtc)

    learners = [] 
    i = 0

    # create one learner instance per slice
    for _ in range(n_embb):
        sv = SVvariable() # create support vector memory
        kernel = GaussianKernel(sv,1) # kernel
        algorithm = Projectron(kernel) # online classifier
        initial_action = rng.integers(embb_a[0], embb_a[1])
        sec = rng.integers(embb_sec[0], embb_sec[1])
        learner = Learner(algorithm, slice(i,i+embb_dim), initial_action, sec)
        learners.append(learner)
        i += embb_dim

    for _ in range(n_mmtc):
        sv = SVvariable()
        kernel = GaussianKernel(sv,1)
        algorithm = Projectron(kernel)
        initial_action = rng.integers(mmtc_a[0], mmtc_a[1])
        sec = rng.integers(mmtc_sec[0], mmtc_sec[1])
        learner = Learner(algorithm, slice(i,i+mmtc_dim), initial_action, sec)
        learners.append(learner)
        i += mmtc_dim

    kbrl_agent = KBRL_Control(learners, n_prbs, alfa = alfa, accuracy_range = accuracy_range)

    return kbrl_agent


# ------------ QR Learner initialization values ------------------

# Initial random action range
embb_a = (2, 8)
mmtc_a = (2, 10)

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
        'cbr_traffic': 1.5e6 * time_per_step,
        'cbr_th': 1.5e6 * time_per_step,
        'cbr_prb': n_prbs * slots_per_step,
        'cbr_queue': 100e4 * slots_per_step,
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
        # The learning algorithm for this specific SLA
        # Input dimension is state_dim + 1 (for the action)
        # 1. Create the dedicated memory store (SV) for this learner
        sv_store = SV(dimension=embb_dim+1, budget=qr_params['budget'])

        # 2. Create the kernel object
        kernel = SimpleGaussianKernel(gamma=qr_params['gamma'])
        #algorithm = OnlineQuantileRegressor(input_dim=embb_dim + 1, quantile=quantile)
        algorithm = KernelizedOnlineQuantileRegressor(sv=sv_store, kernel=kernel, quantile=quantile, learning_rate=qr_params['learning_rate'])
        initial_action = rng.integers(embb_a[0], embb_a[1])
        
        # The learner holds the algorithm and the SLA definition
        learner = QR_Learner(
            algorithm=algorithm, 
            indexes=slice(i, i + embb_dim), 
            initial_action=initial_action,
            sla_threshold=embb_sla['threshold'] * time_per_step / norm_const_embb['cbr_th'],
            constraint_type=embb_sla['type'],
            kpi_key=embb_sla['kpi_key'], # Key to find the true KPI value from the env's info dict
            kpi_index = index
        )
        learners.append(learner)
        i += embb_dim
        index += 1

    # Create one learner instance per mMTC slice (with its success rate SLA)
    for slice_idx in range(n_mmtc):
        sv_store = SV(dimension=mmtc_dim+1, budget=qr_params['budget'])
        kernel = SimpleGaussianKernel(gamma=qr_params['gamma'])
        #algorithm = OnlineQuantileRegressor(input_dim=mmtc_dim + 1, quantile=quantile)
        algorithm = KernelizedOnlineQuantileRegressor(sv=sv_store, kernel=kernel, quantile=quantile, learning_rate=qr_params['learning_rate'])
        initial_action = rng.integers(mmtc_a[0], mmtc_a[1])
        
        learner = QR_Learner(
            algorithm=algorithm, 
            indexes=slice(i, i + mmtc_dim), 
            initial_action=initial_action,
            sla_threshold=mmtc_sla['threshold'],
            constraint_type=mmtc_sla['type'],
            kpi_key=mmtc_sla['kpi_key'], # Assumes unique keys for KPIs
            kpi_index = index
        )
        learners.append(learner)
        i += mmtc_dim

    # The main controller wraps all the individual learners
    qr_agent = QR_Control(rng, learners, n_prbs, exploration_factor=qr_params['exploration_factor'], resource_cost_factor=qr_params['resource_cost_factor'])

    return qr_agent