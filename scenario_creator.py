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
from schedulers import ProportionalFair, ProportionalFair_PowerSpreading
from channel_models import SINRSelectiveFading, MCSCodeset, SNRGenerator


# ----------------- scenario parameters ------------------------

scenario_1 = { 'n_prbs': 100, 'n_embb': 3, 'n_mmtc': 0}
scenario_2 = { 'n_prbs': 150, 'n_embb': 3, 'n_mmtc': 2}
scenario_3 = { 'n_prbs': 100, 'n_embb': 1, 'n_mmtc': 4}
scenario_4 = { 'n_prbs': 70,  'n_embb': 1, 'n_mmtc': 1}
scenarios = [scenario_1, scenario_2, scenario_3, scenario_4]

# -------------------- eMBB parameters -------------------------

CBR_description = { # GBR traffic
#    'lambda': 1.0/60.0, # low traffic
    'lambda': 1.0/60.0, # UE arrivals: Poisson process with arrival rate = 2 users / min
    't_mean': 60.0, # UE connection time: Exponentially distributed with mean = 30 secs
    'bit_rate': [0.6e6, 0.6e6, 1.6e6] # slightly larger than SLA
}

state_variables_embb = ['5th_cbr_th', 'cbr_queue', 'cbr_snr', 'cbr_ue'] # , 'cbr_queue' , 'cbr_prb', '50th_cbr_th', 'std_cbr_th', 'fair_cbr_prb', 'starve_cbr_prb' , 'cbr_prb'

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

def create_env(rng, all_scenarios = scenarios, n = 0, slots_per_step = 50, 
               propagation_type = 'macro_cell_urban_2GHz', L1_level = True, penalty = 100, quantile = 5):
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
        'cbr_traffic': [0.75e6 * time_per_step, 0.75e6 * time_per_step, 1.25e6 * time_per_step],
        'cbr_th': [0.75e6 * time_per_step, 0.75e6 * time_per_step, 1.256 * time_per_step],
        'cbr_prb': n_prbs * slots_per_step, # total prb each step
        'cbr_queue': 10e4 * slots_per_step,
        'cbr_snr': 35 * slots_per_step,
        'cbr_ue': 10
    }

    # -------------------- mMTC normalization constants -----------------------

    norm_const_mmtc = {
        'devices': 100 * slots_per_step,
        'avg_rep': 100 * slots_per_step,
        'delay': 100 * slots_per_step
    }

    SLA_embb = { # overall
    'cbr_th': [0.5e6  * time_per_step / norm_const_embb['cbr_th'][0], 0.5e6  * time_per_step / norm_const_embb['cbr_th'][1], 1e6  * time_per_step / norm_const_embb['cbr_th'][2]], # normed total throughput of slice each step?
    'cbr_prb': [15  * slots_per_step / norm_const_embb['cbr_prb'], 20  * slots_per_step / norm_const_embb['cbr_prb'], 35  * slots_per_step / norm_const_embb['cbr_prb']], # normed average 30  GBR authorized capacity 20 RBs/subframe
    'cbr_queue': 10e4 * slots_per_step / norm_const_embb['cbr_queue'], # normed average 5e4 Maximum average queue per GBR user: 100Kbit/UE
    'vbr_th': 10e4, # 10e6  # total throughput of slice ?
    'vbr_prb': 30, # 40 non-GBR QoS compliant capacity 30RBs/subframe
    'vbr_queue': 15e4 # Maximum average queue per non-GBR user: 150Kbit/UE
    }

    # ------------------- auxiliary functions -----------------------

    def new_slice_mmtc(id, rng):
        return SliceRANmMTC(rng, id, SLA_mmtc, MTC_description, state_variables_mmtc, norm_const_mmtc, slots_per_step)

    def new_slice_embb(id, rng, user_counter):
        return SliceRANeMBB(rng, user_counter, id, SLA_embb, CBR_description, 
                            state_variables_embb, norm_const_embb, slots_per_step, quantile = quantile)

    # ------------------- environment creation ------------------------

    snr_generator = SINRSelectiveFading(rng, propagation_type, n_prbs = n_prbs)

    mcs_codeset = MCSCodeset()

    scheduler = ProportionalFair(mcs_codeset)
    #scheduler = ProportionalFair_PowerSpreading(mcs_codeset)

    user_counter = count()

    slices_l1 = []

    if L1_level: # each slice has its own L1 resources
        index = 0
        for id in range(n_embb):
            slices_ran_embb = [new_slice_embb(id, rng, user_counter)]
            slice_l1_embb = SliceL1eMBB(rng, snr_generator, 0, slices_ran_embb, scheduler, l1sliceid = index)
            slices_l1.append(slice_l1_embb)
            index += 1

        for id in range(n_mmtc):
            slices_ran_mmtc = [new_slice_mmtc(id, rng)]
            slice_l1_mmtc = SliceL1mMTC(5, slices_ran_mmtc)
            slices_l1.append(slice_l1_mmtc)
            index += 1

    else: # slices are multiplexed in the L1 (the scheduler should handle ues from different slices) 

        slices_ran_embb = [new_slice_embb(id, rng, user_counter) for id in range(n_embb)]
        slice_l1_embb = SliceL1eMBB(rng, snr_generator, 0, slices_ran_embb, scheduler)
        slices_l1 = [slice_l1_embb]

        if n_mmtc > 0:
            slices_ran_mmtc = [new_slice_mmtc(id, rng) for id in range(n_mmtc)]
            slice_l1_mmtc = SliceL1mMTC(5, slices_ran_mmtc)
            slices_l1.append(slice_l1_mmtc)

    node = NodeB(slices_l1, slots_per_step, n_prbs) # create gNB

    node_env = gym.make('gym_ran_slice:RanSlice-v1', node_b = node, penalty = penalty)

    return node_env