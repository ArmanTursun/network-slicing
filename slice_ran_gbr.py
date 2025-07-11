#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Arman
Classes:

UE
SliceRANmMTC
SliceRANeMBB

"""
DEBUG = True
CBR = 0
VBR = 1

import numpy as np
from traffic_generators import VbrSource, CbrSource

# You can place this class in a utility file or at the top of your environment script.
import numpy as np

class OnlineVariance:
    """
    Implements Welford's algorithm to compute mean and variance in a single pass.
    This avoids storing all data points in memory.
    """
    def __init__(self):
        self.count = 0
        self.mean = 0.0
        # M2 is used instead of S for the standard algorithm notation
        self.M2 = 0.0

    def update(self, new_value):
        self.count += 1
        delta = new_value - self.mean
        self.mean += delta / self.count
        delta2 = new_value - self.mean
        self.M2 += delta * delta2

    def reset(self):
        """ Resets the tracker to its initial state. """
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0

    @property
    def variance(self):
        """ Returns the final population variance. """
        if self.count < 2:
            return 0.0
        return self.M2 / self.count

    @property
    def std_dev(self):
        """ Returns the final population standard deviation. """
        return np.sqrt(self.variance)

class UE:
    '''
    eMBB UE contains a traffic source that can be CRB (GBR) or VBR (non-GBR)
    '''
    def __init__(self, id, slice_ran_id, traffic_source, type, window = 50, slot_length = 1e-3):
        self.id = id
        self.slice_ran_id = slice_ran_id
        self.traffic_source = traffic_source
        self.type = type
        self.th = 0
        self.b = 1/window
        self.a = 1 - self.b
        self.queue = 0
        self.slot_length = slot_length
        self.max_tx_power_mW = 20 # Example: 200mW is 23 dBm

        # per subframe variables
        self.snr = 0 # real error values per prb
        self.e_snr = 0 # estimated error
        self.new_bits = 0 # incoming bits
        self.bits = 0 # assigned bits
        self.prbs = 0 # assigned prbs
        self.p = 0 # reception probability
    
    def estimate_snr(self, snr):
        self.snr = snr
        self.e_snr = round(np.mean(snr))
        snr_noise = 0
        #snr_noise = np.random.randint(-5, 6) # SNR suddenly change
        self.e_snr += snr_noise

    def traffic_step(self):
        self.new_bits = self.traffic_source.step()
        self.queue += self.new_bits
    
    def transmission_step(self, received):
        if not received:
            self.bits = 0
        self.queue = max(self.queue - self.bits, 0)
        self.th = self.a * self.th + self.b * self.bits / self.slot_length

    def __repr__(self):
        return 'UE {}'.format(self.id)

class MTCdevice:
    def __init__(self, id, repetitions, slice_ran_id):
        self.id = id
        self.repetitions = repetitions
        self.slice_ran_id = slice_ran_id
    def __repr__(self):
        return 'MTC {}'.format(self.id)

class SliceRANmMTC:
    '''
    Generates message arrivals at the mMTC devices
    according to the characteristics defined in MTC_description:
    - n_devices: total number of devices
    - repetition_set: possible repetitions
    - period_set: possible times between message arrivals
    '''
    def __init__(self, rng, id, SLA, MTCdescription, state_variables, norm_const, slots_per_step):
        self.type = 'mMTC'
        self.rng = rng
        self.id = id
        self.SLA = SLA
        self.state_variables = state_variables # ['devices', 'avg_rep', 'delay']
        self.norm_const = norm_const # 100 all
        self.slots_per_step = slots_per_step

        self.n_devices = MTCdescription['n_devices']
        self.repetition_set = MTCdescription['repetition_set']
        self.period_set = MTCdescription['period_set']

        self.reset()

    def reset(self):
        self.reset_state()
        self.reset_info()
        self.period = np.ones((self.n_devices), dtype=np.int64)
        self.t_to_arrival = np.zeros((self.n_devices), dtype=np.int64)
        self.devices = []
        for i in range(self.n_devices):
            repetitions = self.rng.choice(self.repetition_set)
            self.period[i] = self.rng.choice(self.period_set)
            self.t_to_arrival[i] = 1 + self.rng.choice(np.arange(self.period[i]))
            self.devices.append(MTCdevice(i, repetitions, self.id))

    def slot(self):
        self.slot_counter += 1

        # advance time
        self.t_to_arrival -= 1

        # arrivals
        arrival_list = []
        arrivals = self.t_to_arrival == 0
        indices = np.where(arrivals)

        # print('indices = {}'.format(indices))
        for i in indices[0]:
            arrival_list.append(self.devices[i])

        # prepare for next arrival (deterministic inter arrival time)
        self.t_to_arrival[arrivals] = self.period[arrivals]

        return arrival_list, []

    def reset_info(self):
        self.info = {'delay': 0, 'avg_rep': 0, 'devices': 0}
        self.slot_counter = 0

    def reset_state(self):
        self.state = np.full((len(self.state_variables)), 0, dtype = np.float64)

    def get_n_variables(self):
        return len(self.state_variables)

    def get_state(self):
        '''convert the info into a normalized vector'''
        for i, var in enumerate(self.state_variables):
            self.state[i] = self.info[var] / self.norm_const[var]        
        return self.state

    def update_info(self, delay, avg_rep, devices):
        self.info['delay'] += delay
        self.info['avg_rep'] += avg_rep
        self.info['devices'] += devices
        

    def compute_reward(self):
        '''assesses SLA violations'''
        SLA_fulfilled = self.info['delay']/self.slots_per_step < self.SLA['delay']
        return not(SLA_fulfilled)

class SliceRANeMBB:
    '''
    Generates arrivals and departures of eMBB ues.
    CBR traffic parameters are given in CBR_description
    '''
    def __init__(self, rng, user_counter, id, SLA, CBR_description, state_variables, norm_const, slots_per_step, quantile = 5, slot_length = 1e-3):
        self.type = 'eMBB'
        self.rng = rng
        self.user_counter = user_counter
        self.id = id
        self.slot_length = slot_length
        self.slots_per_step = slots_per_step
        self.observation_time = slots_per_step * slot_length
        self.SLA = SLA # service level agreement description
        self.state_variables = state_variables
        self.norm_const = norm_const
        self.quantile = quantile

        self.cbr_arrival_rate = CBR_description['lambda']
        self.cbr_mean_time = CBR_description['t_mean']
        self.cbr_bit_rate = CBR_description['bit_rate'][self.id]

        self.slot_counter = 0
        self.remaining_time = {}
        self.cbr_steps_next_arrival = 0
        self.cbr_ues = {}

        self.info = {'cbr_traffic': {}, 'cbr_th': {}, 'fair_cbr_prb': 0, 'starve_cbr_prb': 0, 'cbr_prb': {}, 'cbr_queue':{}, 'cbr_snr': {}, 'cbr_ue': 0}
        self.ue_time = {}

        self.reset()

    def reset(self):
        
        self.reset_state()
        self.reset_info()

    def get_n_variables(self):
        return len(self.state_variables)

    def cbr_cac(self):
        '''Admission control for CBR users'''
        slots = max(self.slot_counter,1)
        #time = slots * self.slot_length
        cbr_prb = sum(self.info['cbr_prb']) / slots
        #cbr_th = sum(self.info['cbr_th']) / time if len(self.info['cbr_th']) > 0 else 0
        if cbr_prb >= self.SLA['cbr_prb'][self.id] or len(self.cbr_ues) >= 5: #or cbr_th >= self.SLA['cbr_th'][self.id]:
            return False
        return True

    def cbr_arrivals(self):
        
        if self.cbr_steps_next_arrival == 0:
            
            # generate next arrival
            inter_arrival_time = self.rng.exponential(1.0 / self.cbr_arrival_rate)
            inter_arrival_time = np.rint(inter_arrival_time / self.slot_length)
            
            self.cbr_steps_next_arrival = inter_arrival_time

            if self.cbr_cac(): # check admission control
                # generate new user
                ue_list = []
                for i in range(1): # 3 ues per slice
                    ue_id = next(self.user_counter)
                    cbr_source = CbrSource(bit_rate = self.cbr_bit_rate)
                    #print(self.cbr_bit_rate)
                    ue = UE(ue_id, self.id, cbr_source, CBR, window = self.slots_per_step)
                    self.cbr_ues[ue_id] = ue

                    # generate holding time
                    holding_time = self.rng.exponential(self.cbr_mean_time)
                    holding_time = np.rint(holding_time / self.slot_length)
                    self.remaining_time[ue_id] = holding_time
                    ue_list.append(ue)
                    for key in self.info.keys():
                        if key != 'fair_cbr_prb' and key != 'starve_cbr_prb' and key != 'cbr_ue':
                            self.info[key][ue.id] = 0
                    self.info['cbr_ue'] += 1
                    self.ue_time[ue.id] = 1

                #return [ue] # return user
                return ue_list 
        else:
            self.cbr_steps_next_arrival -= 1    
            for ue_id in self.ue_time.keys():
                self.ue_time[ue_id] += 1
        return []

    def departures(self):
        departures = []
        current_ids = list(self.remaining_time.keys())
        for id in current_ids:
            self.remaining_time[id] -= 1 # assume ue does not leave
            if self.remaining_time[id] == 0:
                departures.append(id)
                del self.remaining_time[id] # delete timer
                self.cbr_ues.pop(id, None) # or here  
                self.del_ue_from_info(id)  
                self.ue_time.pop(id, None) # or here
        return departures   

    def del_ue_from_info(self, id):
        for cur_info in self.info.keys():
            if cur_info != 'fair_cbr_prb' and cur_info != 'starve_cbr_prb' and cur_info != 'cbr_ue':
                if id in self.info[cur_info]:                
                    self.info[cur_info].pop(id, None)
                else:
                    print('ERROR: UE is not found in info')
        self.info['cbr_ue'] -= 1

    def slot(self):
        self.slot_counter += 1
        arrivals = self.cbr_arrivals()
        #arrivals.extend(self.vbr_arrivals())
        departures = self.departures()
        return arrivals, departures

    def reset_info(self):
        for key in self.info.keys():
            if key == 'cbr_ue':
                continue
            elif key == 'fair_cbr_prb' or key == 'starve_cbr_prb':
                self.info[key] = 0
            else:
                for ue in self.info[key].keys():
                    self.info[key][ue] = 0
        for ue_id in self.ue_time.keys():
            self.ue_time[ue_id] = 0

    def reset_state(self):
        self.state = np.full((len(self.state_variables)), 0, dtype = np.float64)
    
    def update_info(self):
        #queue = 0
        #snr = 0
        #n = 0
        prbs_this_slot = []
        for ue in self.cbr_ues.values(): # norm is total each step
            if ue.id not in self.info['cbr_traffic']:
                self.info['cbr_traffic'][ue.id] = ue.new_bits / self.norm_const['cbr_traffic'][self.id] 
            else:
                self.info['cbr_traffic'][ue.id] += ue.new_bits / self.norm_const['cbr_traffic'][self.id] 
            
            if ue.id not in self.info['cbr_th']:
                self.info['cbr_th'][ue.id] = ue.bits / self.norm_const['cbr_th'][self.id]
            else:
                self.info['cbr_th'][ue.id] += ue.bits / self.norm_const['cbr_th'][self.id] 
            
            prbs_this_slot.append(ue.prbs)
            if ue.id not in self.info['cbr_prb']:
                self.info['cbr_prb'][ue.id] = ue.prbs / self.norm_const['cbr_prb'] 
            else:
                self.info['cbr_prb'][ue.id] += ue.prbs / self.norm_const['cbr_prb'] 
            
            if ue.id not in self.info['cbr_queue']:
                self.info['cbr_queue'][ue.id] = ue.queue / self.norm_const['cbr_queue']
            else:
                self.info['cbr_queue'][ue.id] += ue.queue / self.norm_const['cbr_queue']
            
            if ue.id not in self.info['cbr_snr']:
                self.info['cbr_snr'][ue.id] = ue.e_snr / self.norm_const['cbr_snr']
            else:
                self.info['cbr_snr'][ue.id] += ue.e_snr / self.norm_const['cbr_snr']

        current_fairness = np.std(prbs_this_slot) if len(prbs_this_slot) > 0 else 0
        current_starvation = np.percentile(prbs_this_slot, 5) if len(prbs_this_slot) > 0 else 0
        self.info['fair_cbr_prb'] += current_fairness / self.norm_const['cbr_prb']
        self.info['starve_cbr_prb'] += current_starvation / self.norm_const['cbr_prb']

    def compute_reward(self):
        '''assesses SLA violations''' # SLA normed total each step
        th_violation = 0
        prb_violation = 0
        queue_violation = 0
        for ue in self.info['cbr_th'].keys():
            cur_time = self.ue_time[ue] / self.slots_per_step
            if self.info['cbr_th'][ue] / cur_time < self.SLA['cbr_th'][self.id]:
                th_violation += 1
        for ue in self.info['cbr_prb'].keys():
            if self.info['cbr_prb'][ue] < self.SLA['cbr_prb'][self.id]:
                prb_violation += 1
        for ue in self.info['cbr_queue'].keys():
            if self.info['cbr_queue'][ue] > self.SLA['cbr_queue']:
                queue_violation += 1
        total_violation = th_violation #+ queue_violation #+ prb_violation
        #cbr_th = self.info['cbr_th'] >= self.SLA['cbr_th']
        #cbr_prb = self.info['cbr_prb'] > self.SLA['cbr_prb']
        #print(self.info['cbr_traffic']/self.observation_time, self.info['cbr_th']/self.observation_time, self.info['cbr_prb']/self.slots_per_step)
        #cbr_queue = self.info['cbr_queue'] < self.SLA['cbr_queue']
        # the slice has to guarantee the objective delay for cbr and vbr if their traffics do not surpass the maximum     
        #cbr_fulfilled = cbr_th #or cbr_queue #or cbr_prb 
        #SLA_fulfilled = cbr_fulfilled
        return 1 if total_violation > 0 else 0

    def get_state(self):
        '''converts the info into a normalized vector'''
        # ['5th_cbr_th', '50th_cbr_th', 'std_cbr_th', 'fair_cbr_prb', 
        # 'starve_cbr_prb', 'cbr_queue', 'cbr_snr', 'cbr_ue'] #  
        for i, var in enumerate(self.state_variables):
            all_val = np.full(len(self.cbr_ues), 0, dtype = np.float64)
            if var == 'cbr_ue': 
                self.state[i] = self.info[var] / self.norm_const['cbr_ue']
            elif var == 'cbr_prb':
                for j, ue in enumerate(self.info[var].keys()):
                    all_val[j] = self.info[var][ue] 
                    self.state[i] = np.sum(all_val)
            elif var == 'fair_cbr_prb':
                self.state[i] = self.info[var]
            elif var == 'starve_cbr_prb':
                self.state[i] = self.info[var]
            elif var == 'cbr_queue':
                for j, ue in enumerate(self.info[var].keys()):
                    all_val[j] = self.info[var][ue] 
                self.state[i] = np.percentile(all_val, 95) if len(all_val) > 0 else 0
            elif var == '5th_cbr_th':
                for j, ue in enumerate(self.info['cbr_th'].keys()):
                    cur_time = self.ue_time[ue] / self.slots_per_step
                    all_val[j] = self.info['cbr_th'][ue] / cur_time
                self.state[i] = np.percentile(all_val, 5) if len(all_val) > 0 else 0
            elif var == '50th_cbr_th':
                for j, ue in enumerate(self.info['cbr_th'].keys()):
                    cur_time = self.ue_time[ue] / self.slots_per_step
                    all_val[j] = self.info['cbr_th'][ue] / cur_time
                self.state[i] = np.percentile(all_val, 50) if len(all_val) > 0 else 0
            elif var == 'std_cbr_th':
                for j, ue in enumerate(self.info['cbr_th'].keys()):
                    cur_time = self.ue_time[ue] / self.slots_per_step
                    all_val[j] = self.info['cbr_th'][ue] / cur_time
                self.state[i] = np.std(all_val) if len(all_val) > 0 else 0
            else: # average along ues
                for j, ue in enumerate(self.info[var].keys()):
                    all_val[j] = self.info[var][ue] 
                    self.state[i] = np.percentile(all_val, 50) if len(all_val) > 0 else 0
                #avg_val = sum(all_val) / len(self.cbr_ues.values()) if len(self.cbr_ues.values()) > 0 else 0
                #min_val = min(all_val) if all_val else 0
                #self.state[i] = round(min_val, 2)        
        return self.state

