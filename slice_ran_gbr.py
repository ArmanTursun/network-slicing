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
    def __init__(self, rng, user_counter, id, SLA, CBR_description, state_variables, norm_const, slots_per_step, slot_length = 1e-3):
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

        self.cbr_arrival_rate = CBR_description['lambda']
        self.cbr_mean_time = CBR_description['t_mean']
        self.cbr_bit_rate = CBR_description['bit_rate'][self.id]

        self.slot_counter = 0
        self.remaining_time = {}
        self.cbr_steps_next_arrival = 0
        self.cbr_ues = {}

        self.info = {'cbr_traffic': {}, 'cbr_th': {}, 'cbr_prb': {}, 'cbr_queue':{}, 'cbr_snr': {}}

        self.reset()

    def reset(self):
        
        self.reset_state()
        self.reset_info()

    def get_n_variables(self):
        return len(self.state_variables)

    def cbr_cac(self):
        '''Admission control for CBR users'''
        slots = max(self.slot_counter,1)
        time = slots * self.slot_length
        cbr_prb = sum(self.info['cbr_prb']) / len(self.info['cbr_prb']) / slots if len(self.info['cbr_prb']) > 0 else 0
        cbr_th = sum(self.info['cbr_th']) / len(self.info['cbr_th']) / time if len(self.info['cbr_th']) > 0 else 0
        if cbr_prb >= self.SLA['cbr_prb'] or cbr_th >= self.SLA['cbr_th'][self.id]:
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
                        self.info[key][ue.id] = 0

                #return [ue] # return user
                return ue_list 
        else:
            self.cbr_steps_next_arrival -= 1    
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
        return departures   

    def del_ue_from_info(self, id):
        for cur_info in self.info.keys():
            if id in self.info[cur_info]:
                self.info[cur_info].pop(id, None)
            else:
                print('ERROR: UE is not found in info')

    def slot(self):
        self.slot_counter += 1
        arrivals = self.cbr_arrivals()
        #arrivals.extend(self.vbr_arrivals())
        departures = self.departures()
        return arrivals, departures

    def reset_info(self):
        for key in self.info.keys():
            for ue in self.info[key].keys():
                self.info[key][ue] = 0
        #self.info = {'cbr_traffic': {}, 'cbr_th': {}, 'cbr_prb': {}, 'cbr_queue':{}, 'cbr_snr': {}}
        #self.slot_counter = 0

    def reset_state(self):
        self.state = np.full((len(self.state_variables)), 0, dtype = np.float64)
    
    def update_info(self):
        #queue = 0
        #snr = 0
        #n = 0
        for ue in self.cbr_ues.values(): # norm is total each step
            if ue.id not in self.info['cbr_traffic']:
                self.info['cbr_traffic'][ue.id] = ue.new_bits / self.norm_const['cbr_traffic'][self.id] 
            else:
                self.info['cbr_traffic'][ue.id] += ue.new_bits / self.norm_const['cbr_traffic'][self.id] 
            
            if ue.id not in self.info['cbr_th']:
                self.info['cbr_th'][ue.id] = ue.bits / self.norm_const['cbr_th'][self.id]
            else:
                self.info['cbr_th'][ue.id] += ue.bits / self.norm_const['cbr_th'][self.id] 
            
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
            #queue += ue.queue
            #snr += ue.e_snr
            #n += 1
        #n = max(n,1)
        #self.info['cbr_queue'] += (queue/n / self.norm_const['cbr_queue'])
        #self.info['cbr_snr'] += (snr/n / self.norm_const['cbr_snr']) 

    def compute_reward(self):
        '''assesses SLA violations''' # SLA normed total each step
        th_violation = 0
        prb_violation = 0
        queue_violation = 0
        for ue in self.info['cbr_th'].keys():
            if self.info['cbr_th'][ue] < self.SLA['cbr_th'][self.id]:
                th_violation += 1
        for ue in self.info['cbr_prb'].keys():
            if self.info['cbr_prb'][ue] < self.SLA['cbr_prb']:
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
        return total_violation

    def get_state(self):
        '''converts the info into a normalized vector'''
        for i, var in enumerate(self.state_variables):
            if var == 'cbr_traffic': # demand
                self.state[i] = round(self.SLA['cbr_th'][self.id], 2)
            elif var == 'cbr_ue': # minum throughput along ues
                self.state[i] = round(len(self.cbr_ues) / self.norm_const['cbr_ue'], 2)
            else: # average along ues
                all_val = [self.info[var][ue] for ue in self.info[var].keys()]    
                #avg_val = sum(all_val) / len(self.cbr_ues.values()) if len(self.cbr_ues.values()) > 0 else 0
                min_val = min(all_val) if all_val else 0
                self.state[i] = round(min_val, 2)        
        return self.state

