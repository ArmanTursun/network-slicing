#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on June 3, 2025

@author: Arman
"""

import numpy as np

'''proportional fair scheduler for average cqi reports'''
class ProportionalFair:
    def __init__(self, mcs_codeset, granularity = 2, slot_length = 1e-3, window = 50, sym_per_prb = 158):
        self.granularity = granularity
        self.mcs_codeset = mcs_codeset
        self.b = 1/window
        self.a = 1 - self.b
        self.sym_per_prb = sym_per_prb
        self.slot_length = slot_length

    def allocate(self, ues, n_prb, error_bound = 0.1):
        '''
        Updates the following variables of the ues:
        - ue.bits : assigned bits in this subframe
        - ue.prbs : assigned prbs in this subframe
        - ue.p : reception probability
        '''
        # create auxiliary data structures
        n_ues = len(ues)
        ue_rbs = np.zeros(n_ues, dtype = np.int32)
        ue_mcs = np.zeros(n_ues, dtype = np.int32)
        ue_queue = np.zeros(n_ues, dtype = np.int32)
        ue_rate = np.zeros(n_ues, dtype = np.int32)
        ue_bits = np.zeros(n_ues, dtype = np.int32)
        ue_th = np.zeros(n_ues)

        # extract ue information
        for i, ue in enumerate(ues):
            ue_th[i] = max(ue.th, 1) # to avoid division by zero
            ue_queue[i] = ue.queue
            # determine the mcs given the objective and the estimated snr
            ue_mcs[i], bits_per_sym = self.mcs_codeset.mcs_rate_vs_error(ue.e_snr, error_bound)
            # achievable rate for the ue
            ue_rate[i] = self.sym_per_prb * bits_per_sym
        
        # loop over the resources
        for r in range(0, n_prb, self.granularity):
            # prbs to be allocated in this iteration
            prbs = min(n_prb - r, self.granularity)

            # selected user for this resource (remove users without data)
            index = np.argmax(ue_rate * (ue_queue > 0)/ ue_th)
            
            # assign the resource to this ue
            ue_rbs[index] += prbs

            # update queue and throughput of this user
            tx_bits = min(prbs * ue_rate[index], ue_queue[index])
            ue_queue[index] -= tx_bits
            ue_bits[index] += tx_bits

            # update the estimated throughput with current allocation
            ue_th[index] = self.a * ue_th[index] + self.b * ue_bits[index] / self.slot_length

        # update ues
        prb_i = 0
        for i, ue in enumerate(ues):
            prbs = ue_rbs[i]
            ue.prbs = prbs
            ue.bits = ue_bits[i]
            if prbs:
                #snr_values = ue.snr[prb_i: prb_i + prbs]
                snr_values = [ue.e_snr]
                ue.p = self.mcs_codeset.response(ue_mcs[i], snr_values)
            else:
                ue.p = 0
            prb_i += prbs


class ProportionalFair_PowerSpreading:
    """
    A Proportional Fair (PF) scheduler that incorporates a realistic, non-monotonic
    throughput model based on UE power spreading.
    """
    def __init__(self, mcs_codeset, granularity=2, slot_length=1e-3, window=50, sym_per_prb=168):
        self.granularity = granularity
        self.mcs_codeset = mcs_codeset
        self.b = 1 / window
        self.a = 1 - self.b
        self.sym_per_prb = sym_per_prb
        self.slot_length = slot_length

    def _calculate_potential_bits(self, ue, n_prbs, base_snr_vector):
        """
        Calculates the total bits for a UE for a given number of PRBs,
        considering the UE's power constraint. It also returns the effective SINR
        values for each allocated PRB. This is the core non-monotonic logic.
        """
        if n_prbs == 0 or not hasattr(ue, 'max_tx_power_mW'):
            return 0, np.array([])

        # 1. Calculate Power per PRB based on the total allocation
        power_per_prb = ue.max_tx_power_mW / n_prbs

        # 2. A smart scheduler picks the PRBs with the best channel quality first.
        sorted_indices = np.argsort(base_snr_vector)[::-1]
        selected_prb_indices = sorted_indices[:n_prbs]

        total_bits = 0
        effective_sinr_values = []
        
        for i in selected_prb_indices:
            # 3. Calculate effective SINR for each selected PRB
            base_snr = base_snr_vector[i]
            effective_sinr_linear = power_per_prb * base_snr
            effective_sinr_values.append(effective_sinr_linear)

            # 4. Find the bits per symbol from the MCS codeset based on SINR
            bits_per_symbol = self.mcs_codeset.get_bits_from_sinr(effective_sinr_linear)
            total_bits += bits_per_symbol * self.sym_per_prb
            
        return total_bits, np.array(effective_sinr_values)

    def allocate(self, ues, n_prb, ue_snr_map, error_bound=0.1):
        """
        Allocates PRBs to UEs based on the PF metric, using the non-monotonic model.
        """
        n_ues = len(ues)
        if n_ues == 0:
            return

        # Initialize local arrays for this scheduling instance
        ue_rbs = np.zeros(n_ues, dtype=np.int32)
        ue_bits = np.zeros(n_ues, dtype=np.int32)
        ue_queue = np.array([ue.queue for ue in ues], dtype=np.int32)
        ue_th = np.array([max(ue.th, 1) for ue in ues])

        # Main resource allocation loop, assigning small chunks of PRBs at a time
        for r in range(0, n_prb, self.granularity):
            prbs_to_assign = min(n_prb - r, self.granularity)

            marginal_rates = np.zeros(n_ues)
            all_potential_bits = np.zeros(n_ues) # Store results to avoid recalculation

            # Evaluate the marginal gain for each UE if they were to receive this chunk
            for i, ue in enumerate(ues):
                if ue_queue[i] > 0:
                    current_bits = ue_bits[i]
                    potential_bits, _ = self._calculate_potential_bits(ue, ue_rbs[i] + prbs_to_assign, ue_snr_map[ue.id])
                    all_potential_bits[i] = potential_bits
                    
                    marginal_gain_bits = potential_bits - current_bits
                    marginal_rates[i] = marginal_gain_bits / self.slot_length

            if np.sum(marginal_rates) <= 0:
                continue # No user can benefit from more resources
            
            # Choose the user with the best PF metric (marginal rate / historic rate)
            best_ue_index = np.argmax(marginal_rates / ue_th)

            # Reuse the calculated value instead of recomputing
            new_total_bits = all_potential_bits[best_ue_index]

            # Determine how many bits to actually transmit (cannot exceed queue)
            assignable_bits = new_total_bits - ue_bits[best_ue_index]
            tx_bits_this_iter = min(assignable_bits, ue_queue[best_ue_index])

            # Update local state for the next scheduling iteration
            ue_rbs[best_ue_index] += prbs_to_assign
            ue_bits[best_ue_index] += tx_bits_this_iter
            ue_queue[best_ue_index] -= tx_bits_this_iter
            ue_th[best_ue_index] = self.a * ue_th[best_ue_index] + self.b * ue_bits[best_ue_index] / self.slot_length

        # Final loop to update the actual UE objects with the results of the scheduling
        for i, ue in enumerate(ues):
            ue.prbs = ue_rbs[i]
            ue.bits = ue_bits[i]

            if ue.prbs > 0:
                # Get the final effective SINR values for the user's total allocation
                _final_bits, final_sinrs = self._calculate_potential_bits(ue, ue.prbs, ue_snr_map[ue.id])
                
                # Determine the effective MCS that was used
                effective_bits_per_symbol = ue.bits / (ue.prbs * self.sym_per_prb)
                mcs_index = self.mcs_codeset.get_mcs_from_rate(effective_bits_per_symbol)

                # Calculate the final reception probability based on the real conditions
                ue.p = self.mcs_codeset.response(mcs_index, final_sinrs)
            else:
                ue.p = 0.0