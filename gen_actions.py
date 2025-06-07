"""
@author: Arman

This script generates action space

"""

import numpy as np
import math

def generate_random_actions(n_actions=10, n_slices=5, total_prbs=200):
    actions = []
    for _ in range(n_actions):
        # Choose a random target sum ≤ total_prbs
        target = np.random.randint(10, total_prbs + 1)
        
        # Generate random floats, normalize, scale to target
        raw = np.random.randint(low = 2, high = 10, size = n_slices)
        if raw.sum() == 0:
            raw += 1e-6  # avoid division by zero
        normalized = raw / raw.sum()
        prb_allocation = np.ceil(normalized * target).astype(np.int32)

        # Adjust for rounding error
        diff = target - prb_allocation.sum()
        for i in np.argsort(-normalized):
            if diff <= 0:
                break
            prb_allocation[i] += 1
            diff -= 1

        actions.append(prb_allocation)
    return actions

def _generate_recursive_combinations_le_multiple_of(
    slice_idx,
    current_sum,
    combination_so_far,
    all_combinations,
    n_slices,
    total_prbs_limit,
    # This min_val_for_slice_loop is the actual starting multiple for the current slice's loop
    min_val_for_slice_loop, 
    multiple_of # The step and base for multiples
):
    """
    Helper recursive function to generate combinations where sum <= total_prbs_limit
    and each element is a multiple of 'multiple_of'.

    Args:
        slice_idx (int): Current index of the slice being populated (0 to n_slices-1).
        current_sum (int): The sum of elements in combination_so_far.
        combination_so_far (list): The current combination being built.
        all_combinations (list): List to store all valid combinations found.
        n_slices (int): The total number of elements each combination should have.
        total_prbs_limit (int): The sum of elements in a combination must be <= this.
        min_val_for_slice_loop (int): The smallest multiple of 'multiple_of' that this
                                      slice's value can take (and is >= original min_prb_per_slice).
        multiple_of (int): Each element in the combination must be a multiple of this number.
    """

    if slice_idx == n_slices:
        all_combinations.append(list(combination_so_far))
        return

    remaining_slices_to_fill = n_slices - (slice_idx + 1)
    
    # Max value for current_slice (val) such that:
    # current_sum + val + remaining_slices_to_fill * min_val_for_slice_loop <= total_prbs_limit
    # val <= total_prbs_limit - current_sum - remaining_slices_to_fill * min_val_for_slice_loop
    # This max_val_for_current_slice is the absolute upper bound. The loop will pick multiples.
    max_val_for_current_slice = total_prbs_limit - current_sum - (remaining_slices_to_fill * min_val_for_slice_loop)

    # Iterate through possible values for the current slice.
    # Values must be multiples of 'multiple_of', starting from 'min_val_for_slice_loop'.
    for val in range(min_val_for_slice_loop, max_val_for_current_slice + 1, multiple_of):
        combination_so_far.append(val)
        _generate_recursive_combinations_le_multiple_of(
            slice_idx + 1,
            current_sum + val,
            combination_so_far,
            all_combinations,
            n_slices,
            total_prbs_limit,
            min_val_for_slice_loop, # Next slice also starts its search from this minimum multiple
            multiple_of
        )
        combination_so_far.pop() # Backtrack

def generate_combinations_sum_le(n_slices, total_prbs, min_prb_per_slice=2, multiple_of=1):
    """
    Generates all combinations of 'n_slices' integers where each integer
    is at least 'min_prb_per_slice', is a multiple of 'multiple_of',
    and their sum is less than or equal to 'total_prbs'.

    Args:
        n_slices (int): The number of integers in each combination.
        total_prbs (int): The sum of integers in a combination must be <= this value.
        min_prb_per_slice (int, optional): The minimum value for each integer in the
                                           combination. Defaults to 2.
        multiple_of (int, optional): Each integer in the combination must be a multiple
                                     of this number. Defaults to 1 (any integer).

    Returns:
        list: A list of numpy.ndarray (dtype=np.int32), where each array is a valid
              combination. Returns an empty list if no such combinations are possible.
    """
    if n_slices < 0:
        raise ValueError("n_slices cannot be negative.")
    if min_prb_per_slice <= 0:
        raise ValueError("min_prb_per_slice must be positive.")
    if multiple_of <= 0:
        raise ValueError("multiple_of must be positive.")

    if total_prbs < 0:
        # If n_slices is 0, sum is 0. 0 <= negative is false.
        # If n_slices > 0, min sum will be positive, so it cannot be <= negative total_prbs.
        return []

    if n_slices == 0:
        # Sum is 0. Valid if 0 <= total_prbs. 'multiple_of' is irrelevant for empty combination.
        if 0 <= total_prbs:
            return [np.array([], dtype=np.int32)]
        else:
            return []

    # Calculate the smallest value that any slice can take.
    # It must be >= min_prb_per_slice AND a multiple of 'multiple_of'.
    if multiple_of == 1:
        effective_min_val_per_slice = min_prb_per_slice
    else:
        # Smallest multiple of 'multiple_of' that is >= 'min_prb_per_slice'
        effective_min_val_per_slice = int(math.ceil(float(min_prb_per_slice) / multiple_of) * multiple_of)

    min_possible_sum = n_slices * effective_min_val_per_slice
    if min_possible_sum > total_prbs:
        return []

    all_combinations_list = []
    current_combination_temp = []

    _generate_recursive_combinations_le_multiple_of(
        slice_idx=0,
        current_sum=0,
        combination_so_far=current_combination_temp,
        all_combinations=all_combinations_list,
        n_slices=n_slices,
        total_prbs_limit=total_prbs,
        min_val_for_slice_loop=effective_min_val_per_slice,
        multiple_of=multiple_of
    )

    return [np.array(combo, dtype=np.int32) for combo in all_combinations_list]

if __name__ == '__main__':
    combinations = generate_combinations_sum_le(n_slices=3, total_prbs=50, min_prb_per_slice=2, multiple_of=2)
    print(len(combinations))

    #for combo in combinations10:
        #print(f"{combo} (sum: {np.sum(combo)})")
    # Expected:
    # [2 2] (sum: 4)
    # [2 4] (sum: 6)
    # [2 6] (sum: 8)
    # [2 8] (sum: 10)
    # [4 2] (sum: 6)
    # [4 4] (sum: 8)
    # [4 6] (sum: 10)
    # [6 2] (sum: 8)
    # [6 4] (sum: 10)
    # [8 2] (sum: 10)

