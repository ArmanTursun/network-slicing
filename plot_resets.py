import numpy as np
import math
import matplotlib.pyplot as plt
import os
import sys
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

def average_per_window(values, window):
    values = np.array(values)
    n = len(values)
    trimmed_length = (n // window) * window  # Drop extra elements
    reshaped = values[:trimmed_length].reshape(-1, window)
    return reshaped.mean(axis=1)

WINDOW = 100
# Define your source and target folders
source_folder = './results/scenario_0/QR_15_1_1_2_005_1'
# Generate distinct colors from a colormap
color_list = plt.cm.tab10.colors  # up to 10 distinct colors; you can also use tab20, Set3, etc.
color_map = {i: color_list[i % len(color_list)] for i in range(10)}

fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(25, 11), constrained_layout=False)
axs = axs.flatten()
fig.subplots_adjust(top=0.78)

labels = []

# Loop over the two files
for i in range(10):
    labels.append('result_' + str(i))
    filename = f'results_{i}.npz'
    source_path = os.path.join(source_folder, filename)

    # Load both source and target
    source_data = np.load(source_path, allow_pickle=True)

    # Get 'reset' array from source
    _violations_total = source_data['violation']
    _violations_total = np.array([np.sum(r) for r in _violations_total], dtype=np.int16)
    violations_total = average_per_window(_violations_total.cumsum(), WINDOW)

    viol_raw = source_data['violation']
    _violations = np.array([np.array(r, dtype=np.int16) for r in viol_raw])  # object array
    _violations = np.stack(_violations, axis=0)  # now shape (T, K)
    _violations = _violations.T
    violations = [average_per_window(v.cumsum(), WINDOW) for v in _violations]

    violations_idx = i

    steps = np.arange(len(violations_total))

    axs[violations_idx].set_title(labels[i], fontsize=14)
    axs[violations_idx].plot(steps, violations_total, label = f'{labels[i]}_total', linewidth = 2, color=color_map[9])
    for j, v in enumerate(violations):
        axs[violations_idx].plot(np.arange(len(v)), v, label = f'{labels[i]}_slice{j}', linewidth=2)
    axs[violations_idx].set_xlabel('Epoch', fontsize=14)  # Add an x-label to the axes.
    axs[violations_idx].set_ylabel('violations', fontsize=14)
    axs[violations_idx].set_ylim((0,1000)) # 15000
    axs[violations_idx].set_yticks(np.arange(0, 1001, 100))
    axs[violations_idx].legend(loc='upper left', fontsize=14)
    #axs[1].legend(loc='best')
    axs[violations_idx].grid()
    #if i == 9:
    #    ncol = math.ceil(5)
    #    fig.legend(labels, loc='upper center', ncol=ncol, bbox_to_anchor=(0.5, 1.0), frameon=True, fontsize=14)
    #    fig.tight_layout(rect=[0, 0, 1, 0.8])

    fig.savefig(source_folder + '/results', format='png')

