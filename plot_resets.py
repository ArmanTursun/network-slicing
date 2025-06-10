import numpy as np
import os

# Define your source and target folders
source_folder = 'SPPO_m05_05_001_196'
target_folder = 'SPPO_m05_05_196_1024_kmean_10_001'

# Loop over the two files
for i in range(2):
    filename = f'history_{i}.npz'
    source_path = os.path.join(source_folder, filename)
    target_path = os.path.join(target_folder, filename)

    # Load both source and target
    source_data = np.load(source_path)
    target_data = np.load(target_path)

    # Get 'reset' array from source
    if 'reset' not in source_data:
        print(f"'reset' not found in {source_path}, skipping.")
        continue
    reset_array = source_data['reset']

    # Copy all target data into a dict
    target_dict = {k: target_data[k] for k in target_data.files}
    target_dict['reset'] = reset_array

    # Save back to the target path (overwrite)
    np.savez(target_path, **target_dict)
    print(f"Updated {target_path} with 'reset' from {source_path}")
