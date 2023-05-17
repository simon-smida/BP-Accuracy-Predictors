# File: split_train_val.py
# Description: 
#   This script loads the NAS-Bench-101 dataset, generates various splits of the data,
#   and removes architectures with validation accuracy less than a certain threshold. The
#   results are then saved for future use in Neural Architecture Search experiments.

# Inspired by: NPNAS (https://github.com/ultmaster/neuralpredictor.pytorch)
# License: MIT License

import numpy as np
import h5py


def denoise_nasbench(metrics, threshold=0.8):
    """Remove architectures with validation accuracy less than threshold from the NAS-Bench-101 dataset.

    Arguments:
        metrics -- numpy array containing metrics of NAS-Bench-101 architectures

    Keyword Arguments:
        threshold -- the validation accuracy threshold (default: {0.8})

    Returns:
        indices -- indices of architectures with validation accuracy greater than threshold
    """
    # Extrac the validation metrics (final epoch, final run, validation accuracy)
    val_metrics = metrics[:, -1, :, -1, 2]
    index = np.where(val_metrics[:, 0] > threshold)
    return index[0]


# Load NAS-Bench-101 dataset
with h5py.File("data/nasbench101.hdf5", mode="r") as f:
    # Get the total number of unique architectures in the dataset
    total_count = len(f["hash"][()])
    # Load the metrics for each architecture
    metrics = f["metrics"][()]

# Generate splits - indices of architectures in the dataset
random_state = np.random.RandomState(0)
result = dict()
for n_samples in [172, 334, 860, 2000, 5000, 10000]:
    split = random_state.permutation(total_count)[:n_samples]
    result[str(n_samples)] = split

# Generate denoised splits (only architectures with validation accuracy above a threshold)
valid91 = denoise_nasbench(metrics, threshold=0.91)
for n_samples in [172, 334, 860, 2000, 5000, 10000]:
    # Intersection between the original split and the denoised data
    result["91-" + str(n_samples)
           ] = np.intersect1d(result[str(n_samples)], valid91)
result["denoise-91"] = valid91

result["denoise-80"] = denoise_nasbench(metrics)

# Save the splits (both normal and denoised) for future use
np.savez("data/train.npz", **result)
