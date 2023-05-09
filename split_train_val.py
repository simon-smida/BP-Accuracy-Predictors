import numpy as np
import h5py


<<<<<<< HEAD
def denoise_nasbench(metrics, threshold=0.8):
    val_metrics = metrics[:, -1, :, -1, 2]
    index = np.where(val_metrics[:, 0] > threshold)
    return index[0]


with h5py.File("data/nasbench.hdf5", mode="r") as f:
    total_count = len(f["hash"][()])
    metrics = f["metrics"][()]
random_state = np.random.RandomState(0)
result = dict()
for n_samples in [172, 334, 860]:
    split = random_state.permutation(total_count)[:n_samples]
    result[str(n_samples)] = split

# >91
valid91 = denoise_nasbench(metrics, threshold=0.91)
for n_samples in [172, 334, 860]:
    result["91-" + str(n_samples)] = np.intersect1d(result[str(n_samples)], valid91)
result["denoise-91"] = valid91

result["denoise-80"] = denoise_nasbench(metrics)
np.savez("data/train.npz", **result)
=======
# Extract valid architectures from nasbench
def denoise_nasbench(metrics, threshold=0.8):
    """Denoise the nasbench dataset by removing architectures with validation accuracy below a threshold.

    Arguments:
        metrics -- architecture metrics

    Keyword Arguments:
        threshold -- threshold for validation accuracy (default: {0.8})

    Returns:
        index -- indices of valid architectures
    """
    
    print("--------Metrics--------")
    print(metrics.shape)
    
    # Extract the last validation accuracy metric for each accuracy
    val_metrics = metrics[:, -1, :, -1, 2]
    print("--------Val Metrics--------")
    print(val_metrics.shape)
    print(val_metrics[0])
    print("Min validation accuracy: ", np.min(val_metrics))
    print("Max validation accuracy: ", np.max(val_metrics))
    
    #index = np.where(val_metrics[:, 0] > threshold)
    
    # Find the average validation accuracy for each accuracy
    avg_val_metrics = np.mean(val_metrics, axis=1) 
    print("--------Avg Val Metrics--------")
    print(avg_val_metrics.shape)
    print(avg_val_metrics[36])
    
    # Find the indices of architectures with average validation accuracy above the threshold
    index = np.where(avg_val_metrics > threshold)
    print("--------Index--------")
    print("# of architectures with acc above threshold: ", len(index[0]))
    print("# of architectures with acc below the threshold: ", len(avg_val_metrics) - len(index[0]))
    print(index)

    # Return the indices of the valid architectures
    return index[0] 


with h5py.File("data/nasbench.hdf5", mode="r") as f:
    # Get the  number of architectures in the dataset
    total_count = len(f["hash"][()])
    # Load the metrics for each architecture into a numpy array
    metrics = f["metrics"][()]
    
    
# Initialize a random state with seed 0 - this is used to randomly sample architectures
random_state = np.random.RandomState(0)

# Create an empty dictionary to store the results
result = dict()

# Generate three different data splits with 172, 334, and 860 architectures
for n_samples in [172, 334, 860]:
    # Select random architectures from the dataset to create the split
    split = random_state.permutation(total_count)[:n_samples]
    # Store the indices of the selected architectures in the result dictionary with the key being the number of architectures in the split
    result[str(n_samples)] = split

# > 91
# Extract indices of architectures with average validation accuracy above 0.91
valid91 = denoise_nasbench(metrics, threshold=0.91)

# Get the intersection of the indices of valid architectures with the indices of each data split
for n_samples in [172, 334, 860]:
    # Create a key for the intersection of the valid architectures with the given data split
    key = "91-" + str(n_samples)
    # Get the intersection of the indices and store it in the result dictionary with the key
    result[key] = np.intersect1d(result[str(n_samples)], valid91)
    

# Store the indices of architectures with average validation accuracy above 0.91 in the result dictionary
result["denoise-91"] = valid91

# Store the indices of architectures with average validation accuracy above 0.8 in the result dictionary
result["denoise-80"] = denoise_nasbench(metrics)

# Save the result dictionary as a .npz file - numpy binary file
np.savez("data/train.npz", **result)
>>>>>>> origin/main
