import h5py
import numpy as np
from nasbench import api
from tqdm import tqdm
from collections import deque

# Constants
NASBENCH_FULL_TFRECORD = "data/nasbench_full.tfrecord"
NASBENCH_HDF5 = "data/nasbench101.hdf5"

LABEL2ID = {
    "input": -1,
    "output": -2,
    "conv3x3-bn-relu": 0,
    "conv1x1-bn-relu": 1,
    "maxpool3x3": 2
}


def calculate_depth(adjacency_matrix):
    """Calculate the depth of a given adjacency matrix.

    Arguments:
        adjacency_matrix -- numpy array representing the adjacency matrix of the network.

    Returns:
        depth -- integer representing the depth of the network.
    """
    n = len(adjacency_matrix)
    depths = [0] * n
    
    def dfs(vertex):
        if depths[vertex] != 0:
            return depths[vertex]
        
        max_depth = 0
        for adjacent_vertex, connected in enumerate(adjacency_matrix[vertex]):
            if connected:
                max_depth = max(max_depth, dfs(adjacent_vertex))
        
        depths[vertex] = max_depth + 1
        return depths[vertex]
    
    return dfs(0) - 2  # Subtract 2 to exclude input and output nodes

# TODO: check if this is correct/needed
    # n = len(adjacency_matrix)
    # depths = [0] * n

    # def dfs(vertex):
    #     if depths[vertex] != 0:
    #         return depths[vertex]

    #     max_depth = 0
    #     for adjacent_vertex, connected in enumerate(adjacency_matrix[vertex]):
    #         if connected:
    #             max_depth = max(max_depth, dfs(adjacent_vertex))

    #     depths[vertex] = max_depth + 1
    #     return depths[vertex]

    # max_depth = 0
    # for i in range(1, n - 1):
    #     max_depth = max(max_depth, dfs(i))

    # return max_depth


def pad_adjacency_and_operations(metadata, vertices_count):
    """Pad adjacency matrix and operations list with zeros up to the maximum number of vertices (7).

    Arguments:
        metadata -- dictionary containing the adjacency matrix and operations list.
        vertices_count -- integer representing the number of vertices in the current network.

    Returns:
        adjacency_padded -- numpy array representing the padded adjacency matrix.
        operations_padded -- numpy array representing the padded operations list.
    """

    adjacency = np.array(metadata["module_adjacency"], dtype=np.int8)
    adjacency_padded = np.zeros((7, 7), dtype=np.int8)
    adjacency_padded[:adjacency.shape[0], :adjacency.shape[1]] = adjacency

    operations = np.array([LABEL2ID[t] for t in metadata["module_operations"]], dtype=np.int8)
    operations_padded = np.zeros((7,), dtype=np.int8)
    operations_padded[:vertices_count] = operations

    return adjacency_padded, operations_padded

def convert_metrics(metrics_data):
    """Convert the metrics data into a numpy array.

    Arguments:
        metrics_data -- dictionary containing the metrics data.

    Returns:
        converted_metrics_list -- list of numpy arrays representing the converted metrics.
    """

    converted_metrics_list = []
    epochs = [4, 12, 36, 108]

    for epoch in epochs:
        converted_metrics = []

        for seed in range(3):
            cur = metrics_data[epoch][seed]
            converted_metrics.append(np.array([[cur[t + "_training_time"],
                                                cur[t + "_train_accuracy"],
                                                cur[t + "_validation_accuracy"],
                                                cur[t + "_test_accuracy"]] for t in ["halfway", "final"]
                                               ], dtype=np.float32))
        converted_metrics_list.append(converted_metrics)

    return converted_metrics_list

# Main script
def main():
    nasbench = api.NASBench(NASBENCH_FULL_TFRECORD)

    # Initialize lists to store data
    metrics = []
    operations = []
    adjacency = []
    trainable_parameters = []
    hashes = []
    num_vertices = []
    network_depth = []

    # Iterate through all available architectures
    for hashval in tqdm(nasbench.hash_iterator()):
        metadata, metrics_data = nasbench.get_metrics_from_hash(hashval)

        # Store data for each architecture
        hashes.append(hashval.encode())
        trainable_parameters.append(metadata["trainable_parameters"])
        vertices_count = len(metadata["module_operations"])
        num_vertices.append(vertices_count)
        assert vertices_count <= 7

        adjacency_padded, operations_padded = pad_adjacency_and_operations(metadata, vertices_count)
        adjacency.append(adjacency_padded)
        operations.append(operations_padded)

        metrics.append(convert_metrics(metrics_data))

        network_depth.append(calculate_depth(adjacency_padded))

    # Convert lists to numpy arrays
    hashes = np.array(hashes)
    operations = np.stack(operations)
    adjacency = np.stack(adjacency)
    trainable_parameters = np.array(trainable_parameters, dtype=np.int32)
    metrics = np.array(metrics, dtype=np.float32)
    num_vertices = np.array(num_vertices, dtype=np.int8)
    network_depth = np.array(network_depth, dtype=np.int8)

    # Save data to HDF5 file
    with h5py.File(NASBENCH_HDF5, "w") as fp:
        fp.create_dataset("hash", data=hashes)
        fp.create_dataset("num_vertices", data=num_vertices)
        fp.create_dataset("trainable_parameters", data=trainable_parameters)
        fp.create_dataset("adjacency", data=adjacency)
        fp.create_dataset("operations", data=operations)
        fp.create_dataset("metrics", data=metrics)
        fp.create_dataset("network_depth", data=network_depth)

if __name__ == "__main__":
    main()