import h5py
import numpy as np
from torch.utils.data import Dataset


class NASBench101Dataset(Dataset):
    """
    A PyTorch Dataset class for the NAS-Bench-101 dataset (benchmark).
    
    Args:
        hdf5_file (str): The path to the hdf5 file containing the preprocessed NAS-Bench-101 data.
        split (str, optional): The dataset split to use. Choices are 'train', 'test', or 'all'. Default is 'all'.
    """

    # TODO: how we got these values?
    MEAN = 0.908192
    STD = 0.023961

    def __init__(self, hdf5_file, split="all"):
        super().__init__()
        
        self.hash2id = dict()
        self.hdf5_file = hdf5_file
        self.seed = 0
        
        with h5py.File(self.hdf5_file, "r") as f:
            for i, h in enumerate(f["hash"][()]):
                self.hash2id[h.decode()] = i
            self.num_vertices = f["num_vertices"][()]
            self.trainable_parameters = f["trainable_parameters"][()]
            self.adjacency = f["adjacency"][()]
            self.operations = f["operations"][()]
            self.metrics = f["metrics"][()]
            self.depth = f["network_depth"][()]
            
            for i, h in enumerate(f["hash"][()]):
                self.hash2id[h.decode()] = i

        indices = np.arange(len(self.hash2id))
        np.random.seed(42)  # Set seed for reproducibility
        np.random.shuffle(indices)

        if split != "all" and split != "train" and split != "test":
            self.indices = np.load("data/train.npz")[str(split)]
        elif split == "train":
            self.indices = indices[:int(0.8 * len(indices))]
        elif split == "test":
            self.indices = indices[int(0.8 * len(indices)):]
        else: # "all"
            self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """
        Get an item (architecture) from the dataset by index.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            dict: A dictionary containing the following keys:
                - "num_vertices": The number of vertices in the architecture.
                - "trainable_parameters": The number of trainable parameters of the architecture.
                - "adjacency": The adjacency matrix of the architecture.
                - "operations": The operations of the architecture.
                - "mask": The mask of the architecture.
                - "val_acc": The validation accuracy of the architecture.
                - "test_acc": The test accuracy of the architecture.
                - "depth": The depth of the architecture.
                - "num_connections": The number of connections in the architecture.
                - "avg_connections_per_vertex": The average number of connections per vertex in the architecture.
                - "conv3x3_count": The number of 3x3 convolutions in the architecture.
                - "conv1x1_count": The number of 1x1 convolutions in the architecture.
                - "maxpool3x3_count": The number of 3x3 max pooling operations in the architecture.
        """
              
        index = self.indices[idx]
        
        # Extract features and target
        val_acc, test_acc = self.metrics[index, -1, 0, -1, 2:] # seed 0

        if self._is_acc_blow(val_acc):
            val_acc = self.resample_acc(index, split="val")
        if self._is_acc_blow(test_acc):
            test_acc = self.resample_acc(index, split="test")
        
        num_vertices = self.num_vertices[index] # <=7 (or just 7? to have same size for all)
        adjacency = self.adjacency[index], #self.adjacency[index, :num_vertices, :num_vertices]
        ops_onehot = self._create_onehot_operations(self.operations[index])
        trainable_parameters = self.trainable_parameters[index]
            
        mask = np.array([i < num_vertices for i in range(7)], dtype=np.float32)
        if num_vertices < 7:
            ops_onehot[num_vertices:] = 0.
        
    
        conv3x3_count, conv1x1_count, maxpool3x3_count = self._get_ops_counts(ops_onehot)
        num_connections = self._count_connections(adjacency)
    
        result = {
            "num_vertices": num_vertices,
            "trainable_parameters": trainable_parameters,
            "adjacency": adjacency,
            "operations": ops_onehot,
            #"operations": self.operations[index],
            "mask": mask,
            "val_acc": val_acc,#self.normalize(val_acc),   # normalized
            "test_acc": test_acc,#self.normalize(test_acc), # normalized
            "depth": self.depth[index],
            "num_connections": num_connections,
            "avg_connections_per_vertex": self._count_avg_connections_per_vertex(num_connections, num_vertices),
            "conv3x3_count": conv3x3_count,
            "conv1x1_count": conv1x1_count,
            "maxpool3x3_count": maxpool3x3_count,
            "training_time": self.metrics[index, -1, 0, -1, 0],
        }

        return result
        
    @staticmethod
    def normalize(value):
        """
        Normalize the given value using the precomputed mean and standard deviation.

        Args:
            value (float): The value to normalize.

        Returns:
            float: The normalized value.
        """
        return (value - NASBench101Dataset.MEAN) / NASBench101Dataset.STD

    @staticmethod
    def denormalize(value):
        """
        Denormalize the given value using the precomputed mean and standard deviation.

        Args:
            value (float): The value to denormalize.

        Returns:
            float: The denormalized value.
        """
        return value * NASBench101Dataset.STD + NASBench101Dataset.MEAN

    def mean_acc(self):
        """
        Compute the mean accuracy of the dataset.

        Returns:
            float: The mean accuracy.
        """
        return np.mean(self.metrics[:, -1, self.seed, -1, 2])

    def std_acc(self):
        """
        Compute the standard deviation of accuracy of the dataset.

        Returns:
            float: The standard deviation of accuracy.
        """
        return np.std(self.metrics[:, -1, self.seed, -1, 2])
    
    def resample_acc(self, index, split="val"):
        # when val_acc or test_acc are out of range
        assert split in ["val", "test"]
        split = 2 if split == "val" else 3
        for seed in range(3):
            acc = self.metrics[index, -1, seed, -1, split]
            if not self._is_acc_blow(acc):
                return acc
        return np.array(self.MEAN)

    def _is_acc_blow(self, acc):
        return acc < 0.2

    @staticmethod
    def _create_onehot_operations(operations, num_operations=5):
        """
        Convert the operation indices to one-hot encoded format.

        Args:
            operations (numpy.ndarray): The operation indices.
            num_operations (int, optional): The number of possible operations. Default is 5.

        Returns:
            numpy.ndarray: The one-hot encoded operations.
        """
        # encoding = {
        #     "input": -1,
        #     "output": -2,
        #     "conv3x3-bn-relu": 0,
        #     "conv1x1-bn-relu": 1,
        #     "maxpool3x3": 2
        # }
        onehot_array = np.zeros((len(operations), num_operations), dtype=np.float32)
        for i, operation in enumerate(operations):
            operation_idx = operation + 2  # Add 2 to match the encoding
            onehot_array[i, operation_idx] = 1.0
        return onehot_array
   
    @staticmethod
    def _count_operation_types(onehot_operations):
        operation_counts = np.sum(onehot_operations, axis=0)
        return operation_counts
    
    @staticmethod
    def _count_connections(adjacency):
        return np.sum(adjacency)

    @staticmethod
    def _count_avg_connections_per_vertex(num_connections, num_vertices):
        return num_connections / num_vertices
    
    @staticmethod
    def _get_ops_counts(ops_onehot):
        # Indices of operations in one-hot encoding
        # Input = 0
        # Output = 1
        # Conv3x3 = 2
        # Conv1x1 = 3
        # MaxPool3x3 = 4
        conv3x3_count = np.sum(ops_onehot[:, 2])
        conv1x1_count = np.sum(ops_onehot[:, 3])
        maxpool3x3_count = np.sum(ops_onehot[:, 4])
        return conv3x3_count, conv1x1_count, maxpool3x3_count
        
        # operation_counts = [NASBench101Dataset._count_operation_types(ops) for ops in ops_list]
        # print(operation_counts)
        # conv3x3_count = np.array([op_count[2] for op_count in operation_counts])
        # conv1x1_count = np.array([op_count[3] for op_count in operation_counts])
        # maxpool3x3_count = np.array([op_count[4] for op_count in operation_counts])
        # return conv3x3_count, conv1x1_count, maxpool3x3_count