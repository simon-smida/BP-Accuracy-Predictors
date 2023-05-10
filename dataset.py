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
            
            # TODO: review depth
            # Store depth for each architecture
            self.depth = np.zeros(len(self.hash2id), dtype=np.int32)
            for i, h in enumerate(f["hash"][()]):
                self.hash2id[h.decode()] = i
                #self.depth[i] = self._network_depth(self.adjacency[i, :self.num_vertices[i], :self.num_vertices[i]])

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
                - num_vertices (int): The number of vertices in the architecture.
                - trainable_parameters (int): The number of trainable parameters in the architecture.
                - adjacency (np.ndarray): The adjacency matrix of the architecture.
                - operations_onehot (np.ndarray): The operations of the architecture in one-hot encoding.
                - operations (np.ndarray): The operations of the architecture.
                - encoded_onehot_flat (np.ndarray): The flattened one-hot encoding of the architecture.
                - encoded_gcn (np.ndarray): The GCN encoding of the architecture.
                - mask (np.ndarray): The mask of the architecture.
                - val_acc (float): The validation accuracy of the architecture.
                - test_acc (float): The test accuracy of the architecture.
                - depth (int): The depth of the architecture.
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
        
        result = {
            "num_vertices": num_vertices,
            #"trainable_parameters": trainable_parameters,
            "adjacency": adjacency,
            "operations": ops_onehot, #TODO beware of this (we want this)
            #"operations": self.operations[index],
            "mask": mask,
            "val_acc": val_acc,#self.normalize(val_acc),   # normalized
            "test_acc": test_acc,#self.normalize(test_acc), # normalized
            "depth": self.depth[index]
        }

        return result
    
    # TODO add # of weights (scaled?)
    @staticmethod
    def _encode_onehot_flat(adjacency, ops_onehot):
        """Encoding of the adjacency matrix and the operations into a single vector.
        - useful for MLP, LR, RF, XGB

        Arguments:
            adjacency -- adjacency matrix
            ops_onehot -- operations one-hot encoded (including input and output)

        Returns:
            vector -- flattened vector of the adjacency matrix and the operations
        """
        adj_flat = adjacency[0].flatten()
        ops_flat = ops_onehot.flatten()
        return np.concatenate([adj_flat, ops_flat])
    
    @staticmethod
    def _encode_gcn(num_vertices, adjacency, ops_onehot, mask, val_acc):
        """
        Input:
        GCN: a list of categorical ops starting from 0
        """
        #  for i in range(len(self)):
        #     yield self._encode_gcn(
        #         self.num_vertices[i], 
        #         self.adjacency[i], 
        #         self._create_onehot_operations(self.operations[i]), 
        #         self.mask[i], 
        #         self.val_acc[i]
        #     )
        dic = {
            "num_vertices": num_vertices,
            "adjacency": adjacency,
            "operations": ops_onehot,
            "mask": mask,
            "val_acc": val_acc,  # normalized
        }
        return dic
    
    def get_flat_features(self):
        """
        Get list of flattened features for each architecture in the dataset.
        """
        for i in range(len(self)):
            yield self._encode_onehot_flat(self.adjacency[i], self._create_onehot_operations(self.operations[i]))
            
    def get_gcn_features(self):
        """
        Get list of GCN features for each architecture in the dataset.
        """
        for i in range(len(self)):
            yield self._encode_gcn(
                self.num_vertices[i], 
                self.adjacency[i], 
                self._create_onehot_operations(self.operations[i]), 
                self.mask[i], 
                self.val_acc[i]
            )
        
    def get_targets(self):
        """
        Get list of targets for each architecture in the dataset.
        """
        for idx in self.indices:
            val_acc = self.metrics[idx, -1, 0, -1, 2] # seed 0
            if self._is_acc_blow(val_acc):
                val_acc = self.resample_acc(idx, split="val")
            normalized_val_acc = self.normalize(val_acc)
            yield normalized_val_acc



# OPS = [CONV3X3, CONV1X1, MAXPOOL3X3]
# OPS_INCLUSIVE = [INPUT, OUTPUT, *OPS]

# NUM_VERTICES = 7
# OP_SPOTS = NUM_VERTICES - 2
# MAX_EDGES = 9

# def encode_adj(spec):
#     """
#     ADJACENCY_ONE_HOT: compute adjacency matrix + op list encoding
#     """
#     matrix, ops = spec["matrix"], spec["ops"]
#     op_dict = {CONV1X1: [0, 0, 1], CONV3X3: [0, 1, 0], MAXPOOL3X3: [1, 0, 0]}
#     encoding = []
#     for i in range(NUM_VERTICES - 1):
#         for j in range(i + 1, NUM_VERTICES):
#             encoding.append(matrix[i][j])
#     for i in range(1, NUM_VERTICES - 1): # skip input and output
#         encoding = [*encoding, *op_dict[ops[i]]]
#     return encoding

# def encode_gcn(spec):
#     """
#     Input:
#     GCN: a list of categorical ops starting from 0
#     """
#     matrix, ops = spec["matrix"], spec["ops"]
#     op_map = [OUTPUT, INPUT, *OPS]
#     ops_onehot = np.array(
#         [[i == op_map.index(op) for i in range(len(op_map))] for op in ops],
#         dtype=np.float32,
#     )

#     dic = {
#         "num_vertices": 7,
#         "adjacency": matrix,
#         "operations": ops_onehot,
#         "mask": np.array([i < 7 for i in range(7)], dtype=np.float32),
#         "val_acc": 0.0,
#     }
#     return dic

    
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
    def _network_depth(adjacency_matrix):
        """
        Compute the depth of the network given its adjacency matrix.

        Args:
            adjacency_matrix (numpy.ndarray): The adjacency matrix of the network.

        Returns:
            int: The depth of the network.
        """
        def dfs(node, depth, visited):
            visited[node] = True
            max_depth = depth
            for i, conn in enumerate(adjacency_matrix[node]):
                if conn and not visited[i]:
                    max_depth = max(max_depth, dfs(i, depth + 1, visited))
            return max_depth

        input_node = 0  # The index of the input node in the adjacency matrix
        visited = [False] * len(adjacency_matrix)
        depth = dfs(input_node, 0, visited)

        return depth

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