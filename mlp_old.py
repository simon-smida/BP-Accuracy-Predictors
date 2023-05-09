import copy
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle
from predictor import Predictor


class MLPPredictor(Predictor):
    def __init__(self, input_dim=None):
        super().__init__()
        self.input_dim = input_dim
        self.default_hyperparams = {
            "hidden_layers": [64, 64],
            "activation": "relu",
            "lr": 1e-3,
            "epochs": 100,
            "batch_size": 32,
        }
        self.hyperparams = copy.deepcopy(self.default_hyperparams)

    def set_random_hyperparams(self):
        """
        Sets random hyperparameters for the MLP model.
        """
        self.hyperparams = copy.deepcopy(self.default_hyperparams)

        # Randomly choose the number of hidden layers
        n_hidden_layers = np.random.choice([1, 2, 3])

        # Randomly choose the size of each hidden layer
        hidden_layers = []
        for _ in range(n_hidden_layers):
            hidden_layers.append(np.random.choice([32, 64, 128, 256]))

        self.hyperparams['hidden_layers'] = hidden_layers

        # Randomly choose the activation function
        activation = np.random.choice(['relu', 'tanh'])
        self.hyperparams['activation'] = activation

        # Randomly choose the learning rate
        lr = 10 ** np.random.uniform(-5, -1)
        self.hyperparams['lr'] = lr

        return self.hyperparams

    def save(self, filepath):
        torch.save(self.model.state_dict(), filepath)

    def load(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found at '{filepath}'")
        self.model.load_state_dict(torch.load(filepath))

    def fit(self, x, y):
        
        if self.input_dim is None:
            self.input_dim = x.shape[1]
        
        self.model = self._build_model()
        self.model.train()

        dataset = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
        dataloader = DataLoader(dataset, batch_size=self.hyperparams['batch_size'], shuffle=True)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.hyperparams['lr'])

        for epoch in range(self.hyperparams['epochs']):
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y.unsqueeze(1))
                loss.backward()
                optimizer.step()

    def refit(self, x, y):
        self.fit(x, y)

    def query(self, x):
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(torch.tensor(x, dtype=torch.float32)).numpy().squeeze()
        return predictions

    def _build_model(self):
        layers = []
        prev_size = self.input_dim

        for size in self.hyperparams['hidden_layers']:
            layers.append(nn.Linear(prev_size, size))
            if self.hyperparams['activation'] == 'relu':
                layers.append(nn.ReLU())
            elif self.hyperparams['activation'] == 'tanh':
                layers.append(nn.Tanh())
            prev_size = size

        layers.append(nn.Linear(prev_size, 1))
        model = nn.Sequential(*layers)
        return model