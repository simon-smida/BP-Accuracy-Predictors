import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from predictor import Predictor
from utils import get_logger, AverageMeterGroup


# Set device to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MLPModel(nn.Module):
    def __init__(self, input_dims, num_layers=3, layer_width=[10,10,10], output_dims=1, activation="relu"):
        super().__init__()

        # Activation function mapping
        activation_mapping = {
            "relu": nn.ReLU,
            "sigmoid": nn.Sigmoid,
            "tanh": nn.Tanh
        }

        if activation not in activation_mapping:
            raise ValueError(f"Invalid activation function: {activation}")

        # List of numbers of neurons in each layer
        all_units = [input_dims] + num_layers * [layer_width]

        # Sequential - stack layers in order
        # nn.Linear(# of input neurons, # of output neurons) + applies a linear transformation
        hidden_layers = [
            nn.Sequential(nn.Linear(all_units[i], all_units[i + 1]), activation_mapping[activation]())
            for i in range(num_layers)  # for each hidden layer
        ]

        # Add the output layer at the end of the list of hidden layers
        self.layers = nn.Sequential(
            *hidden_layers, nn.Linear(all_units[-1], output_dims))

    def forward(self, x):
        """Forward pass - pass 'x' through each layer sequentially"""
        return self.layers(x)


class MLPPredictor(Predictor):
    def __init__(self, input_dims=None, ss_type="nasbench101", encoding_type="adj_onehot"):
        super().__init__(ss_type=ss_type, encoding_type=encoding_type)

        self.model = None
        self.input_dims = input_dims

        self.default_hyperparams = {
            "model_params": {
                "num_layers": 20,
                "layer_width": 20,
                "output_dims": 1,
                "activation": "relu",
            },
            "batch_size": 32,
            "learning_rate": 0.001,
            "epochs": 100,
            "device": device,
        }

        self.hyperparams = None

    def save(self, file_path):
        """Save the trained model to a file."""
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))
        torch.save(self.model.state_dict(), file_path)

    def load(self, file_path):
        """Load a trained model from a file."""
        if self.hyperparams is None:
            self.hyperparams = self.default_hyperparams.copy()

        if self.input_dims is None:
            raise RuntimeError(
                "The input_dims attribute should be set before loading the model.")

        self.model = MLPModel(input_dims=self.input_dims,
                              **self.hyperparams['model_params'])
        self.model.load_state_dict(torch.load(file_path))
        self.model.to(self.hyperparams['device'])

    def fit(self, xtrain, ytrain):
        """Train the model on the given data"""
        if self.hyperparams is None:
            self.hyperparams = self.default_hyperparams.copy()


        # Create a dataset from tensors
        train_data = TensorDataset(torch.FloatTensor(xtrain),
                                   torch.FloatTensor(ytrain))
        # Create a data loader from the dataset
        data_loader = DataLoader(
            train_data, batch_size=self.hyperparams['batch_size'], shuffle=True)

        num_layers = self.hyperparams['model_params']['num_layers']
        layer_width = self.hyperparams['model_params']['layer_width']
        if self.model is None:
            # self.model = MLPModel(
            #     input_dims=xtrain.shape[1], **self.hyperparams['model_params'])
            self.model = MLPModel(
                input_dims=xtrain.shape[1],
                num_layers=num_layers,
                layer_width=num_layers*[layer_width]
            )
            self.model.to(self.hyperparams['device'])

        # Model training: loss, optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.hyperparams['learning_rate'])

        # Logging setup
        logger = get_logger()

        # Train the model
        for epoch in range(self.hyperparams['epochs']):
            self.model.train()  # Set the model to train mode
            meters = AverageMeterGroup()
            # Iterate over data batches
            for x, y in data_loader: # x: batch of inputs, y: batch of targets
                x = x.to(self.hyperparams['device'])
                y = y.to(self.hyperparams['device'])

                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass + backward pass + optimize
                y_pred = self.model(x).squeeze()
                loss = criterion(y_pred, y)
                loss.backward()
                optimizer.step()

                # Update meters
                meters.update({"loss": loss.item()})

            # Log training progress
            #logger.info(f"Epoch {epoch + 1}, Loss: {meters['loss'].avg:.4f}")


    def predict(self, xtest):
        """Predict on the given data"""
        self.model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            xtest = torch.tensor(xtest, dtype=torch.float32).to(
                self.hyperparams['device'])
            y_pred = self.model(xtest).cpu().numpy()

        return y_pred.squeeze()

    def set_random_hyperparams(self):
        """Set random hyperparameters for the model"""
        if self.hyperparams is None:
            params = self.default_hyperparams.copy()
        else:
            params = {
                "model_params": {
                    "num_layers": int(np.random.choice(range(5, 25))),
                    "layer_width": int(np.random.choice(range(5, 25))),
                    "output_dims": 1,
                    "activation": "relu",
                },
                "batch_size": 32,
                "learning_rate": np.random.choice([0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]),
                "epochs": 100,
                "device": device,
            }

        self.hyperparams = params
        return params

    def __repr__(self):
        """Representation of the predictor"""
        hyperparams = self.default_hyperparams if self.hyperparams is None else self.hyperparams
        return f"MLPPredictor(input_dims={self.input_dims}, ss_type='{self.ss_type}', encoding_type='{self.encoding_type}', hyperparams={hyperparams})"

    def __str__(self):
        """String representation of the predictor"""
        hyperparams = self.default_hyperparams if self.hyperparams is None else self.hyperparams
        return f"MLPPredictor for {self.ss_type} search space with {self.encoding_type} encoding.\nCurrent hyperparameters: {hyperparams}"
