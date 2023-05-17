# File: utils.py
# Description: 
#   This Python script contains a variety of utility functions used for performance measurement,
#   feature extraction, data visualization, cross-validation, logging, and metric computation.
#   It is primarily used for analyzing and predicting the performance of models. 
#   Functions in this script can measure training and query time of predictors, extract flat or
#   Graph Convolutional Network (GCN) features from NAS datasets, visualize data with scatter plots, perform
#   k-fold cross validation, log events, and compute performance metrics.

import sys
import logging
import time
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from collections import OrderedDict
from sklearn.model_selection import KFold
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler

# Constants
IMG_PATH = "imgs"

## ------------------- MEASURE PERFORMANCE -------------------
def measure_training_time(predictor, train_features, train_targets, num_runs=3):
    """Measure training time of a predictor - average of num_runs runs."""
    times = []

    for i in range(num_runs):
        # Measure time
        start = time.time()
        # Fit predictor
        predictor.fit(train_features, train_targets)
        end = time.time()
        times.append(end - start)

    # Return the average time of num_runs runs
    return np.mean(times)

def measure_query_time(predictor, test_features, num_runs=3):
    """Measure query time of a predictor - average of num_runs runs."""
    times = []

    for i in range(num_runs):
        # Measure time
        start = time.time()
        # Query predictor
        predictor.predict(test_features)
        end = time.time()
        times.append(end - start)

    # Return the average time of num_runs runs
    return np.mean(times)


## ------------------- FEATURE EXTRACTION -------------------
def get_targets(dataset):
    """Get targets (validation accuracy) for each architecture in the dataset."""
    dataset_size = len(dataset)
    targets = np.zeros(dataset_size)

    for i in range(dataset_size):
        sample = dataset[i]
        targets[i] = sample['val_acc']

    return targets

# NOTE: slower than get_targets()
def get_targets2(dataset):
    """Generator for targets (validation accuracy) for each architecture in the dataset."""
    dataset_size = len(dataset)
    for i in range(dataset_size):
        sample = dataset[i]
        yield sample['val_acc']

def get_flat_features(dataset):
    """Get flattened features (adjacency matrix + operations) for each architecture in the dataset."""
    features = []
    for arch in dataset:
        # Flatten and concatenate adjacency and operations arrays directly
        f = np.concatenate((
            arch["adjacency"][0].flatten(), 
            arch["operations"].flatten())
        )
        features.append(f)
    return np.array(features)

# NOTE: slower than get_flat_features()
def get_flat_features2(dataset):
    """Generator for flattened features (adjacency matrix + operations) for each architecture in the dataset."""
    dataset_size = len(dataset)
    adj_matrices = np.zeros((dataset_size, 7, 7))
    ops = np.zeros((dataset_size, 7, 5))

    features = []
    for i in range(dataset_size):
        
        # Adjacency and operations are one-hot encoded already (see dataset.py)
        arch = dataset[i]
        adj_matrices[i] = arch["adjacency"][0]
        ops[i] = arch["operations"]
        
        # Flatten and concatenate
        adj_flat = adj_matrices[i].flatten()
        ops_flat = ops[i].flatten()

        f = np.concatenate((adj_flat, ops_flat))
        yield f

def get_flat_features_boosted(dataset):
    """Get flattened features (adjacency matrices, operations, trainable parameters, conv3x3 count) for each architecture in the dataset."""

    # Initialize arrays for trainable_parameters and conv3x3_count
    trainable_parameters = np.array([arch["trainable_parameters"] for arch in dataset])
    conv3x3_count = np.array([arch["conv3x3_count"] for arch in dataset])

    # Initialize StandardScaler and fit on entire datasets
    scaler = StandardScaler()
    trainable_parameters_scaled = scaler.fit_transform(trainable_parameters.reshape(-1, 1)).reshape(-1)
    conv3x3_count_scaled = scaler.fit_transform(conv3x3_count.reshape(-1, 1)).reshape(-1)

    features = []
    for i, arch in enumerate(dataset):
        # Flatten and concatenate adjacency and operations arrays directly
        f = np.concatenate((
            arch["adjacency"][0].flatten(),
            arch["operations"].flatten(),
            [trainable_parameters_scaled[i]],
            [conv3x3_count_scaled[i]]
        ))
        features.append(f)

    return np.array(features)

def get_gcn_features(dataset):
    """Get GCN features (adjacency matrix, operations, mask) for each architecture in the dataset."""
    dataset_size = len(dataset)
    gcn_features = []
    
    for i in range(dataset_size):
        arch = dataset[i]

        gcn_f = {
            "num_vertices": arch["num_vertices"],
            "adjacency": arch["adjacency"][0],
            "operations": arch["operations"],
            "mask": arch["mask"],
            "val_acc": arch["val_acc"]
        }

        gcn_features.append(gcn_f)

    return gcn_features

## --------------------- VISUALIZATION ---------------------
def scatter_plot(
    y_true, y_pred, title,
    b=0.05, cmap=None, s=4, a=0.75,
    xlabel='Actual values', ylabel='Predicted values',
    save=False, filename='example.png', dense=False):
    """Create a scatter plot with gradient coloring based on the density estimate of the data points.

    Arguments:
        y_true -- actual values
        y_pred -- predicted values

    Keyword Arguments:
        b -- bandwidth for the kernel density estimate (default: {0.05})
        cmap -- colormap for the scatter plot (default: {'viridis'})
        s -- size of the scatter plot points (default: {4})
        a -- alpha value for the scatter plot points (default: {0.75})
        xlabel -- label for the x-axis (default: {'Actual values'})
        ylabel -- label for the y-axis (default: {'Predicted values'})
        save -- save the figure to a file (default: {False})
        filename -- name of the file to save the figure to (default: {'example.png'})
        dense -- show the density estimate as a heatmap (default: {False})
    """

    z = None
    if dense:
        cmap = 'viridis'
        # Combine the Actual values and Predicted values into a single 2D array
        data = np.vstack([y_true, y_pred]).T

        # Compute the density estimate
        kde = KernelDensity(kernel='gaussian', bandwidth=b).fit(data)
        z = np.exp(kde.score_samples(data))

    # Create a scatter plot with gradient coloring based on the density estimate
    plt.scatter(y_true, y_pred, c=z, cmap=cmap,
                s=s, edgecolors='face', alpha=a)
    plt.xlabel(xlabel)    
    plt.ylabel(ylabel)

    plt.plot([0, 100], [0, 100], 'r-', lw=0.5, label='y=x')
    plt.legend(loc='lower right')
    
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.grid(linewidth=0.25, alpha=0.5, color='lightgray')

    # Add title
    plt.title(title)

    if dense:
        plt.colorbar(label='Density')

    if save:
        plt.savefig(IMG_PATH + '/' + filename, dpi=300, bbox_inches='tight')

    plt.show()
    
def scatter_plot_nice(
    y_true, y_pred, title,
    b=0.05, cmap=None, s=4, a=0.75, top_lim=96,
    xlabel='Validation accuracy [%]', ylabel='Predictions [%]',
    save=False, filename='example.png', dense=False):
    """Scatter and zoomed-in scatter plot with optional density estimate.

    Arguments:
        y_true -- true values
        y_pred -- predicted values
        title -- title of the plot

    Keyword Arguments:
        b -- bandwidth for the density estimate (default: {0.05})
        cmap -- colormap for the density estimate (default: {None})
        s -- size of the points (default: {4})
        a -- alpha value for the points (default: {0.75})
        top_lim -- upper limit for the zoomed-in plot (default: {96})
        xlabel -- label for the x-axis (default: {'Actual values [%]'})
        ylabel -- label for the y-axis (default: {'Predicted values [%]'})
        save -- save the plot to a file (default: {False})
        filename -- name of the file to save the plot to (default: {'example.png'})
        dense -- use a density estimate (default: {False})
    """

    z = None
    if dense:
        cmap = 'viridis'
        data = np.vstack([y_true, y_pred]).T
        kde = KernelDensity(kernel='gaussian', bandwidth=b).fit(data)
        z = np.exp(kde.score_samples(data))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Create the basic scatter plot
    sc1 = ax1.scatter(y_true, y_pred, c=z, cmap=cmap, s=s, alpha=a)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.set_xlim(20, 100)
    ax1.set_ylim(20, 100)
    ax1.set_title('Complete Range', fontsize=10)
    line, = ax1.plot([20, 100], [20, 100], color='red', linestyle='-', linewidth=0.5, label='y=x')

    # Create the zoomed-in scatter plot
    sc2 = ax2.scatter(y_true, y_pred, c=z, cmap=cmap, s=s, alpha=a)
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel)
    ax2.set_xlim(90, top_lim)
    ax2.set_ylim(90, top_lim)
    ax2.set_title('Zoomed in [90, ' + str(top_lim) + ']', fontsize=10)  
    ax2.plot([90, top_lim], [90, top_lim], color='red', linestyle='-', linewidth=0.5)

    if dense:
        # Add colorbar to the second plot
        plt.colorbar(sc2, ax=ax2, label='Point Density Estimate')

    ax1.grid(linewidth=0.5, alpha=0.8, color='lightgray', linestyle='--')
    ax2.grid(linewidth=0.5, alpha=0.8, color='lightgray', linestyle='--')

    # Add legend for the red line y=x
    ax1.legend(handles=[line], loc='lower right')

    # Add title
    fig.suptitle(title, fontsize=16)

    if save:
        plt.savefig(IMG_PATH + '/' + filename, dpi=300, bbox_inches='tight')

    plt.show()

def plot_linear_regression(show=True, save=False, filename='linear_reg.pdf'):
    """Plot a linear regression model fitted to a synthetic dataset.

    Keyword Arguments:
        show -- show the plot (default: {True})
        save -- save the plot to a file (default: {False})
        filename -- name of the file to save the plot to (default: {'linear_reg.pdf'})
    """

    # Generate synthetic dataset
    np.random.seed(42)
    x = np.random.rand(50)
    y = 2 * x + 1 + np.random.normal(0, 0.1, 50)

    # Fit linear regression model
    coefficients = np.polyfit(x, y, 1)
    linear_regression = np.poly1d(coefficients)

    # Calculate predictions
    predictions = linear_regression(x)

    # Plot the data points and the fitted line
    plt.scatter(x, y, label='Data points', color='blue')
    plt.plot(x, predictions, color='red', label='Fitted line')
    plt.grid(linewidth=0.5, alpha=0.5, color='lightgray')

    # Plot the differences between predictions and actual values
    for i in range(len(x)):
        plt.plot([x[i], x[i]], [y[i], predictions[i]],
                 color='green', linestyle='dashed', linewidth=1)

    # Create a custom legend entry for the differences
    differences_legend = mlines.Line2D(
        [], [], color='green', linestyle='dashed', linewidth=1, label='Differences')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(handles=[plt.scatter([], [], color='blue', label='Data points'),
                        mlines.Line2D([], [], color='red',label='Fitted line'),
                        differences_legend])

    # with Differences between Predictions and Actual Values
    plt.title('Linear Regression')

    # Show or save the figure
    if save:
        plt.savefig(IMG_PATH + '/' + filename)
    if show:
        plt.show()

def plot_relu(show=True, save=False, filename='relu.pdf'):
    """Plot the ReLU activation function.

    Keyword Arguments:
        show -- show the plot (default: {True})
        save -- save the plot to a file (default: {False})
        filename -- name of the file to save the plot to (default: {'relu.pdf'})
    """
    # Define the ReLU function
    def relu(x):
        return np.maximum(0, x)

    # Create the x values from -10 to 10
    x = np.linspace(-10, 10, 1000)

    # Compute the ReLU values for the x values
    y = relu(x)

    # Create the plot
    plt.plot(x, y, label="ReLU(x) = max(0, x)")
    plt.xlabel("x")
    plt.ylabel("ReLU(x)")
    plt.grid(linewidth=0.5, alpha=0.9, color='lightgray')
    plt.xlim(-10, 10)
    plt.ylim(-1, 10)
    plt.title("ReLU Activation Function")

    # Add the label in the top-left corner
    plt.legend(loc="upper left")

    # Show or save the figure
    if save:
        plt.savefig(IMG_PATH + '/' + filename)
    if show:
        plt.show()

def show_correlation_matrix(data, title="Correlation Matrix Heatmap",
                            text_out=False, plot_out=True, show=True,
                            save=False, filename="correlation_matrix.png"):
    """Visualize the correlation matrix of a Pandas DataFrame (text or plot output)

    Arguments:
        data -- Pandas DataFrame

    Keyword Arguments:
        title -- title of the plot (default: {"Correlation Matrix Heatmap"})
        text_out -- print the correlation matrix (default: {False})
        plot_out -- show the correlation matrix plot (default: {True})
        show -- show the plot (default: {True})
        save -- save the plot (default: {False})
        save_path -- path to save the plot (default: {"correlation_matrix.png"})
    """
    # Calculate the correlation matrix
    correlation_matrix = data.corr(method="pearson")

    if text_out:  # Print the correlation matrix
        print(correlation_matrix)

    # Create a heatmap of the correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True,
                cmap="coolwarm", vmin=-1, vmax=1)

    # Customize the plot (optional)
    plt.title(title)
    plt.xticks(rotation=45, ha='right')

    if save: # Save the plot
        plt.savefig(IMG_PATH + '/' + filename, bbox_inches='tight')

    if plot_out: # Show the plot
        plt.show()

## -------------------- CROSS VALIDATION --------------------
# Inspired by: NASLib (https://github.com/automl/NASLib)
# License: Apache License 2.0
def cross_validation(xtrain, ytrain, predictor, k=3, score_metric="kendalltau"):
    """Cross validation for a accuracy predictors

    Arguments:
        xtrain -- training data
        ytrain -- training labels
        predictor -- predictor to be evaluated
        k -- number of folds (default: 3)

    Keyword Arguments:
        score_metric -- score metric to be used (default: {"kendalltau"})

    Returns:
        validation_score -- mean validation score
    """

    validation_score = []
    kfold = KFold(n_splits=k)

    for train_indices, validation_indices in kfold.split(xtrain):
        xtrain_i = xtrain[train_indices]
        ytrain_i = ytrain[train_indices]
        xval_i = xtrain[validation_indices]
        yval_i = ytrain[validation_indices]

        predictor.fit(xtrain_i, ytrain_i)
        ypred_i = predictor.predict(xval_i)

        # If the predictor is an ensemble, take the mean
        if len(ypred_i.shape) > 1:
            ypred_i = np.mean(ypred_i, axis=0)

        # Calculate the score specified by a user
        if score_metric == "pearson":
            score_i = np.abs(np.corrcoef(yval_i, ypred_i)[1, 0])
        elif score_metric == "mae":
            score_i = np.mean(abs(ypred_i - yval_i))
        elif score_metric == "rmse":
            score_i = metrics.mean_squared_error(
                yval_i, ypred_i, squared=False)
        elif score_metric == "spearman":
            score_i = stats.spearmanr(yval_i, ypred_i)[0]
        elif score_metric == "kendalltau":
            score_i = stats.kendalltau(yval_i, ypred_i)[0]

        # Add the score to the list of scores
        validation_score.append(score_i)

    return np.mean(validation_score)

## ------------------------- LOGGING -------------------------
# Inspired by: NASLib (https://github.com/automl/NASLib)
# License: Apache License 2.0
def get_logger():
    """Get logger for the current module

    Returns:
        logger -- logger for the current module
    """
    time_format = "%m/%d %H:%M:%S"
    fmt = "[%(asctime)s] %(levelname)s (%(name)s) %(message)s"
    formatter = logging.Formatter(fmt, time_format)
    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    
    return logger

## ------------------------- METRICS -------------------------
# Inspired by: NASLib (https://github.com/automl/NASLib)
# License: Apache License 2.0
class NamedAverageMeter:
    """Computes and stores the average and current value, ported from naszilla repo"""

    def __init__(self, name, fmt=":f"):
        """Initialization of AverageMeter

        Arguments:
            name -- Name to display. 

        Keyword Arguments:
            fmt -- Format string to print the values. (default: {":f"})
        """
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        """Reset the meter"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Update the meter"""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        """String representation of the meter"""
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        """Summary of the meter"""
        fmtstr = "{name}: {avg" + self.fmt + "}"
        return fmtstr.format(**self.__dict__)

# Inspired by: NASLib (https://github.com/automl/NASLib)
# License: Apache License 2.0
class AverageMeterGroup:
    """Average meter group for multiple average meters, ported from Naszilla repo."""

    def __init__(self):
        self.meters = OrderedDict()

    def update(self, data, n=1):
        """Update the meter"""
        for k, v in data.items():
            if k not in self.meters:
                self.meters[k] = NamedAverageMeter(k, ":4f")
            self.meters[k].update(v, n=n)

    def __getattr__(self, item):
        """Get the attribute"""
        return self.meters[item]

    def __getitem__(self, item):
        """Get the item"""
        return self.meters[item]

    def __str__(self):
        """String representation of the meter"""
        return "  ".join(str(v) for v in self.meters.values())

    def summary(self):
        """Summary of the meter"""
        return "  ".join(v.summary() for v in self.meters.values())