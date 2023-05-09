import sys
import numpy as np
import logging
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.neighbors import KernelDensity
from collections import OrderedDict
from scipy import stats
from sklearn.model_selection import KFold



# TODO: make me relative
OVERLEAF_PATH = "C:/Users/simon/Desktop/VUT/bacalarka/overleaf/imgs"


def get_val_accs_weights(dataset):
    dataset_size = len(dataset)
    val_accs = np.zeros(dataset_size)
    weights = np.zeros(dataset_size)

    for i in range(dataset_size):
        sample = dataset[i]
        val_accs[i] = sample['val_acc']
        weights[i] = sample['trainable_parameters']  
    return val_accs, weights

def get_targets(dataset):
    """Get list of targets for each architecture in the dataset."""
    dataset_size = len(dataset)
    for i in range(dataset_size):
        sample = dataset[i]
        yield sample['val_acc']

def get_flat_features(dataset):
    """Get list of flattened features for each architecture in the dataset."""
    dataset_size = len(dataset)
    adj_matrices = np.zeros((dataset_size, 7, 7))
    ops = np.zeros((dataset_size, 7, 5))

    features = []
    for i in range(dataset_size):
        
        # Adjacency and operations should be one-hot encoded already
        arch = dataset[i]
        adj_matrices[i] = arch["adjacency"][0] # TODO: why [0], because it is a list
        ops[i] = arch["operations"]
        
        # Flatten and concatenate
        adj_flat = adj_matrices[i].flatten()
        ops_flat = ops[i].flatten()
        
        f = np.concatenate((adj_flat, ops_flat))
        yield f
    
def get_targets2(dataset):
    dataset_size = len(dataset)
    targets = np.zeros(dataset_size)

    for i in range(dataset_size):
        sample = dataset[i]
        targets[i] = sample['val_acc']

    return targets

def get_flat_features2(dataset):
    dataset_size = len(dataset)

    adj_matrices = np.zeros((dataset_size, 7, 7))
    ops = np.zeros((dataset_size, 7, 5))

    for i in range(dataset_size):
        arch = dataset[i]
        adj_matrices[i] = arch["adjacency"][0] # TODO: why [0], because it is a list
        ops[i] = arch["operations"]

    # Flatten and concatenate using NumPy operations
    adj_matrices_flat = adj_matrices.reshape(dataset_size, -1)
    ops_flat = ops.reshape(dataset_size, -1)
    features = np.concatenate((adj_matrices_flat, ops_flat), axis=1)

    return features

def get_gcn_features(dataset):
    dataset_size = len(dataset)
    
    for i in range(dataset_size):
        arch = dataset[i]
        
        gcn_f = {
            "num_vertices": arch["num_vertices"],
            "adjacency": arch["adjacency"][0], # TODO: beware of this [0]
            "operations": arch["operations_onehot"],
            "mask": arch["mask"],
            "val_acc": arch["val_acc"]
        }
        
        yield gcn_f
        
    

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
    plt.scatter(y_true, y_pred, c=z, cmap=cmap, s=s, edgecolors='face', alpha=a)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot([0, 100], [0, 100], color='darkgray', linestyle='-', linewidth=0.5)
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.grid(linewidth=0.25, alpha=0.5, color='lightgray')
    
    # Add title
    plt.title(title)
    
    if dense:
        plt.colorbar(label='Density')
    
    if save:
        plt.savefig(filename)
    
    plt.show()


def scatter_plot2(
    y_true, y_pred, title,
    b=0.05, cmap=None, s=4, a=0.75, 
    xlabel='Actual values [%]', ylabel='Predicted values [%]', 
    save=False, filename='example.png', dense=False):
    """Create a scatter plot, optionally with gradient coloring based on the density estimate of the data points.

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
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Create the basic scatter plot
    sc1 = ax1.scatter(y_true, y_pred, c=z, cmap=cmap, s=s, edgecolors='face', alpha=a)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.plot([0, 100], [0, 100], color='darkgray', linestyle='-', linewidth=0.5)
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 100)

    # Create the zoomed-in scatter plot
    sc2 = ax2.scatter(y_true, y_pred, c=z, cmap=cmap, s=s, edgecolors='face', alpha=a)
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel)
    ax2.plot([90, 100], [90, 100], color='darkgray', linestyle='-', linewidth=0.5)
    
    # TODO add threshold for the lower bound
    ax2.set_xlim(90, 97)
    ax2.set_ylim(90, 97)
    
    ax1.grid(linewidth=0.25, alpha=0.5, color='lightgray')
    ax2.grid(linewidth=0.25, alpha=0.5, color='lightgray')

    # Add a colorbar for the density
    # fig.colorbar(sc1, ax=[ax1, ax2], label='Density')

    # if dense:
    #     # Add a colorbar
    #     fig.subplots_adjust(right=1.95)
    #     cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
    #     fig.colorbar(hb2, cax=cbar_ax, label='Density')

    # Add title
    fig.suptitle(title)
     
    if save:
        plt.savefig(filename)
    
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
        plt.plot([x[i], x[i]], [y[i], predictions[i]], color='green', linestyle='dashed', linewidth=1)

    # Create a custom legend entry for the differences
    differences_legend = mlines.Line2D([], [], color='green', linestyle='dashed', linewidth=1, label='Differences')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(handles=[plt.scatter([], [], color='blue', label='Data points'),
                        mlines.Line2D([], [], color='red', label='Fitted line'),
                        differences_legend])
    
    plt.title('Linear Regression') # with Differences between Predictions and Actual Values
    
    # Show or save the figure
    if save:
        plt.savefig(OVERLEAF_PATH + '/' + filename)
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
        plt.savefig(OVERLEAF_PATH + '/' + filename)
    if show:
        plt.show()
  
def cross_validation_old(xtrain, ytrain, predictor, split_indices, score_metric="kendalltau"):
    """Cross validation for a predictor

    Arguments:
        xtrain -- training data
        ytrain -- training labels
        predictor -- predictor to be evaluated
        split_indices -- split indices for cross validation

    Keyword Arguments:
        score_metric -- score metric to be used (default: {"kendalltau"})

    Returns:
        validation_score -- mean validation score
    """
    validation_score = []
    # Print predictor hyperparameters
    #print("Hyperparams: ", predictor.get_hyperparams())

    for train_indices, validation_indices in split_indices:
        xtrain_i = [xtrain[j] for j in train_indices]
        ytrain_i = [ytrain[j] for j in train_indices]
        xval_i = [xtrain[j] for j in validation_indices]
        yval_i = [ytrain[j] for j in validation_indices]

        #print("train_indices", train_indices)
        #print("validation_indices", validation_indices)

        predictor.fit(xtrain_i, ytrain_i)
        ypred_i = predictor.predict(xval_i)

        # If the predictor is an ensemble, take the mean
        if len(ypred_i.shape) > 1:
            ypred_i = np.mean(ypred_i, axis=0)

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

        validation_score.append(score_i)

    #print("Validation scores: ", validation_score)
    return np.mean(validation_score)

def cross_validation(xtrain, ytrain, predictor, k=3, score_metric="kendalltau"):
    """Cross validation for a predictor

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
    # Print predictor hyperparameters
    #print("Hyperparams: ", predictor.get_hyperparams())

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


        validation_score.append(score_i)

    return np.mean(validation_score)


def generate_kfold(n, k):
    """Generate kfold indices

    Arguments:
        n -- number of training examples
        k -- number of folds
    Returns:
        kfold_indices: a list of len k. Each entry takes the form
        (training indices, validation indices)
    """
    assert k >= 2
    kfold_indices = []

    indices = np.array(range(n))
    fold_size = n // k

    fold_indices = [
        indices[i * fold_size: (i + 1) * fold_size] for i in range(k - 1)]
    fold_indices.append(indices[(k - 1) * fold_size:])

    for i in range(k):
        training_indices = [fold_indices[j] for j in range(k) if j != i]
        validation_indices = fold_indices[i]
        kfold_indices.append(
            (np.concatenate(training_indices), validation_indices))

    return kfold_indices

def loguniform(low=0, high=1, size=None):
    return np.exp(np.random.uniform(np.log(low), np.log(high), size))

class NamedAverageMeter:
    """Computes and stores the average and current value, ported from naszilla repo"""

    def __init__(self, name, fmt=":f"):
        """
        Initialization of AverageMeter
        Parameters
        ----------
        name : str
            Name to display.
        fmt : str
            Format string to print the values.
        """
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = "{name}: {avg" + self.fmt + "}"
        return fmtstr.format(**self.__dict__)

class AverageMeterGroup:
    """Average meter group for multiple average meters, ported from Naszilla repo."""

    def __init__(self):
        self.meters = OrderedDict()

    def update(self, data, n=1):
        for k, v in data.items():
            if k not in self.meters:
                self.meters[k] = NamedAverageMeter(k, ":4f")
            self.meters[k].update(v, n=n)

    def __getattr__(self, item):
        return self.meters[item]

    def __getitem__(self, item):
        return self.meters[item]

    def __str__(self):
        return "  ".join(str(v) for v in self.meters.values())

    def summary(self):
        return "  ".join(v.summary() for v in self.meters.values())
    
def get_logger():
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