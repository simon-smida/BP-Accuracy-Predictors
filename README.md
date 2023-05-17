# Design of accuracy predictors for CNNs in NAS
This project focuses on designing, implementing and comparing accuracy predictors (ML models) for Convolutional Neural Networks (CNNs) in the context of Neural Architecture Search (NAS).

## Description
This repository hosts the essential source code, datasets, and utility tools needed for constructing and evaluating accuracy predictors for CNNs using a range of machine learning models. The primary objectives include examining diverse input features, comparing distinct methodologies for developing model-based predictors, and enhancing the efficiency of NAS by delivering precise and computationally effective performance estimates for a variety of CNN architectures.

![predictor](https://github.com/xsmida03/BP-Accuracy-Predictors/blob/main/imgs/predictor.png)

## Search Space
The chosen dataset for trained CNN architectures and their corresponding metrics is [NAS-Bench-101](https://github.com/google-research/nasbench), which consists of 423,624 architectures.

To utilize the NAS-Bench-101 API, it is advised to clone the up-to-date (as of May 10, 2023) NAS-Bench-101 repository from [here](https://github.com/xsmida03/nasbench). In the scope of this project, the NAS-Bench-101 API is only required for converting the official `nasbench_full.tfrecord` file to a faster `hdf5` file format.
- official `tfrecord` file (slow): https://storage.googleapis.com/nasbench/nasbench_full.tfrecord
- `hdf5` file (fast): TODO

To convert between the tfrecord and hdf5 file formats yourself, you can utilize the [`tfrecord_to_hdf5.py`](https://github.com/xsmida03/BP-Accuracy-Predictors/blob/main/tfrecord_to_hdf5.py) script provided.

## Dependencies
- `PyTorch` - follow [official guide](https://pytorch.org/get-started/locally/) according to your system
  - not included in `requirements.txt`
  - cuda recommended
- `NAS-Bench-101` - only if you want to run `tfrecord_to_hdf5.py` (convert `tfrecord` to `hdf5`)
  - follow the steps from [this repository](https://github.com/xsmida03/nasbench) to install nas-bench-101 api 

## Getting Started
1. Clone the repository: `git clone https://github.com/xsmida03/BP-Accuracy-Predictors.git`
2. Go to the main directory: `cd BP-Accuracy-Predictors`
3. Create virtual environment: `virtualenv venv`
4. Activate the environment: 
    - `./venv/Scripts/activate` (Windows)
    - `source ./venv/bin/activate` (Linux)
6. Make sure you have installed necessary dependencies (above)
7. Install necessary packages: `pip install -r requirements.txt`
8. Recommended: review the jupyter notebooks:
    - `analysis.ipynb`
    - `correlation_analysis.ipynb`
    - `hyperparameter_tuning.ipynb`

## Example training
- `XGBPredictor` training (on 172 NAS-Bench-101 architectures) and applying to predict accuracies of the whole dataset
```python
import numpy as np

from dataset import NASBench101Dataset
from utils import get_targets, get_flat_features
from predictors.xgb import XGBPredictor

# Load NAS-Bench-101 dataset (172 training samples)
dataset = NASBench101Dataset('data/nasbench101.hdf5', "172") 
dataset_all = NASBench101Dataset('data/nasbench101.hdf5', "all")

# Get the features
features_train = get_flat_features(dataset)
features_all = get_flat_features(dataset_all)
targets_train = get_targets(dataset)

# XGB-based predictor
xgb_predictor = XGBPredictor()

# Training
xgb_predictor.fit(features_train, targets_train)

# Prediction
xgb_predictor.predict(features_all)
```
![xgboost](https://github.com/xsmida03/BP-Accuracy-Predictors/blob/main/imgs/xgb_predictor100k.png)

## Awards
**[Excel@fit](https://excel.fit.vutbr.cz/)**: Awarded by an [expert panel](https://excel.fit.vutbr.cz/vysledky/#oceneni-odbornym-panelem) (see [paper](https://excel.fit.vutbr.cz/submissions/2023/082/82.pdf) and [poster](https://excel.fit.vutbr.cz/submissions/2023/082/82_poster.pdf) for more information)

## License
This project is released under the [MIT License](https://opensource.org/license/mit/).
