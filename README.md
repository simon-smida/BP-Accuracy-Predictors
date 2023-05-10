# Design of accuracy predictors for CNNs in NAS
This project focuses on designing, implementing and comparing accuracy predictors (ML models) for Convolutional Neural Networks (CNNs) in the context of Neural Architecture Search (NAS).

## Overview
This repository hosts the essential source code, datasets, and utility tools needed for constructing and evaluating accuracy predictors for CNNs using a range of machine learning models. The primary objectives include examining diverse input features, comparing distinct methodologies for developing model-based predictors, and enhancing the efficiency of NAS by delivering precise and computationally effective performance estimates for a variety of CNN architectures.

![predictor](https://github.com/xsmida03/BP-Accuracy-Predictors/blob/main/imgs/predictor.png)

## Project Details
### Dataset - `NAS-Bench-101`
The chosen dataset for trained CNN architectures and their corresponding metrics is [NAS-Bench-101](https://github.com/google-research/nasbench), which consists of 423,624 architectures.

To utilize the NAS-Bench-101 API, it is advised to clone the up-to-date (as of May 10, 2023) NAS-Bench-101 repository from [here](https://github.com/xsmida03/nasbench). In the scope of this project, the NAS-Bench-101 API is only required for converting the official `nasbench_full.tfrecord` file to a faster `hdf5` file format.
- official `tfrecord` file (slow): https://storage.googleapis.com/nasbench/nasbench_full.tfrecord
- `hdf5` file (fast): TODO

To convert between the tfrecord and hdf5 file formats yourself, you can utilize the [`tfrecord_to_hdf5.py`](https://github.com/xsmida03/BP-Accuracy-Predictors/blob/main/tfrecord_to_hdf5.py) script provided.

## Getting Started
1. Clone the repository: `git clone https://github.com/xsmida03/BP-Accuracy-Predictors.git`
2. Install required packages: `pip install -r requirements.txt`
3. Look at the main script, `analysis.ipynb` - run cells, make custom modifications

## Awards
**[Excel@fit](https://excel.fit.vutbr.cz/)**: Awarded by an [expert panel](https://excel.fit.vutbr.cz/vysledky/#oceneni-odbornym-panelem) (see [paper](https://excel.fit.vutbr.cz/submissions/2023/082/82.pdf) and [poster](https://excel.fit.vutbr.cz/submissions/2023/082/82_poster.pdf) for more information)

## License
This project is released under the [MIT License](https://opensource.org/license/mit/).
