# DGTANet

Dynamic Graph Temporal-Aware Attention Network for Multi-Horizon Quantitative Precipitation Forecasting.

This repository contains the official implementation of the paper:

**"Dynamic Graph and Temporal Convolution Network for Multi-Horizon Precipitation Forecasting in Complex Climatic Regions"** (submitted to *Computers & Geosciences*).

## Model Overview

DGTANet jointly models evolving spatial correlations and multi-scale temporal patterns for precipitation forecasting across diverse climatic regimes. It integrates:

- Dynamic Graph Learning Module
- Dilated Temporal Convolution Module
- Gated Fusion Layer

## Requirements

- Python >= 3.8
- PyTorch >= 1.10
- Numpy
- Pandas
- PyYAML

Install dependencies:

```bash
pip install -r requirements.txt
```

## Data Preparation

Prepare your GFS multi-variable dataset as a NumPy array with shape:

```
[num_samples, num_nodes, num_features]
```

Save it as `data/data.npy`.

## Training

```bash
python train.py
```

## Notes

- This code provides the structure and training process supporting the results reported in the paper.
- Users are expected to prepare their own datasets for training and evaluation.

## License

This project is licensed under the MIT License.
