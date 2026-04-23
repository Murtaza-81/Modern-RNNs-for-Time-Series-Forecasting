# CS511 Assignment 2: Modern RNNs for Time Series Forecasting

## Overview
This project implements and compares two recurrent neural network architectures for multivariate time series forecasting on the ETTh1 (Electricity Transformer Temperature) dataset.

## Models Implemented
- **GRU Baseline**: Standard Gated Recurrent Unit with 2 layers and MLP head
- **iGRU Variant**: Improved GRU with cross-channel mixing block to capture inter-variable dependencies

## Key Features
Complete data pipeline (70/10/20 train/val/test split)
Z-score normalization using training statistics only
Sliding window dataset (L=96, H=24)
Multi-step forecasting (predict 24 hours ahead)
Early stopping to prevent overfitting
Comprehensive metrics (RMSE, MAE, MAPE)
Efficiency profiling (parameters, training time, inference time)
Reproducible results (fixed random seed 42)

## Results

| Model | RMSE | MAE | MAPE | Parameters | Inference (ms/batch) |
|-------|------|-----|------|------------|---------------------|
| GRU | 0.2221 | 0.1670 | 19.72% | 41,848 | 128.3 |
| iGRU | 0.2252 | 0.1635 | 18.21% | 41,974 | 104.4 |

## Key Findings
- **iGRU achieves 7.7% lower MAPE** than baseline GRU
- **18.6% faster inference** with only 0.3% more parameters
- Cross-channel mixing effectively captures correlations between the 7 input features

## Dataset
**ETTh1 (Electricity Transformer Temperature - Hourly)**
- 17,420 samples (July 2016 - June 2018)
- 7 features: HUFL, HULL, MUFL, MULL, LUFL, LULL, OT
- Target: OT (Oil Temperature)

## Requirements
- Python 3.8+
- PyTorch 2.0+
- NumPy, Pandas, Matplotlib, scikit-learn

## How to Run

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/rnn-time-series-forecasting.git
cd rnn-time-series-forecasting

# Install dependencies
pip install -r requirements.txt

# Run complete experiment
python run_experiments.py

# Or run individual components
python train.py          # Train both models
python evaluate.py       # Evaluate on test set
python efficiency.py     # Profile model efficiency
