# Anomaly Detection in Network Traffic using LSTM Autoencoder

---

## Project Overview

This project implements unsupervised anomaly detection using a Long Short-Term Memory (LSTM) Autoencoder. The model learns to reconstruct normal network traffic sequences and flags anomalies based on high reconstruction error.

---

## Key Features

- Uses LSTM Autoencoder for sequence-based anomaly detection
- Trained only on normal traffic data
- Flags anomalies using reconstruction error thresholding
- Achieves 89% accuracy and 0.95 AUC on test data

---

## Dataset

- Dataset: NSL-KDD
- Source: [NSL-KDD on Kaggle](https://www.kaggle.com/datasets/hassan06/nslkdd))
- Contains network traffic records labeled as normal or attack types

---

## Tech Stack

- Python
- TensorFlow
- Pandas
- NumPy
- Matplotlib

---

## How It Works

1. Load and preprocess the NSL-KDD dataset
2. Apply one-hot encoding to categorical features
3. Scale numerical features using MinMaxScaler
4. Train an LSTM Autoencoder using only normal data
5. Compute reconstruction error on test data
6. Mark samples exceeding the threshold as anomalies

---

## Results

- Accuracy: 89%
- AUC (ROC): 0.95
- Precision: ~90%
- Recall: ~88%

Includes plots:
- ROC Curve
- Loss over epochs
- Reconstruction error distribution

---

## Getting Started

### Requirements

Install dependencies:

```bash
pip install tensorflow pandas numpy scikit-learn matplotlib
```

Run the Project:

```bash
python lstm_autoencoder_anomaly_detection.py
```

