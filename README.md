Anomaly Detection in Network Traffic using an LSTM Autoencoder
This project implements an unsupervised anomaly detection system for network intrusion detection using a Long Short-Term Memory (LSTM) Autoencoder. The model is trained on the NSL-KDD dataset to learn the patterns of normal network traffic and identify deviations that signify potential attacks.

Table of Contents
Problem Statement

Core Idea

Methodology

Dataset

Data Preprocessing

Model Architecture

How It Works

Results

Getting Started

Prerequisites

Installation

Usage

Project Structure

License

Problem Statement
With the increasing sophistication of cyber threats, traditional signature-based Intrusion Detection Systems (IDS) are often insufficient. They struggle to detect novel, zero-day attacks for which no predefined rules exist. This project addresses the need for a more dynamic and intelligent detection system that can identify anomalous behavior without prior knowledge of attack signatures.

Core Idea
The fundamental principle is to use an unsupervised learning approach. We train an LSTM Autoencoder exclusively on normal network traffic. The model learns to reconstruct this normal data with very high accuracy, resulting in a low reconstruction error. When the model encounters anomalous (attack) traffic, its patterns will deviate significantly from what it learned. This causes the model to struggle with reconstruction, leading to a high reconstruction error, which serves as a clear indicator of an anomaly.

Methodology
Dataset
The project utilizes the NSL-KDD dataset, an improved version of the classic KDD '97 dataset. It's preferred for this task because it has addressed some of the original dataset's inherent issues, such as the removal of redundant and duplicate records, which allows for a more robust and unbiased model evaluation.

You can download the dataset from the University of New Brunswick.

Data Preprocessing
The raw data was transformed into a suitable format for the LSTM model through the following steps:

Binary Classification: The 23 distinct traffic labels (1 normal, 22 attack types) were mapped into a binary format: 0 for Normal and 1 for Attack.

Label Encoding: Categorical features (e.g., protocol_type, service, flag) were converted into numerical representations using sklearn.preprocessing.LabelEncoder.

Feature Scaling: All numerical features were scaled to a range of [0, 1] using sklearn.preprocessing.MinMaxScaler. This is crucial for neural networks to prevent features with large value ranges from disproportionately influencing the model's learning.

Reshaping for LSTM: The data was reshaped into a 3D tensor of shape (samples, timesteps, features), which in this case was (n_samples, 1, 41), to match the input requirements of the LSTM layers.

Model Architecture
The model is a deep LSTM Autoencoder built with TensorFlow/Keras. It consists of an encoder and a decoder.

Encoder: Compresses the input data into a lower-dimensional latent representation.

Input Layer (41 features)

LSTM Layer (128 units)

LSTM Layer (64 units) -> Latent Space

Decoder: Attempts to reconstruct the original input data from the latent representation.

RepeatVector Layer

LSTM Layer (64 units)

LSTM Layer (128 units)

TimeDistributed Dense Layer (41 units) -> Reconstructed Output

Input (41 features) -> LSTM(128) -> LSTM(64) -> [Latent Vector] -> RepeatVector -> LSTM(64) -> LSTM(128) -> Output (41 features)

How It Works
The anomaly detection process follows these two phases:

Training Phase: The LSTM autoencoder is trained only on normal data. The objective is to minimize the reconstruction error (Mean Squared Error) between the input and the output. This forces the model to learn the intricate patterns of normal behavior.

Inference & Detection Phase:

The trained model is used to reconstruct new, unseen network traffic (from the test set, which contains both normal and attack data).

The reconstruction error is calculated for each data point.

A threshold is determined by finding the value that best separates the reconstruction errors of normal and attack data (optimized by maximizing the F1-score).

If a data point's reconstruction error is above the threshold, it is flagged as an Anomaly (Attack). Otherwise, it is considered Normal.

Results
The model's performance was evaluated on the KDDTest+ dataset. The optimal threshold for distinguishing anomalies was found to be 0.08.

Performance Metrics
Metric

Score

Accuracy

92.42%

Precision

96.95%

Recall

87.79%

F1-Score

92.15%

Key Visualizations
1. Reconstruction Error Distribution

This plot clearly shows that the reconstruction error for normal traffic is tightly concentrated at a low value, while the error for attack traffic is much higher and more spread out. This separation is what makes the detection possible.

2. Confusion Matrix

The confusion matrix provides a detailed look at the classification results, showing a high number of true positives and true negatives.

Getting Started
Prerequisites
This project requires Python 3.8+ and the following libraries:

TensorFlow

NumPy

Pandas

Scikit-learn

Matplotlib

Seaborn

Installation
Clone the repository:

git clone https://github.com/your-username/lstm-autoencoder-ids.git
cd lstm-autoencoder-ids

Install the required packages:

pip install -r requirements.txt

(You may need to create a requirements.txt file containing the libraries listed above)

Usage
Download the NSL-KDD dataset and place the KDDTrain+.txt and KDDTest+.txt files in the project's root directory.

Open and run the lstm_autoencoder.ipynb Jupyter Notebook to train the model, evaluate its performance, and see the visualizations.

Project Structure
.
├── lstm_autoencoder.ipynb      # Jupyter Notebook with all the code
├── KDDTrain+.txt               # Training data (must be downloaded)
├── KDDTest+.txt                # Testing data (must be downloaded)
└── README.md                   # This file
