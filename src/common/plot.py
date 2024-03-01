import os
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import logging
import sys
# Configure the logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

# Path to the folder containing subfolders "eng" and "non-eng"
folder_path = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/datasets/x-vector-embeddings/displace-2lang-2sec-xVector-embeddings_full_fast"
logging.info(f"Loadding xVector embedding from {folder_path}")

lang2id = {"eng":0,"not-eng":1}


def read_hidden_features_with_labels(folder_path):
    hidden_features = []
    labels = []
    for label in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label)
        if os.path.isdir(label_path):
            for filename in os.listdir(label_path):
                file_path = os.path.join(label_path, filename)
                if filename.endswith(".csv"):
                    # Read the CSV file and reshape it as a 512-dimensional flattened row vector
                    hidden_feature_matrix = pd.read_csv(file_path, header=None).to_numpy().flatten()
                    if len(hidden_feature_matrix) != 512:
                        print(f"Skipping {file_path}: Expected 512-dimensional vector, got {len(hidden_feature_matrix)}")
                        continue
                    hidden_features.append(hidden_feature_matrix)
                    labels.append(lang2id[label])  # Use folder names as labels
    return np.array(hidden_features), np.array(labels)
# Function to preprocess hidden features
def preprocess_hidden_features(hidden_features):
    # Standardize the features
    scaler = StandardScaler()
    hidden_features_scaled = scaler.fit_transform(hidden_features)
    return hidden_features_scaled, scaler

# Read hidden features and labels from CSV files
hidden_features, labels = read_hidden_features_with_labels(folder_path)
print(f"Hidden features size is {hidden_features.shape} and lables are {labels.shape}")

# Preprocess hidden features
hidden_features_scaled, scaler = preprocess_hidden_features(hidden_features)

# Apply t-SNE for dimensionality reduction
tsne = TSNE(n_components=2, random_state=42)
embedded_features = tsne.fit_transform(hidden_features_scaled)

print("The embedded features are: ",embedded_features)

# Save the array to a .npz file
pathToSave = os.path.join("/nlsasfs/home/nltm-st/sujitk/yash-mtp/logs/plots","tsne-xVectorembedding.npz")
np.savez(pathToSave, data=embedded_features,  labels=labels)

logging.info(f"The array of hidden features reduced dimensional is saved to {pathToSave}")




# # Path to the folder containing subfolders "eng" and "non-eng"
# folder_path = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/datasets/wav2vec2/displace-2lang-2sec-HiddenFeatures-wave2vec2_full_fast"
# logging.info(f"Loadding hidden features from {folder_path}")

# lang2id = {"eng":0,"not-eng":1}

# # Function to read hidden features and labels from CSV files
# def read_hidden_features_with_labels(folder_path):
#     hidden_features = []
#     labels = []
#     for label in os.listdir(folder_path):
#         label_path = os.path.join(folder_path, label)
#         if os.path.isdir(label_path):
#             for filename in os.listdir(label_path):
#                 file_path = os.path.join(label_path, filename)
#                 if filename.endswith(".csv"):
#                     hidden_feature_matrix = pd.read_csv(file_path, header=None).values
#                     hidden_features.append(hidden_feature_matrix.reshape(-1))
#                     labels.append(lang2id[label])  # Use folder names as labels
#     return np.array(hidden_features), np.array(labels)

# # Function to preprocess hidden features
# def preprocess_hidden_features(hidden_features):
#     # Standardize the features
#     scaler = StandardScaler()
#     hidden_features_scaled = scaler.fit_transform(hidden_features)
#     return hidden_features_scaled, scaler

# # Read hidden features and labels from CSV files
# hidden_features, labels = read_hidden_features_with_labels(folder_path)
# print(f"Hidden features size is {hidden_features.shape} and lables are {labels.shape}")

# # Preprocess hidden features
# hidden_features_scaled, scaler = preprocess_hidden_features(hidden_features)

# # Apply t-SNE for dimensionality reduction
# tsne = TSNE(n_components=2, random_state=42)
# embedded_features = tsne.fit_transform(hidden_features_scaled)

# print("The embedded features are: ",embedded_features)

# # Save the array to a .npz file
# pathToSave = os.path.join("/nlsasfs/home/nltm-st/sujitk/yash-mtp/logs/plots","tsne-hiddenFeatures.npz")
# np.savez(pathToSave, data=embedded_features,  labels=labels)

# logging.info(f"The array of hidden features reduced dimensional is saved to {pathToSave}")