# Refence: https://syftbox-documentation.openmined.org/cpu-tracker-2

import os
import re
import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path
from syftbox.lib import Client
from sklearn.preprocessing import LabelEncoder

API_NAME = os.getenv("API_NAME")

def network_participants(datasite_path: Path) -> list[str]:
    """
    Network Participants Discovery:
    Retrieves a list of user directories (participants) in a given datasite path. 
    This function scans the network for all available peers by looking at directories in the datasite path.
    
    By looking for "api_data / API_NAME" only those from the specific app will be considered.
    
    """

    entries = os.listdir(datasite_path)
    users = []

    for entry in entries:
        if Path(datasite_path / entry / "api_data" / API_NAME).is_dir():
            users.append(entry)

    return users

def extract_number(file_name):
    match = re.search(r'_(\d+)\.joblib$', file_name)
    return int(match.group(1)) if match else -1


def get_users_mlp_parameters(
        datasites_path: Path, peers: list[str]
) -> tuple[list, list]:
    """
    This method retrieve the parameters from the local trained MLP. Those files have the names:
    - netflix_mlp_weights_<NUM_SAMPLES>.joblib
    - netflix_mlp_bias_<NUM_SAMPLES>.joblib

    Returns a tuple of (weights, biases) from all participants
    """
    
    weights = []
    biases = []

    for peer in peers:
        dir = datasites_path / peer / "api_data" / API_NAME

        weight = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f)) and "mlp_weights" in f]
        bias = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f)) and "mlp_bias" in f]
        weight = max(weight, key=extract_number, default=None)  # get the greater 
        bias = max(bias, key=extract_number, default=None)      # get the greater

        weights.append(dir / weight)
        biases.append(dir / bias)

    return weights, biases

def weighted_average(parameters, samples):
    total_samples = sum(samples)
    weighted_params = [
        np.multiply(param, n / total_samples) for param, n in zip(parameters, samples)
    ]
    return np.sum(weighted_params, axis=0)


def mlp_fedavg(weights: list, biases: list) -> tuple[list, list]:
    """
    FedAvg computes the weighted average of parameters (weights and biases) from multiple users.
    The weights for averaging are proportional to the number of samples each user has.
    """
    samples = [extract_number(str(n)) for n in weights]
    
    weight_matrices = [joblib.load(weight_path) for weight_path in weights]
    bias_vectors = [joblib.load(bias_path) for bias_path in biases]

    fedavg_weights = [weighted_average([w[layer] for w in weight_matrices], samples) for layer in range(len(weight_matrices[0]))]
    fedavg_biases = [weighted_average([b[layer] for b in bias_vectors], samples) for layer in range(len(bias_vectors[0]))]

    return fedavg_weights, fedavg_biases
    
def create_vocab():
    df = pd.read_csv("./data/netflix_series_2024-12.csv.zip")

    label_encoder = LabelEncoder()
    label_encoder.fit(df['Title'])

    vocab_mapping = {title: idx for idx, title in enumerate(label_encoder.classes_)}
    
    with open('./data/vocabulary.json', 'w') as f:
        json.dump(vocab_mapping, f)

if __name__ == "__main__":
    client = Client.load()

    # Create a Vocabulary of TV Series
    create_vocab()

    datasite_path = Path(client.datasite_path.parent)   # automatically retrieve datasites path

    peers = network_participants(datasite_path)         # check participant of netflix trend
    
    # MLP use case -> FedAvg
    weights, biases = get_users_mlp_parameters(datasite_path, peers)    # MLP: retrieve the path to weights and bias
    fedavg_weights, fedavg_biases = mlp_fedavg(weights, biases)
    
    output_mlp_fedavg = client.datasite_path / "public"
    
    joblib.dump(fedavg_weights, output_mlp_fedavg / "netflix_mlp_fedavg_weights.joblib")
    joblib.dump(fedavg_biases, output_mlp_fedavg / "netflix_mlp_fedavg_biases.joblib")

    