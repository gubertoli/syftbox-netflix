import json
import os
import numpy as np
import pandas as pd
from pathlib import Path

API_NAME = os.getenv("API_NAME")
AGGREGATOR_DATASITE = os.getenv("AGGREGATOR_DATASITE")

def calculate_top5(files: list[Path], destination_folder: Path, vocab: Path):
    """
    Calculates the top-5 most seen (number of episodes) series.
    """
    data = []

    for vector in files:
        if vector.is_file(): data.append(np.load(vector))

    result = np.vstack(data)
    series_totals = result.sum(axis=0)

    # Load the series mapping (index to name) from the JSON file
    try:
        with open(vocab, 'r') as f:
            series_mapping = json.load(f)

        # Reverse the mapping to go from index to name
        index_to_name = {v: k for k, v in series_mapping.items()}
    except:
        print(f"> Error: {API_NAME} | Aggregator: {AGGREGATOR_DATASITE} | Unable to open vocab -> {str(vocab)}")

    destination_folder.mkdir(parents=True, exist_ok=True)
    # Get the indices of the top-5 most-watched series
    top5_indices = np.argsort(series_totals)[-5:][::-1]
    top5_names = [index_to_name[idx] for idx in top5_indices]
    top5_values = series_totals[top5_indices]

    csv_file_path = "./aggregator/data/netflix_series_2024-12.csv.zip"

    try:
        df = pd.read_csv(csv_file_path, compression='zip')
    except Exception as e:
        print(f"> Error: Unable to read the CSV from {csv_file_path}. Error: {e}")
        return
    
    top5_data = []
    for idx, name, count in zip(top5_indices, top5_names, top5_values):
        # Locate the row in the DataFrame with the matching title
        row = df[df["Title"].str.strip() == name].iloc[0]
        entry = {
            "id": int(idx),  # Use the index from `top5_indices`
            "name": name,
            "language": row["Language"],
            "rating": row["Rating"],
            "imdb": row["IMDB"] if pd.notna(row["IMDB"]) else "N/A",  # Default to N/A if IMDB is missing
            "img": row["Cover URL"],
            "count": int(count)
        }
        top5_data.append(entry)

    # Save to a JSON file
    try:
        with open(destination_folder / "top5_series.json", 'w') as f:
            json.dump(top5_data, f, indent=4)
        print(f"Top 5 series saved to {destination_folder / 'top5_series.json'}")
    except Exception as e:
        print(f"> Error: Unable to save the JSON. Error: {e}")

def dp_top5_series(datasites_path: Path, peers: list[str], min_participants: int):
    """
    Retrieves the path of all available participants with DP vectors of TV series seens episodes.
    """
    available_dp_vectors = []
    dp_file = "top5_series_dp.npy"

    for peer in peers:
        dir: Path = datasites_path / peer / "api_data" / API_NAME
        file: Path = dir / dp_file
        
        if file.exists():
            # Backwards compatibility - No profile hierarchy
            available_dp_vectors.append(file)
        else:
            # Current version - Profile hierarchy
            # Iterate through all profiles. Get all folders that start with "profile_"
            flr_prefix = "profile_"
            profiles = [f for f in os.listdir(dir) if os.path.isdir(os.path.join(dir, f)) and f.startswith(flr_prefix)]

            # Sort the profiles by the number at the end of the folder name
            profiles = sorted(profiles, key=lambda x: int(x.split("_")[-1]))

            for profile in profiles:
                profile_file = dir / profile / dp_file
                
                if not profile_file.exists():
                    print(f"DP file not found for {profile}. Skipping...")
                    continue
                    
                available_dp_vectors.append(profile_file)

    if len(available_dp_vectors) < min_participants:  
        print(f"{API_NAME} | Aggregator | There are no sufficient partcipants \
                (Available: {len(available_dp_vectors)}| Required: {min_participants})")
        return len(available_dp_vectors)
    else:
        destination_folder: Path = ( datasites_path / AGGREGATOR_DATASITE / "private" / API_NAME )
        vocab: Path = datasites_path / AGGREGATOR_DATASITE / "api_data" / API_NAME / "shared" / "tv-series_vocabulary.json"
        calculate_top5(available_dp_vectors, destination_folder, vocab)

    return len(available_dp_vectors)
