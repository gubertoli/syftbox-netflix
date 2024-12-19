import os
import re
import json
import pandas as pd
import numpy as np
from pathlib import Path
from rapidfuzz import process

class SequenceData:
    """
    This class creates, from the original data, an ordered dataframe from oldest to newest,
    with the attributes First_Seen (date), a number of episodes seen.
    """
    def __init__(self, dataset: np.ndarray):
        self.dataset = dataset
        self.aggregated_data = self.process_dataset()

    def extract_features(self, df):
        # Extract show name and season from title
        df['show'] = df['Title'].apply(lambda x: x.split(':')[0] if ':' in x else x)
        df['season'] = df['Title'].apply(lambda x: 
            int(re.search(r'Season (\d+)', x).group(1)) if re.search(r'Season (\d+)', x) else 0)
        
        # Convert date strings to datetime objects
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
        
        # Extract temporal features
        df['day_of_week'] = df['Date'].dt.dayofweek
        # df['hour'] = df['Date'].dt.hour
    
        return df

    def process_dataset(self):
        """
        This method get the original data and organizes sequentially by oldest to the newest seen TV Series.
        Aggregating the number of episodes seen (Total_Views) and the stating date (First_Seen).
        """
        df = pd.DataFrame(self.dataset, columns=["Title", "Date"])
        df = self.extract_features(df).copy()
        df_aggregated = (
            df.groupby("show")
            .agg(Total_Views=("Date", "size"), First_Seen=("Date", "min"))
            .reset_index()
        )
        df_aggregated = df_aggregated.sort_values(by="First_Seen", ascending=True).reset_index(drop=True)
        
        df_filtered = df_aggregated[df_aggregated["Total_Views"] > 1].reset_index(drop=True)
        return df_filtered
    

## ==================================================================================================
## Data Processing - View Count Vectors
## ==================================================================================================

def match_title(title, vocabulary: dict, threshold=80):
    # Exact match
    if title in vocabulary:
        return vocabulary[title]
    
    # Fuzzy match
    vocab_keys = list(vocabulary.keys())  # Convert keys to list for fuzzy matching
    match_result = process.extractOne(title, vocab_keys)
    
    # Extract only the best match and score
    if match_result is not None:
        best_match, score = match_result[:2]  # Unpack the first two values
        if score >= threshold:
            return vocabulary[best_match]
    
    # If no match, return -1
    return -1

def create_view_counts_vector(restricted_shared_folder, aggregated_data: pd.DataFrame) -> np.ndarray:
    # TODO: load vocabulary from aggregator (LATER BE UPDATED TO RETRIEVE FROM AGGREGATOR'S PUBLIC SITE)
    try:
        shared_file = os.path.join(restricted_shared_folder, "tv-series_vocabulary.json")
        with open(shared_file, "r", encoding="utf-8") as file:
            vocabulary = json.load(file)
    except:
        # TODO: to remove once available in the Aggregator
        print("Dev Note -> Could not find the shared file. Loading from local file.")
        with open("./aggregator/data/tv-series_vocabulary.json", "r", encoding="utf-8") as file:
            vocabulary = json.load(file)

    aggregated_data["ID"] = aggregated_data["show"].apply(lambda x: match_title(x, vocabulary))
    
    vector_size = len(vocabulary) 
    sparse_vector = np.zeros(vector_size, dtype=int)

    for _, row in aggregated_data.iterrows():
        if row["ID"] != -1:
            sparse_vector[row["ID"]] += row["Total_Views"]

    unmatched_titles = aggregated_data[aggregated_data["ID"] == -1]["show"].tolist()
    print(">> (create_view_counts_vector) Unmatched Titles:", unmatched_titles)

    return sparse_vector