import os
import sys
import csv
import joblib
import json
import numpy as np
import pandas as pd
import unicodedata
import re
from rapidfuzz import process
from datetime import datetime
from pathlib import Path
from syftbox.lib import Client, SyftPermission
from collections import Counter
from dotenv import load_dotenv
from fetcher import NetflixFetcher
from utils.ml import train_model, SequenceModel
from utils.checks import is_file_modified_today

# Load environment variables
load_dotenv()

API_NAME = os.getenv("API_NAME")
AGGREGATOR_DATASITE = os.getenv("AGGREGATOR_DATASITE")
CSV_NAME = os.getenv("NETFLIX_CSV")
OUTPUT_DIR = os.getenv("OUTPUT_DIR")

def create_private_folder(path: Path, client: Client) -> Path:
    """
    Create a private folder within the specified path.

    This function creates a directory structure containing the NetflixViewingHistory.csv.
    """

    netflix_datapath: Path = path / "private" / "netflix_data"
    os.makedirs(netflix_datapath, exist_ok=True)

    # Set the default permissions
    permissions = SyftPermission.datasite_default(email=client.email)
    permissions.save(netflix_datapath)  # update the ._syftperm

    return netflix_datapath

def create_public_folder(path: Path, client: Client) -> None:
    """
    Create a API public folder within the specified path.

    This function creates a directory for receiving the private enhanced version \
    of the viewing history.
    """

    os.makedirs(path, exist_ok=True)

    # Set default permissions for this folder
    permissions = SyftPermission.datasite_default(email=client.email)
    permissions.read.append(AGGREGATOR_DATASITE) # set read permission to the aggregator
    permissions.save(path)

def load_csv_to_numpy(file_path: str) -> np.ndarray:
    """
    Load a CSV file into a NumPy array, handling quoted fields.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        np.ndarray: A 2D NumPy array containing the data from the CSV.
    """
    cleaned_data = []

    with open(file_path, mode="r", encoding="utf-8") as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header
        for row in reader:
            cleaned_data.append(row)

    return np.array(cleaned_data)

def extract_titles(history: np.ndarray) -> np.ndarray:
    """
    Extract and reduce titles from the viewing history.
    """
    return np.array([title.split(":")[0] if ":" in title else title for title in history[:, 0]])


def convert_dates_to_weeks(history: np.ndarray) -> np.ndarray:
    """
    Convert viewing dates to ISO week numbers.
    """
    return np.array([
        datetime.strptime(date, "%d/%m/%Y").isocalendar()[1]
        for date in history[:, 1]
    ])


def aggregate_title_week_counts(reduced_data: np.ndarray) -> np.ndarray:
    """
    Aggregate the reduced viewing history by counting occurrences for each title and week.

    Args:
        reduced_data (np.ndarray): A 2D array with titles and weeks.

    Returns:
        np.ndarray: A 2D array with aggregated counts for each title and week combination.
    """
    counts = Counter(map(tuple, reduced_data))
    aggregated_data = np.array([[title, week, str(count)] for (title, week), count in counts.items()])
    return aggregated_data


def orchestrate_reduction(history: np.ndarray) -> np.ndarray:
    """
    Orchestrates the reduction process for Netflix viewing history.
    """
    titles = extract_titles(history)
    weeks = convert_dates_to_weeks(history)
    return np.column_stack((titles, weeks))

def download_daily_data():
    """
    Download Netflix data into today's subfolder.
    """
    downloader = NetflixFetcher()
    downloader.run()


def get_latest_file(subfolder_path):
    """
    Get the latest file in the subfolder by datetime in filename.
    """
    netflix_csv = os.path.splitext(CSV_NAME)[0] + "_"

    files = [f for f in os.listdir(subfolder_path) if os.path.isfile(os.path.join(subfolder_path, f)) and netflix_csv in f]
    if not files:
        raise FileNotFoundError(f"No files found in {subfolder_path}")

    def extract_datetime(filename):
        try:
            date_str = filename.replace(netflix_csv, "").replace(".csv", "")
            return datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            return datetime.min

    files.sort(key=lambda f: extract_datetime(f), reverse=True)
    
    return os.path.join(subfolder_path, files[0])


def get_or_download_latest_data():
    """
    Ensure latest Netflix data exists or download it.
    """
    datapath = os.path.expanduser(OUTPUT_DIR)
    today_date = datetime.now().strftime("%Y-%m-%d")
    netflix_csv = os.path.splitext(CSV_NAME)[0]

    filename = f"{netflix_csv}_{today_date}.csv"
    file_path = os.path.join(datapath, filename)

    if not os.path.exists(file_path):
        download_daily_data()

    return get_latest_file(datapath)

def join_viewing_history_with_netflix(reduced_history, netflix_show_data):
    """
    Join the reduced viewing history with Netflix show data based on titles.

    Args:
        reduced_history (np.ndarray): Viewing history data.
        netflix_show_data (np.ndarray): Netflix titles data.

    Returns:
        list: Joined data as a list of dictionaries with keys from both datasets.
    """
    # Reduce the viewing history to relevant titles
    my_titles = set([str(title) for title in reduced_history[:, 0]])

    # Get Netflix show titles
    netflix_titles_dict = {
        str(row[2]): row for row in netflix_show_data
    }  # Use title as key for efficient lookups

    # Perform an inner join
    joined_rows = []

    for row in reduced_history:
        title = str(row[0])
        if title in netflix_titles_dict:
            netflix_row = netflix_titles_dict[title]
            # Combine viewing history row and Netflix data row
            joined_rows.append(np.concatenate((row, netflix_row)))

    joined_data = np.array(joined_rows)
    my_found_titles = set([str(title) for title in joined_data[:, 0]])

    print(f"Found Titles: {len(my_found_titles)}")
    print(f"Not Found Titles: {len(my_titles) - len(my_found_titles)}")

    return joined_data

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

def create_view_counts_vector(aggregated_data: pd.DataFrame) -> np.ndarray:
    # load vocabulary from aggregator (LATER BE UPDATED TO RETRIEVE FROM AGGREGATOR'S PUBLIC SITE)
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


def main():

    try:
        latest_data_file = get_or_download_latest_data()
        print(f"Process completed. Latest data file is: {latest_data_file}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
        
    client = Client.load()

    restricted_public_folder = client.api_data(API_NAME)    # create an API
    create_public_folder(restricted_public_folder, client)  # create the dedicated API folder

    private_folder = create_private_folder(client.datasite_path, client)

    # First column is title and second column the date
    viewing_history = load_csv_to_numpy(latest_data_file)

    # Reduce the original information
    reduced_history = orchestrate_reduction(viewing_history)

    netflix_file_path = 'data/netflix_titles.csv'
    netflix_show_data = load_csv_to_numpy(netflix_file_path)
    my_shows_data = join_viewing_history_with_netflix(reduced_history, netflix_show_data)

    # Next, to perform something useful with this joined (enhanced) data
    # This is an enhanced data compared with the retrieved viewing history from Netflix website
    # show_id,type,title,director,cast,country,date_added,release_year,rating,duration,listed_in,description
    #
    # Useful for Embeddings and more complex learning

    private_shows_file: Path = private_folder / "my_shows_data_full.npy"
    np.save(str(private_shows_file), my_shows_data)

    ##############

    # Aggregate the reduced information
    aggregated_history = aggregate_title_week_counts(reduced_history)

    # Saving the reduced Viewing History.
    public_file: Path = restricted_public_folder / "netflix_reduced.npy"
    np.save(str(public_file), reduced_history)

    # Saving the aggregated Viewing History
    aggregated_file: Path = restricted_public_folder / "netflix_aggregated.npy"
    np.save(str(aggregated_file), aggregated_history)

    # Saving the full Viewing History.
    private_file: Path = private_folder / "netflix_full.npy"
    np.save(str(private_file), viewing_history)

    # Train a MLP as recommender in the data
    mlp, _, _, num_samples = train_model(latest_data_file)

    mlp_weights: Path = restricted_public_folder / f"netflix_mlp_weights_{num_samples}.joblib"
    mlp_bias: Path = restricted_public_folder / f"netflix_mlp_bias_{num_samples}.joblib"
    joblib.dump(mlp.coefs_, str(mlp_weights))
    joblib.dump(mlp.intercepts_, str(mlp_bias))

    # Train locally a sequence model to predict next series (filter by > 1 episodes)
    # - loaded with the original NetflixViewingHistory.csv
    sequence_recommender = SequenceModel(viewing_history)
    
    view_counts_vector = create_view_counts_vector(sequence_recommender.aggregated_data)
    private_tvseries_views_file: Path = private_folder / "tvseries_views_sparse_vector.npy"
    np.save(str(private_tvseries_views_file), view_counts_vector)
    

if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)
