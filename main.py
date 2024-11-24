import os
import sys
import csv
import datetime
import numpy as np
from pathlib import Path
from syftbox.lib import Client, SyftPermission


API_NAME = "netflix_trend_participant"
AGGREGATOR_DATASITE = "gubertoli@gmail.com"
DATA_PATH = "~/Downloads/NetflixViewingHistory.csv" # change here with your data retrieved from Netflix


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



def load_netflix_history(file_path: str) -> np.ndarray:
    """
    Loads Netflix viewing history into a NumPy array, handling quoted fields.

    Args:
        file_path (str): Path to the NetflixViewingHistory.csv file.

    Returns:
        np.ndarray: A 2D NumPy array with Title and Date columns.
    """

    cleaned_data = []

    with open(file_path, mode="r", encoding="utf-8") as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header
        for row in reader:
            cleaned_data.append(row)

    return np.array(cleaned_data)


def reduce_information(history: np.ndarray) -> np.ndarray:
    """
    In this function (by now) reduces too much detail about the viewing history.
    """
    
    # Get only the title, for instance:
    # From: "The Blacklist: Season 1: Wujing (No. 84)" | To: "The Blacklist"
    # Other case like movies or documentaries, get the full title
    titles = np.array([title.split(":")[0] if ":" in title else title for title in history[:, 0]])

    # Convert dates to week numbers
    weeks = np.array([
        datetime.datetime.strptime(date, "%d/%m/%Y").isocalendar()[1]
        for date in history[:, 1]
    ])


    reduced_data = np.column_stack((titles, weeks))

    return reduced_data


def main():
    full_datapath = os.path.expanduser(DATA_PATH)

    if not os.path.exists(full_datapath):
        raise FileNotFoundError(f"Error: The specified data path '{full_datapath}' does not exist.")

    client = Client.load()

    restricted_public_folder = client.api_data(API_NAME)    # create an API
    create_public_folder(restricted_public_folder, client)  # create the dedicated API folder

    private_folder = create_private_folder(client.datasite_path, client)

    # First column is title and second column the date
    viewing_history = load_netflix_history(full_datapath)

    # Reduce the original information
    reduced_history = reduce_information(viewing_history)

    # Saving the reduced Viewing History.
    public_file: Path = restricted_public_folder / "netflix_reduced.npy"
    np.save(str(public_file), reduced_history)

    # Saving the full Viewing History.
    private_file: Path = private_folder / "netflix_full.npy"
    np.save(str(private_file), viewing_history)



if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)