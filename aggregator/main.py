# Refence: https://syftbox-documentation.openmined.org/cpu-tracker-2

import os
import numpy as np
from pathlib import Path

API_NAME = os.getenv("API_NAME")

def network_participants(datasite_path: Path) -> list[str]:
    """
    Network Participants Discovery:
    Retrieves a list of user directories (participants) in a given datasite path. 
    This function scans the network for all available peers by looking at directories in the datasite path.
    """

    entries = os.listdir(datasite_path)
    users = []

    for entry in entries:
        if Path(datasite_path / entry).is_dir():
            users.append(entry)

    return users


def get_users_history(
        datasites_path: Path, peers: list[str]
) -> tuple[float, list[str]]:
    
    for peer in peers:
        view_history_file = (
            datasites_path / peer / "api_data" / API_NAME / "netflix_aggregated.npy"    # other options are netflix_reduced.npy or MLP weights
        )

        if not view_history_file.exists():
            continue

        try:
            data = np.load(view_history_file)

            # TODO :THE AGGREGATOR PROCESSING