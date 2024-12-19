import os
from pathlib import Path
from syftbox.lib import Client, SyftPermission
from participant_utils.checks import should_run

API_NAME = os.getenv("API_NAME")

def setup_environment(client, api_name, aggregator_path, profile):
    """
    Set up public and private folders for data storage.

    Args:
        client: Client instance for managing API and datasite paths.

    Returns:
        tuple: Paths to restricted public and private folders.
    """

    def create_private_folder(path: Path, client: Client, profile) -> Path:
        """
        Create a private folder within the specified path.

        This function creates a directory structure containing the NetflixViewingHistory.csv.
        """

        netflix_datapath: Path = path / "private" / API_NAME / profile

        os.makedirs(netflix_datapath, exist_ok=True)

        # Set the default permissions
        permissions = SyftPermission.datasite_default(email=client.email)
        permissions.save(netflix_datapath)  # update the ._syftperm

        return netflix_datapath

    def create_public_folder(path: Path, client: Client, aggregator_path) -> None:
        """
        Create a API public folder within the specified path.

        This function creates a directory for receiving the private enhanced version \
        of the viewing history.
        """

        os.makedirs(path, exist_ok=True)

        # Set default permissions for this folder
        permissions = SyftPermission.datasite_default(email=client.email)
        permissions.read.append(aggregator_path) # set read permission to the aggregator
        permissions.save(path)

    datasites_path = Path(client.datasite_path.parent)
    restricted_shared_folder = Path(datasites_path / aggregator_path / "api_data" / api_name / "shared")
    restricted_public_folder = client.api_data(api_name) / profile

    create_public_folder(restricted_public_folder, client, aggregator_path)
    private_folder = create_private_folder(client.datasite_path, client, profile)
    return restricted_shared_folder, restricted_public_folder, private_folder