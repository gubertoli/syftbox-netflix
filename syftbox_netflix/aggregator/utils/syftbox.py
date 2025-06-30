import os
from pathlib import Path

import yaml
from syft_core import Client as SyftboxClient
from syft_core.permissions import SyftPermission


def participants_datasets(
    datasites_path: Path, dataset_name="Netflix Data", dataset_format="CSV"
) -> list[str]:
    """
    Check for "Netflix Data" from datasites/<user>/public/datasets.yaml
    """
    entries = sorted(os.listdir(datasites_path))
    users = []

    for entry in entries:
        datasets_yaml = Path(datasites_path / entry / "public" / "datasets.yaml")
        if datasets_yaml.is_file():
            with open(datasets_yaml, "r") as file:
                data = yaml.safe_load(file)

                for dataset in data.get("datasets", []):
                    if (
                        dataset.get("name") == dataset_name
                        and dataset.get("format") == dataset_format
                        and "path" in dataset
                    ):
                        users.append(entry)

    return users


def network_participants(datasite_path: Path, api_name: str) -> list[str]:
    """
    Network Participants Discovery:
    Retrieves a list of user directories (participants) in a given datasite path.
    This function scans the network for all available peers by looking at directories in the datasite path.

    By looking for "app_data / APP_NAME" only those from the specific app will be considered.

    """

    entries = sorted(os.listdir(datasite_path))
    users = []

    for entry in entries:
        if Path(datasite_path / entry / "app_data" / api_name).is_dir():
            users.append(entry)

    return users


def create_shared_folder(
    path: Path, api_name: str, client: SyftboxClient, participants: list
) -> Path:
    """
    Create a shared folder accessible to participants only with the computations
    """

    shared_datapath: Path = path / "app_data" / api_name / "shared"
    os.makedirs(shared_datapath, exist_ok=True)

    # Set the default permissions
    permissions = SyftPermission.datasite_default(context=client, dir=shared_datapath)
    for participant in participants:  # set read permission to participants
        permissions.add_rule(path="**", user=participant, permission="read")
    permissions.save(
        shared_datapath
    )  # update the syftperm.yaml (previously ._syftperm)

    return shared_datapath, permissions

def create_recommendation_dataset(
    path: Path, api_name: str, client: SyftboxClient, participants: list, permissions
) -> Path:
    """
    Create a recommendation dataset based on participant interactions for further analyses.
    """

    file_datapath: Path = path / "app_data" / api_name / "shared"
    os.makedirs(file_datapath, exist_ok=True)

    file_path = file_datapath / "recommendations.csv"
    if not file_path.exists():
        with open(file_path, "w") as f:
            f.write("timestamp,username,raw_list,reranked_list,item_from_column,clicked_item\n")

    for participant in participants:  # set read permission to participants
        permissions.add_rule(path="recommendations.csv", user=participant, permission="write")
    permissions.save(
        file_datapath
    )  # update the syftperm.yaml (previously ._syftperm)

    return file_datapath
