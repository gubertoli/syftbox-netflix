import argparse
import logging
import os
import sys
import subprocess
from pathlib import Path
import json

import numpy as np
from dotenv import load_dotenv
from syft_core import Client as SyftboxClient
from syft_core import SyftClientConfig
from syftbox_netflix.aggregator.pets.svd_recommender import local_recommendation

from .federated_analytics import data_processing as fa
from .federated_analytics.dp_series import run_top5_dp
from .federated_learning.sequence_data import SequenceData, create_view_counts_vector
from .federated_learning.svd_participant_finetuning import participant_fine_tuning
from .participant_utils.data_loading import (
    get_or_download_latest_data,
    load_csv_to_numpy,
)
from .participant_utils.syftbox import setup_environment

# Set up logging with colors
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.DEBUG,
    datefmt="%Y-%m-%d %H:%M:%S",
)

# A ANSI escape codes for colors
COLORS = {
    "DEBUG": "\033[94m",  # Blue
    "INFO": "\033[92m",  # Green
    "WARNING": "\033[93m",  # Yellow
    "ERROR": "\033[91m",  # Red
    "RESET": "\033[0m",  # Reset
}


class ColoredFormatter(logging.Formatter):
    def format(self, record):
        color = COLORS.get(record.levelname, COLORS["RESET"])
        reset = COLORS["RESET"]
        message = super().format(record)
        # return f"{color}{message}{reset}"
        return f"{message}"


console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(
    ColoredFormatter("%(asctime)s - %(levelname)s - %(message)s")
)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.handlers = [console_handler]

# Load environment variables
load_dotenv()
APP_NAME = os.getenv("APP_NAME", "syftbox-netflix")
AGGREGATOR_DATASITE = os.getenv("AGGREGATOR_DATASITE")
CSV_NAME = os.getenv("NETFLIX_CSV", "NetflixViewingHistory.csv")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", ".")


def run_federated_analytics(restricted_public_folder, private_folder, viewing_history):
    logging.info("Running Federated Analytics...")
    logging.debug(f"User viewing history contains {len(viewing_history)} entries.")

    # Reduce and aggregate the original information
    reduced_history = fa.orchestrate_reduction(viewing_history)
    logging.debug(f"Reduced history created with {len(reduced_history)} entries.")

    aggregated_history = fa.aggregate_title_week_counts(reduced_history)
    logging.debug(f"Aggregated history contains {len(aggregated_history)} records.")

    # Infer ratings as per viewing patterns
    ratings_dict = fa.calculate_show_ratings(aggregated_history)
    logging.debug("Ratings inferred from viewing patterns.")

    netflix_file_path = "data/netflix_titles.csv"
    netflix_show_data = load_csv_to_numpy(netflix_file_path)
    logging.debug(f"Loaded Netflix show data from {netflix_file_path}.")

    # title_genre_dict = fa.create_title_field_dict(netflix_show_data, title_col=2, field_col=10)
    # logging.info("Created title-to-genre mapping dictionary.")

    user_information = fa.add_column_from_dict(
        aggregated_history, ratings_dict, key_col=0, new_col_name="rating"
    )
    logging.debug("Added user ratings to aggregated history.")

    # Enhanced data compared with the retrieved viewing history
    my_shows_data = fa.join_viewing_history_with_netflix(
        user_information, netflix_show_data
    )
    logging.debug("Joined viewing history with Netflix show data.")

    # Save data
    fa.save_npy_data(private_folder, "netflix_reduced.npy", reduced_history)
    fa.save_npy_data(private_folder, "netflix_aggregated.npy", user_information)
    fa.save_npy_data(private_folder, "netflix_full.npy", viewing_history)
    fa.save_npy_data(private_folder, "data_full.npy", my_shows_data)
    fa.save_npy_data(private_folder, "ratings.npy", ratings_dict)
    logging.info("All processed data saved successfully.")


def run_federated_learning(
    restricted_shared_folder,
    restricted_public_folder,
    private_folder,
    viewing_history,
    latest_data_file,
    datasite_parent_path,
):
    logging.info("Running Federated Learning...")

    # # Train and save MLP model --> Demonstration, as we moved to SVD Process instead
    # mlp.train_and_save_mlp(latest_data_file, private_folder, restricted_public_folder)
    # logging.info("MLP model trained and saved.")

    # Create sequence data
    sequence_recommender = SequenceData(viewing_history)
    logging.debug("Sequence data created from viewing history.")

    view_counts_vector = create_view_counts_vector(
        restricted_shared_folder, sequence_recommender.aggregated_data
    )
    private_tvseries_views_file: Path = (
        private_folder / "tvseries_views_sparse_vector.npy"
    )
    np.save(str(private_tvseries_views_file), view_counts_vector)
    logging.debug(f"View counts vector saved to {private_tvseries_views_file}.")


def main(profile, profile_id):
    logging.info("Starting local webpage...")
    local_path = os.path.dirname(os.path.abspath(__file__))  # get path of this main.py
    assigned_port = os.getenv("SYFTBOX_ASSIGNED_PORT", "8080")

    command = [
        "uv", "run",
        "uvicorn", "app:app",
        "--reload",
        "--host", "0.0.0.0",
        "--port", assigned_port
    ]

    # Run it from the script's directory
    subprocess.Popen(command, cwd=local_path)

    logging.info(f"Starting process for profile: {profile} (ID: {profile_id})")

    config = SyftClientConfig.load()
    client = SyftboxClient(config)
    profile_masked_name = f"profile_{profile_id}"
    datapath = os.path.join(OUTPUT_DIR, profile_masked_name)

    # Set up environment
    restricted_shared_folder, restricted_public_folder, private_folder = (
        setup_environment(client, APP_NAME, AGGREGATOR_DATASITE, profile_masked_name)
    )

    print(
        "Setting up Environment with",
        "restricted_shared_folder",
        restricted_shared_folder,
        "restricted_public_folder",
        restricted_public_folder,
        "private_folder",
        private_folder,
    )

    # Optional/Experimental - use public yaml to download dataset. Configure it here.
    if profile == "demo_profile":
        logging.info("Using demo profile configuration.")
        yml_custom_config = {
            "client_datasite_path": Path("data/demo_profile"),
            "dataset_name": "Netflix Data",
            "dataset_format": "CSV",
        }
    else:
        logging.info("Using profile configuration.")
        yml_custom_config = {
            "client_datasite_path": client.datasite_path,
            "dataset_name": "Netflix Data",
            "dataset_format": "CSV",
        }

    logging.info("Environment setup completed.")

    latest_data_file, viewing_history = get_or_download_latest_data(
        datapath, CSV_NAME, profile, experimental_config=yml_custom_config
    )
    logging.info("Latest data retrieved successfully.")

    # Run analytics and learning processes
    run_federated_analytics(restricted_public_folder, private_folder, viewing_history)
    run_federated_learning(
        restricted_shared_folder,
        restricted_public_folder,
        private_folder,
        viewing_history,
        latest_data_file,
        client.datasite_path.parent,
    )

    run_top5_dp(
        private_folder / "tvseries_views_sparse_vector.npy",
        restricted_public_folder,
        verbose=False,
    )
    logging.info("Top-5 DP process completed.")


    # Removed for prototype
    # finetuned_flag_path = os.path.join(
    #     restricted_public_folder, "svd_training", "local_finetuning_succeed.txt"
    # )
    # if os.path.exists(finetuned_flag_path):
    #     logging.info(f"Fine-tuning already completed for {profile_masked_name}: {finetuned_flag_path}.")
    # else:
    #     logging.warning(
    #         f"Fine-tuning not yet triggered for {profile_masked_name}. Starting process..."
    #     )

    logging.info(f"Starting SVD Recommendation Engine fine-tuning process for {profile_masked_name}...")
    participant_fine_tuning(
        profile_id,
        private_folder,
        restricted_shared_folder,
        restricted_public_folder,
        epsilon=1,
        noise_type="gaussian",
        clipping_threshold=None,
        plot=False,
        dp_all=False,
    )
    logging.info("Fine-tuning completed successfully.")

    # Save the version of the last running process
    current_version = 1.01
    version_file = os.path.join(restricted_public_folder, "version.txt")
    with open(version_file, "w") as f:
        f.write(str(current_version))

    try:
        logging.debug("Performing Local SVD Recommandations...")
        ## Note, this is a local recommendation, not a federated one
        ## Only works for cases where the aggregator account is also a participant, and uses the aggregator's private data to generate recommendations
        ## This is for DEMO purposes while maintaining privacy of the participants
        participant_private_path = (
            Path(client.config.data_dir) / "private" / APP_NAME / "profile_0"
        )
        datasites_path = Path(client.datasite_path.parent)
        shared_folder_path: Path = (datasites_path / AGGREGATOR_DATASITE / "app_data" / APP_NAME / "shared")
        tv_vocab = {}

        try:
            json_file_path = shared_folder_path / "tv-series_vocabulary.json"
            with open(json_file_path, 'r', encoding='utf-8') as f:
                tv_vocab = json.load(f)
        except Exception as e:
            logging.error(
            f"Could not find TV vocabulary file: {e}"
        )
        
        raw_recommendations, reranked_recommendations = local_recommendation(
            participant_private_path,
            shared_folder_path,
            tv_vocab,
            exclude_watched=True,
        )

        raw_results_path = participant_private_path / "raw_recommendations.json"
        try:
            os.makedirs(os.path.dirname(raw_results_path), exist_ok=True)
            with open(raw_results_path, 'w') as f:
                json.dump(raw_recommendations, f, indent=4)
        except (IOError, TypeError) as e:
            logging.error(
                f"ERROR: Failed to write local raw recommendations to output file: {e}"
            )

        reranked_results_path = participant_private_path / "reranked_recommendations.json"

        try:
            os.makedirs(os.path.dirname(reranked_results_path), exist_ok=True)
            with open(reranked_results_path, 'w') as f:
                json.dump(reranked_recommendations, f, indent=4)
        except (IOError, TypeError) as e:
            logging.error(
                f"ERROR: Failed to write local reranked recommendations to output file: {e}"
            )

    except Exception as e:
        logging.error(
            f"An unexpected error occurred during Local SVD Recommandations: {e}"
        )


if __name__ == "__main__":
    try:
        # Parse arguments
        parser = argparse.ArgumentParser(
            description="Run the main function with a participant's profile name argument."
        )
        parser.add_argument(
            "--profile",
            default="demo_profile",
            help="Participant profile name (default: demo_profile)",
        )
        parser.add_argument(
            "--profile_id",
            default="demo",
            help="Participant profile ID (default: demo)",
        )
        args = parser.parse_args()

        sys.exit(0)
        main(args.profile, args.profile_id)
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        sys.exit(1)
