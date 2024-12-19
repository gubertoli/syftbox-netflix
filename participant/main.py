import os
import argparse
import sys
import logging
import numpy as np
from pathlib import Path
from syftbox.lib import Client

# Package utils
from participant_utils.syftbox import setup_environment
from participant_utils.data_loading import load_csv_to_numpy, get_or_download_latest_data

# Package functions
import federated_analytics.data_processing as fa
import federated_learning.mlp_model as mlp
from federated_learning.sequence_data import SequenceData, create_view_counts_vector
from federated_learning.svd_participant_finetuning import participant_fine_tuning
from federated_analytics.dp_series import run_top5_dp

from dotenv import load_dotenv

# Set up logging with colors
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.DEBUG,
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ANSI escape codes for colors
COLORS = {
    "DEBUG": "\033[94m",  # Blue
    "INFO": "\033[92m",   # Green
    "WARNING": "\033[93m",  # Yellow
    "ERROR": "\033[91m",   # Red
    "RESET": "\033[0m",    # Reset
}

class ColoredFormatter(logging.Formatter):
    def format(self, record):
        color = COLORS.get(record.levelname, COLORS["RESET"])
        reset = COLORS["RESET"]
        record.msg = f"{color}{record.msg}{reset}"
        return super().format(record)

# Apply the colored formatter
for handler in logging.getLogger().handlers:
    handler.setFormatter(ColoredFormatter("%(asctime)s - %(levelname)s - %(message)s"))

# Load environment variables
load_dotenv()
API_NAME = os.getenv("API_NAME")
AGGREGATOR_DATASITE = os.getenv("AGGREGATOR_DATASITE")
CSV_NAME = os.getenv("NETFLIX_CSV")
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

    netflix_file_path = 'data/netflix_titles.csv'
    netflix_show_data = load_csv_to_numpy(netflix_file_path)
    logging.debug(f"Loaded Netflix show data from {netflix_file_path}.")

    # title_genre_dict = fa.create_title_field_dict(netflix_show_data, title_col=2, field_col=10)
    # logging.info("Created title-to-genre mapping dictionary.")

    user_information = fa.add_column_from_dict(
        aggregated_history, ratings_dict, key_col=0, new_col_name='rating'
    )
    logging.debug("Added user ratings to aggregated history.")

    # Enhanced data compared with the retrieved viewing history
    my_shows_data = fa.join_viewing_history_with_netflix(user_information, netflix_show_data)
    logging.debug("Joined viewing history with Netflix show data.")

    # Save data
    fa.save_npy_data(private_folder, "netflix_reduced.npy", reduced_history)
    fa.save_npy_data(private_folder, "netflix_aggregated.npy", user_information)
    fa.save_npy_data(private_folder, "netflix_full.npy", viewing_history)
    fa.save_npy_data(private_folder, "data_full.npy", my_shows_data)
    fa.save_npy_data(private_folder, "ratings.npy", ratings_dict)
    logging.info("All processed data saved successfully.")

def run_federated_learning(restricted_shared_folder, restricted_public_folder, private_folder, viewing_history, latest_data_file, datasite_parent_path):
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
    private_tvseries_views_file: Path = private_folder / "tvseries_views_sparse_vector.npy"
    np.save(str(private_tvseries_views_file), view_counts_vector)
    logging.debug(f"View counts vector saved to {private_tvseries_views_file}.")

def main(profile, profile_id):
    logging.info(f"Starting process for profile: {profile} (ID: {profile_id})")

    client = Client.load()
    profile_masked_name = f'profile_{profile_id}'
    datapath = os.path.join(OUTPUT_DIR, profile_masked_name)

    # Set up environment
    restricted_shared_folder, restricted_public_folder, private_folder = setup_environment(
        client, API_NAME, AGGREGATOR_DATASITE, profile_masked_name
    )

    # Optional/Experimental - use public yaml to download dataset. Configure it here.
    if profile == "demo_profile":
        logging.info("Using demo profile configuration.")
        yml_custom_config = {'client_datasite_path': Path('data/demo_profile'), 'dataset_name': "Netflix Data", 'dataset_format': "CSV"}
    else:
        logging.info("Using profile configuration.")
        yml_custom_config = {'client_datasite_path': client.datasite_path, 'dataset_name': "Netflix Data", 'dataset_format': "CSV"}

    logging.info("Environment setup completed.")

    # Retrieve data
    latest_data_file, viewing_history = get_or_download_latest_data(
        datapath, CSV_NAME, profile, experimental_config=yml_custom_config
    )
    logging.info("Latest data retrieved successfully.")

    # Run analytics and learning processes
    run_federated_analytics(restricted_public_folder, private_folder, viewing_history)
    run_federated_learning(
        restricted_shared_folder, restricted_public_folder, private_folder,
        viewing_history, latest_data_file, client.datasite_path.parent
    )

    run_top5_dp(private_folder / "tvseries_views_sparse_vector.npy", restricted_public_folder, verbose=False)
    logging.info("Top-5 DP process completed.")

    finetuned_flag_path = os.path.join(restricted_public_folder, "svd_training", "local_finetuning_succeed.txt")
    if os.path.exists(finetuned_flag_path):
        logging.info(f"Fine-tuning already completed for {profile_masked_name}.")
    else:
        logging.warning(f"Fine-tuning not completed for {profile_masked_name}. Starting process...")
        logging.info(f"Starting SVD Recommendation Engine fine-tuning process for {profile_masked_name}...")
        participant_fine_tuning(
            profile_id, private_folder, restricted_shared_folder, restricted_public_folder,
            epsilon=1, noise_type="gaussian", clipping_threshold=None, plot=False, dp_all=False
        )
        logging.info("Fine-tuning completed successfully.")

    # Save the version of the last running process
    current_version = 1.01
    version_file = os.path.join(restricted_public_folder, "version.txt")
    with open(version_file, "w") as f:
        f.write(str(current_version))

if __name__ == "__main__":
    try:
        # Parse arguments
        parser = argparse.ArgumentParser(description="Run the main function with a participant's profile name argument.")
        parser.add_argument("--profile", default="demo_profile", help="Participant profile name (default: demo_profile)")
        parser.add_argument("--profile_id", default="demo", help="Participant profile ID (default: demo)")
        args = parser.parse_args()

        # Execute main
        main(args.profile, args.profile_id)
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        sys.exit(1)