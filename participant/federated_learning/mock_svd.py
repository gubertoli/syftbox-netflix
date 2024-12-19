# Currently not working as in development.
# This is an exploratory process for functions that are being refactored into the main codebase.

import os
import json
import numpy as np
import shutil
import diffprivlib.tools as dp
import matplotlib.pyplot as plt
import copy

from participant.main import setup_environment
from syftbox.lib import Client

from participant.federated_learning.svd_participant_finetuning import participant_fine_tuning
from participant.federated_learning.svd_participant_local_recommendation import compute_recommendations
from participant.federated_learning.svd_server_initialisation import initialize_item_factors
from participant.federated_learning.svd_server_aggregation import aggregate_item_factors
from participant.server_utils.data_loading import load_tv_vocabulary, load_imdb_ratings, load_global_item_factors

from dotenv import load_dotenv
load_dotenv()

def normalize_string(s):
    """
    """
    return s.replace('\u200b', '').lower()

def server_initialization(save_to:str, tv_series_path:str, imdb_ratings_path:str):

    # Step 1: Load vocabulary and IMDB ratings
    tv_vocab = load_tv_vocabulary(tv_series_path)
    imdb_ratings = load_imdb_ratings(imdb_ratings_path)

    # Step 2: Load and normalize IMDB ratings
    imdb_data = np.load(imdb_ratings_path, allow_pickle=True).item()
    imdb_ratings = {normalize_string(title): float(rating) for title, rating in imdb_data.items() if rating}

    # Step 2: Initialize item factors
    V = initialize_item_factors(tv_vocab, imdb_ratings)

    # Step 4: Save the initialized model
    os.makedirs(save_to, exist_ok=True)
    np.save(os.path.join(save_to, "global_V.npy"), V)

    print("Server initialization complete. Item factors (V) are saved.")

def server_aggregate(updates, save_to, weights=None, learning_rate=1.0, epsilon=1.0, clipping_threshold=0.5):
    """
    Orchestrates the server aggregation process:
    1. Loads current global item factors.
    2. Calls `aggregate_item_factors` to perform the aggregation.
    3. Saves the updated global item factors.

    Args:
        updates (list[dict]): List of delta dictionaries from participants.
        weights (list[float]): List of weights for each participant. If None, equal weights are assumed.
        learning_rate (float): Scaling factor for the aggregated deltas.
        epsilon (float): Privacy budget for differential privacy.
        clipping_threshold (float): Clipping threshold for updates.
        save_to (str): Path to save the updated global item factors.
    """
    global_V_path = os.path.join(save_to, "global_V.npy")

    # Step 1: Load current global item factors
    V = load_global_item_factors(global_V_path)

    # Step 2: Aggregate updates
    V = aggregate_item_factors(
        V, updates, weights=weights, learning_rate=learning_rate, epsilon=epsilon, clipping_threshold=clipping_threshold
    )

    # Step 3: Save the updated global item factors
    os.makedirs(os.path.dirname(global_V_path), exist_ok=True)
    np.save(global_V_path, V)

    print("Server aggregation complete. Global item factors (V) updated.")


def local_recommendation(user_id, tv_vocab, user_ratings, exclude_watched=True,
                         global_path="mock_dataset_location/tmp_model_parms",
                         local_root_path="mock_dataset_location/tmp_model_parms"):
    """Main entry point for local recommendation generation."""
    # Step 1: Load data
    local_path = os.path.join(local_root_path, user_id)
    global_V_path = os.path.join(global_path, "global_V.npy")
    user_U_path = os.path.join(local_path, "U.npy")
    
    user_U = np.load(user_U_path)
    global_V = np.load(global_V_path)

    # Step 2: Run process to compute recommendations
    recommendations = compute_recommendations(user_U, global_V, tv_vocab, user_ratings, exclude_watched=exclude_watched)

    # Step 3: Write or return recommendations
    print("Recommended based on most recently watched:")
    for i, (show, score) in enumerate(recommendations):
        print(f"\t{i+1} => {show}: {score:.4f}")
    return recommendations

def run_process():

    API_NAME = os.getenv("API_NAME")
    AGGREGATOR_DATASITE = os.getenv("AGGREGATOR_DATASITE")
    NETFLIX_PROFILES = os.getenv("NETFLIX_PROFILES")
    netflix_profiles_list = NETFLIX_PROFILES.split(",")

    test_user_name = netflix_profiles_list[0]
    test_user_id = 'profile_0'
    test_user = test_user_id
    # test_user = f'{test_user}_mock0' # For testing with simulations

    client = Client.load()

    restricted_shared_folders = {}
    restricted_public_folders = {}
    private_folders = {}

    for profile_id, profile in enumerate(netflix_profiles_list):
        profile_masked_name = f'profile_{profile_id}'
        restricted_shared_folders[profile_masked_name], restricted_public_folders[profile_masked_name], private_folders[profile_masked_name] = setup_environment(client, API_NAME, AGGREGATOR_DATASITE, profile_masked_name)

    ########################################
    # Step 0: Model Initialisation and Fine-Tuning
    ########################################

    # Clear folder
    fldr_base = "mock_dataset_location/tmp_model_parms"
    user_ids = list(private_folders.keys())
    # user_ids = netflix_profiles_list[:-1] # Exclude the last user for testing

    for user_id in user_ids:
        fldr_user = os.path.join(fldr_base, user_id)
        if os.path.exists(fldr_user):
            shutil.rmtree(fldr_user)

    # Server initialisation
    server_initialization(save_to="mock_dataset_location/tmp_model_parms", 
                           tv_series_path="aggregator/data/tv-series_vocabulary.json", 
                           imdb_ratings_path="data/imdb_ratings.npy")
    backup_global_v = np.load("mock_dataset_location/tmp_model_parms/global_V.npy") # For analytics

    delta_V = {}
    for user_id in user_ids:
        # Fine-tuning of the item embeddings with user data
        global_path = "mock_dataset_location/tmp_model_parms"
        save_path="mock_dataset_location/tmp_model_parms"
        restricted_folder = private_folders[user_id]
        delta_V[user_id] = participant_fine_tuning(user_id, private_folders[user_id], global_path, restricted_folder, epsilon=1, noise_type="gaussian", clipping_threshold=None, plot=False, dp_all=False) #0.36

    # # Dictionary to store all mocked user IDs and map them to original user IDs
    # mocked_to_original_mapping = {}

    # # Single dictionary to store all user_id deltas
    # delta_V = {}

    # # Mocking the process 50 times
    # for i in range(50):
    #     for original_user_id in private_folders.keys():
    #         # Create a mocked user ID
    #         mocked_user_id = f"{original_user_id}_mock{i}"

    #         # Map mocked user ID to original user ID
    #         mocked_to_original_mapping[mocked_user_id] = original_user_id

    #         # Use the original user ID to retrieve the private folder
    #         private_folder = private_folders[original_user_id]

    #         # Perform fine-tuning
    #         delta_V[mocked_user_id] = participant_fine_tuning(
    #             mocked_user_id,
    #             private_folder,
    #             epsilon=10,
    #             clipping_threshold=None,
    #             plot=False
    #         )


    ########################################
    # Step 1: Local Recommendation Computation
    ########################################

    with open("aggregator/data/tv-series_vocabulary.json", "r") as f:
        tv_vocab = json.load(f)

    # Example user data
    my_activity_path = os.path.join(private_folders[test_user], 'netflix_aggregated.npy')
    my_activity = np.load(my_activity_path, allow_pickle=True) # Title, Week, Rating

    my_activity_formatted = np.empty(my_activity.shape, dtype=object)
    my_activity_formatted[:, 0] = my_activity[:, 0]  # Show name remains as string
    my_activity_formatted[:, 1] = my_activity[:, 1].astype(int)  # Week number as int
    my_activity_formatted[:, 2] = my_activity[:, 2].astype(int)  # View times as int
    my_activity_formatted[:, 3] = my_activity[:, 3].astype(float)  # Ratings as float

    print("Vanilla Recommendations (IMDB)...")
    top_6 = local_recommendation(test_user, tv_vocab, user_ratings=my_activity_formatted)

    print("Updating Global Model with user deltas...")
    # Server aggregation
    # server_aggregate([delta_V[user_ids[0]], delta_V[user_ids[1]]])
    delta_V_list = list(delta_V.values())
    save_path="mock_dataset_location/tmp_model_parms"
    server_aggregate(delta_V_list, save_to=save_path, epsilon=None, clipping_threshold=None)

    print("Federated Recommendations (IMDB)...")
    top_6 = local_recommendation(test_user, tv_vocab, user_ratings=my_activity_formatted)


    # Logs
    global_V_path = os.path.join(save_path, "global_V.npy")
    global_V = np.load(global_V_path)
    print("Global V shape:", global_V.shape)

    top_show = top_6[0][0]
    top_show_id = tv_vocab[top_show]
    print(f"Top show '{top_show}' has item_id={top_show_id}")

    # # Debug: Check the actual rating for the top show
    # for user in user_ids:
    #     user_rating_path = os.path.join(private_folders[user], 'ratings.npy')
    #     user_rating = np.load(user_rating_path, allow_pickle=True).item()
    #     print(f"Actual rating for '{top_show}' by {user}: {user_rating.get(top_show, "Not rated")}")


    import pandas as pd
    # Prepare the top 6 show IDs
    top_shows = [show[0] for show in top_6]
    ratings_matrix = []

    for user in user_ids:
        # Path to the user's ratings
        user_rating_path = os.path.join(private_folders[user], 'ratings.npy')
        
        # Load the user's ratings
        user_rating = np.load(user_rating_path, allow_pickle=True).item()
        
        # Collect ratings for the top 6 shows
        user_ratings = [float(user_rating.get(show_id, np.nan)) for show_id in top_shows]
        ratings_matrix.append(user_ratings)

    # Add imdb rating
    imdb_path = os.path.join("data", 'imdb_ratings.npy')
    imdb_data = np.load(imdb_path, allow_pickle=True).item()
    imdb_data = {title: float(rating)/2 if rating else np.nan for title, rating in imdb_data.items() if rating}

    imdb_ratings = [imdb_data.get(show, np.nan) for show in top_shows]
    ratings_matrix.append(imdb_ratings)

    average_ratings = np.nanmean(ratings_matrix, axis=0)
    ratings_matrix.append(average_ratings)

    # Convert to a DataFrame for better visualization
    ratings_df = pd.DataFrame(ratings_matrix, index=user_ids + ['imdb', 'average'], columns=[show[0] for show in top_6])
    print(ratings_df)


    # Embeddings before and after fine-tuning
    print("Top Show V factors before fine-tuning:")
    print(backup_global_v[top_show_id])

    print("Top Show V factors after fine-tuning:")
    print(global_V[top_show_id])

    pass

    ########################################
    # NOTE -> BELOW HERE IS IN DEVELOPMENT!!!
    ########################################




    ########################################
    # Step 2: User Chooses Something Outside Our Predictions
    ########################################

    # Let's say the user picks a show not in top_5, e.g., "Pedro Páramo" is re-watched or a new title "100 Humans".
    # For demonstration:
    user_new_choice = top_6[-1][0]
    if user_new_choice not in [t for (t, _) in top_6[:5]]:
        print(f"\nUser selected '{user_new_choice}' which was not in the top 5 predictions.")

    # Let's assume the user watched and implicitly "rated" it.
    # Row to append
    new_rating = 3.6
    new_row = np.array([user_new_choice, 47, 1, new_rating], dtype=object)
    my_activity_formatted = np.vstack([my_activity_formatted, new_row])

    print(f"---->Mock user activity updated with the new show={user_new_choice} and rating={new_rating}.")

    # print("Recalculating recommendations after user interaction to verify consistency...")
    # top_6 = local_recommendation(user1_id, tv_vocab, user_ratings=my_activity_formatted)

    # We now have an additional data point. The user updates their model locally.

    ########################################
    # Step 3: Local Update (Incremental Training)
    ########################################

    # Load model parameters
    load_from = "mock_dataset_location/tmp_model_parms"
    local_U_path = os.path.join(load_from, test_user, f"{test_user}_U.npy")
    global_V_path = os.path.join(load_from, "global_V.npy")

    local_U = np.load(local_U_path)
    global_V = np.load(global_V_path)

    # Identify item_id for the newly chosen item
    # If user_new_choice not in tv_vocab, add it dynamically:
    if user_new_choice not in tv_vocab:
        print(f"Item '{user_new_choice}' not in vocabulary. Adding it now.")

        # Find a new item_id for this show
        new_item_id = max(tv_vocab.values()) + 1
        tv_vocab[user_new_choice] = new_item_id

        # Initialize item factors randomly
        k = local_U.shape[0]  # latent dimension (assuming global_U is [k])
        new_item_factors = np.random.normal(scale=0.01, size=(k,))
        # Expand global_V to accommodate this new item
        # Assuming global_V is shape [num_items, k]
        # We'll need to append a new row
        global_V = np.vstack([global_V, new_item_factors[np.newaxis, :]])
        
        print(f"Assigned item_id={new_item_id} for new show '{user_new_choice}'")
    else:
        new_item_id = tv_vocab[user_new_choice]

    # Now we have new_item_id for the chosen show.
    # Perform a mini step of gradient descent to incorporate the new rating
    alpha = 0.01
    lambda_reg = 0.1

    # Current prediction before update
    pred_before = local_U.dot(global_V[new_item_id])
    error = new_rating - pred_before

    # Compute gradients
    U_u_grad = error * global_V[new_item_id] - lambda_reg * local_U
    V_i_grad = error * local_U - lambda_reg * global_V[new_item_id]

    # Store the old item factors to compute delta
    old_V_item = global_V[new_item_id].copy()

    # Update locally
    local_U += alpha * U_u_grad
    global_V[new_item_id] += alpha * V_i_grad

    # Compute the delta for the item factor
    delta_V = global_V[new_item_id] - old_V_item

    ########################################
    # Step 4: Send Updates (Delta) Back to Server
    ########################################

    # In a real federated scenario, we wouldn't send the raw delta as is,
    # we might send gradient updates or encrypted parameters.
    # For demonstration, let's just print what would be sent.
    print("\nSending updates back to the server:")
    print(f"Item factor delta for item_id {new_item_id} ({user_new_choice}): {delta_V}")

    # Server-side pseudo-code to handle updates:
    # In reality, the server would:
    # - Load the currently stored global_V
    # - Apply the delta to the corresponding item_id

    # Mock server aggregation:
    save_to = "mock_dataset_location/tmp_model_parms"
    os.makedirs(save_to, exist_ok=True)

    # Mock server load:
    server_global_V = np.load(os.path.join(save_to, "global_V.npy"))
    server_local_U = np.load(os.path.join(save_to, test_user, f"{test_user}_U.npy"))

    # If we added a new item not previously in server_global_V, we need to align the dimensions.
    # Assume server_global_V shape: [N_items, k]
    # If new_item_id >= server_global_V.shape[0], we must expand server_global_V as well.
    if new_item_id >= server_global_V.shape[0]:
        # Expand server_global_V to accommodate new_item_id
        rows_to_add = new_item_id - server_global_V.shape[0] + 1
        additional_rows = np.random.normal(scale=0.01, size=(rows_to_add, server_global_V.shape[1]))
        server_global_V = np.vstack([server_global_V, additional_rows])

    # Apply delta:
    server_global_V[new_item_id] += delta_V

    # Save updated global parameters:
    np.save(os.path.join(save_to, test_user, f"{test_user}_U.npy"), server_local_U)
    np.save(os.path.join(save_to, "global_V.npy"), server_global_V)

    print("\nServer: Applied client delta updates to global parameters and re-saved.")

    # At this point, the server’s global model now reflects the user’s latest interaction
    # with the newly chosen item, and this item is also integrated into the vocabulary.

    ### Re-run local recommendation to see if the new item is now recommended
    print("Recalculating recommendations after user interaction and model update to verify consistency...")
    local_recommendation(test_user, tv_vocab, user_ratings=my_activity_formatted[:-1])


if __name__ == "__main__":
    run_process()

########################################
# Notes:
########################################
# - This code is conceptual. In a real federated learning framework:
#   - The user factors (global_U here) might be kept entirely local and not shared at all.
#   - Only item factors or gradient updates would be shared, and potentially in a privacy-preserving manner.
# - If multiple users exist, the server collects such deltas from all users and aggregates them.
#   E.g., server_global_V = average of all user updates for each item.

# - If the user picks something unexpected, it affects the local model.
#   Over time, these adjustments help the global model better reflect real user preferences. 
