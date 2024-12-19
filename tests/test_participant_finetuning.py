import os
import shutil
import unittest
import numpy as np
from participant.federated_learning.svd_participant_finetuning import (
    save_training_results,
    prepare_training_data,
    perform_local_training
)

class TestParticipantFineTuning(unittest.TestCase):

    def setUp(self):
        # Setup sandboxed test environment
        self.sandbox_dir = "test_sandbox/participant_datasite"
        self.user_id = "test_user"
        self.shared_path = os.path.join(self.sandbox_dir, "shared")
        self.restricted_path = os.path.join(self.sandbox_dir, "restricted", self.user_id)
        self.private_folder = os.path.join(self.sandbox_dir, "private", self.user_id)

        # Create sandbox directories
        os.makedirs(self.shared_path, exist_ok=True)
        os.makedirs(os.path.join(self.private_folder, "svd_training"), exist_ok=True)
        os.makedirs(os.path.join(self.restricted_path, "svd_training"), exist_ok=True)

        # Declare input/output paths
        self.global_V_path = os.path.join(self.shared_path, "global_V.npy")
        self.participant_V_path = os.path.join(self.private_folder, "svd_training", f"updated_V.npy")
        self.user_matrix_path = os.path.join(self.private_folder, "svd_training", f"U.npy")
        self.delta_V_path = os.path.join(self.restricted_path, "svd_training", f"delta_V.npy")

        # Mock data
        self.tv_vocab = {"show1": 0, "show2": 1}
        self.final_ratings = {"show1": 4.5, "show2": 3.0}
        self.V = np.random.normal(size=(2, 10))  # 2 items, latent dimension 10
        self.U_u = np.random.normal(size=(10,))

        # Save mock data
        np.save(self.global_V_path, self.V)
        np.save(os.path.join(self.private_folder, "ratings.npy"), self.final_ratings)
        np.save(self.user_matrix_path, self.U_u)

    def tearDown(self):
        # Remove sandbox directory after each test
        if os.path.exists(self.sandbox_dir):
            shutil.rmtree(self.sandbox_dir)

    def test_prepare_training_data(self):
        # Prepare training data
        train_data = prepare_training_data(self.user_id, self.tv_vocab, self.final_ratings)

        # Expected result
        expected_data = [
            (self.user_id, 0, 4.5),  # "show1"
            (self.user_id, 1, 3.0),  # "show2"
        ]
        self.assertEqual(train_data, expected_data)

    def test_perform_local_training(self):
        # Prepare training data
        train_data = prepare_training_data(self.user_id, self.tv_vocab, self.final_ratings)

        # Perform training
        initial_V_returned, updated_V, updated_U_u = perform_local_training(
            train_data, self.V, self.U_u, alpha=0.01, lambda_reg=0.1, iterations=10
        )

        # Ensure initial_V is returned unchanged
        np.testing.assert_array_equal(initial_V_returned, self.V)

        # Ensure updated_V and updated_U_u are modified
        self.assertFalse(np.allclose(updated_V, self.V))
        self.assertFalse(np.allclose(updated_U_u, self.U_u))

    def test_save_training_results(self):
        # Save training results
        save_training_results(self.user_id, self.private_folder, self.restricted_path, self.V, self.final_ratings, self.U_u)

        # Check that the files exist
        self.assertTrue(os.path.exists(self.participant_V_path))
        self.assertTrue(os.path.exists(self.delta_V_path))
        self.assertTrue(os.path.exists(self.user_matrix_path))

        # Validate the saved data
        np.testing.assert_array_equal(np.load(self.participant_V_path), self.V)
        saved_delta = np.load(self.delta_V_path, allow_pickle=True).item()
        self.assertEqual(saved_delta.keys(), self.final_ratings.keys())  # Check keys match
        np.testing.assert_array_equal(np.load(self.user_matrix_path), self.U_u)