import os
import shutil
import unittest
import numpy as np
import json
from participant.participant_utils.data_loading import (
    load_tv_vocabulary,
    load_participant_ratings,
    load_global_item_factors,
    load_or_initialize_user_matrix,
    initialize_user_matrix,
)

class TestLoadingFunctions(unittest.TestCase):

    def setUp(self):
        # Setup sandboxed test environment
        self.sandbox_dir = "test_sandbox"
        self.vocabulary_path = os.path.join(self.sandbox_dir, "tv_vocabulary.json")
        self.private_folder = os.path.join(self.sandbox_dir, "private")
        self.save_path = os.path.join(self.sandbox_dir, "tmp_model_parms")
        self.user_id = "test_user"
        self.global_V_path = os.path.join(self.save_path, "global_V.npy")
        self.user_matrix_path = os.path.join(self.save_path, "U.npy")

        # Create directories
        os.makedirs(self.private_folder, exist_ok=True)
        os.makedirs(self.save_path, exist_ok=True)

        # Mock data
        self.tv_vocab = {"show1": 0, "show2": 1}
        self.ratings = {"show1": 4.5, "show2": 3.0}
        self.V = np.random.normal(size=(2, 10))  # 2 items, latent dimension 10

        # Write mock data
        with open(self.vocabulary_path, "w") as f:
            json.dump(self.tv_vocab, f)
        np.save(os.path.join(self.private_folder, "ratings.npy"), self.ratings)
        np.save(self.global_V_path, self.V)

    def tearDown(self):
        # Remove sandbox directory after each test
        if os.path.exists(self.sandbox_dir):
            shutil.rmtree(self.sandbox_dir)

    def test_load_tv_vocabulary(self):
        # Load vocabulary
        loaded_vocab = load_tv_vocabulary(self.vocabulary_path)

        # Assert vocabulary matches
        self.assertEqual(loaded_vocab, self.tv_vocab)

    def test_load_participant_ratings(self):
        # Load participant ratings
        loaded_ratings = load_participant_ratings(self.private_folder)

        # Assert ratings match
        self.assertEqual(loaded_ratings, self.ratings)

    def test_load_global_item_factors(self):
        # Load global item factors
        loaded_V = load_global_item_factors(self.save_path)

        # Assert global factors match
        np.testing.assert_array_equal(loaded_V, self.V)

    def test_initialize_user_matrix(self):
        # Initialize user matrix
        latent_dim = 10
        U_u = initialize_user_matrix(self.user_id, latent_dim, save_path=self.save_path)

        # Assert file was created
        self.assertTrue(os.path.exists(self.user_matrix_path))

        # Assert matrix dimensions
        self.assertEqual(U_u.shape[0], latent_dim)

    def test_load_or_initialize_user_matrix_initialize(self):
        # Remove user matrix if exists
        if os.path.exists(self.user_matrix_path):
            os.remove(self.user_matrix_path)

        # Load or initialize user matrix
        latent_dim = 10
        U_u = load_or_initialize_user_matrix(self.user_id, latent_dim, save_path=self.save_path)

        # Assert file was created
        self.assertTrue(os.path.exists(self.user_matrix_path))

        # Assert matrix dimensions
        self.assertEqual(U_u.shape[0], latent_dim)

    def test_load_or_initialize_user_matrix_load(self):
        # Pre-save a user matrix
        latent_dim = 10
        pre_saved_matrix = np.random.normal(size=(latent_dim,))
        np.save(self.user_matrix_path, pre_saved_matrix)

        # Load or initialize user matrix
        U_u = load_or_initialize_user_matrix(self.user_id, latent_dim, save_path=self.save_path)

        # Assert loaded matrix matches pre-saved matrix
        np.testing.assert_array_equal(U_u, pre_saved_matrix)

        # Assert no re-initialization
        self.assertTrue(os.path.exists(self.user_matrix_path))


if __name__ == "__main__":
    unittest.main()