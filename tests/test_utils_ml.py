import unittest
import pandas as pd
import numpy as np
import os
from unittest.mock import patch
from pathlib import Path
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from datetime import datetime
from utils.ml import SequenceModel, extract_features, prepare_data, train_model, get_recommendation

class TestMLRoutines(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Set up the sandbox directory for tests.
        """
        cls.sandbox_path = Path("test_sandbox/utils_ml")
        if cls.sandbox_path.exists():
            for item in cls.sandbox_path.iterdir():
                if item.is_dir():
                    os.rmdir(item)
                else:
                    item.unlink()
        cls.sandbox_path.mkdir(parents=True, exist_ok=True)
        cls.file_path = cls.sandbox_path / "test_dataset.csv"

    @classmethod
    def tearDownClass(cls):
        """
        Clean up the sandbox directory after tests.
        """
        if cls.sandbox_path.exists():
            for item in cls.sandbox_path.iterdir():
                if item.is_dir():
                    os.rmdir(item)
                else:
                    item.unlink()
            cls.sandbox_path.rmdir()

    def setUp(self):
        """
        Set up a mock dataset for testing.

        Includes dates with day > 12 to validate correct datetime parsing.
        """
        self.dataset = np.array([
            ["Show A: Season 1", "01/01/2023"], # Sunday
            ["Show A: Season 1", "13/01/2023"], # Friday, Day > 12
            ["Show B: Season 1", "25/01/2023"], # Wednesday, Day > 12
            ["Show A: Season 2", "04/01/2023"], # Wednesday
            ["Show C", "15/01/2023"],           # Sunday, Day > 12
            ["Show B: Season 1", "26/01/2023"], # Thursday, Day > 12
        ])
        df = pd.DataFrame(self.dataset, columns=["Title", "Date"])
        df.to_csv(self.file_path, index=False)


    def test_extract_features(self):
        """
        Test feature extraction from raw dataset.

        Verifies that features like 'show', 'season', and 'day_of_week'
        are correctly extracted from input data and that datetime parsing
        handles various formats, including dates with day > 12.

        Example:
            Input:  Title               Date
                    Show A: Season 1    13/01/2023
            Output: show="Show A", season=1, day_of_week=4
        """
        df = pd.DataFrame(
            [
                ["Show A: Season 1", "13/01/2023"],  # Day > 12
                ["Show B", "25/01/2023"],           # Day > 12
                ["Show A: Season 2", "03/01/2023"],
                ["Show C", "15/01/2023"],           # Day > 12
            ],
            columns=["Title", "Date"],
        )
        df_processed = extract_features(df)

        # Validate extracted features
        self.assertIn("show", df_processed.columns)
        self.assertIn("season", df_processed.columns)
        self.assertIn("day_of_week", df_processed.columns)

        # Check parsing and feature extraction
        self.assertEqual(df_processed.loc[0, "show"], "Show A")
        self.assertEqual(df_processed.loc[0, "season"], 1)
        self.assertEqual(df_processed.loc[0, "day_of_week"], 4)  # Friday
        self.assertEqual(df_processed.loc[1, "show"], "Show B")
        self.assertEqual(df_processed.loc[1, "season"], 0)
        self.assertEqual(df_processed.loc[1, "day_of_week"], 2)  # Wednesday

    def test_process_dataset(self):
        """
        Test SequenceModel data processing and aggregation.

        Verifies that the dataset is sorted by 'First_Seen' and filtered
        to include only shows with multiple views.

        Example:
            Input: Multiple entries of "Show A" and "Show B".
            Output: Aggregated views and earliest dates per show.
        """
        model = SequenceModel(self.dataset)
        df_aggregated = model.aggregated_data

        # Ensure consistent date formats in expected results
        expected = pd.DataFrame(
            {
                "show": ["Show A", "Show B"],
                "Total_Views": [3, 2],
                "First_Seen": [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-25")],
            }
        )

        # Convert First_Seen to datetime64[ns] to ensure format consistency
        expected["First_Seen"] = pd.to_datetime(expected["First_Seen"])

        # Assert equality of DataFrames
        pd.testing.assert_frame_equal(df_aggregated, expected)

    def test_prepare_data(self):
        """
        Test data preparation for model training.

        Ensures that the feature matrix (X) and target variable (y) are
        constructed correctly from the input dataset.

        Example:
            Input: Titles with encoded 'show', 'season', and 'day_of_week'.
            Output: X=[[show_encoded, season, day_of_week]], y=[next_show_encoded]
        """
        df = pd.DataFrame(self.dataset, columns=["Title", "Date"])
        df.to_csv(self.file_path, index=False)

        X, y, le_show = prepare_data(self.file_path)

        # Expected X and y
        expected_X = np.array(
            [
                [0, 1, 6],  # Show A, Season 1, Sunday
                [0, 1, 4],  # Show A, Season 1, Friday
                [1, 1, 2],  # Show B, Season 1, Wednesday
                [0, 2, 2],  # Show A, Season 2, Wednesday
                [2, 0, 6],  # Show C, None, Sunday
            ]
        )

        expected_y = np.array([0, 1, 0, 2, 1])

        # Debugging logs
        print("Actual X:")
        print(X)
        print("Expected X:")
        print(expected_X)
        print("Actual y:")
        print(y)
        print("Expected y:")
        print(expected_y)

        # Validate results
        np.testing.assert_array_equal(X, expected_X)
        np.testing.assert_array_equal(y, expected_y)

    def test_train_model(self):
        """
        Test the train_model function for training an MLP classifier.

        Verifies that the model, scaler, and label encoder are correctly
        returned and that the number of samples matches the dataset size.

        Example:
            Input: Dataset with features and labels split into training and testing.
            Output: Trained MLPClassifier, StandardScaler, and LabelEncoder.
        """
        df = pd.DataFrame(self.dataset, columns=["Title", "Date"])
        df.to_csv(self.file_path, index=False)

        mlp, scaler, le_show, num_samples = train_model(self.file_path)

        # Validate outputs
        self.assertIsInstance(mlp, MLPClassifier)
        self.assertIsInstance(scaler, StandardScaler)
        self.assertIsInstance(le_show, LabelEncoder)
        self.assertEqual(num_samples, len(self.dataset) - 1)

    @patch("utils.ml.get_current_day_of_week")
    def test_get_recommendation(self, mock_get_current_day_of_week):
        """
        Test the get_recommendation function for generating show recommendations.

        Verifies:
            1. The model can predict a valid recommendation.
            2. The recommendation aligns with the shows in the dataset based on day_of_week.
        """
        # Prepare mock data for training
        X_mock = np.array([
            [0, 1, 6],  # Show A, Season 1, Sunday
            [1, 2, 0],  # Show B, Season 2, Monday
            [2, 3, 1],  # Show C, Season 3, Tuesday
        ])
        y_mock = np.array([1, 2, 0])  # Encoded target: next show

        # Train MLP model
        mlp = MLPClassifier(hidden_layer_sizes=(64, 32), random_state=42)
        mlp.fit(X_mock, y_mock)

        # Train scaler and label encoder
        scaler = StandardScaler()
        scaler.fit(X_mock)
        le_show = LabelEncoder()
        le_show.fit(["Show A", "Show B", "Show C"])

        # Input for recommendation
        last_watched = "Show A: Season 1"

        # Define test cases for each day of the week
        test_cases = [
            ((datetime(2023, 1, 1), 6), "Show B"),  # Sunday
            ((datetime(2023, 1, 2), 0), "Show C"),  # Monday
        ]

        for mock_date, expected_recommendation in test_cases:
            with self.subTest(day_of_week=mock_date[1]):
                # Mock the current day and day of the week
                mock_get_current_day_of_week.return_value = mock_date

                # Get recommendation
                recommendation = get_recommendation(mlp, scaler, le_show, last_watched)

                # Assertions
                self.assertIn(recommendation, ["Show A", "Show B", "Show C"])  # Validate recommendation is in dataset
                self.assertEqual(recommendation, expected_recommendation)  # Validate correct recommendation


if __name__ == "__main__":
    unittest.main()
