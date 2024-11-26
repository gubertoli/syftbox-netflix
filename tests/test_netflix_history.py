import unittest
from unittest.mock import patch
import numpy as np
from main import (
    load_netflix_history,
    extract_titles,
    convert_dates_to_weeks,
    orchestrate_reduction,
    aggregate_title_week_counts
)

class TestNetflixHistory(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Setup class-wide resources before running the tests.
        """
        print("Setting up class resources...")
        cls.test_history = np.array([
            ["The Blacklist: Season 1", "01/01/2023"],
            ["The Blacklist: Season 1", "02/01/2023"],
            ["Movie Title", "02/04/2023"],
            ["Another Movie: Season 4: Chapter 1", "21/10/2023"],
            ["Another Movie: Season 4: Chapter 2", "22/10/2023"],
        ])
        cls.expected_reduced = np.array([
            ["The Blacklist", "52"],  # ISO week of 01/01/2023
            ["The Blacklist", "1"],   # ISO week of 02/01/2023
            ["Movie Title", "13"],    # ISO week of 02/04/2023
            ["Another Movie", "42"],  # ISO week of 21/10/2023
            ["Another Movie", "42"],  # ISO week of 21/10/2023
        ])

    @classmethod
    def tearDownClass(cls):
        """
        Clean up class-wide resources after tests.
        """
        print("Cleaning up class resources...")
        del cls.test_history
        del cls.expected_reduced

    @patch("builtins.open", new_callable=unittest.mock.mock_open, read_data="Title,Date\nMovie1,01/01/2023\nMovie2,02/01/2023")
    def test_load_netflix_history(self, mock_open):
        file_path = "/fake/path/NetflixViewingHistory.csv"
        result = load_netflix_history(file_path)

        expected = np.array([
            ["Movie1", "01/01/2023"],
            ["Movie2", "02/01/2023"],
        ])
        np.testing.assert_array_equal(result, expected)
        mock_open.assert_called_once_with(file_path, mode="r", encoding="utf-8")

    def test_extract_titles(self):
        result = extract_titles(self.test_history)
        expected = np.array([
            "The Blacklist",
            "The Blacklist",
            "Movie Title",
            "Another Movie",
            "Another Movie",
        ])
        np.testing.assert_array_equal(result, expected)

    def test_convert_dates_to_weeks(self):
        result = convert_dates_to_weeks(self.test_history)
        expected = np.array([
            52,  # ISO week of 01/01/2023
            1,   # ISO week of 02/01/2023
            13,  # ISO week of 02/04/2023
            42,  # ISO week of 21/10/2023
            42,  # ISO week of 21/10/2023
        ])
        np.testing.assert_array_equal(result, expected)

    def test_orchestrate_reduction(self):
        result = orchestrate_reduction(self.test_history)
        np.testing.assert_array_equal(result, self.expected_reduced)

    def test_aggregate_title_week_counts(self):
        reduced_data = self.expected_reduced

        result = aggregate_title_week_counts(reduced_data)

        expected = np.array([
            ["The Blacklist", "52", "1"],
            ["The Blacklist", "1", "1"],
            ["Movie Title", "13", "1"],
            ["Another Movie", "42", "2"],
        ])

        np.testing.assert_array_equal(result, expected)

if __name__ == "__main__":
    unittest.main()
