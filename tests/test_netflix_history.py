import os
import unittest
import numpy as np
from unittest.mock import patch, MagicMock
from datetime import datetime
from main import (
    load_csv_to_numpy,
    extract_titles,
    convert_dates_to_weeks,
    orchestrate_reduction,
    aggregate_title_week_counts,
    get_or_download_latest_data,
    download_daily_data,
    get_latest_file,
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
        result = load_csv_to_numpy(file_path)

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

    @patch("main.NetflixFetcher")
    def test_download_daily_data(self, mock_fetcher):
        # Simulate NetflixFetcher.run
        mock_fetcher_instance = MagicMock()
        mock_fetcher.return_value = mock_fetcher_instance

        # Run the function
        download_daily_data()

        # Assert the downloader's `run` method was called
        mock_fetcher_instance.run.assert_called_once()

    @patch("main.CSV_NAME", "NetflixViewingActivity.csv")
    @patch("os.listdir", return_value=[
        "NetflixViewingActivity_2024-11-27.csv",
        "NetflixViewingActivity_2024-11-26.csv",
        "NetflixViewingActivity_2024-11-25.csv",
    ])
    @patch("os.path.isfile", return_value=True)
    def test_get_latest_file(self, mock_isfile, mock_listdir):
        """
        Test get_latest_file with mocked directory and files.
        """
        subfolder_path = "/mocked/subfolder"

        # Expected result
        expected_file = "/mocked/subfolder/NetflixViewingActivity_2024-11-27.csv"

        # Run the function
        result = get_latest_file(subfolder_path)

        # Assertions
        self.assertEqual(result, expected_file)
        mock_listdir.assert_called_once_with(subfolder_path)

    @patch("main.get_latest_file")
    @patch("main.download_daily_data")
    @patch("os.path.exists")
    @patch("main.OUTPUT_DIR", "/mocked/netflix_data")
    @patch("main.CSV_NAME", "NetflixViewingActivity.csv")
    @patch("main.datetime")  # Mock datetime module inside the function
    def test_get_or_download_latest_data_file_exists(self, mock_datetime, mock_exists, mock_download, mock_get_latest):
        """
        Test when today's file already exists.
        """
        # Mock today's date
        mock_datetime.now.return_value = datetime(2024, 11, 27)
        mock_datetime.strftime = datetime.strftime  # Ensure strftime works as expected

        today_date = "2024-11-27"
        file_path = f"/mocked/netflix_data/NetflixViewingActivity_{today_date}.csv"

        # Mock os.path.exists to simulate the file exists
        mock_exists.return_value = True

        # Mock get_latest_file to return the existing file
        mock_get_latest.return_value = file_path

        # Run the function
        result = get_or_download_latest_data()

        # Assertions
        mock_download.assert_not_called()  # Download should not be triggered
        mock_get_latest.assert_called_once_with("/mocked/netflix_data")
        self.assertEqual(result, file_path)

    @patch("main.get_latest_file")
    @patch("main.download_daily_data")
    @patch("os.path.exists")
    @patch("main.OUTPUT_DIR", "/mocked/netflix_data")
    @patch("main.CSV_NAME", "NetflixViewingActivity.csv")
    @patch("main.datetime")  # Mock datetime module inside the function
    def test_get_or_download_latest_data_file_missing(self, mock_datetime, mock_exists, mock_download, mock_get_latest):
        """
        Test when today's file does not exist, and download is triggered.
        """
        # Mock today's date
        mock_datetime.now.return_value = datetime(2024, 11, 27)
        mock_datetime.strftime = datetime.strftime  # Ensure strftime works as expected

        today_date = "2024-11-27"
        file_path = f"/mocked/netflix_data/NetflixViewingActivity_{today_date}.csv"

        # Mock os.path.exists to simulate the file does not exist
        mock_exists.side_effect = lambda path: path != file_path

        # Mock get_latest_file to return the new file after download
        mock_get_latest.return_value = file_path

        # Run the function
        result = get_or_download_latest_data()

        # Assertions
        mock_download.assert_called_once()  # Ensure download is triggered
        mock_get_latest.assert_called_once_with("/mocked/netflix_data")
        self.assertEqual(result, file_path)

if __name__ == "__main__":
    unittest.main()
