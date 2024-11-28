import os
import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime
from fetcher.netflix_fetcher import NetflixFetcher  # Replace with the actual file name

class TestNetflixFetcherRename(unittest.TestCase):
    @patch.dict(os.environ, {
        "OUTPUT_DIR": "/tmp"
    })
    @patch("fetcher.netflix_fetcher.datetime")  # Mock datetime in the module where it is used
    @patch("os.makedirs", return_value=None)  # Mock directory creation
    @patch("os.rename", return_value=None)    # Mock file renaming
    @patch("os.listdir", return_value=["viewing_activity.csv"])  # Mock the download folder contents
    @patch("time.sleep", return_value=None)  # Avoid actual sleeps
    def test_rename_downloaded_file(self, mock_sleep, mock_listdir, mock_rename, mock_makedirs, mock_datetime):
        """Test renaming and moving the downloaded file."""
        fetcher = NetflixFetcher()

        # Mock current datetime for consistent test output
        mock_datetime.now.return_value = datetime(2024, 11, 27, 14, 30, 45)
        mock_datetime.strftime = datetime.strftime

        fetcher.rename_downloaded_file()

        # Dynamically construct the expected path
        expected_output_dir = os.getenv("OUTPUT_DIR")
        expected_subfolder = os.path.join(expected_output_dir, "2024-11-27")

        # Ensure the subfolder is created
        mock_makedirs.assert_called_once_with(expected_subfolder, exist_ok=True)

        # Ensure the file is renamed
        mock_rename.assert_called_once_with(
            os.path.join(expected_output_dir, "viewing_activity.csv"),
            os.path.join(expected_subfolder, "Netflix_Viewing_Activity_2024-11-27_14-30-45.csv")
        )

    @patch("os.listdir", return_value=[])  # Simulate no files found
    @patch("time.sleep", return_value=None)  # Avoid actual sleeps
    def test_rename_no_file_found(self, mock_sleep, mock_listdir):
        """Test behavior when no downloaded file is found."""
        fetcher = NetflixFetcher()

        # Call the function
        with self.assertLogs(level="INFO") as log:
            fetcher.rename_downloaded_file()

        # Ensure the correct log message is output
        self.assertIn("Download file not found. Please check the download directory.", log.output[-1])

    def tearDown(self):
        """Clean up any test-specific effects."""
        pass

if __name__ == "__main__":
    unittest.main()
