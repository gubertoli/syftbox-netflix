import os
import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime
from fetcher.netflix_fetcher import NetflixFetcher  # Replace with the actual file name

class TestNetflixFetcherRename(unittest.TestCase):
    @patch.dict(os.environ, {
        "OUTPUT_DIR": "/mocked/output/dir"
    })
    @patch("fetcher.netflix_fetcher.datetime")  # Mock datetime in the module where it is used
    @patch("os.rename", return_value=None)    # Mock file renaming
    @patch("os.listdir", return_value=["viewing_activity.csv"])  # Mock the download folder contents
    @patch("time.sleep", return_value=None)  # Avoid actual sleeps
    def test_rename_downloaded_file(self, mock_sleep, mock_listdir, mock_rename, mock_datetime):
        """Test renaming the downloaded file with the current date."""
        fetcher = NetflixFetcher()
        fetcher.csv_name = "viewing_activity.csv"  # Set the expected CSV name
        
        # Mock current datetime for consistent test output
        mock_datetime.now.return_value = datetime(2024, 11, 27, 14, 30, 45)
        mock_datetime.strftime = datetime.strftime

        fetcher.rename_downloaded_file()

        # Ensure the file is renamed with the expected name and date
        mock_rename.assert_called_once_with(
            os.path.join(fetcher.output_dir, "viewing_activity.csv"),
            os.path.join(fetcher.output_dir, "NetflixViewingHistory_2024-11-27.csv")
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
