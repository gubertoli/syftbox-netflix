import os
import time
import logging
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys


class NetflixFetcher:
    def __init__(self):
        """Initialize the downloader with environment variables."""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Create handler and formatter
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        
        # Add the handler to the logger
        self.logger.addHandler(console_handler)

        self.logger.info("Initializing Netflix Fetcher")

        self.email = os.getenv("NETFLIX_EMAIL")
        self.password = os.getenv("NETFLIX_PASSWORD")
        self.profile = os.getenv("NETFLIX_PROFILE")
        self.output_dir = os.getenv("OUTPUT_DIR")
        self.driver_path = os.getenv("CHROMEDRIVER_PATH")
        self.csv_name = os.getenv("NETFLIX_CSV")
        self.driver = None

    def setup_driver(self):
        """Set up the Chrome WebDriver."""
        chrome_options = Options()
        prefs = {
            "download.default_directory": self.output_dir,
            "download.prompt_for_download": False,
        }

        self.logger.info("Downloading Netflix Viewing Activity to %s", self.output_dir)

        chrome_options.add_experimental_option("prefs", prefs)
        chrome_options.add_argument("--headless")  # Run in headless mode
        chrome_service = Service(self.driver_path)
        self.driver = webdriver.Chrome(service=chrome_service, options=chrome_options)

    def login(self):
        """Log in to Netflix."""
        print(f"🍿 Downloading Netflix Activity for: {self.email}, Profile {self.profile}")
        self.driver.get("https://www.netflix.com/login")
        email_input = self.driver.find_element(By.NAME, "userLoginId")
        password_input = self.driver.find_element(By.NAME, "password")
        email_input.send_keys(self.email)
        password_input.send_keys(self.password)
        print(">> Logging In")
        password_input.send_keys(Keys.ENTER)
        time.sleep(3)

    def switch_profile(self):
        """Switch to the specified Netflix profile."""
        print(">> Switching Profiles")
        self.driver.get(f"https://www.netflix.com/SwitchProfile?tkn={self.profile}")
        time.sleep(3)

    def download_viewing_activity(self):
        """Download the viewing activity for the current profile."""
        print(">> Getting Viewing Activity")
        self.driver.get("https://www.netflix.com/viewingactivity")
        time.sleep(3)
        self.driver.find_element(By.LINK_TEXT, "Download all").click()
        time.sleep(10)
        self.rename_downloaded_file()

    def rename_downloaded_file(self):
        """Rename the downloaded file into a subfolder with the date and include datetime in the name."""
        print(">> Renaming downloaded file")
        downloaded_file = None

        # Wait until the file appears in the output directory
        for _ in range(20):  # Poll for 20 seconds
            files = os.listdir(self.output_dir)
            for file in files:
                if os.path.basename(file) == self.csv_name:  # Assuming Netflix downloads a CSV file
                    downloaded_file = file
                    break
            if downloaded_file:
                break

        if downloaded_file:
            # Rename the file with the datetime
            date_str = datetime.now().strftime("%Y-%m-%d")
            new_file_name = f"NetflixViewingHistory_{date_str}.csv"
            old_path = os.path.join(self.output_dir, downloaded_file)
            new_path = os.path.join(self.output_dir, new_file_name)

            os.rename(old_path, new_path)
            print(f"File renamed to: {new_path}")
        else:
            self.logger.info("Download file not found. Please check the download directory.")


    def close(self):
        """Close the WebDriver."""
        if self.driver:
            self.driver.quit()

    def run(self):
        """Execute the full routine."""
        try:
            self.setup_driver()
            self.login()
            self.switch_profile()
            self.download_viewing_activity()
        finally:
            self.close()
