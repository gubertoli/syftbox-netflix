# SyftBox App for Netflix Viewing History 🍿

This project is a proof of concept utilizing [SyftBox](https://syftbox-documentation.openmined.org/) from [OpenMined](https://openmined.org/) to process 🔒 private data. The use case focuses on analyzing the [Netflix viewing history](https://help.netflix.com/en/node/101917) provided by users. This effort is part of the [#30DaysOfFLCode](https://info.openmined.org/30daysofflcode) initiative.

[![Join OpenMined on Slack](https://img.shields.io/badge/Join%20Us%20on-Slack-blue)](https://slack.openmined.org/)

## 🎯 Goals

The primary aim is to apply 🛡️ privacy-enhancing technologies to derive aggregate information from Netflix viewing history while safeguarding personal details. Some possible insights include (**_ideas are welcome_**):

- **Most common show viewed in the last week**
- **Viewing trends among participants**
- **Am I watching too much in comparison with others?**
- **Watching more due to sickness/injury?** [(source)](https://www.kaggle.com/code/nachoco/netflix-viewing-analysis-with-injury)

---

## Installation & Requirements
**_Tested on Linux and macOS._**

Download your Netflix viewing activity as a CSV file from your Netflix account. See [How to download your Netflix viewing history](https://help.netflix.com/en/node/101917). Once downloaded, place the CSV file into the folder specified by the `OUTPUT_DIR` variable in your `.env` file. 

If you have data from **multiple profiles**, save each profile's CSV in its own subfolder within `OUTPUT_DIR`:

```bash
OUTPUT_DIR/
├── profile_0/
│   └── ViewingActivity.csv
├── profile_1/
│   └── ViewingActivity.csv
└── ...

```

### 1. Start SyftBox
Install and start SyftBox by running this command:

   ```bash
   curl -LsSf https://syftbox.openmined.org/install.sh | sh
   ```
### 2. Install app on SyftBox
From terminal, once SyftBox is running:
   ```bash
   syftbox app install svenlankester/syftbox-netflix
   ```

### 3. Set Up the Environment
Configure this app inside SyftBox.

1. Navigate to the `syftbox-netflix` directory:
   ```bash
   cd /SyftBox/apps/syftbox-netflix
   ```
2. Open the `.env` file in a text editor and **define at least** `OUTPUT_DIR`. This is the directory to make available your `NetflixViewingHistory.csv` downloaded manually, if not available, a dummy file will be created. 

Optionally, fill in your Netflix account information to automate viewing history retrieval. For this option, you need the ChromeDriver.

- **For macOS:**
  ```bash
  brew install chromedriver
   ```

- **For Ubuntu/Linux:**
   ```bash
   sudo apt-get install chromium-driver
   ```

#### A more complete `.env` example:
   ```
   APP_NAME="syftbox-netflix"                            # Mandatory
   AGGREGATOR_DATASITE="<aggregator-datasite-email>"     # Mandatory
   NETFLIX_EMAIL="<your-netflix-email@provider.com>"
   NETFLIX_PASSWORD="<your-password>"
   NETFLIX_PROFILE="<profile-name>"
   NETFLIX_CSV="NetflixViewingHistory.csv"               # Mandatory
   OUTPUT_DIR="/home/<your-username>/Downloads/"         # Mandatory
   AGGREGATOR_DATA_DIR="data/"                           # Mandatory
   ```

# Viewing History Data format (Netflix)
The data provided by Netflix (Viewing History) is a comma-separated file (CSV), organized by Title and Date:

   ```
   Title,Date
   "Show Name: Season X: Episode Name","DD/MM/YYYY"
   ...
   ```

   > :warning: The retrieved data might be in the format `MM/DD/YY`. The current implementation is capable of both, if your viewing history has other data format, changes are needed.
---


Feel free to reach out with questions or suggestions to improve this project.
