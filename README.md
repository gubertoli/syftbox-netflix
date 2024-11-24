# SyftBox for Netflix Viewing History 🍿

This project is a proof of concept utilizing [SyftBox](https://syftbox-documentation.openmined.org/) from [OpenMined](https://openmined.org/) to process 🔒 private data. The use case focuses on analyzing the [Netflix viewing history](https://help.netflix.com/en/node/101917) provided by users. This effort is part of the [#30DaysOfFLCode](https://info.openmined.org/30daysofflcode) initiative.

> ✋ If you are interested in joining this work, check out the Matchmaking spreadsheet for #30DaysOfFLCode [here](https://docs.google.com/spreadsheets/d/1euxZMxQXwctjRt_MVLqnqkuBqpXKuGagLReYANXj1i8/edit?gid=78639164#gid=78639164).

[![Join OpenMined on Slack](https://img.shields.io/badge/Join%20Us%20on-Slack-blue)](https://slack.openmined.org/)

## 🎯 Goals

The primary aim is to apply 🛡️ privacy-enhancing technologies to derive aggregate information from Netflix viewing history while safeguarding personal details. Some possible insights include:

- **Most common show viewed in the last week**
- **Viewing trends among participants**

---

## 🚧 Current Status

The project is currently focused on reducing the granularity of the Netflix viewing history entries. For example:

- **From:** `🎬 The Blacklist: Season 1: Wujing (No. 84) 📅 21/11/2024`
- **To:** `🎬 The Blacklist 📆 Week 47`

---

## Requirements

1. **Netflix Viewing History:** Users must have their Netflix viewing history file. Instructions to retrieve this file are available [here](https://help.netflix.com/en/node/101917).
2. **Set Up ****`DATA_PATH`****:** In `main.py`, set the `DATA_PATH` variable to point to your downloaded viewing history file.

---

## Loading to SyftBox

1. 📂 Copy the following files into the SyftBox API folder:

   - 📄 `requirements.txt`
   - 📄 `run.sh`
   - 📄 `main.py` (ensure it is updated as needed for your setup)

2. The target directory should be:

   - `/SyftBox/apis/netflix_trend_participant`

3. Logs for debugging and status updates can be found in:

   - `/SyftBox/apis/netflix_trend_participant/logs/netflix_trend_participant.log`

---

## 📁 Generated Files

1. **Reduced Viewing History:**

   - 📂 File: `/SyftBox/datasites/<your-email>/api_data/netflix_trend_participant/netflix_reduced.npy`
   - Contains the aggregated and reduced version of the Netflix viewing history, accessible to participants.

2. **Full Viewing History:**

   - 📂 File: `/SyftBox/datasites/<your-email>/private/netflix_data/netflix_full.npy`
   - Contains the full version of the Netflix viewing history, stored privately and not accessible externally.

---

## 🔮 Future Work

- **Noise Addition:** Implement differential privacy by adding noise to the reduced data.
- 📈 **Trend Analysis:** Develop algorithms for analyzing viewing trends across participants while preserving privacy.
- 🤖 **Automation:** Streamline the workflow to minimize manual setup requirements.

---

Feel free to reach out with questions or suggestions to improve this project.
