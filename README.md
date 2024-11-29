# SyftBox for Netflix Viewing History 🍿

This project is a proof of concept utilizing [SyftBox](https://syftbox-documentation.openmined.org/) from [OpenMined](https://openmined.org/) to process 🔒 private data. The use case focuses on analyzing the [Netflix viewing history](https://help.netflix.com/en/node/101917) provided by users. This effort is part of the [#30DaysOfFLCode](https://info.openmined.org/30daysofflcode) initiative.

> ✋ If you are interested in joining this work, check out the Matchmaking spreadsheet for #30DaysOfFLCode [here](https://docs.google.com/spreadsheets/d/1euxZMxQXwctjRt_MVLqnqkuBqpXKuGagLReYANXj1i8/edit?gid=78639164#gid=78639164).

[![Join OpenMined on Slack](https://img.shields.io/badge/Join%20Us%20on-Slack-blue)](https://slack.openmined.org/)

## 🎯 Goals

The primary aim is to apply 🛡️ privacy-enhancing technologies to derive aggregate information from Netflix viewing history while safeguarding personal details. Some possible insights include (**_ideas are welcome_**):

- **Most common show viewed in the last week**
- **Viewing trends among participants**
- **Am I watching too much in comparison with others?**
- **Watching more due to sickness/injury?** [(source)](https://www.kaggle.com/code/nachoco/netflix-viewing-analysis-with-injury)

## Sketch Preview

Below is the sketch preview for insights generated on 2024-11-28:

![Sketch for 2024-11-28](aggregator/static/sketch-2024-11-28.png)


---

## Requirements
_Tested on Linux and MacOS._

0. Install the `chromedriver` to perform a daily retrieval of your Netflix viewing history:
   ```bash
   brew install chromedriver  # MacOS
   ```
   ```bash
   sudo apt-get install chromium-driver  # Ubuntu
   ```

1. **Start SyftBox:**
   ```bash
   curl -LsSf https://syftbox.openmined.org/install.sh | sh
   ```
2. **Copy this repository to SyftBox:** 

   - Copy this repository to your SyftBox `apis` folder:
   ```bash
   cd ~/SyftBox/apis
   git clone https://github.com/gubertoli/syftbox-netflix.git netflix_trend_participant
   ```
3. **⚠️ Set Up the Environment:** 

   - You shall create a `.env` file with your data in the `netflix_trend_participant`. 
   ```bash
   cd netflix_trend_participant
   cp .env.example .env
   ```

   - Edit the `.env` with your Netflix personal information.

### Data format (Netflix)
A comma-separated file (CSV), organized by Title and Date:

```
Title,Date
"The Blacklist: Season 1: Wujing (No. 84)","21/11/2024"
"Buy Now: The Shopping Conspiracy","20/11/2024"
"Jake Paul vs. Mike Tyson","16/11/2024"
"Murder Mindfully: Breathing","15/11/2024"
...
```

---

## 📁 Generated Files

1. **Aggregated / PET Files:**

   - 📂 Path: `/SyftBox/datasites/<your-email>/api_data/netflix_trend_participant/`
   - This folder contains the aggregated and/or processed (privacy enhanced) Netflix viewing history, accessible to aggregator. For instance, parameters of machine learning models for federated learning or differential private metrics.

2. **Full Viewing History:**

   - 📂 File: `/SyftBox/datasites/<your-email>/private/netflix_data/netflix_full.npy`
   - Contains the full version of the Netflix viewing history, stored privately and not accessible externally. This could be used as a starting point for PETs evaluations,

---

## 🔮 Future Work

- **Noise Addition:** Implement differential privacy by adding noise to the reduced data.
- 📈 **Trend Analysis:** Develop algorithms for analyzing viewing trends across participants while preserving privacy.
- 🤖 **Automation:** Streamline the workflow to minimize manual setup requirements.

---

Feel free to reach out with questions or suggestions to improve this project.
