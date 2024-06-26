# Classifying fatigue from PPG signals

The goal of this project is to differentiate between physically fatigued vs. non-fatigued states from only a photoplethysmography (PPG) signal. PPG data is collected from a wrist-worn device, normalised and filtered, then features are extracted and passed into a range of supervised ML classifier models.

## Contents

1. [Project Description](#project-description)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Contributing](#contributing)
5. [License](#license)
6. [Contact](#contact)


## Repo structure
Lots of clarifying notes + descriptions in the code. Summary:

- `data/` - folder containing raw .txt files of PPG signals
- `scripts/` - folder containing python scripts
  - `PPG_data.py` - master script that extracts and gathers PPG data. Run this as a first step after gathering the raw PPG data. Outputs a CSV file containing dataframe of features for each signal
  - `PPG_full_analysis.py` - extracts features specific to PPG cycles. Called automatically by PPG_data.py
  - `HRV_analysis.py` - extracts features specific to HRV (heart rate variability). Called automatically by PPG_data.py
  - `PPG_models.py` - takes CSV output from PPG_data.py and trains data on a supervised ML model specified by user, then tests the model on a previously unseen split
  - `general_peak_filtering.py` - general peak-finding algorithm tailored to PPG

## Installation
### 1. Clone this repository:
   ```
   git clone https://github.com/yourusername/PPG_fatigue.git
   cd PPG_fatigue
   ```

### 2. Create and activate virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate
   ```

### 3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage
### 1. Run main script to extract and process data 

   ```
    python scripts/PPG_data.py
   ```
When prompted, type `run new`

The output will be a CSV file

## Usage
1. Collect multiple PPG signals (ideally 10s or 100s) from wearable wrist device using Arduino or similar - make sure as close to zero motion artifacts as possible by keeping body still throughout
2. Save down raw signal values in .txt files
3. Clean signals and perform feature extraction
4. Train features on supervised models
5. Test model to get model accuracy, precision, recall, ROC AUC and probability gradient plots for the classes

6. Goal: Use Supervised ML models to determine between different classes of photoplethysmography (PPG) biosignals (fatigued vs non-fatigued)

Fatigued = reading taken <2hrs after an intense training session (sprinting + heavy weights)

Non-fatigued = no kind of physical exercise for >12hrs prior

^^ these were my project definitions - can also be used for testing mental fatigue / stress as well as physical fatigue


# Results
Will post up soon

# Sources of inspiration
