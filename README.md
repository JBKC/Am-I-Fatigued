# PPG_fatigue

1. [Project Description](#project-description)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Contributing](#contributing)
5. [License](#license)
6. [Contact](#contact)

Goal: Use Supervised ML models to determine between different classes of photoplethysmography (PPG) biosignals (fatigued vs non-fatigued)

Fatigued = reading taken <2hrs after an intense training session (sprinting + heavy weights)

Non-fatigued = no kind of physical exercise for >12hrs prior

^^ these were my project definitions - can also be used for testing mental fatigue / stress as well as physical fatigue

# Methodology
1. Collect multiple PPG signals (ideally 10s or 100s) from wearable wrist device using Arduino or similar - make sure as close to zero motion artifacts as possible by keeping body still throughout
2. Save down raw signal values in .txt files
3. Clean signals and perform feature extraction
4. Train features on supervised models
5. Test model to get model accuracy, precision, recall, ROC AUC and probability gradient plots for the classes

# Files in this repo
Lots of notes + descriptions in the code. Summary here:

- PPG_data.py - master file that extracts and gathers PPG data. Run this as a first step after gathering the raw PPG data. Outputs a CSV file containing dataframe of features for each signal
- PPG_full_analysis.py - extracts features specific to PPG cycles. Called automatically by PPG_data.py
- HRV_analysis.py - extracts features specific to HRV (heart rate variability). Called automatically by PPG_data.py
- PPG_models.py - takes CSV from PPG_data.py and trains data on model specified by user, then tests the model on a previously unseen split
- general_peak_filtering.py - general peak-finding algorithm tailored to PPG


# Results
Will post up soon

# Sources of inspiration
