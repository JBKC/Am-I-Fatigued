# Classifying fatigue from PPG signals

The goal of this project is to differentiate between physically fatigued vs. non-fatigued states from only a photoplethysmography (PPG) signal. PPG data is collected from a wrist-worn device, normalised and filtered, then features are extracted and passed into a range of supervised ML classifier models which runs nested cross validation over train/test splits and optimal hyperparameter combinations.


## Contents

1. [Project structure](#project-structure)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Results](#results)
5. [Sources of inspiration](#sources-of-inspiration)



## Project structure
Lots of clarifying notes + descriptions in the code. Summary:

- `data/` - folder containing raw .txt files of PPG signals taken from my wrist. "Fatigued" data was taken <2hrs after an intense training session (sprinting + heavy weights); "Non-fatigued" is defined as having done no kind of physical exercise for >12hrs prior to reading.
- `scripts/` - folder containing python scripts
  - `PPG_data.py` - master script that extracts and gathers PPG data. Run this as a first step after gathering the raw PPG data. Outputs a CSV file containing dataframe of features for each signal
  - `PPG_full_analysis.py` - extracts features specific to PPG cycles. Called automatically by PPG_data.py
  - `HRV_analysis.py` - extracts features specific to HRV (heart rate variability). Called automatically by PPG_data.py
  - `PPG_models.py` - takes CSV output from PPG_data.py and trains data on a supervised ML model specified by user, then tests the model on a previously unseen split
  - `general_peak_filtering.py` - general peak-finding algorithm tailored to PPG


## Installation
### 1. Clone this repository:
   ```
   git clone https://github.com/JBKC/PPG_fatigue.git
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

Run the main file in the terminal:

   ```
    python scripts/PPG_data.py
   ```
When prompted, type `run new`. You'll see the names of the filenames printed on the terminal as each session is processed.

Output is a CSV file named 'ppg_dataframe.csv'

### 2. (Optional) Plot XY scatter data

Run the same file again:
   ```
    python scripts/PPG_data.py
   ```
When prompted, type `plot results` followed by desired X axis and Y axis (must be typed exactly as in dataframe)

### 3. Train models on data

Run the models file:
   ```
    python scripts/PPG_models.py
   ```
When prompted, choose which model to train the data on exactly as appears in the options:
- `svm` = Support Vector Machine
- `random forest` = Random Forest Algorithm
- `knn` = K Nearest Neighbours
- `qda` = Quadratic Discrimnant Analysis
- `gaussian nb` = Gaussian Naive Bayes
- `gaussian pc` = Gaussian Process Classification
  
Output is the chosen hyperparameters, model performance table (accuracy, precision, recall, f1-score), confusion matrix values, ROC AUC (with ROC curve plot).

Also outputs a probability countour plot over the training data  for `svm`, `knn`, `qda`, `gaussian nb` & `gaussian pc` models (close the ROC curve plot to see).



# Results

TBU


# Sources of inspiration
Some papers used to flesh out this project:
- [Measuring Photoplethysmogram-Based Stress-Induced Vascular Response Index to Assess Cognitive Load and Stress](https://dl.acm.org/doi/10.1145/2702123.2702399)
- [Fatigue Estimation Using Peak Features from PPG Signals](https://www.mdpi.com/2227-7390/11/16/3580)
- [An Overview of Heart Rate Variability Metrics and Norms](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5624990/)
- [Heart rate variability: Standards of measurement, physiological interpretation, and clinical use](https://academic.oup.com/eurheartj/article/17/3/354/485572)
- [Nonlinear Methods to Assess Changes in Heart Rate Variability in Type 2 Diabetic Patients](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4062368/)
  

