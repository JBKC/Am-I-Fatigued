'''
Used to upload PPG data from a .txt file and extract statistics
Output is a CSV containing the dataframe of results
Also the option to plot data from an existing CSV file
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import PPG_full_analysis as full_stats
import os
import emd
import seaborn as sns

def get_signal(file):
    '''
    Extract time series signal from defined file

    :param: file
        filepath of .txt file containing signal data
    :return: signal, fs
        array of the signal y values and the sampling frequency
    '''

    # note - adjust this function to fit your own data

    with open(file, 'r', encoding='latin-1') as f:
        lines = f.readlines()
        # total_time = int(lines[1].strip()) / 1000
        signal = lines[2:-1]
        signal = [int(line.strip()) for line in signal]         # Convert lines from string to float
        signal = np.array(signal)
        # fs = len(signal) / total_time                         # sampling frequency (manually extracted)
        fs = 97.3                                               # hardcoded for the sample data
    return signal, fs

def normalise_signal(signal):
    '''
    Prepare the signal by normalising
    :param signal: time series signal
    :return: normalised signal
    '''

    mean_val = np.mean(signal)
    std_dev = np.std(signal)
    signal = (signal - mean_val) / std_dev

    return signal

def ensemble_sift(signal):
    '''
    Use Empirical Model Decomposition to clean signal of high-frequency noise
    :param signal: time series signal
    :return imf: range of IMFs extracted from signal
    :return imf_hmnc: range of IMFs with final IMF combined
    '''

    # parameters for emd.sift.ensemble_sift()
    sd_thresh = 0.05
    max_imfs = 6
    nensembles = 20
    nprocesses = 6
    ensemble_noise = 0.5

    imf_opts = {'sd_thresh': sd_thresh}
    imf = emd.sift.ensemble_sift(signal, max_imfs=max_imfs, nensembles=nensembles, nprocesses=nprocesses,
                                 ensemble_noise=ensemble_noise, imf_opts=imf_opts)

    # sum IMFs 3 through 6 (ie ignoring the high frequency noise of IMFs 1&2)
    key_imf = imf[:, 2:6].sum(axis=1)

    return key_imf

def flatten_dict(d, parent_key='', sep=' '):
    '''
    General function for flattening nested dictionaries into one level
    :param d: dictionary
    :param parent_key: set to ''
    :param sep: set to ''
    :return dict(items): flattened dictionary
    '''

    items = []

    # iterate through key-value pairs
    for k, v in d.items():
        # create new key based on dictionary key
        new_key = f"{parent_key}{sep}{k}" if parent_key else k  # iterate over key-value pairs
        # check if the value is a dictionary itself (nested dict)
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def run_new_full(folder):
    '''
    Main function for running a fresh analysis on a set of signals
    Searches and categorises each signal file based on title

    :param folder: Folder containing the .txt files of the signals
    :param return: dataframe with all statistics (
    '''

    df = pd.DataFrame()

    # search through folder for fatigued vs. non-fatigued - adjust search parameters as necessary
    for filename in os.listdir(folder):
        if filename == '.DS_Store':
            continue  # skip processing .DS_Store file
        if 'pre-training' in filename or 'standard' in filename or 'no training' in filename:
            state = 0  # 1 = fatigued, 0 = standard
        elif 'post-training' in filename:
            state = 1
        else:
            continue

        filepath = os.path.join(folder, filename)
        if os.path.isfile(filepath):  # check if it's a file (not a subdirectory)
            filename = os.path.splitext(filename)[0]  # get rid of .txt
            print(filename)

            # pull signal data in
            signal, fs = get_signal(filepath)
            signal = ensemble_sift(normalise_signal(signal))

            # extract the signal features + save into dict
            params = [signal, fs]

            # call PPG_full_analysis.py to pull in 3 dictionaries of stats
            signal_stats, fft_stats, cycle_stats = full_stats.main(params)

            # flatten + combine dictionaries
            combined_dict = {**flatten_dict(signal_stats), **flatten_dict(fft_stats), **flatten_dict(cycle_stats)}
            combined_dict = {'Filename': filename, 'State': state, **combined_dict}

            # append to main dataframe
            sub_df = pd.DataFrame([combined_dict])
            df = pd.concat([df, sub_df], ignore_index=True)

    # save locally to CSV
    df.to_csv('hrv_ppg.csv', index=False)
    print(df)
    return df

def plot_results(x_data, y_data):
    '''
    Plot scatter of fatigued vs. non-fatigued state (1 datapoint per signal)
    :param: x_data, y_data
        Statistics to be plotted on x and y axes to explore relationship and notice any clustering patterns
    '''

    # read saved CSV
    df = pd.read_csv('hrv_ppg.csv')

    # Define the unique states and a color palette
    unique_states = df['State'].unique()
    sns.set(style='dark', context='talk')
    palette = sns.color_palette("hsv", len(unique_states))

    # Initialize the plot
    plt.figure(figsize=(12, 8))

    # Iterate through each unique state and plot
    for i, state in enumerate(unique_states):
        state_data = df[df['State'] == state]
        plt.scatter(state_data[x_data], state_data[y_data], color=palette[i], label=state)

    # Add labels and title
    plt.xlabel(x_data)
    plt.ylabel(y_data)
    plt.legend(title='State')
    plt.show()

    return

def main():
    '''
    Switch between running new analysis and plotting existing data from CSV
    '''

    function_call = input("Options: run new, plot results\n"
                       "Type option here: ").strip().lower()

    # call the corresponding function
    if function_call == "run new":
        script_dir = os.path.dirname(os.path.abspath(__file__))
        folder = os.path.join(script_dir, '..', 'data')
        run_new_full(folder)
        
    if function_call == "plot results":
        x_data = input("X data (must be typed exactly as in dataframe): ").strip()
        y_data = input("Y data (must be typed exactly as in dataframe): ").strip()
        plot_results(x_data, y_data)
        
    else:
        print(f"Option '{function_call}' is not supported.")

    return

# switch between modes
if __name__ == '__main__':
    main()
