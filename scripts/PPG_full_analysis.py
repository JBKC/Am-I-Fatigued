'''
Doing feature extraction from time series PPG signals
To be called from "PPG_data.py"
'''

import numpy as np
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis
from scipy.integrate import simpson
import general_peak_filtering as general_peak_filtering
import HRV_analysis as hrv_stats

def signal_stats(params):
    '''
    High level time-domain stats across the whole signal
    :param: signal, fs
        Takes in time-series signal and sampling frequency input from PPG_data.py

    :return: signal_dict
        Dictionary containing:
        from this function:
            'Mean': mean of the signal
            'Stdev': standard deviation of the signal
        from hrv_analysis.py:
            'HR': heart rate
            'RMSSD': root mean squared of successive differences between heartbeats
            'SDRR': standard deviation of intervals
            'pNN50' % of differences in the signal that are greater than 50 milliseconds
            'Triangular Index'
            'SD1': standard deviation of points from the short diagonal in the Poincare plot
            'SD2': standard deviation of points from the long diagonal in the Poincare plot
            'SD Ratio': SD1 / SD2
            'ApEn': approximate entropy
            'SampEn': sample entropy
            'LF Peak Idx': index of the peak frequency in the low frequency range
            'LF Peak Freq': value of the peak frequency in the low frequency range
            'LF Power': power in the low frequency range
            'LF Rel Power': relative power in the low frequency range
            'HF Peak Idx': index of the peak frequency in the high frequency range
            'HF Peak Freq': value of the peak frequency in the high frequency range
            'HF Power': power in the high frequency range
            'HF Rel Power': relative power in the low frequency range
            'LFHF Ratio': LF Power / HF Power
    '''

    signal, fs = params

    signal_dict = {
        'Mean': np.mean(signal),
        'Stdev': np.std(signal),
    }

    # pull hrv stats from hrv_analysis file
    hrv_results = hrv_stats.main(params)

    # update dictionary
    for result in hrv_results:
        signal_dict.update(result)

    return signal_dict

def FFT(params):
    '''
    High level frequency-domain stats
    :param: signal, fs
        Takes in time-series signal and sampling frequency input from PPG_data.py

    :return: fft_stats
        Dictionary containing spectral centroid and spectral entropy
    '''

    def roi(array,frequency):
        '''
        Take region of interest, i.e. 0-5Hz
        :param array:
        :return: cropped array
        '''

        return np.squeeze(array[:int(len(signal)*frequency/fs)])

    signal, fs = params

    # take FFT and get the power of each frequency bucket
    Pxx = np.abs(np.fft.fft(signal)) ** 2 / len(signal)
    # get the frequencies
    f = np.fft.fftfreq(len(signal), 1 / fs)

    # take 0-5Hz range (region of interest for PPG)
    Pxx = roi(Pxx,5)
    f = roi(f,5)

    # get spectral centroid of 0-5Hz region
    spectral_centroid = np.sum(f * Pxx) / np.sum(Pxx)
    # normalize the power spectral density + get spectral entropy
    Pxx = Pxx / np.sum(Pxx)
    spectral_entropy = -np.sum(Pxx * np.log2(Pxx + np.finfo(float).eps))  # Add eps for numerical stability

    fft_stats = {
        'Spectral Centroid': spectral_centroid,
        'Spectral Entropy': spectral_entropy
    }

    return fft_stats

def get_cycles(params):
    '''
    Divides signal into individual peak to peak cycles by:
    1. finding all time domain peaks
    2. filtering down to just pulse (systolic) peaks
    3. using these to find the pulse onsets (start of cycle) and so divide up the signal

    :param: signal, fs
        Takes in time-series signal and sampling frequency input from PPG_data.py

    :return: list(zip(cycles_idx, cycles))
        A zipped list of the global cycle indices and their corresponding y values
    '''

    def find_pulse_onset(signal, peaks, window_size=20):
        '''
        Gets the location of the pulse onset (start of the cycle) by finding the local minima before the peak
        Uses a recursive window

        :param signal: the time series signal
        :param peaks: indices of the location of the signal pulse peaks
        :param window_size: size of windows
        :return local_minima: location of pulse onset
        '''

        local_minima = []

        for peak in peaks:
            # set current peak to local minimum
            local_min = peak
            while True:
                # create window behind current peak to find the min value (onset)
                window_start = max(local_min - window_size, 0)
                window = signal[window_start:local_min+1]
                if len(window) == 0:
                    break  # no more window to check

                # find the index of the smallest value in the window
                min_idx = np.argmin(window) + window_start

                if min_idx == local_min:
                    break  # stop if the current position is the smallest in the window

                local_min = min_idx  # update local_min to the new minimum position

            local_minima.append(local_min)
        local_minima = sorted(set(local_minima))            # remove duplicates

        return local_minima

    def divide_cycles():
        '''
        Split signal into cycles, determined by pulse onset

        :return: list(zip(cycles_idx, cycles))
            A zipped list of all global cycle indices and their corresponding y values
        '''

        cycles = []
        cycles_idx = []

        for i in range(1, len(pulse_onsets) - 1):       # ignore first and last peaks
            start = pulse_onsets[i] +1                  # first pulse onset, +1 index
            end = pulse_onsets[i + 1]                   # second pulse onset
            cycle = signal[start:end]
            cycles.append(cycle + abs(cycle[0]))        # normalise cycle by setting each occurence of pulse onset to 0 on y axis
            cycles_idx.append(list(range(start,end)))

        return list(zip(cycles_idx, cycles))

    signal, fs = params

    # find all peaks
    sys_peaks = find_peaks(signal)[0]

    # filter the peaks to only contain pulse peaks, using general_peak_filtering.py
    sys_peaks = sys_peaks[signal[sys_peaks] > 0.95 * np.mean(signal[sys_peaks])]
    sys_peaks = general_peak_filtering.main(signal,sys_peaks,fs)

    # find pulse onsets - i.e. local minima before each systolic peak
    pulse_onsets = find_pulse_onset(signal, sys_peaks)

    return divide_cycles()

def cycle_stats(cycles):
    '''
    Performs analysis on a single peak to peak PPG cycle
    :param: cycles:
        Takes in indices and y values of each cycle from get_cycles function

    :return: cycle_dict
        Dictionary of all cycle stats for a signal (i.e. number of values for each statistic = number of cycles in signal)
        Containing:
            - 'Skew Cycle': skewness of the cycle's waveform
            - 'Kurt Cycle': kurtosis of the cycle's waveform
            - 'Area Cycle': total area under a cycle's waveform
            - 'SVRI': Stress induced vascular index
            - 'R Index': Fatigue index (ratio between systolic and diastolic peak heights)
            - 'TPA': Total peripheral resistance
            - 'Peak Index': local index of systolic peak for each cycle (i.e. what number sample it is)
            - 'Cycle Length': total length of cycle (measured in number of sampled datapoints, not in time)

    '''

    def error_handling(func,data):
        try:
            return abs(func(data))
        except:
            return 0

    def find_di(cycle,sys_peak_idx):
        '''
        Find the diastolic peak and dicrotic notch using rate of change of gradient
        :param cycle, sys_peak_idx:
            Takes in cycle data and indices of the systolic peaks

        :return:
            Indices of the diastolic peak and dicrotic notch

        '''

        cycle = cycle[1][sys_peak_idx:]             # take portion of cycle after peak
        grads = np.gradient(cycle)                  # gradient at each point
        acc = np.gradient(grads)                    # rate of change of gradient

        notch_idx = np.argmax(acc)                  # dicrotic notch == location where 2nd deriv of gradient is highest
        peak_idx = np.argmin(acc[notch_idx:])       # dicrotic peak == location where 2nd deriv is lowest between systolic peak and notch

        return peak_idx+notch_idx+sys_peak_idx, notch_idx+sys_peak_idx      # convert to cycle-level indices

    # get systolic peaks + total cycle length
    sys_peaks = [max(cycle[1]) for cycle in cycles]
    sys_peak_idx = [np.argmax(cycle[1]) for cycle in cycles]            # local indices within each cycle
    cycle_length = [len(cycle[0]) for cycle in cycles]

    # get distolic peaks + notches
    di_stats = [find_di(cycle, sys_idx) for cycle, sys_idx in zip(cycles, sys_peak_idx)]
    di_peak_idx, di_notch_idx = zip(*di_stats)
    di_peaks = [cycle[1][idx] if not np.isnan(idx) else np.nan for cycle, idx in zip(cycles, di_peak_idx)]

    # define the areas under the signal before / after the systolic peak using simpson's rule
    pre_sys_area = [error_handling(lambda x: simpson(cycle[1][:idx + 1]), cycle[1]) for cycle, idx in
                    zip(cycles, sys_peak_idx)]
    post_sys_area = [error_handling(lambda x: simpson(cycle[1][idx:]), cycle[1]) for cycle, idx in
                     zip(cycles, sys_peak_idx)]
    # define the areas under the signal before / after the dicrotic notch
    pre_di_area = [error_handling(lambda x: simpson(cycle[1][:idx + 1]), cycle[1]) for cycle, idx in
                   zip(cycles, di_notch_idx)]
    post_di_area = [error_handling(lambda x: simpson(cycle[1][idx:]), cycle[1]) for cycle, idx in
                    zip(cycles, di_notch_idx)]

    # get skewness, kurtosis and total area of each cycle
    skew_cycle = [error_handling(skew, cycle[1]) for cycle in cycles]
    kurt_cycle = [error_handling(lambda x: kurtosis(x, fisher=False), cycle[1]) for cycle in cycles]
    area_cycle = [error_handling(simpson, cycle[1]) for cycle in cycles]

    # get advanced PPG stats:
    # SVRI (stress induced vascular index)
    svri = [np.nan if post_sys == 0 else pre_sys / post_sys if np.isfinite(pre_sys / post_sys) else np.nan
            for pre_sys, post_sys in zip(pre_sys_area, post_sys_area)]
    # fatigue index
    r_idx = [np.nan if sys_peak == 0 else di_peak / sys_peak if np.isfinite(di_peak / sys_peak) else np.nan
             for sys_peak, di_peak in zip(sys_peaks, di_peaks)]
    # TPA (total peripheral resistance)
    tpa = [np.nan if post_di == 0 else pre_di / post_di if np.isfinite(pre_di / post_di) else np.nan
           for pre_di, post_di in zip(pre_di_area, post_di_area)]

    # store values in dictionary
    cycle_dict = {
        'Skew Cycle': skew_cycle,
        'Kurt Cycle': kurt_cycle,
        'Area Cycle': area_cycle,
        'SVRI': svri,
        'R Index': r_idx,
        'TPA': tpa,
        'Peak Index': sys_peak_idx,
        'Cycle Length': cycle_length
    }

    return cycle_dict

def cycle_summarize(dict):
    '''
    Turns individual cycle stats into a series of single summary values expressing the distribution of the cycle data across a full signal

    :param: dict
        Dictionary of stats from cycle_stats function (containing "number of cycles * number of keys in dictionary" pieces of data)
    :return:
        Dictionary of summary stats (containing 1 number per signal)
    '''

    result = {}

    for key, value in dict.items():
        result[key] = {
            'Mean': np.nanmean(value),
            'Median': np.nanmedian(value),
            'Range': np.nanmax(value) - np.nanmin(value),
            'StdDev': np.nanstd(value),
            'IQR': np.percentile(value, 75) - np.percentile(value, 25),
            'Skewness': skew(value, nan_policy='omit'),
            'Kurtosis': kurtosis(value, nan_policy='omit')
        }

    return result

################################################################################################

def main(params):
    '''
    Main function
    :param: signal, fs
        Takes in time-series signal and sampling frequency input from PPG_data.py

    :return: signal_dict, fft_dict, cycle_dict
        3 dictionaries containing time domain, frequency domain, and individual cycle stats

    '''
    # extract features from PPG signal
    signal_dict = signal_stats(params)
    fft_dict = FFT(params)
    cycle_dict = cycle_summarize(cycle_stats(get_cycles(params)))

    return signal_dict, fft_dict, cycle_dict

if __name__ == '__main__':
    pass
