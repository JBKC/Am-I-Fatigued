'''
Performing HRV analysis on time series PPG data
To be called from "PPG_data.py"
'''

import numpy as np
from scipy.signal import welch, find_peaks
import general_peak_filtering as general_peak_filtering
import EntropyHub as EH
from scipy.optimize import minimize
from spectrum import pburg

def peaks_stats(signal,fs):
    '''
    Extract peak locations, intervals and interval differences
    :param signal: time series signal
    :param fs: sampling frequency
    :return: peak locations, peak intervals, difference in length between successive intervals
    '''
    # returns RR / sys peaks + related stats
    peaks, _ = find_peaks(signal)                               # scipy function (_ value returns a dictionary of properties)
    peaks = general_peak_filtering.main(signal, peaks, fs)      # refine peak selection
    intervals = np.diff(peaks)                                  # absolute intervals between peaks
    diffs = abs(np.diff(intervals))                             # difference in successive intervals
    return peaks, intervals, diffs

def hrv_time(signal,fs):
    '''
    Get HRV time series features
    :param signal: raw signal values
    :param fs: sampling frequency

    :return: time_stats
        Dictionary containing RMSSD, SDRR, pNN50, triangular_index, TINN
    '''

    # get peaks, intervals and differences
    rr_peaks, intervals, diffs = peaks_stats(signal,fs)

    # get basic heartrate
    secs = len(signal) / fs                 # total time in seconds
    HR = len(rr_peaks)*60 / secs            # mean HR

    # time domain HRV features
    RMSSD = np.sqrt(np.mean(diffs ** 2))                                            # root mean square of successive differences
    SDRR = np.std(intervals)                                                        # standard deviation of intervals
    pNN50 = len([diff for diff in diffs if diff > (fs / 20)]) / len(diffs)          # % of differences > 50ms

    # geometric stats from histogram
    bin_width = fs * 7.8125 / 1000          # 7.8125 is the standard
    histogram, bin_edges = np.histogram(intervals, bins=int((max(intervals) - min(intervals)) / bin_width))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2          # clever formula - deep it

    N = len(intervals)              # Total number of NN intervals
    H = max(histogram)              # Maximum height of the histogram
    triangular_index = N / H        # HRV triangular index

    time_stats = {
        "HR": HR,
        "RMSSD": RMSSD,
        "SDRR": SDRR,
        "pNN50": pNN50,
        "Triangular Index": triangular_index
    }

    # find TINN (triangular interpolation) using an optimisation function
    # note for a low sampling rate or short signal, this stat should be ignored
    def triangular_func(x, left, peak, right):
        # optimally finds the left and right points of the triangle (selects bin that is furthest from peak either side)
        func = np.minimum((x - left) / (peak - left), (right - x) / (right - peak))
        # return an array that is triangular function (y values) for the given bin centers (x values)
        return np.maximum(0, func)

    def error_function(params):
        left, peak, right = params
        # set x as the bin centers
        triangle_approx = triangular_func(bin_centers, left, peak, right)
        # scale the triangular approximation to match the histogram
        triangle_approx_scaled = triangle_approx * max(histogram) / max(triangle_approx)
        # return mean square error
        return np.sum((histogram - triangle_approx_scaled) ** 2)

        # initial guess for the parameters (left, peak, right)
        initial_guess = [min(bin_centers), bin_centers[np.argmax(histogram)], max(bin_centers)]

        # perform optimization (scipy package)
        result = minimize(
            error_function,
            initial_guess,
            # set reasonable bounds for each of left, peak, right
            bounds=[(min(bin_centers), bin_centers[np.argmax(histogram)]),
                    (bin_centers[np.argmax(histogram)] - bin_width, bin_centers[np.argmax(histogram)] + bin_width),
                    (bin_centers[np.argmax(histogram)], max(bin_centers))]
        )

        # Extract the optimal parameters (.x picks out the part of relevance from the scipy result array)
        left_opt, peak_opt, right_opt = result.x
        # Calculate the TINN as the width of the optimal triangle
        TINN = right_opt - left_opt

        time_stats['TINN'] = TINN

    return time_stats

def hrv_nonlinear(signal,fs):
    '''
    Get the non-linear HRV features
    :param signal: time series signal
    :param fs: sampling frequency

    :return:
        Dictionary containing SD1, SD2, SD Ratio, ApEn, SampEn
    '''

    rr_peaks, intervals, diffs = peaks_stats(signal,fs)

    # poincare plots
    intervals = intervals - np.mean(intervals)      # normalise intervals
    rr1 = intervals[:-1]
    rr2 = intervals[1:]

    # extra forward shifts to compare - optional
    rr3 = intervals[2:].tolist()
    rr4 = intervals[3:].tolist()
    rr5 = intervals[4:].tolist()

    # SD1 & SD2 (rr1 vs. rr2 only)
    sd1 = np.std(rr1-rr2) / np.sqrt(2)                                  # stdev of the points rotated 45deg onto axis
    sd2 = np.sqrt(2*np.std(rr1)**2 - 0.5*np.std(rr1-rr2)**2)

    non_lin_stats = {
        "SD1": sd1,
        "SD2": sd2,
        "SD Ratio": sd1/sd2
    }

    # calculate approximate entropy and sample entropy
    m = 2
    r = 0.15

    apen, _ = EH.ApEn(signal, m=m, tau=1, r=r, Logx=np.exp(1))  # tau = time delay, logx = log base to use
    sampen, _, _ = EH.SampEn(signal, m=m, tau=1, r=r, Logx=np.exp(1))

    # save only the last item in the result (see EntropyHub documentation)
    apen = apen[-1]
    sampen = sampen[-1]

    non_lin_stats['ApEn'] = apen
    non_lin_stats['SampEn'] = sampen

    return non_lin_stats

def hrv_frequency(signal,fs):
    '''
    Calculate the frequency stats using FFT
    :param signal: time series signal
    :param fs: sampling frequency

    :return:
    '''

    # dictionary for frequency bands of interest
    freq_bands = {'LF': (0.04, 0.15),           # low frequency
                  'HF': (0.15, 0.4)}            # high frequency

    # get power spectrum from FFT & normalise
    Pxx = np.abs(np.fft.fft(signal))**2 / len(signal)
    f = np.fft.fftfreq(len(signal), 1 / fs)             # frequency axis

    freq_stats = {}
    total_power = 0

    # extract stats from frequency domain
    for band, (low, high) in freq_bands.items():
        # Find the GLOBAL indices corresponding to the frequency band
        band_idx = np.where((f >= low) & (f <= high))[0]
        peak_idx = band_idx[np.argmax(Pxx[band_idx])]              # for plotting
        peak = f[peak_idx]                                         # peak frequency
        abs_power = np.sum(Pxx[peak_idx])
        total_power += abs_power
        freq_stats[band] = {'Peak Idx': peak_idx, 'Peak Freq': peak, 'Power': abs_power}

    # get relative power + add to dictionary
    for band in freq_stats:
        rel_power = freq_stats[band]['Power'] / total_power  # Relative power
        freq_stats[band]['Rel Power'] = rel_power

    lfhf_ratio = freq_stats['LF']['Power'] / freq_stats['HF']['Power']
    freq_stats['LFHF Ratio'] = lfhf_ratio

    return freq_stats

################################################################################################

def main(params):
    '''
    Main function
    :param: signal, fs
        Takes in time-series signal and sampling frequency input from PPG_data.py

    :return: signal_dict, fft_dict, cycle_dict
        3 dictionaries containing time domain, non-linear, and frequency HRV stats

    '''
    # extract features from PPG signal
    signal, fs = params
    time_stats = hrv_time(signal, fs)
    non_lin_stats = hrv_nonlinear(signal, fs)
    freq_stats = hrv_frequency(signal, fs)

    return time_stats, non_lin_stats, freq_stats

if __name__ == '__main__':
    pass