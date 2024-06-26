'''
Simple peak detection algorithm to find systolic peaks in PPG
Starts with scipy-found peaks and iteratively makes sure it's the systolic by windowing
'''

import numpy as np

def refine_peaks(signal,peaks,window_size):
    '''
    Function that refines the peaks from all peaks down to only the most prominent ones (systolic peaks)
    Uses a window to look backwards and forwards to make sure diastolic peaks / noise peaks aren't picked up in error
    :param signal: time domain function
    :param peaks: indices of peaks in signal
    :param window_size: size of search window in number of sample points
    :return: sorted(set(final_peaks)): final list of peaks
    '''

    # find the highest value in each window
    final_peaks = []

    for peak in peaks:
        # set current peak to local max
        local_max = peak
        while True:
            # create window around current peak to find the min value (onset)
            window_start = max(local_max - window_size, 0)  # ensure we don't go out of bounds
            window_end = min(local_max + window_size, len(signal))
            window = signal[window_start:window_end]
            if len(window) == 0:
                break  # no more window to check

            # Find the index of the smallest value in the window
            max_idx = np.argmax(window) + window_start

            if max_idx == local_max:
                break  # stop if the current position is the smallest in the window

            local_max = max_idx  # Update local_min to the new minimum position

        final_peaks.append(local_max)

    return sorted(set(final_peaks))            # remove duplicates


def main(signal,peaks,fs):
    '''
    Main function
    :param signal: time domain function
    :param peaks: indices of all peaks in signal
    :param fs: sampling frequency
    :return final_peaks: indices of peaks
    '''

    # set the likely maximum bpm that will be picked up by PPG
    max_bpm = 180
    # set window size (in number of sampled datapoints
    window_size = int(fs * 60 / max_bpm)
    # trigger iterative refinement
    final_peaks = refine_peaks(signal,peaks,window_size)

    return final_peaks


if __name__ == '__main__':
    main()






