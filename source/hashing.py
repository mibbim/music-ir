from collections import namedtuple
from typing import List, Dict, Tuple, Any

import numpy as np

from source.ListDict import ListDict
from source.fingerprint import get_peak_coordinates

Anchorpoint_data = namedtuple('Anchorpoint_data', ["constallation", "frequencies", "time"])
Star = namedtuple('Star', ["frequency", "time"])


def select_fanout_windows_peaks(peaks: np.ndarray,
                                fanout_window: int = 10) -> List[Anchorpoint_data]:
    """Select fanout peaks.

    Args:
        peaks (np.ndarray): 2D array of peaks.
        fanout_window (int): Fanout.
    Returns:

    """
    peaks_freq, peaks_time = get_peak_coordinates(peaks).T
    sorting_mask = np.argsort(peaks_time)
    sorted_freq = peaks_freq[sorting_mask]
    sorted_time = peaks_time[sorting_mask]
    unique_sorted_time = np.unique(sorted_time)
    fanout_windows_data: List[Anchorpoint_data] = [None] * unique_sorted_time.shape[0]

    for i, t1 in enumerate(unique_sorted_time):  # using set to remove duplicates
        window = np.logical_and(sorted_time > t1, sorted_time < t1 + fanout_window)
        f1 = sorted_freq[sorted_time == t1]  # getting all the peak frequencies
        constellation = {Star(f, t) for f, t, in zip(sorted_freq[window], sorted_time[window])}
        fanout_windows_data[i] = Anchorpoint_data(constellation, f1, t1)
    return fanout_windows_data


# def select_fanout_windows_peaks_opt(peaks: np.ndarray, fanout_window: int = 10) -> List[
#     Anchorpoint_data]:
#     peaks_freq, peaks_time = get_peak_coordinates(peaks).T
#     sorting_mask = np.argsort(peaks_time)
#     sorted_freq = peaks_freq[sorting_mask]
#     sorted_time = peaks_time[sorting_mask]
#     unique_sorted_time = set(sorted_time)
#     fanout_windows_data = [None] * len(unique_sorted_time)
#
#     for i, t1 in enumerate(unique_sorted_time):
#         window = (sorted_time > t1) & (sorted_time < t1 + fanout_window)
#         f1 = sorted_freq[sorted_time == t1]
#         constellation = {Star(f, t) for f, t, in zip(sorted_freq[window], sorted_time[window])}
#         fanout_windows_data[i] = Anchorpoint_data(constellation, f1, t1)
#     return fanout_windows_data


def hash_fanout_windows(constellations_data: List[Anchorpoint_data], song_id) \
        -> Dict[Tuple[int, int, int], Tuple[int, Any]]:
    """
    Hash fanout windows.
    """
    fanout_windows_hash = {}
    for constellation, f1, t1 in constellations_data:
        delta_ts = {star.time - t1 for star in constellation}
        freqs = {star.frequency for star in constellation}
        fanout_windows_hash.update({(f, f2, delta_t): (t1, song_id)
                                    for f2, delta_t in zip(freqs, delta_ts)
                                    for f in f1})
    return fanout_windows_hash


def hash_fanout_windows_listdict(constellations_data: List[Anchorpoint_data], song_id) \
        -> ListDict[Tuple[int, int, int], List[Tuple[int, Any]]]:
    """
    Hash fanout windows.
    """
    fanout_windows_hash = ListDict()
    for constellation, f1, t1 in constellations_data:
        delta_ts = {star.time - t1 for star in constellation}
        freqs = {star.frequency for star in constellation}
        fanout_windows_hash.update({(f, f2, delta_t): [(t1, song_id)]
                                    for f2, delta_t in zip(freqs, delta_ts)
                                    for f in f1})
    return fanout_windows_hash
