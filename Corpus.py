from abc import abstractmethod
from pathlib import PosixPath
from typing import Dict, Tuple

import librosa
import numpy as np
from matplotlib import mlab

from fingerprint import detect_peaks
from hashing import hash_fanout_windows, \
    select_fanout_windows_peaks_opt

SongID = int | None
Hash = Tuple[int, int, int]


def handle_stereo_signal(signal: np.ndarray) -> np.ndarray:
    if len(signal.shape) == 2:
        return signal.sum(axis=1)
    return signal


class Corpus:
    def __init__(self,
                 fanout_window: int = 10,
                 wsize: int = 4086,
                 wratio: float = 0.5
                 ):
        self.wsize = wsize
        self.wratio = wratio
        self.fanout_window = fanout_window

    @abstractmethod
    def add_song(self, filepath: PosixPath):
        raise NotImplementedError

    @abstractmethod
    def recognize(self, input_signal: np.ndarray,
                  sample_rate: int,
                  n_matches_threshold: int = 10):
        raise NotImplementedError

    @staticmethod
    def _preprocess_signal(signal: np.ndarray) -> np.ndarray:
        signal = handle_stereo_signal(signal)  # Stereo to mono conversion
        return signal

    def _compute_db_spectrum(self,
                             signal: np.ndarray,
                             sr: int) -> np.ndarray:
        spectrum, frequencies, time = mlab.specgram(
            signal,
            NFFT=self.wsize,
            Fs=sr,
            window=mlab.window_hanning,
            noverlap=int(self.wsize * self.wratio)
        )
        return librosa.power_to_db(spectrum, ref=np.max)

    def _get_hashes(self,
                    signal: np.ndarray,
                    sample_rate: int,
                    song_id: int | None) -> Dict[Hash, Tuple[int, SongID]]:
        db_spectrum = self._compute_db_spectrum(signal, sample_rate)
        peaks = detect_peaks(db_spectrum)
        fan_win_data = select_fanout_windows_peaks_opt(peaks, fanout_window=self.fanout_window)
        return hash_fanout_windows(fan_win_data, song_id)
