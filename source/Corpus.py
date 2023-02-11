from abc import abstractmethod
from pathlib import PosixPath, Path
from typing import Dict, Tuple

import librosa
import numpy as np
from matplotlib import mlab

from fingerprint import detect_peaks
from hashing import hash_fanout_windows, select_fanout_windows_peaks
from load_utils import load_mp3

SongID = int | None | str
Hash = Tuple[int, int, int]


def handle_stereo_signal(signal: np.ndarray) -> np.ndarray:
    if len(signal.shape) == 2:
        return signal.sum(axis=1)
    return signal


class Corpus:
    def __init__(self,
                 fanout_window: int = 10,
                 spec_window_size: int = 4086,
                 spec_window_overlap_ratio: float = 0.5
                 ):
        self.spec_window_size = spec_window_size
        self.spec_window_overlap_ratio = spec_window_overlap_ratio
        self.fanout_window = fanout_window

    @abstractmethod
    def add_song(self, filepath: PosixPath):
        raise NotImplementedError

    @abstractmethod
    def recognize(self,
                  input_signal: np.ndarray,
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
            NFFT=self.spec_window_size,
            Fs=sr,
            window=mlab.window_hanning,
            noverlap=int(self.spec_window_size * self.spec_window_overlap_ratio)
        )
        return librosa.power_to_db(spectrum, ref=np.max)

    def _get_hashes(self,
                    signal: np.ndarray,
                    sample_rate: int,
                    song_id: SongID = None) -> Dict[Hash, Tuple[int, SongID]]:
        db_spectrum = self._compute_db_spectrum(signal, sample_rate)
        peaks = detect_peaks(db_spectrum)
        fan_win_data = select_fanout_windows_peaks(peaks, fanout_window=self.fanout_window)
        return hash_fanout_windows(fan_win_data, song_id)


def find_song(path: Path, corpus: Corpus, seconds=3, verbose=False):
    signal, sr = load_mp3(path)
    start = np.random.randint(0, signal.shape[0] - sr * seconds)  # 200
    recognized = corpus.recognize(signal[start: start + sr * seconds], sr)
    if verbose:
        print(recognized)
    return recognized


if __name__ == "__main__":
    pass
