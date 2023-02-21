from abc import abstractmethod
from pathlib import PosixPath, Path
from typing import Dict, Tuple, List

import librosa
import numpy as np
from matplotlib import mlab

from source.fingerprint import detect_peaks
from source.hashing import hash_fanout_windows, select_fanout_windows_peaks
from source.load_utils import load_mp3

SongID = int | None | str
Hash = Tuple[int, int, int]


def handle_stereo_signal(stereo_signal: np.ndarray) -> np.ndarray:
    """
    Handles stereo signals by summing the two channels into a mono signal.
    :param stereo_signal: the input signal.
    :return: The mono signal.
    """
    if len(stereo_signal.shape) == 2:
        return stereo_signal.sum(axis=1)
    return stereo_signal


class Corpus:
    def __init__(self,
                 fanout_window: int = 10,
                 spec_window_size: int = 4086,
                 spec_window_overlap_ratio: float = 0.5
                 ):
        """
        Initialize the `DictCorpus` object.

        :param fanout_window: (optional) the size of the time window of each peak constellation.
        :type fanout_window: int, optional
        :param spec_window_size: (optional) the window size to use in the spectrogram creation.
        :type spec_window_size: int, optional
        :param spec_window_overlap_ratio: (optional) the overlap ratio to use in the hashing process.
        :type spec_window_overlap_ratio: float, optional
        """
        self.spec_window_size = spec_window_size
        self.spec_window_overlap_ratio = spec_window_overlap_ratio
        self.fanout_window = fanout_window

    @property
    def song_ids(self) -> List[SongID]:
        raise NotImplementedError

    @property
    def corpus(self) -> Dict[Hash, Tuple[int, SongID]]:
        raise NotImplementedError

    @abstractmethod
    def add_song(self, filepath: PosixPath):
        """
        Adds a song to the fingerprint database and indexes its hashes in the corpus.

        :param filepath: The filepath to the MP3 file to add to the database.
        :type filepath: PosixPath
        :return: None
        """
        raise NotImplementedError

    @abstractmethod
    def recognize(self,
                  input_signal: np.ndarray,
                  sample_rate: int,
                  n_matches_threshold: int = 10):
        """
        Recognize the song from an audio signal.

        Args:
            input_signal (array-like): The audio signal to be recognized.
            sample_rate (int): The sample rate of the audio signal.
            n_matches_threshold (int, optional): The minimum number of matches required for the
                                            recognition to be considered successful. Defaults to 10.
        """
        raise NotImplementedError

    @staticmethod
    def _preprocess_signal(signal: np.ndarray) -> np.ndarray:
        """
        Preprocess the signal to be used in the fingerprinting process.
        :param signal: the signal to preprocess.
        :type signal: np.ndarray
        :return: the preprocessed signal.
        :rtype: np.ndarray
        """
        signal = handle_stereo_signal(signal)  # Stereo to mono conversion
        return signal

    def _compute_db_spectrum(self,
                             signal: np.ndarray,
                             sr: int) -> np.ndarray:
        """
        Compute the spectrogram of the input audio signal.
        It uses the `mlab.specgram` function from `matplotlib` to compute it.

        :param signal: 1D numpy array representing the audio signal.
        :param sr: The sample rate of the audio signal.
        :return: Spectrogram of the input audio signal in dB scale.
        :type: np.ndarray
        """
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
        """
        Compute hashes for a given audio signal. In order to do that, it computes the spectrogram
         of the signal, detects peaks in it and then hashes the peaks and finally stores the
         hashes in a inverted index.

        Args:
            signal (np.ndarray): Audio signal
            sample_rate (int): Sample rate of the audio signal
            song_id (SongID, optional): Song identifier. Defaults to None.

        Returns:
            Dict[Hash, Tuple[int, SongID]]: A dictionary mapping hashes to tuples containing the frequency bin index and
            the song identifier.
        """
        db_spectrum = self._compute_db_spectrum(signal, sample_rate)
        peaks = detect_peaks(db_spectrum)
        fan_win_data = select_fanout_windows_peaks(peaks, fanout_window=self.fanout_window)
        return hash_fanout_windows(fan_win_data, song_id)

    def info(self):
        """
        Returns a formatted string with some statistics about the corpus.
        """
        lens = [len(l) for l in self.corpus.values()]
        result = f"""
    number of songs: {len(self.song_ids)}
    number of hashes: {len(self.corpus)}
    average anchor point per hash: {np.mean([len(l) for l in self.corpus.values()])}
    max anchor point per hash: {np.max([len(l) for l in self.corpus.values()])}
    min anchor point per hash: {np.min([len(l) for l in self.corpus.values()])}
"""
        return result


def find_song(path: Path, corpus: Corpus, seconds=3, verbose=False):
    """
    Testing function.
    Identify a song based on its audio content by comparing it to a reference corpus.
    To make the comparison more robust, it randomly selects a 3-second clip from the
    input audio file and compares it to the reference corpus.

    :param path: Path object representing the location of the audio file to be recognized.
    :param corpus: A reference corpus of audio tracks used to identify the input audio file.
    :param seconds: The length of the audio clip in seconds, to be used for recognition.
    :param verbose: If set to True, the function will print the recognition results.
    :return: A dictionary with information about the recognized song, such as its id and the recognition score.
    """
    signal, sr = load_mp3(path)
    start = np.random.randint(0, signal.shape[0] - sr * seconds)  # 200
    recognized = corpus.recognize(signal[start: start + sr * seconds], sr)
    if verbose:
        print(recognized)
    return recognized


if __name__ == "__main__":
    pass
