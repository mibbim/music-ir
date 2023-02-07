from collections import Counter
from operator import itemgetter
from typing import Tuple, List, Dict

import numpy as np

from Corpus import Corpus, Hash, SongID
from ListDict import ListDict
from pathlib import PosixPath

from fingerprint import detect_peaks
from hashing import select_fanout_windows_peaks, hash_fanout_windows_listdict
from load_utils import load_mp3

TOT_HASHES = 0


class ListDictCorpus(Corpus):
    """
    This implementation uses a dictionary to store the hashes.
    It is however not working properly because of the high number of conflicts in hashes
    that result in hashes overwriting.
    In order to work properly, the dictionary should use as values a list of values
    instead of a single values.
    """

    def __init__(self,
                 fanout_window: int = 10,
                 wsize: int = 4086,
                 wratio: float = 0.5):
        super().__init__(fanout_window, wsize, wratio)
        self.corpus: ListDict[Hash, List[Tuple[int, SongID]]] = ListDict()
        self.song_ids = []
        self.song_paths: Dict[SongID, PosixPath] = {}

    def add_song(self, filepath: PosixPath):
        global TOT_HASHES
        song_id = len(self.song_ids)
        self.song_ids.append(song_id)
        self.song_paths[song_id] = filepath
        signal, sample_rate = load_mp3(filepath)
        signal = self._preprocess_signal(signal)  # stereo to mono conversion
        hashes = self._get_hashes(signal, sample_rate, song_id)
        TOT_HASHES += len(hashes)
        self.corpus.update(hashes)
        return

    def _get_matches(self, hashes: ListDict) -> List[Tuple[int, SongID]]:
        """

        :param hashes:
        :return: List of tuples (time_difference, song_id)
        """
        raise NotImplementedError
        # deltas = [((match[0] - t), match[1])
        #           for hash_tuple, (t, _) in hashes.items()
        #           if (match := self.corpus.get(hash_tuple, None))]
        # return deltas

    def _retrieve_best_guess(self, matching_hashes: List[Tuple[int, SongID]],
                             ) -> Tuple[SongID, int, int]:

        try:
            champion = Counter(map(itemgetter(0), matching_hashes)).most_common(1)[0]
        except IndexError:
            return None, 0, -1
        best_alignment_delta, _ = champion

        best_song_id, n_matches = Counter(
            [song_id for delta, song_id in matching_hashes if
             delta == best_alignment_delta]).most_common(1)[0]
        return best_song_id, n_matches, best_alignment_delta

    def _get_hashes(self,
                    signal: np.ndarray,
                    sample_rate: int,
                    song_id: SongID = None) -> ListDict[Hash, List[Tuple[int, SongID]]]:
        db_spectrum = self._compute_db_spectrum(signal, sample_rate)
        peaks = detect_peaks(db_spectrum)
        fan_win_data = select_fanout_windows_peaks(peaks, fanout_window=self.fanout_window)
        return hash_fanout_windows_listdict(fan_win_data, song_id)

    def recognize(self, input_signal, sample_rate,
                  n_matches_threshold: int = 10):
        input_signal = self._preprocess_signal(input_signal)
        input_hashes = self._get_hashes(input_signal, sample_rate, song_id=None)
        matching_hashes = self._get_matches(input_hashes)
        song_id, n_matches, alignment = self._retrieve_best_guess(matching_hashes)
        if n_matches < n_matches_threshold:
            return None, 0, -1
        matching_confidence = n_matches / len(input_hashes)
        return self.song_paths[song_id], matching_confidence, alignment


if __name__ == "__main__":
    import pickle
    from pathlib import Path
    import os
    from tqdm import tqdm
    from Corpus import find_song

    corpus_file = Path("corpus_0_ld.pickle")

    try:
        with open(corpus_file, "rb") as f:
            corpus = pickle.load(f)
    except FileNotFoundError:
        print("Corpus not found, creating new one...")
        corpus = ListDictCorpus()
        folder_path = Path('data/000')
        for path in tqdm(os.listdir(folder_path)):
            corpus.add_song(folder_path / path)
        print(TOT_HASHES, sum([len(v) for v in corpus.corpus.values()]))
        with open(corpus_file, "wb") as f:
            pickle.dump(corpus, f)

    for i in range(100):
        find_song(PosixPath("data/000/000190.mp3"), corpus, verbose=True)
