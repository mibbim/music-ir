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
    ListDictCorpus

    A subclass of Corpus for storing and searching for audio fingerprints in a dictionary of list
    data structure.

    Attributes:
    corpus (ListDict[Hash, List[Tuple[int, SongID]]]): The hash map of audio fingerprints and their
    corresponding timestamps and song IDs.
    song_ids (List[SongID]): List of song IDs.
    song_paths (Dict[SongID, PosixPath]): Dictionary mapping song IDs to file paths.

    """

    def __init__(self,
                 fanout_window: int = 10,
                 spec_window_size: int = 4086,
                 spec_window_overlap_ratio: float = 0.5):
        """
        Initialize the `ListDictCorpus` object.

        :param fanout_window: (optional) the size of the time window of each peak constellation.
        :type fanout_window: int, optional
        :param spec_window_size: (optional) the window size to use in the spectrogram creation.
        :type spec_window_size: int, optional
        :param spec_window_overlap_ratio: (optional) the weight ratio to use in the hashing process.
        :type spec_window_overlap_ratio: float, optional
        """
        super().__init__(fanout_window, spec_window_size, spec_window_overlap_ratio)
        self.corpus: ListDict[Hash, List[Tuple[int, SongID]]] = ListDict()
        self.song_ids = []
        self.song_paths: Dict[SongID, PosixPath] = {}

    def add_song(self, filepath: PosixPath):
        # global TOT_HASHES
        song_id = len(self.song_ids)
        self.song_ids.append(song_id)
        self.song_paths[song_id] = filepath
        signal, sample_rate = load_mp3(filepath)
        signal = self._preprocess_signal(signal)  # stereo to mono conversion
        hashes = self._get_hashes(signal, sample_rate, song_id)
        # TOT_HASHES += len(hashes)
        self.corpus.update(hashes)
        return

    def _get_matches(self, hashes: ListDict) -> List[Tuple[int, SongID]]:
        """

        :param hashes:
        :return: List of tuples (time_difference, song_id)
        """
        deltas = []
        for hash_tuple, times in hashes.items():
            for (t, _) in times:
                if matches := self.corpus.get(hash_tuple, None):
                    deltas.extend([(match[0] - t, match[1]) for match in matches])
        return deltas

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

    def merge(self, other_corpus):
        self.corpus.update(other_corpus.corpus)
        present_tracks = len(self.song_ids)
        self.song_ids.extend([si + present_tracks for si in other_corpus.song_ids])
        self.song_paths.update(
            {si + present_tracks: p for si, p in other_corpus.song_paths.items()})


if __name__ == "__main__":
    import pickle
    from pathlib import Path
    import os
    from Corpus import find_song

    # corpus_file = Path("corpus_0_ld.pickle")
    # all_corpora = ListDictCorpus()
    # for i in tqdm(range(31), desc="Corpora", position=0):
    #     corpus_file = Path(f"corpus_{i}_ld.pickle")
    #
    #     try:
    #         with open(corpus_file, "rb") as f:
    #             corpus = pickle.load(f)
    #
    #     except FileNotFoundError:
    #         print(f"Corpus {corpus_file} not found, creating new one...")
    #         corpus = ListDictCorpus()
    #         folder_path = Path(f'data/{i:03d}')
    #
    #         for path in tqdm(os.listdir(folder_path), leave=False, desc=f"Corpus {i}", position=1):
    #             corpus.add_song(folder_path / path)
    #         # print(TOT_HASHES, sum([len(v) for v in corpus.corpus.values()]))
    #         with open(corpus_file, "wb") as f:
    #             pickle.dump(corpus, f)
    #
    #     all_corpora.merge(corpus)

    # with open("first_30_corpora.pickle", "wb") as f:
    #     pickle.dump(all_corpora, f)

    with open("first_30_corpora.pickle", "rb") as f:
        all_corpora = pickle.load(f)


    def select_random_song() -> Path:
        import random
        folder = random.choice(os.listdir("data"))
        if folder.endswith(".zip"):
            return select_random_song()
        song = random.choice(os.listdir(f"data/{folder}"))
        return Path(f"data/{folder}/{song}")


    for i in range(100):
        random_song = select_random_song()
        recognized, _, _ = find_song(random_song, all_corpora, verbose=True)
        if random_song != recognized:
            print("Wrong song recognized!")
