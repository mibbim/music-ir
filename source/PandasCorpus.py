import os
import pickle
from pathlib import Path, PosixPath
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from Corpus import Corpus, Hash, SongID, find_song
from load_utils import load_mp3


class PandasCorpus(Corpus):
    """
    A corpus that uses a Pandas DataFrame to store the hashes.
    Note: This implementation is not optimized for speed.

    """

    def __init__(self,
                 fanout_window: int = 10,
                 spec_window_size: int = 4086,
                 spec_window_overlap_ratio: float = 0.5):
        super().__init__(fanout_window, spec_window_size, spec_window_overlap_ratio)
        self._corpus = pd.DataFrame(columns=['song_id', 'time'])
        raise NotImplementedError

    @property
    def song_ids(self) -> List[SongID]:
        return self._corpus['song_id']

    @property
    def corpus(self) -> pd.DataFrame:
        return self._corpus

    @staticmethod
    def dict_to_corpus(hashes: Dict[Hash, Tuple[int, SongID]]) -> pd.DataFrame:
        return pd.DataFrame.from_dict(hashes,
                                      orient='index',
                                      columns=['time', 'song_id'])

    def add_song(self, song_path: Path):
        song_id = self._get_song_id(song_path)
        signal, sample_rate = load_mp3(song_path)
        signal = self._preprocess_signal(signal)
        hashes = self._get_hashes(signal, sample_rate, song_id)
        self._corpus = pd.concat(
            (self._corpus, self.dict_to_corpus(hashes)))

    def _get_matches(self, hashes: Dict[Hash, Tuple[int, SongID]]) -> pd.DataFrame:
        """

        :param hashes:
        :return: Df of tuples (time_difference, song_id)
        """
        matches = self._corpus.loc[
            self._corpus.index.intersection(hashes.keys())]  # Returns a copy, not a view
        matches.loc[:, "time"] -= [hashes[i][0] for i in matches.index]
        matches.columns = ['song_id', 'time_diff', ]
        return matches  # deltas

    @staticmethod
    def _get_song_id(song_path: Path):
        return song_path.stem

    def recognize(self, input_signal: np.ndarray,
                  sample_rate: int,
                  n_matches_threshold: int = 10):
        input_signal = self._preprocess_signal(input_signal)
        input_hashes = self._get_hashes(input_signal, sample_rate)
        matching_hashes = self._get_matches(input_hashes)
        song_id, n_matches, alignment = self._retrieve_best_guess(matching_hashes, )
        if n_matches < n_matches_threshold:
            return None, 0, -1
        matching_confidence = n_matches / len(input_hashes)
        return song_id, matching_confidence, alignment

    def _retrieve_best_guess(self, matching_hashes: pd.DataFrame, check_neighbour=False):
        counts = matching_hashes.time_diff.value_counts(sort=True)
        if check_neighbour:
            raise NotImplementedError
        best_alignment_delta, champion_count = counts.index[0], counts.iloc[0]
        best_songs = matching_hashes.loc[
            matching_hashes.time_diff == best_alignment_delta, 'song_id'
        ].value_counts(sort=True)
        best_song_id, n_matches = best_songs.index[0], best_songs.iloc[0]
        return best_song_id, n_matches, best_alignment_delta


if __name__ == '__main__':
    corpus = PandasCorpus()
    folder_path = Path('../data/000')

    corpus_file = Path("corpus_pd_0_5.pickle")

    try:
        with open(corpus_file, "rb") as f:
            corpus = pickle.load(f)
    except FileNotFoundError:
        print("Corpus not found, creating new one...")
        for track_path in tqdm(os.listdir(folder_path)):
            corpus.add_song(folder_path / track_path)
        with open(corpus_file, "wb") as f:
            pickle.dump(corpus, f)

    for i in range(20):
        find_song(PosixPath("../data/000/000190.mp3"), corpus)
    print()
