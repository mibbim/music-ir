import os
import pickle
from pathlib import Path, PosixPath
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from Corpus import Corpus, Hash, SongID, find_song
from load_utils import load_mp3


class PandasCorpus(Corpus):
    def __init__(self,
                 fanout_window: int = 10,
                 wsize: int = 4086,
                 wratio: float = 0.5):
        super().__init__(fanout_window, wsize, wratio)
        self.corpus = pd.DataFrame(columns=['song_id', 'time'])

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
        self.corpus = pd.concat(
            (self.corpus, self.dict_to_corpus(hashes)))

    def _get_matches(self, hashes: Dict[Hash, Tuple[int, SongID]]) -> pd.DataFrame:
        """

        :param hashes:
        :return: Df of tuples (time_difference, song_id)
        """
        matches = self.corpus.loc[
            self.corpus.index.intersection(hashes.keys())]  # Returns a copy, not a view
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
    folder_path = Path('data/000')

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
        find_song(PosixPath("data/000/000190.mp3"), corpus)
    print()
