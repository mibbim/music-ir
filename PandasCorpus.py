from pathlib import Path
from typing import Dict, Tuple, List

import pandas as pd
from tqdm import tqdm

from Corpus import Corpus, Hash, SongID
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

    def _get_matches(self, hashes: Dict) -> List[Tuple[int, SongID]]:
        """

        :param hashes:
        :return: List of tuples (time_difference, song_id)
        """
        deltas = []
        for hash_tuple, (t, _) in hashes.items():
            matches = self.corpus.loc[[hash_tuple]]
            for _, row in matches.iterrows():
                deltas.append(((row['time'] - t), row['song_id']))
        return deltas

    @staticmethod
    def _get_song_id(song_path: Path):
        return song_path.stem

    # def recognize(self, input_signal: np.ndarray,
    #               sample_rate: int,
    #               n_matches_threshold: int = 10):


if __name__ == '__main__':
    corpus = PandasCorpus()
    folder_path = Path('data/000')
    for song_path in tqdm(folder_path.glob('*.mp3')):
        corpus.add_song(song_path)
    print(corpus.corpus)
