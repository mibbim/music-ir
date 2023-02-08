from collections import Counter
from operator import itemgetter
from pathlib import PosixPath
from typing import Tuple, List, Dict

import numpy as np
from tqdm import tqdm

from Corpus import Corpus, Hash, SongID
from ListDict import ListDict
from fingerprint import detect_peaks
from hashing import select_fanout_windows_peaks, hash_fanout_windows_listdict
from load_utils import load_mp3


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
        """
        Adds a song to the fingerprint database and indexes its hashes in the corpus.

        :param filepath: The filepath to the MP3 file to add to the database.
        :type filepath: PosixPath
        :return: None
        """
        song_id = len(self.song_ids)
        self.song_ids.append(song_id)
        self.song_paths[song_id] = filepath
        signal, sample_rate = load_mp3(filepath)
        signal = self._preprocess_signal(signal)  # stereo to mono conversion
        hashes = self._get_hashes(signal, sample_rate, song_id)
        self.corpus.update(hashes)

    def _get_matches(self, hashes: ListDict) -> List[Tuple[int, SongID]]:
        """
        Find matching hashes in the corpus.

        Given a list of hashes, this method finds corresponding matches in the corpus and returns
        the time difference and the `SongID` for each match.

        Args:
        hashes (ListDict): A list of hashes representing audio segments.

        Returns:
        List[Tuple[int, SongID]]: A list of tuples, where each tuple contains the time difference
                                    and the `SongID` of a match.
        """
        deltas = []
        for hash_tuple, times in hashes.items():
            for (t, _) in times:
                if matches := self.corpus.get(hash_tuple, None):
                    deltas.extend([(match[0] - t, match[1]) for match in matches])
        return deltas

    @staticmethod
    def _retrieve_best_guess(matching_hashes: List[Tuple[int, SongID]],
                             ) -> Tuple[SongID, int, int]:

        """
        Retrieve the song ID that matches the most number of hashes in the input hashes.

        Parameters
        ----------
        matching_hashes : list of tuple
            List of tuples with (delta, song_id) where delta is the difference in position of the
            hash and song_id is the identifier of the song in the corpus.

        Returns
        -------
        tuple
            Tuple of the song_id with the highest number of matches, the number of matches, and the
            delta that provided the highest number of matches.
        """
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
        """Compute hashes for a given signal.

        Compute hashes for a given signal, as a pre-processing step before recognition.
        This involves computing the db spectrum of the signal, detecting peaks in the spectrum,
        selecting fanout windows based on the detected peaks and finally, hashing the selected
        fanout windows.

        Args:
        signal (np.ndarray): Input signal for which to compute the hashes.
        sample_rate (int): Sample rate of the input signal.
        song_id (SongID, optional): ID of the song that the input signal belongs to. Defaults to None.

        Returns:
        ListDict[Hash, List[Tuple[int, SongID]]]: A list-dict object, containing hashes as keys and
        corresponding fanout windows (constellation) as values, along with their time offsets in the
        song and the ID of the song that the signal belongs to.
        """

        db_spectrum = self._compute_db_spectrum(signal, sample_rate)
        peaks = detect_peaks(db_spectrum)
        fan_win_data = select_fanout_windows_peaks(peaks, fanout_window=self.fanout_window)
        return hash_fanout_windows_listdict(fan_win_data, song_id)

    def recognize(self, input_signal: np.ndarray,
                  sample_rate: int,
                  n_matches_threshold: int = 10) -> Tuple[PosixPath | None, float, int]:

        """Recognize the song from an audio signal.

        Args:
            input_signal (array-like): The audio signal to be recognized.
            sample_rate (int): The sample rate of the audio signal.
            n_matches_threshold (int, optional): The minimum number of matches required for the
                                            recognition to be considered successful. Defaults to 10.

        Returns:
            tuple: (path, confidence, alignment)
                path (str or None): The path of the recognized song, or None if the recognition
                                        is not successful.
                confidence (float): The confidence of the recognition, or 0 if the
                                        recognition is not successful.
                alignment (int or -1): The alignment of the signal, or -1 if the recognition
                                        is not successful.
        """
        input_signal = self._preprocess_signal(input_signal)
        input_hashes = self._get_hashes(input_signal, sample_rate, song_id=None)
        matching_hashes = self._get_matches(input_hashes)
        song_id, n_matches, alignment = self._retrieve_best_guess(matching_hashes)
        if n_matches < n_matches_threshold:
            return None, 0., -1
        matching_confidence = n_matches / len(input_hashes)
        return self.song_paths[song_id], matching_confidence, alignment

    def get_attributes_with_song_id_offset(self, offset: int) -> Tuple[
        ListDict[Hash, List[Tuple[int, SongID]]], List[SongID], Dict[SongID, PosixPath]
    ]:
        """
        Returns the corpus, song_ids, and song_paths of the instance with a given offset added
        to song_ids.

        :param offset: The integer offset that will be added to the song_ids.
        :type offset: int
        :return: The corpus, song_ids, and song_paths with song_ids increased by offset.
        :rtype: Tuple[ListDict[Hash, List[Tuple[int, SongID]]], List[SongID], Dict[SongID, PosixPath]]
        """
        corpus = ListDict(
            {k: [(t, si + offset) for t, si in v] for k, v in self.corpus.items()})
        song_ids = [si + offset for si in self.song_ids]
        song_paths = {si + offset: p for si, p in self.song_paths.items()}
        return corpus, song_ids, song_paths

    def merge(self, other_corpus):
        """Merge two corpora.

        This method merges the songs of the current corpus with the songs of another
        corpus `other`. The resulting corpus contains songs from both corpora.

        Args:
            other_corpus (ListDictCorpus): The other corpus to be merged with the current corpus.

        Returns:
            None: The method modifies the current corpus in place, no new instance is created.
        """
        offset = len(self.song_ids)
        c, ids, paths = other_corpus.get_attributes_with_song_id_offset(offset)

        self.corpus.update(c)
        self.song_ids.extend(ids)
        self.song_paths.update(paths)


def create_30_ld_corpora(skip_existing: bool = True):
    """
    Creates 30 `ListDictCorpus` objects, one for each of the 30 folders in the `data` directory.

    This function loops through each folder in the `data` directory and adds the songs in the folder to a
    `ListDictCorpus` object. The resulting `ListDictCorpus` objects are then pickled and saved to a file
    with the name `corpus_{i}_ld.pickle`, where `i` is the number of the folder. If the `skip_existing`
    parameter is set to `True` (default), the function will skip creating a `ListDictCorpus` object for a
    folder if a pickled file with the corresponding name already exists.

    It may be useful to run this function once to create the pickled corpora, and then use the pickled
    corpora in subsequent runs of the program. This will save time, as the pickling process can take
    several minutes.

    :param skip_existing: If set to `True`, the function will skip creating a `ListDictCorpus` object for a
    folder if a pickled file with the corresponding name already exists.
    :type skip_existing: bool, optional
    """
    for i in tqdm(range(31), desc="Corpora", position=0):
        corpus_file = Path(f"corpus_{i}_ld.pickle")

        if skip_existing and corpus_file.exists():
            print(f"Corpus {corpus_file} already exists, skipping...")
            continue

        print(f"Corpus {corpus_file} not found, creating new one...")
        corpus = ListDictCorpus()
        folder_path = Path(f'data/{i:03d}')

        for song_path in tqdm(folder_path.iterdir(), desc=f"Folder {i:03d}", position=1):
            corpus.add_song(song_path)

        with open(corpus_file, "wb") as f:
            pickle.dump(corpus, f)


def test_merge():
    c1 = ListDictCorpus()
    c1.corpus = ListDict(
        {(944, 478, 1): [(0, 0)],
         (1486, 478, 1): [(0, 1)]}
    )
    c1.song_ids = [0, 1]
    c1.song_paths = {0: Path("data/000/000000.mp3"),
                     1: Path("data/000/000001.mp3")}
    c2 = ListDictCorpus()
    c2.corpus = ListDict({
        (9440, 4780, 10): [(0, 10)],
        (14860, 4780, 10): [(0, 1)]
    })
    c2.song_ids = [0, 10]
    c2.song_paths = {0: Path("data/001/000000.mp3"),
                     10: Path("data/001/000001.mp3")}
    c1.merge(c2)
    expected_corpus = ListDict({(944, 478, 1): [(0, 0)],
                                (1486, 478, 1): [(0, 1)],
                                (9440, 4780, 10): [(0, 12)],
                                (14860, 4780, 10): [(0, 3)]})
    assert c1.corpus == expected_corpus
    expected_song_ids = [0, 1, 2, 12]
    assert c1.song_ids == expected_song_ids
    expected_song_paths = {0: Path("data/000/000000.mp3"),
                           1: Path("data/000/000001.mp3"),
                           2: Path("data/001/000000.mp3"),
                           12: Path("data/001/000001.mp3")}
    assert c1.song_paths == expected_song_paths


def get_fisrt_30_ld_corpora(skip_existing=True):
    """
    Retrieve the first 30 ListDictCorpora and merge them into a single one.

    :param skip_existing: Whether to skip existing corpora or not.
    :return: Merged ListDictCorpus instance.
    """
    file_path = Path("first_30_corpora.pickle")

    if file_path.exists() and skip_existing:
        with open(file_path, "rb") as f:
            return pickle.load(f)

    all_corpora = ListDictCorpus()

    for i in tqdm(range(31), desc="Corpora", position=0):
        corpus_file = Path(f"corpus_{i}_ld.pickle")

        if skip_existing and corpus_file.exists():
            print(f"Corpus {corpus_file} already exists, skipping...")
            corpus = pickle.load(open(corpus_file, "rb"))

        else:
            print(f"Corpus {corpus_file} not found, creating new one...")
            corpus = ListDictCorpus()
            folder_path = Path(f'data/{i:03d}')

            for song_path in tqdm(folder_path.iterdir(), desc=f"Folder {i:03d}", position=1):
                corpus.add_song(song_path)

            with open(corpus_file, "wb") as f:
                pickle.dump(corpus, f)

        all_corpora.merge(corpus)

    with open(file_path, "wb") as f:
        pickle.dump(all_corpora, f)
    return all_corpora


if __name__ == "__main__":
    import pickle
    from pathlib import Path
    import os
    from Corpus import find_song

    test_merge()

    create_30_ld_corpora(skip_existing=True)
    all_corpora = get_fisrt_30_ld_corpora(skip_existing=True)


    def select_random_song(folders=None) -> Path:
        if folders is None:
            folders = os.listdir("data")
        import random
        folder = random.choice(folders)
        if folder.endswith(".zip") or folder.startswith("."):
            return select_random_song()
        song = random.choice(os.listdir(f"data/{folder}"))
        return Path(f"data/{folder}/{song}")


    for i in range(100):
        random_song = select_random_song()
        recognized, _, _ = find_song(random_song, all_corpora, verbose=True)
        if random_song != recognized:
            print("Wrong song recognized!")
