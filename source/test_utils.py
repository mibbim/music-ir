import os
import pickle
from pathlib import Path

from tqdm import tqdm

from source.ListDictCorpus import ListDictCorpus, root_path, test_merge

root = Path(__file__).parent.parent


def select_random_song(folders=None) -> Path:
    if folders is None:
        folders = os.listdir(root / "data" / "fma_small")
    import random
    folder = random.choice(folders)

    if folder.endswith(".zip") or folder.startswith("."):
        return select_random_song(folders)

    song = random.choice(os.listdir(root / "data" / "fma_small" / folder))
    return root / "data" / "fma_small" / folder / song


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
        corpus_file = root_path / f"corpus_{i}_ld.pickle"

        if skip_existing and corpus_file.exists():
            print(f"Corpus {corpus_file} already exists, skipping...")
            continue

        print(f"Corpus {corpus_file} not found, creating new one...")
        corpus = ListDictCorpus()
        folder_path = root_path / "data" / "fma_small" / f"{i:03d}"

        for song_path in tqdm(folder_path.iterdir(), desc=f"Folder {i:03d}", position=1):
            corpus.add_song(song_path)

        with open(corpus_file, "wb") as f:
            pickle.dump(corpus, f)


def get_first_30_ld_corpora(skip_existing=True, verbose=1, *args, **kwargs):
    """
    Retrieve the first 30 folder ListDict Corpora and merge them into a single one.

    :param skip_existing: Whether to skip existing corpora or not.
    :type skip_existing: bool
    :param verbose: Verbosity level (of progress bars).
                        Level 0 doesn't show any progress bar.
                        Level 1 shows info of the corpora being merged.
                        Level 2 shows info of the songs being indexed.
    :type verbose: int
    :param args: Arguments to pass to the `ListDictCorpus` constructor.
    :type args: list
    :return: Merged ListDictCorpus instance.
    """
    file_path = root_path / "first_30_corpora.pickle"

    if file_path.exists() and skip_existing:
        with open(file_path, "rb") as f:
            return pickle.load(f)

    all_corpora = ListDictCorpus(*args, **kwargs)

    iterator = tqdm(range(31), desc="Corpora", position=0) if verbose > 0 else range(31)
    for i in iterator:
        corpus_file = root_path / f"corpus_{i}_ld.pickle"

        if skip_existing and corpus_file.exists():
            print(f"Corpus {corpus_file} already exists, skipping...")
            corpus = pickle.load(open(corpus_file, "rb"))

        else:
            print(f"Corpus {corpus_file} not found, creating new one...")
            corpus = ListDictCorpus(*args, **kwargs)
            folder_path = root_path / f'data/fma_small/{i:03d}'
            inner_iterator = tqdm(folder_path.iterdir(), desc=f"Folder {i:03d}", position=1,
                                  leave=False) if verbose > 1 else folder_path.iterdir()
            for song_path in inner_iterator:
                corpus.add_song(song_path)

            with open(corpus_file, "wb") as f:
                pickle.dump(corpus, f)

        all_corpora.merge(corpus)

    with open(file_path, "wb") as f:
        pickle.dump(all_corpora, f)
    return all_corpora


def test_for_ListDict_Corpus():
    from Corpus import find_song

    test_merge()

    create_30_ld_corpora(skip_existing=True)
    all_corpora = get_first_30_ld_corpora(skip_existing=True)

    n_tests = 100
    wrong = 0
    for i in range(n_tests):
        random_song = select_random_song()
        recognized, _, _ = find_song(random_song, all_corpora, verbose=True)
        if recognized is None or random_song.name != recognized.name:
            wrong += 1
            print("Wrong song recognized!")
    print(f"Accuracy: {1 - wrong / n_tests}")


if __name__ == '__main__':
    # print(select_random_song())
    test_for_ListDict_Corpus()
