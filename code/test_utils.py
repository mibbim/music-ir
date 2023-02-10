import os
from pathlib import Path


def select_random_song(folders=None) -> Path:
    if folders is None:
        folders = os.listdir("../data")
    import random
    folder = random.choice(folders)
    if folder.endswith(".zip") or folder.startswith("."):
        return select_random_song()
    song = random.choice(os.listdir(f"data/{folder}"))
    return Path(f"data/{folder}/{song}")
