from pathlib import Path
import audio2numpy as a2n


def load_mp3(path: Path):
    signal, sample_rate = a2n.audio_from_file(path.as_posix())
    return signal, sample_rate
