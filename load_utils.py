import librosa


def load_mp3(path):
    signal, sample_rate = librosa.load(path)
    return signal, sample_rate
