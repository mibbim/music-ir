import numpy as np
import matplotlib.pyplot as plt

import plotly.graph_objects as go


def plot_surface(image: np.ndarray, peaks_freq: np.ndarray, peaks_time: np.ndarray):
    """
    Plot surface. Peaks are marked with red dots.
    Args:
        image (np.ndarray): 2D array of image, in our usage a spectrogram.
        peaks_freq (np.ndarray): 1D array of peaks frequencies.
        peaks_time (np.ndarray): 1D array of peaks time.
    """
    peaks_z = image[peaks_freq, peaks_time]
    fig = go.Figure(data=[
        go.Surface(z=image),
        go.Scatter3d(x=peaks_time, y=peaks_freq, z=peaks_z,
                     mode='markers',
                     marker=dict(
                         size=1,
                         color='red',
                         opacity=0.8
                     ))
    ])

    fig.update_layout(title='', autosize=False,
                      width=500, height=500,
                      margin=dict(l=65, r=50, b=65, t=90))

    fig.show()


def plot_spectrogram(spectrum: np.ndarray, frequencies: np.ndarray, time: np.ndarray):
    """Plot spectrogram.

    Args:
        spectrum (np.ndarray): 2D array of spectrum.
        frequencies (np.ndarray): 1D array of frequencies.
        time (np.ndarray): 1D array of time.
    """
    plt.pcolormesh(time, frequencies, spectrum)
    plt.colorbar()
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()
