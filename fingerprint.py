import numpy as np
from scipy.ndimage import generate_binary_structure, maximum_filter, binary_erosion, \
    iterate_structure

CONVOLUTION_REPETITIONS = 10


def get_peak_coordinates(peaks: np.ndarray) -> np.ndarray:
    """Get coordinates of peaks.

    Args:
        peaks (np.ndarray): 2D array of peaks.
    Returns:
        np.ndarray: 2D array of coordinates.
    """
    coordinates = np.argwhere(peaks)
    return coordinates


def detect_peaks(image: np.ndarray) -> np.ndarray:
    """Detect peaks in a 2D image.

    Args:
        image (np.ndarray): 2D image.
    Returns:
        np.ndarray: 2D image mask with peaks marked.
    """
    # scale image to [0, 1], better for peak detection, not necessary
    # image = scale_min_max_image(image)

    # find local maxima
    struct = generate_binary_structure(2, 2)
    neighborhood = iterate_structure(struct, CONVOLUTION_REPETITIONS)
    local_max = maximum_filter(image, footprint=neighborhood) == image

    # find background (pixels with minimum value)
    background = image == image.min(initial=None)
    eroded_background = binary_erosion(
        background, structure=neighborhood, border_value=1)

    # Boolean mask of arr with True at peaks
    # in no signal regions, all the region is considered a peak, so we remove those
    detected_peaks = local_max ^ eroded_background
    return detected_peaks


if __name__ == "__main__":
    pass