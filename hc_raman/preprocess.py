import os
import numpy as np
import tomllib
from .utils import load_raman_file, get_wavenumber_intensity
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import savgol_filter


def filter_data(data, window_length=11, polyorder=3):
    """
    Apply a Savitzky-Golay filter to the data.

    Parameters
    ----------
    data : numpy.ndarray
        Data to be filtered.
    window_length : int, optional
        Length of the filter window (default is 11).
    polyorder : int, optional
        Order of the polynomial to fit to the data (default is 3).

    Returns
    -------
    filtered_data : numpy.ndarray
        Filtered data.
    """
    # Apply a Savitzky-Golay filter
    filtered_data = savgol_filter(data, window_length, polyorder)
    return filtered_data


def normalize_data(data):
    """
    Normalize the data.

    Parameters
    ----------
    data : numpy.ndarray
        Data to be normalized.

    Returns
    -------
    normalized_data : numpy.ndarray
        Normalized data.
    """
    # Normalize the data
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(data.reshape(-1, 1)).flatten()
    return normalized_data


def preprocess_raman_data(file_path, window_length=11, polyorder=3):
    """
    Preprocess Renishaw Raman data.

    Parameters
    ----------
    file_path : str
        Path to the Renishaw Raman file.
    window_length : int, optional
        Length of the filter window (default is 11).
    polyorder : int, optional
        Order of the polynomial to fit to the data (default is 3).

    Returns
    -------
    wavenumber : numpy.ndarray
        Wavenumber values.
    intensity : numpy.ndarray
        Intensity values.
    """
    # Load the Renishaw Raman file
    data = load_raman_file(file_path)
    # Extract the wavenumber and intensity
    wavenumber, intensity = get_wavenumber_intensity(data)
    # Filter the intensity data
    filtered_intensity = filter_data(intensity, window_length, polyorder)
    # Normalize the intensity data
    normalized_intensity = normalize_data(filtered_intensity)
    return wavenumber, normalized_intensity
