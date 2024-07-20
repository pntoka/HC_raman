import os
import numpy as np
import tomllib
from .utils import load_raman_file, get_wavenumber_intensity, get_spectrum_region
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import savgol_filter
import ramanspy


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


def new_preprocess(file_path=None, wavenumber=None, intensity=None, baseline='iarpls', window_length=11, polyorder=3, region="first_order"):
    
    if file_path is not None:
        data = load_raman_file(file_path)
        wavenumber, intensity = get_wavenumber_intensity(data)
    
    if wavenumber is not None and intensity is not None:
        intensity = intensity
        wavenumber = wavenumber

    raman_spectrum = ramanspy.Spectrum(intensity, wavenumber)
    spectrum_regions = get_spectrum_region()
    roi = spectrum_regions["spectrum"]["regions"][region]
    region_val = (roi["min"], roi["max"])

    if baseline == 'iasls':
        baseline_element = ramanspy.preprocessing.baseline.IASLS()
    elif baseline == 'airpls':
        baseline_element = ramanspy.preprocessing.baseline.AIRPLS()
    elif baseline == 'iarpls':
        baseline_element = ramanspy.preprocessing.baseline.IARPLS()
    
    preprocessing_pipeline = ramanspy.preprocessing.Pipeline([
        ramanspy.preprocessing.despike.WhitakerHayes(),
        ramanspy.preprocessing.denoise.SavGol(window_length=window_length, polyorder=polyorder),
        ramanspy.preprocessing.normalise.MinMax(),
        baseline_element,
        ramanspy.preprocessing.misc.Cropper(region=region_val)
    ])
    data = preprocessing_pipeline.apply(raman_spectrum)
    y_data = data.spectral_data
    x_data = data.spectral_axis
    return x_data, y_data
