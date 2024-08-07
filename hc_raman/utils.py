from rsciio.renishaw import file_reader
import os
import tomllib


def load_raman_file(file_path):
    """
    Read Renishaw Raman spectra from a file.

    Parameters
    ----------
    file_path : str
        Path to the Renishaw Raman file.

    Returns
    -------
    data : dict
        Dictionary containing the Raman spectra.
    """
    # Read the Renishaw Raman file
    data = file_reader(file_path)
    return data


def get_wavenumber_intensity(data):
    """
    Extract the wavenumber and intensity from the Renishaw Raman data.

    Parameters
    ----------
    data : dict
        Dictionary containing the Raman spectra.

    Returns
    -------
    wavenumber : numpy.ndarray
        Wavenumber values.
    intensity : numpy.ndarray
        Intensity values.
    """
    # Extract the wavenumber and intensity
    wavenumber = data[0]["axes"][0]["axis"]
    intensity = data[0]["data"]
    return wavenumber, intensity


def get_spectrum_region():
    with open(
        os.path.join(os.path.dirname(__file__), "spectrum_config/spectrum_regions.toml"),
        "rb",
    ) as file:
        spectrum_region = tomllib.load(file)
    return spectrum_region

