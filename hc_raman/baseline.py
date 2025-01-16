from pybaselines.whittaker import airpls, iasls
import os
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit


def airpls_baseline(y_data, lam=10e6):
    """
    Apply the Adaptive iteratively reweighted penalized least squares (airPLS) baseline correction.

    Parameters
    ----------
    y_data : numpy.ndarray
        Data to be corrected.
    lam : float, optional
        Penalty parameter (default is 10e6).

    Returns
    -------
    baseline : numpy.ndarray
        Baseline corrected data.
    """
    # Apply the APLS baseline correction
    baseline = airpls(y_data, lam=lam)
    return baseline[0]


def iasls_baseline(x_data, y_data, lam=10e6, p=1e-2):
    """
    Apply the improved asymmetric least squares (IAsLS) algorithm baseline correction.

    Parameters
    ----------
    x_data : numpy.ndarray
        Wavenumber values.
    y_data : numpy.ndarray
        Data to be corrected.
    lam : float, optional
        Penalty parameter (default is 10e6).
    p : float, optional
        Smoothing parameter (default is 1e-2).

    Returns
    -------
    baseline : numpy.ndarray
        Baseline corrected data.
    """
    # Apply the iasPLS baseline correction
    baseline = iasls(y_data, x_data, lam=lam, p=p)
    return baseline[0]