from .utils import load_raman_file, get_wavenumber_intensity, get_spectrum_region
import os
import numpy as np
import matplotlib.pyplot as plt
import ramanspy

# Fewest points a region may hold before a standard deviation over it stops meaning anything.
MIN_REGION_POINTS = 10


def _build_baseline_element(baseline):
    """Map a baseline name to a ramanspy baseline preprocessing element."""
    if baseline == "iasls":
        return ramanspy.preprocessing.baseline.IASLS()
    elif baseline == "airpls":
        return ramanspy.preprocessing.baseline.AIRPLS()
    elif baseline == "iarpls":
        return ramanspy.preprocessing.baseline.IARPLS()
    raise ValueError(f"Unknown baseline '{baseline}'. Use 'iasls', 'airpls' or 'iarpls'.")


def preprocess(file_path=None, wavenumber=None, intensity=None, baseline='iasls',
               window_length=11, polyorder=3, region="first_order"):
    """
    Preprocess a Raman spectrum for peak fitting using RamanSPy.

    Pipeline: despike -> denoise (Savitzky-Golay) -> MinMax normalise ->
    baseline correction -> crop to region.

    Provide either ``file_path`` or both ``wavenumber`` and ``intensity``.

    Parameters
    ----------
    file_path : str, optional
        Path to a Renishaw Raman file.
    wavenumber, intensity : numpy.ndarray, optional
        Raw spectrum arrays (used when ``file_path`` is not given).
    baseline : str, optional
        Baseline algorithm: 'iasls', 'airpls' or 'iarpls' (default 'iasls').
    window_length, polyorder : int, optional
        Savitzky-Golay denoising parameters.
    region : str, optional
        Named region to crop to (see spectrum_regions.toml, default 'first_order').

    Returns
    -------
    x_data, y_data : numpy.ndarray
        Preprocessed wavenumber and intensity.
    """
    if file_path is not None:
        data = load_raman_file(file_path)
        wavenumber, intensity = get_wavenumber_intensity(data)

    raman_spectrum = ramanspy.Spectrum(intensity, wavenumber)
    spectrum_regions = get_spectrum_region()
    roi = spectrum_regions["spectrum"]["regions"][region]
    region_val = (roi["min"], roi["max"])

    preprocessing_pipeline = ramanspy.preprocessing.Pipeline([
        ramanspy.preprocessing.despike.WhitakerHayes(),
        ramanspy.preprocessing.denoise.SavGol(window_length=window_length, polyorder=polyorder),
        ramanspy.preprocessing.normalise.MinMax(),
        _build_baseline_element(baseline),
        ramanspy.preprocessing.misc.Cropper(region=region_val)
    ])
    data = preprocessing_pipeline.apply(raman_spectrum)
    return data.spectral_axis, data.spectral_data


def conv_preprocess(file_path=None, wavenumber=None, intensity=None, baseline='iarpls',
                    window_length=11, polyorder=3, region=None):
    """
    Preprocess a Raman spectrum for the conventional D/G ratio (no normalisation).

    Pipeline: despike -> denoise (Savitzky-Golay) -> baseline correction ->
    optional crop to region.

    Provide either ``file_path`` or both ``wavenumber`` and ``intensity``.

    Parameters are the same as :func:`preprocess`, except ``region`` is optional
    (no cropping is applied when ``region`` is None).

    Returns
    -------
    x_data, y_data : numpy.ndarray
        Preprocessed wavenumber and intensity.
    """
    if file_path is not None:
        data = load_raman_file(file_path)
        wavenumber, intensity = get_wavenumber_intensity(data)

    raman_spectrum = ramanspy.Spectrum(intensity, wavenumber)

    pipeline_list = [
        ramanspy.preprocessing.despike.WhitakerHayes(),
        ramanspy.preprocessing.denoise.SavGol(window_length=window_length, polyorder=polyorder),
        _build_baseline_element(baseline),
    ]

    if region is not None:
        spectrum_regions = get_spectrum_region()
        roi = spectrum_regions["spectrum"]["regions"][region]
        pipeline_list.append(ramanspy.preprocessing.misc.Cropper(region=(roi["min"], roi["max"])))

    preprocessing_pipeline = ramanspy.preprocessing.Pipeline(pipeline_list)
    data = preprocessing_pipeline.apply(raman_spectrum)
    return data.spectral_axis, data.spectral_data


def _region_mask(spectral_axis, region):
    """Boolean mask selecting a named spectrum region on a spectral axis."""
    roi = get_spectrum_region()["spectrum"]["regions"][region]
    mask = (spectral_axis >= roi["min"]) & (spectral_axis <= roi["max"])
    if mask.sum() < MIN_REGION_POINTS:
        raise ValueError(
            f"Region '{region}' ({roi['min']}-{roi['max']} cm-1) covers only {mask.sum()} points "
            f"of this spectrum, which spans {spectral_axis.min():.0f}-{spectral_axis.max():.0f} "
            "cm-1. Use a region the measurement actually covers."
        )
    return mask


def _safe_ratio(numerator, denominator):
    """Divide two numbers, returning NaN when the denominator is not positive."""
    if denominator <= 0:
        return float("nan")
    return float(numerator / denominator)


def _plot_quality(x_data, y_data, baseline_curve, corrected_data, noise_mask, signal_mask,
                  metrics, title, display_plot=True):
    """Plot the spectrum, the recovered baseline and the quality regions. Returns (fig, ax)."""
    textstr = (f"SNR = {metrics['snr']:.1f} \n noise = {metrics['noise']:.1f} \n "
               f"baseline/signal = {metrics['baseline_ratio']:.2f}")
    props = dict(boxstyle='round', facecolor='gainsboro', alpha=0.5)
    fig, ax = plt.subplots()
    ax.plot(x_data, y_data, c='k', linewidth=1, label="despiked spectrum")
    ax.plot(x_data, baseline_curve, c='r', linestyle='--', label="baseline")
    ax.plot(x_data, corrected_data, c='b', linewidth=1, label="baseline corrected")
    ax.axvspan(x_data[signal_mask].min(), x_data[signal_mask].max(),
               alpha=0.3, color='g', label="signal region")
    ax.axvspan(x_data[noise_mask].min(), x_data[noise_mask].max(),
               alpha=0.3, color='grey', label="noise region")
    ax.set_xlabel("Wavenumber (cm$^-1$)", fontsize=14)
    ax.set_ylabel("Intensity (a.u.)", fontsize=14)
    ax.set_title(title)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
    ax.legend(loc='center right')
    if display_plot:
        plt.show()
    return fig, ax


def get_quality_metrics(file_path=None, wavenumber=None, intensity=None, baseline='iasls',
                        window_length=11, polyorder=3, region="first_order",
                        signal_region="g_band", noise_region="mid",
                        plot=False, display_plot=True):
    """
    Measure the data quality of a Raman spectrum.

    Runs despike -> denoise (Savitzky-Golay) -> baseline correction over the full
    measured range, without normalisation and without cropping, so the baseline
    matches the one :func:`preprocess` fits (it too corrects the baseline before
    cropping). The baseline curve itself is recovered as the difference between
    the spectrum before and after correction, since RamanSPy returns only the
    corrected data.

    The noise is measured on the despiked but *not* denoised spectrum, because
    Savitzky-Golay smoothing removes most of the noise it is meant to measure.
    Every other metric uses the denoised spectrum, matching what is later fitted.

    Provide either ``file_path`` or both ``wavenumber`` and ``intensity``. The
    array form accepts data from any instrument or file format, in either
    wavenumber direction.

    Parameters
    ----------
    file_path : str, optional
        Path to a Renishaw Raman file.
    wavenumber, intensity : numpy.ndarray, optional
        Raw spectrum arrays (used when ``file_path`` is not given). Ascending or
        descending wavenumber are both fine.
    baseline : str, optional
        Baseline algorithm: 'iasls', 'airpls' or 'iarpls' (default 'iasls').
        Pass the algorithm the spectrum will actually be fitted with, so that the
        baseline metrics describe the curve the fit will see (the peak fitting
        entry points default to 'iasls').
    window_length, polyorder : int, optional
        Savitzky-Golay denoising parameters.
    region : str, optional
        Named region the baseline metrics are measured over, i.e. the region to
        be fitted later (default 'first_order').
    signal_region : str, optional
        Named region holding the band used as the signal, normally the G band
        (default 'g_band'). Use 'd_band' for samples where D exceeds G.
    noise_region : str, optional
        Named signal-free region used for the noise estimate (default 'mid',
        1850-2150 cm-1, the silent window between the first- and second-order
        carbon bands).
    plot : bool, optional
        If True, also plot the spectrum, the recovered baseline and the regions.
    display_plot : bool, optional
        If True (default), show the figure when ``plot`` is True.

    Returns
    -------
    metrics : dict
        Quality metrics, or ``(fig, ax, metrics)`` when ``plot=True``:

        snr : float
            ``signal / noise``, the headline quality number.
        signal : float
            Height of the tallest point of the baseline-corrected spectrum in
            ``signal_region``, above the baseline. Negative or near zero for a
            blank measurement, in which case the ratios below are NaN.
        noise : float
            Standard deviation of the despiked spectrum in ``noise_region``
            after removing a linear trend.
        baseline_ratio : float
            Mean baseline over ``region`` divided by ``signal``: how much of the
            measured intensity is background rather than Raman scattering.
        baseline_tilt : float
            Peak-to-peak baseline over ``region`` divided by ``signal``: how much
            the background varies underneath the bands to be fitted, i.e. how
            much the baseline algorithm had to invent.
        n_spike_points : int
            Number of points altered by despiking. A single cosmic ray typically
            alters one to three points.

    Notes
    -----
    ``snr``, ``baseline_ratio`` and ``baseline_tilt`` are dimensionless and
    unaffected by the scale of the input, so they can be compared across
    instruments and units. ``signal`` and ``noise`` are in whatever intensity
    units were supplied and are only comparable within one instrument and one
    set of acquisition settings, which is what makes them useful for spotting an
    exposure or laser power problem across a batch.

    Intensities are not normalised, so ``baseline_ratio`` includes any constant
    offset in the input, such as a detector dark level.
    """
    if file_path is not None:
        data = load_raman_file(file_path)
        wavenumber, intensity = get_wavenumber_intensity(data)

    raman_spectrum = ramanspy.Spectrum(intensity, wavenumber)
    despiked = ramanspy.preprocessing.despike.WhitakerHayes().apply(raman_spectrum)
    denoised = ramanspy.preprocessing.denoise.SavGol(
        window_length=window_length, polyorder=polyorder).apply(despiked)
    corrected = _build_baseline_element(baseline).apply(denoised)

    # ramanspy sorts the spectral axis, so masks must be built from it and not from the input
    x_data = corrected.spectral_axis
    baseline_curve = denoised.spectral_data - corrected.spectral_data

    signal_mask = _region_mask(x_data, signal_region)
    noise_mask = _region_mask(x_data, noise_region)
    region_baseline = baseline_curve[_region_mask(x_data, region)]

    signal = corrected.spectral_data[signal_mask].max()
    noise_trend = np.polyfit(x_data[noise_mask], despiked.spectral_data[noise_mask], 1)
    noise = np.std(despiked.spectral_data[noise_mask]
                   - np.polyval(noise_trend, x_data[noise_mask]))

    metrics = {
        "snr": _safe_ratio(signal, noise),
        "signal": float(signal),
        "noise": float(noise),
        "baseline_ratio": _safe_ratio(region_baseline.mean(), signal),
        "baseline_tilt": _safe_ratio(np.ptp(region_baseline), signal),
        "n_spike_points": int(np.count_nonzero(
            despiked.spectral_data != raman_spectrum.spectral_data)),
    }

    if plot:
        title = os.path.basename(file_path) if file_path is not None else 'Raman spectrum'
        fig, ax = _plot_quality(x_data, despiked.spectral_data, baseline_curve,
                                corrected.spectral_data, noise_mask, signal_mask,
                                metrics, title, display_plot)
        return fig, ax, metrics
    return metrics
