from .utils import load_raman_file, get_wavenumber_intensity, get_spectrum_region
import ramanspy


def _build_baseline_element(baseline):
    """Map a baseline name to a ramanspy baseline preprocessing element."""
    if baseline == "iasls":
        return ramanspy.preprocessing.baseline.IASLS()
    elif baseline == "airpls":
        return ramanspy.preprocessing.baseline.AIRPLS()
    elif baseline == "iarpls":
        return ramanspy.preprocessing.baseline.IARPLS()
    raise ValueError(f"Unknown baseline '{baseline}'. Use 'iasls', 'airpls' or 'iarpls'.")


def preprocess(file_path=None, wavenumber=None, intensity=None, baseline='iarpls',
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
        Baseline algorithm: 'iasls', 'airpls' or 'iarpls' (default 'iarpls').
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
