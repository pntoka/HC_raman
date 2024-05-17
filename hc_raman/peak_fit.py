import os
import numpy as np
from baseline import airpls_baseline, iasls_baseline, nn_baseline, interpolate_data
import tomllib
from lmfit.models import LorentzianModel, GaussianModel
from preprocess import preprocess_raman_data


def get_peaks_config():
    with open(
        os.path.join(os.path.dirname(__file__), "spectrum_config/peaks_config.toml"),
        "rb",
    ) as file:
        peaks_config = tomllib.load(file)
    return peaks_config


def build_lmfit_model(mode="5peaks"):
    peaks_config = get_peaks_config()
    peaks = peaks_config["first_order"]["models"][mode].split("+")
    model_list = []
    for peak in peaks:
        peak_model = eval(
            peaks_config["first_order"]["peaks"][peak]["peak_type"]
            + f"Model(prefix='{peak}_')"
        )
        for hint in peaks_config["first_order"]["peaks"][peak]["param_hints"]:
            peak_model.set_param_hint(
                f"{peak}_{hint}",
                **peaks_config["first_order"]["peaks"][peak]["param_hints"][hint],
            )
        model_list.append(peak_model)
    model = model_list[0]
    for i in range(1, len(model_list)):
        model += model_list[i]
    params = model.make_params()
    return model, params


def fit_model(x_data, y_data, mode="5peaks"):
    """
    Fit the peaks to the Raman spectrum.
    """
    model, params = build_lmfit_model(mode)
    result = model.fit(y_data, params, x=x_data)
    return result


def get_spectrum_region():
    with open(
        os.path.join(os.path.dirname(__file__), "spectrum_config/spectrum_region.toml"),
        "rb",
    ) as file:
        spectrum_region = tomllib.load(file)
    return spectrum_region


def filter_normalize_data(file_path, window_length=11, polyorder=3):
    """'
    Apply preprocessing to the Raman data. Filtering and normalization.
    Returns wavenumber(x_data) and intensity(y_data)
    """
    x_data, y_data = preprocess_raman_data(file_path, window_length, polyorder)
    return x_data, y_data


def baseline_substraction(
    baseline, x_data, y_data, lam=10e6, p=1e-2, size=2200, iter=10
):
    """
    Substract the baseline from the Raman data.
    """
    if baseline == "airpls":
        y_data_substracted = y_data - airpls_baseline(y_data, lam=lam)
    elif baseline == "iasls":
        y_data_substracted = y_data - iasls_baseline(y_data, lam=lam, p=p)
    elif baseline == "nn":
        _, new_y_data = interpolate_data(x_data, y_data, size=size)
        y_data_substracted = new_y_data - nn_baseline(
            x_data, y_data, size=size, iter=iter
        )
    return y_data_substracted


def select_region(x_data, y_data, region="first_order"):
    """
    Select the region of the Raman spectrum.
    """
    spectrum_regions = get_spectrum_region()
    roi = spectrum_regions["spectrum"]["regions"][region]
    x_data_region = x_data[(x_data > roi["min"]) & (x_data < roi["max"])]
    y_data_region = y_data[(x_data > roi["min"]) & (x_data < roi["max"])]
    return x_data_region, y_data_region


def prepare_data(
    file_path,
    baseline,
    window_length=11,
    polyorder=3,
    lam=10e6,
    p=1e-2,
    size=2200,
    iter=10,
    region="first_order",
):
    """
    Prepare the data for the peak fitting.
    """
    x_data, y_data = filter_normalize_data(file_path, window_length, polyorder)
    y_data_substracted = baseline_substraction(
        baseline, x_data, y_data, lam, p, size, iter
    )
    if baseline == "nn":
        x_data, _ = interpolate_data(x_data, y_data, size=size)
    x_data_region, y_data_region = select_region(x_data, y_data_substracted, region)
    return x_data_region, y_data_region


def peak_fit(
    file_path,
    baseline,
    window_length=11,
    polyorder=3,
    lam=10e6,
    p=1e-2,
    size=2200,
    iter=10,
    region="first_order",
    mode="5peaks",
):
    """
    Fit the peaks to the Raman spectrum.
    """
    x_data, y_data = prepare_data(
        file_path, baseline, window_length, polyorder, lam, p, size, iter, region
    )
    result = fit_model(x_data, y_data, mode)
    return result
