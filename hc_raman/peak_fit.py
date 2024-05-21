import os
import numpy as np
from hc_raman.baseline import airpls_baseline, iasls_baseline, nn_baseline, interpolate_data, Convclassifica
import tomllib
from lmfit.models import LorentzianModel, GaussianModel
from hc_raman.preprocess import preprocess_raman_data
import torch
import torch.nn as nn

class Convclassifica(nn.Module):
    def __init__(self):
        super(Convclassifica, self).__init__()
        self.hidden1 = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
            ),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(
                in_features=17568,
                out_features=100,
                bias=True,
            ),
            nn.ReLU(),
        )
        self.classifica = nn.Sequential(nn.Linear(100, 3), nn.Sigmoid())

    def forward(self, x):
        fc1 = self.hidden1(x)
        fc1 = fc1.reshape(fc1.size(0), -1)
        fc2 = self.hidden2(fc1)
        output = self.classifica(fc2)
        return output

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
        os.path.join(os.path.dirname(__file__), "spectrum_config/spectrum_regions.toml"),
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


def baseline_analysis(
        file_path, baseline, window_length=11, polyorder=3,
        lam=10e6, p=1e-2, size=2200, iter=10
):
    assert baseline in ["airpls", "iasls", "nn"]
    x_data, y_data = filter_normalize_data(file_path, window_length, polyorder)

    if baseline == "airpls":
        return x_data, y_data, airpls_baseline(y_data, lam=lam)
    elif baseline == "iasls":
        return x_data, y_data, iasls_baseline(y_data, lam=lam, p=p)
    elif baseline == "nn":
        new_x_data, new_y_data = interpolate_data(x_data, y_data, size=size)
        return x_data, new_y_data, nn_baseline(
            new_x_data, y_data, size=size, iter=iter
        )


def baseline_substraction(
    baseline, x_data, y_data, lam=10e6, p=1e-2, size=2200, iter=10
):
    """
    Substract the baseline from the Raman data.
    """
    assert baseline in ["airpls", "iasls", "nn"]

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
    Fit the peaks to the Raman spectrum with a chosen baseline
    """
    x_data, y_data = prepare_data(
        file_path, baseline, window_length, polyorder, lam, p, size, iter, region
    )
    result = fit_model(x_data, y_data, mode)
    return result
