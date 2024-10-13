import os
import numpy as np
from hc_raman.utils import get_spectrum_region
from hc_raman.baseline import airpls_baseline, iasls_baseline, nn_baseline, interpolate_data, Convclassifica
import tomllib
from lmfit.models import LorentzianModel, GaussianModel
from hc_raman.preprocess import preprocess_raman_data, new_preprocess, conv_preprocess
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

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


def build_lmfit_model(mode="5peaks", region="first_order"):
    peaks_config = get_peaks_config()
    peaks = peaks_config[region]["models"][mode].split("+")
    model_list = []
    for peak in peaks:
        peak_model = eval(
            peaks_config[region]["peaks"][peak]["peak_type"]
            + f"Model(prefix='{peak}_')"
        )
        for hint in peaks_config[region]["peaks"][peak]["param_hints"]:
            peak_model.set_param_hint(
                f"{peak}_{hint}",
                **peaks_config[region]["peaks"][peak]["param_hints"][hint],
            )
        model_list.append(peak_model)
    model = model_list[0]
    for i in range(1, len(model_list)):
        model += model_list[i]
    params = model.make_params()
    return model, params


def fit_model(x_data, y_data, mode="5peaks", region="first_order"):
    """
    Fit the peaks to the Raman spectrum.
    """
    model, params = build_lmfit_model(mode, region)
    result = model.fit(y_data, params, x=x_data)
    return result



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
        return x_data, y_data, iasls_baseline(x_data, y_data, lam=lam, p=p)
    elif baseline == "nn":
        new_x_data, new_y_data = interpolate_data(x_data, y_data, size=size)
        return new_x_data, new_y_data, nn_baseline(
            x_data, y_data, size=size, iter=iter
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
        y_data_substracted = y_data - iasls_baseline(x_data, y_data, lam=lam, p=p)
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


def peak_fit_old(
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
    plot=False
):
    """
    Fit the peaks to the Raman spectrum with a chosen baseline
    """
    x_data, y_data = prepare_data(
        file_path, baseline, window_length, polyorder, lam, p, size, iter, region
    )
    result = fit_model(x_data, y_data, mode, region)
    results_dict = result.params.valuesdict()
    id_ig = results_dict["D_height"] / results_dict["G_height"]
    if plot:
        results_dict = result.params.valuesdict()
        id_ig = results_dict["D_height"] / results_dict["G_height"]
        comps = result.eval_components(x=x_data)
        peaks_config = get_peaks_config()
        peaks = peaks_config[region]["models"][mode].split("+")
        textstr = f"$I_D/I_G$ = {id_ig:.3f} \n R$^2$ = {result.rsquared:.4f}"
        props = dict(boxstyle='round', facecolor='gainsboro', alpha=0.5)
        fig, ax = plt.subplots()
        for peak in peaks:
            ax.plot(x_data, comps[f'{peak}_'], linestyle='--', label=peak)
        ax.scatter(x_data, y_data, c='k', label="raw data", s=1)
        ax.plot(x_data, result.best_fit, label="fit", linestyle='-', c='r')
        ax.set_xlabel("Wavenumber (cm$^-1$)", fontsize=14)
        ax.set_ylabel("Normalized Intensity (a.u.)", fontsize=14)
        ax.set_title(os.path.basename(file_path))
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
        ax.legend()
        plt.show()
        return fig, ax, result

    return result


def get_id_ig(result):
    results_dict = result.params.valuesdict()
    id_ig = results_dict["D_height"] / results_dict["G_height"]
    return id_ig


def auto_peak_fit_old(
    file_path,
    baseline=None,
    window_length=11,
    polyorder=3,
    lam=10e6,
    p=1e-2,
    size=2200,
    iter=10,
    region="first_order",
    mode="5peaks",
    plot=False
):
    """
    Fit the peaks to the Raman spectrum with a airpls baseline or iasls baseline if smaller data
    """
    x_data, _ = filter_normalize_data(file_path, window_length, polyorder)
    if baseline is None and x_data.max() < 2200:
        baseline = "iasls"
    elif baseline is None and x_data.max() > 2200:
        baseline = "airpls"
    x_data, y_data = prepare_data(
        file_path, baseline, window_length, polyorder, lam, p, size, iter, region
    )
    result = fit_model(x_data, y_data, mode, region)
    results_dict = result.params.valuesdict()
    id_ig = results_dict["D_height"] / results_dict["G_height"]
    if plot:
        results_dict = result.params.valuesdict()
        id_ig = results_dict["D_height"] / results_dict["G_height"]
        comps = result.eval_components(x=x_data)
        peaks_config = get_peaks_config()
        peaks = peaks_config[region]["models"][mode].split("+")
        textstr = f"$I_D/I_G$ = {id_ig:.3f} \n R$^2$ = {result.rsquared:.4f}"
        props = dict(boxstyle='round', facecolor='gainsboro', alpha=0.5)
        fig, ax = plt.subplots()
        for peak in peaks:
            ax.plot(x_data, comps[f'{peak}_'], linestyle='--', label=peak)
        ax.scatter(x_data, y_data, c='k', label="raw data", s=1)
        ax.plot(x_data, result.best_fit, label="fit", linestyle='-', c='r')
        ax.set_xlabel("Wavenumber (cm$^-1$)", fontsize=14)
        ax.set_ylabel("Normalized Intensity (a.u.)", fontsize=14)
        ax.set_title(os.path.basename(file_path))
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
        ax.legend()
        plt.show()
        return fig, ax, result

    return result


def peak_fit_from_file(
    file_path, 
    baseline='iasls',
    window_length=11,
    polyorder=3, 
    region="first_order",
    mode="5peaks",
    plot=False
):
    '''Fit peaks to Raman spectrum from file using RamanSPy for preprocessing the data'''

    x_data, y_data = new_preprocess(file_path=file_path, baseline=baseline, region=region,
                                    window_length=window_length, polyorder=polyorder)
    result = fit_model(x_data, y_data, mode, region)
    results_dict = result.params.valuesdict()
    id_ig = results_dict["D_height"] / results_dict["G_height"]
    if plot:
        results_dict = result.params.valuesdict()
        id_ig = results_dict["D_height"] / results_dict["G_height"]
        comps = result.eval_components(x=x_data)
        peaks_config = get_peaks_config()
        peaks = peaks_config[region]["models"][mode].split("+")
        textstr = f"$I_D/I_G$ = {id_ig:.3f} \n R$^2$ = {result.rsquared:.4f}"
        props = dict(boxstyle='round', facecolor='gainsboro', alpha=0.5)
        fig, ax = plt.subplots()
        for peak in peaks:
            ax.plot(x_data, comps[f'{peak}_'], linestyle='--', label=peak)
        ax.scatter(x_data, y_data, c='k', label="raw data", s=1)
        ax.plot(x_data, result.best_fit, label="fit", linestyle='-', c='r')
        ax.set_xlabel("Wavenumber (cm$^-1$)", fontsize=14)
        ax.set_ylabel("Normalized Intensity (a.u.)", fontsize=14)
        ax.set_title(os.path.basename(file_path))
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
        ax.legend()
        plt.show()
        return fig, ax, result

    return result


def peak_fit_from_data(
    wavenumber,
    intensity,    
    baseline='iasls',
    window_length=11,
    polyorder=3, 
    region="first_order",
    mode="5peaks",
    plot=False
):
    '''Fit peaks to Raman spectrum from arrays with data using RamanSPy for preprocessing the data'''
    x_data, y_data = new_preprocess(wavenumber=wavenumber,intensity=intensity, baseline=baseline, region=region,
                                    window_length=window_length, polyorder=polyorder)
    result = fit_model(x_data, y_data, mode, region)
    results_dict = result.params.valuesdict()
    id_ig = results_dict["D_height"] / results_dict["G_height"]
    if plot:
        results_dict = result.params.valuesdict()
        id_ig = results_dict["D_height"] / results_dict["G_height"]
        comps = result.eval_components(x=x_data)
        peaks_config = get_peaks_config()
        peaks = peaks_config[region]["models"][mode].split("+")
        textstr = f"$I_D/I_G$ = {id_ig:.3f} \n R$^2$ = {result.rsquared:.4f}"
        props = dict(boxstyle='round', facecolor='gainsboro', alpha=0.5)
        fig, ax = plt.subplots()
        for peak in peaks:
            ax.plot(x_data, comps[f'{peak}_'], linestyle='--', label=peak)
        ax.scatter(x_data, y_data, c='k', label="raw data", s=1)
        ax.plot(x_data, result.best_fit, label="fit", linestyle='-', c='r')
        ax.set_xlabel("Wavenumber (cm$^-1$)", fontsize=14)
        ax.set_ylabel("Normalized Intensity (a.u.)", fontsize=14)
        ax.set_title('Raman spectrum')
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
        ax.legend()
        plt.show()
        return fig, ax, result

    return result


def get_ratio_from_file(
        file_path,
        baseline='iasls',
        window_length=11,
        polyorder=3,
        region=None,
        plot=False
):
    '''Get the ID/IG ratio from a Raman spectrum file by taking intensity of D peak and G peak'''
    x_data, y_data = conv_preprocess(file_path=file_path, baseline=baseline, window_length=window_length,
                                    polyorder=polyorder, region=region)
    all_data = np.stack((x_data, y_data), axis=-1)
    D_band_condition = (all_data[:, 0] > 1300) & (all_data[:, 0] < 1390)
    G_band_condition = (all_data[:, 0] > 1500) & (all_data[:, 0] < 1670)
    D_band = all_data[D_band_condition]
    G_band = all_data[G_band_condition]
    D_intensity = D_band[:, 1].max()
    G_intensity = G_band[:, 1].max()
    # D_intensity = np.mean(np.sort(D_band[:, 1])[-5:])
    # G_intensity = np.mean(np.sort(G_band[:, 1])[-5:])
    id_ig = D_intensity / G_intensity
    if plot:
        textstr = f"$I_D/I_G$ = {id_ig:.3f}"
        props = dict(boxstyle='round', facecolor='gainsboro', alpha=0.5)
        fig, ax = plt.subplots()
        ax.plot(x_data, y_data, label='Raman spectrum')
        ax.axvspan(1300, 1390, alpha=0.5, color='r', label='D band')
        ax.axvspan(1500, 1620, alpha=0.5, color='g', label='G band')
        # ax.scatter(D_band[:, 0], D_band[:, 1], label='D band', c='r')
        # ax.scatter(G_band[:, 0], G_band[:, 1], label='G band', c='g')
        ax.set_xlabel("Wavenumber (cm$^-1$)", fontsize=14)
        ax.set_ylabel("Normalized Intensity (a.u.)", fontsize=14)
        ax.set_title(os.path.basename(file_path))
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
        ax.legend()
        plt.show()
        return fig, ax, id_ig

    return id_ig