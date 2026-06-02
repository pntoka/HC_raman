import os
import numpy as np
from hc_raman.utils import get_spectrum_region
import tomllib
from lmfit.models import LorentzianModel, GaussianModel
from hc_raman.preprocess import preprocess, conv_preprocess
import matplotlib.pyplot as plt


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


def get_id_ig(result):
    """Compute the I_D/I_G ratio from a fitted lmfit result (D_height / G_height)."""
    results_dict = result.params.valuesdict()
    id_ig = results_dict["D_height"] / results_dict["G_height"]
    return id_ig


def _plot_fit(x_data, y_data, result, region, mode, title, display_plot=True):
    """Plot the fitted peaks, the composite fit and the raw data. Returns (fig, ax)."""
    id_ig = get_id_ig(result)
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
    ax.set_title(title)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
    ax.legend()
    if display_plot:
        plt.show()
    return fig, ax


def peak_fit_from_file(
    file_path,
    baseline='iasls',
    window_length=11,
    polyorder=3,
    region="first_order",
    mode="5peaks",
    plot=False,
    display_plot=True
):
    '''Fit peaks to a Raman spectrum from a file using RamanSPy for preprocessing.

    Returns the lmfit result, or (fig, ax, result) when ``plot=True``.'''
    x_data, y_data = preprocess(file_path=file_path, baseline=baseline, region=region,
                                window_length=window_length, polyorder=polyorder)
    result = fit_model(x_data, y_data, mode, region)
    if plot:
        fig, ax = _plot_fit(x_data, y_data, result, region, mode,
                            os.path.basename(file_path), display_plot)
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
    plot=False,
    display_plot=True
):
    '''Fit peaks to a Raman spectrum from data arrays using RamanSPy for preprocessing.

    Returns the lmfit result, or (fig, ax, result) when ``plot=True``.'''
    x_data, y_data = preprocess(wavenumber=wavenumber, intensity=intensity, baseline=baseline,
                                region=region, window_length=window_length, polyorder=polyorder)
    result = fit_model(x_data, y_data, mode, region)
    if plot:
        fig, ax = _plot_fit(x_data, y_data, result, region, mode,
                            'Raman spectrum', display_plot)
        return fig, ax, result
    return result


def get_ratio_from_file(
        file_path,
        baseline='iasls',
        window_length=11,
        polyorder=3,
        region=None,
        plot=False,
        display_plot=True
):
    '''Get the I_D/I_G ratio (conventional method) from a Raman spectrum file.

    Uses the maximum intensity in the D band vs the G band, without peak fitting.
    Returns the ratio, or (fig, ax, ratio) when ``plot=True``.'''
    x_data, y_data = conv_preprocess(file_path=file_path, baseline=baseline,
                                     window_length=window_length, polyorder=polyorder, region=region)
    all_data = np.stack((x_data, y_data), axis=-1)
    D_band_condition = (all_data[:, 0] > 1300) & (all_data[:, 0] < 1390)
    G_band_condition = (all_data[:, 0] > 1500) & (all_data[:, 0] < 1670)
    D_band = all_data[D_band_condition]
    G_band = all_data[G_band_condition]
    D_intensity = D_band[:, 1].max()
    G_intensity = G_band[:, 1].max()
    id_ig = D_intensity / G_intensity
    if plot:
        textstr = f"$I_D/I_G$ = {id_ig:.3f}"
        props = dict(boxstyle='round', facecolor='gainsboro', alpha=0.5)
        fig, ax = plt.subplots()
        ax.plot(x_data, y_data, label='Raman spectrum')
        ax.axvspan(1300, 1390, alpha=0.5, color='r', label='D band')
        ax.axvspan(1500, 1620, alpha=0.5, color='g', label='G band')
        ax.set_xlabel("Wavenumber (cm$^-1$)", fontsize=14)
        ax.set_ylabel("Normalized Intensity (a.u.)", fontsize=14)
        ax.set_title(os.path.basename(file_path))
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', bbox=props)
        ax.legend()
        if display_plot:
            plt.show()
        return fig, ax, id_ig
    return id_ig
