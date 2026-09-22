import os
import numpy as np
from hc_raman.utils import get_spectrum_region
import tomllib
from lmfit import Parameters, minimize
from lmfit.models import (LorentzianModel, GaussianModel, LinearModel, ConstantModel)
from hc_raman.preprocess import preprocess, conv_preprocess
import matplotlib.pyplot as plt

_PEAK_MODELS = {"Lorentzian": LorentzianModel, "Gaussian": GaussianModel}

# Background shapes a region may declare, with the lmfit parameters each one takes.
_BACKGROUND_MODELS = {"constant": (ConstantModel, ("c",)),
                      "linear": (LinearModel, ("slope", "intercept"))}

BACKGROUND_PREFIX = "bkg_"

# Order in which the staged procedures introduce the bands. G and D carry nearly all
# the signal, so a fit of those two lands in the same place whatever the starting
# point; every later stage then begins next to its own optimum. Peaks not listed here
# are added last, in the order the mode names them.
_STAGE_ORDER = ("G", "D", "D3", "D4", "D2", "D5", "Si1")

# Starts whose chi-square is within this fraction of the best one count as having found
# the same optimum, for the purpose of measuring how flat that optimum is.
_BASIN_TOL = 0.01

# A parameter this close to a bound, as a fraction of the bound's range, counts as
# resting on it.
_BOUND_TOL = 1e-3

# Randomised starts scatter amplitudes by this factor around their configured value,
# and every other bounded parameter across this fraction of its range.
_AMPLITUDE_JITTER = (0.25, 4.0)
_BOX_JITTER = (0.15, 0.85)

_FIT_KWS = {'ftol': 1e-12, 'xtol': 1e-12}

_X_LABEL = "Wavenumber (cm$^-1$)"
_Y_LABEL = "Normalized Intensity (a.u.)"


def get_peaks_config():
    with open(
        os.path.join(os.path.dirname(__file__), "spectrum_config/peaks_config.toml"),
        "rb",
    ) as file:
        peaks_config = tomllib.load(file)
    return peaks_config


def _build_peak_model(peak, peak_config):
    """One lmfit peak model, with the initial values and bounds from its TOML entry."""
    peak_type = peak_config["peak_type"]
    if peak_type not in _PEAK_MODELS:
        raise ValueError(
            f"Peak '{peak}' declares peak_type = '{peak_type}', which is not one of "
            f"{sorted(_PEAK_MODELS)}. Check spectrum_config/peaks_config.toml."
        )
    peak_model = _PEAK_MODELS[peak_type](prefix=f"{peak}_")
    for hint, hint_values in peak_config["param_hints"].items():
        peak_model.set_param_hint(f"{peak}_{hint}", **hint_values)
    return peak_model


def _build_background_model(region_config):
    """
    The background model a region declares, or None when it declares 'none'.

    Without the background term the D4 band ends up acting as a sloping background,
    instead of a band, which is how the D4 amplitude ends up pinned against its bounds
    """
    background_config = region_config.get("background", {})
    background_type = background_config.get("type", "none")
    if background_type == "none":
        return None
    if background_type not in _BACKGROUND_MODELS:
        raise ValueError(
            f"Background type '{background_type}' is not one of "
            f"{sorted(_BACKGROUND_MODELS) + ['none']}. "
            "Check spectrum_config/peaks_config.toml."
        )
    background_class, param_names = _BACKGROUND_MODELS[background_type]
    background_model = background_class(prefix=BACKGROUND_PREFIX)
    for param_name in param_names:
        if param_name in background_config:
            background_model.set_param_hint(f"{BACKGROUND_PREFIX}{param_name}",
                                            **background_config[param_name])
    return background_model


def _build_model_from_peaks(peaks, region_config):
    """Sum the named bands and the region's background term into one model."""
    model = None
    for peak in peaks:
        peak_model = _build_peak_model(peak, region_config["peaks"][peak])
        model = peak_model if model is None else model + peak_model

    background_model = _build_background_model(region_config)
    if background_model is not None:
        model = model + background_model

    return model, model.make_params()


def get_mode_peaks(mode="5peaks", region="first_order"):
    """The band names a mode is made of, in the order the TOML lists them."""
    return get_peaks_config()[region]["models"][mode].split("+")


def build_lmfit_model(mode="5peaks", region="first_order"):
    """
    Build the composite model and parameters for one mode.

    The model is the sum of the bands named by ``mode`` in the region's ``models``
    table, plus an explicit background term with the ``bkg_`` prefix when the region
    declares one. Initial values and bounds come from the TOML, so adjusting the fit
    needs no code change.

    The background bounds in the TOML assume a spectrum normalised so that the tallest
    band is 1.0, which is what :func:`hc_raman.preprocess` produces.
    """
    peaks_config = get_peaks_config()
    return _build_model_from_peaks(get_mode_peaks(mode, region), peaks_config[region])


def params_at_bound(result, tol=_BOUND_TOL):
    """
    Names of the fitted parameters left sitting on one of their bounds.

    A parameter resting on a bound means the optimiser was stopped by the box rather
    than by the data: the band is no longer describing the feature it is named for, and
    the fit wanted to keep going. An empty list is what a healthy fit looks like, so
    this is worth checking before trusting a ratio.
    """
    stuck = []
    for name, param in result.params.items():
        if not param.vary or param.value is None:
            continue
        low, high = param.min, param.max
        if not (np.isfinite(low) and np.isfinite(high)) or high <= low:
            continue
        if min(param.value - low, high - param.value) / (high - low) < tol:
            stuck.append(name)
    return stuck


def _fit_once(model, params, x_data, y_data):
    return model.fit(y_data, params, x=x_data, method='least_squares',
                     fit_kws=_FIT_KWS)


def _stage_peaks(peaks):
    """Cumulative groups of bands, strongest first, one more band per stage."""
    ordered = ([peak for peak in _STAGE_ORDER if peak in peaks]
               + [peak for peak in peaks if peak not in _STAGE_ORDER])
    stages = [ordered[:2]]
    for peak in ordered[2:]:
        stages.append(stages[-1] + [peak])
    return stages


def _jitter_params(params, rng):
    """Move every varying parameter to a random start inside its bounds."""
    for name, param in params.items():
        if not param.vary:
            continue
        if name.endswith("_amplitude") and param.value:
            # Amplitude bounds are far wider than any real band, so scatter around the
            # configured starting value rather than across the whole box.
            value = param.value * rng.uniform(*_AMPLITUDE_JITTER)
        elif np.isfinite(param.min) and np.isfinite(param.max):
            value = param.min + (param.max - param.min) * rng.uniform(*_BOX_JITTER)
        else:
            continue
        param.set(value=float(np.clip(value, param.min, param.max)))


def _warm_start(params, result):
    """Start every shared varying parameter where the previous stage left it."""
    for name, param in result.params.items():
        if name in params and params[name].vary:
            params[name].set(value=float(np.clip(param.value, params[name].min,
                                                 params[name].max)))


def _run_single(x_data, y_data, mode, region, rng=None):
    """One fit of the whole model, from the TOML values or from a random start."""
    model, params = build_lmfit_model(mode, region)
    if rng is not None:
        _jitter_params(params, rng)
    return _fit_once(model, params, x_data, y_data)


def _run_staged(x_data, y_data, mode, region, rng=None):
    """
    Fit the bands the data constrains best first, then let the weaker ones in.

    Only the first stage is randomised: once G and D are placed, every later stage is
    warm-started from the one before it, which is what keeps the optimiser out of the
    distant local minima a cold start can reach.
    """
    region_config = get_peaks_config()[region]
    result = None
    for stage_index, stage in enumerate(_stage_peaks(get_mode_peaks(mode, region))):
        model, params = _build_model_from_peaks(stage, region_config)
        if result is not None:
            _warm_start(params, result)
        if rng is not None and stage_index == 0:
            _jitter_params(params, rng)
        result = _fit_once(model, params, x_data, y_data)
    return result


def _id_ig_or_nan(result):
    try:
        return get_id_ig(result)
    except KeyError:
        return float("nan")


def _attach_diagnostics(result, ratios=(), chisqrs=()):
    """Record which parameters ended on a bound and how flat the optimum was."""
    result.params_at_bound = params_at_bound(result)
    ratios, chisqrs = np.asarray(ratios, float), np.asarray(chisqrs, float)
    finite = ratios[np.isfinite(ratios)]
    result.n_basins = int(len(np.unique(np.round(finite, 2)))) if len(finite) else 1
    result.basin_spread = float("nan")
    if len(chisqrs) > 1:
        near = ratios[chisqrs <= chisqrs.min() * (1 + _BASIN_TOL)]
        near = near[np.isfinite(near)]
        if len(near) > 1 and near.mean():
            result.basin_spread = float(np.ptp(near) / near.mean())
        elif len(near):
            result.basin_spread = 0.0
    return result


def _run_best_of(run_one, x_data, y_data, mode, region, n_starts, seed):
    """Repeat a fitting procedure from reproducible random starts, keep the best."""
    if n_starts < 1:
        raise ValueError(f"n_starts must be at least 1, got {n_starts}.")
    rng = np.random.default_rng(seed)
    best, ratios, chisqrs = None, [], []
    for start in range(n_starts):
        if start == 0:
            # The un-jittered start is the canonical one. If it cannot run, the problem
            # is the model or the data rather than an unlucky starting point, so let the
            # error through instead of reporting it as a failure to converge.
            result = run_one(x_data, y_data, mode, region, None)
        else:
            try:
                result = run_one(x_data, y_data, mode, region, rng)
            except Exception:
                continue
        chisqrs.append(result.chisqr)
        ratios.append(_id_ig_or_nan(result))
        if best is None or result.chisqr < best.chisqr:
            best = result
    return _attach_diagnostics(best, ratios, chisqrs)


def fit_single(x_data, y_data, mode="5peaks", region="first_order"):
    """One fit of the whole model from the initial values in the TOML."""
    return _attach_diagnostics(_run_single(x_data, y_data, mode, region))


def fit_staged(x_data, y_data, mode="5peaks", region="first_order"):
    """Fit G and D first, then introduce the remaining bands one stage at a time."""
    return _attach_diagnostics(_run_staged(x_data, y_data, mode, region))


def fit_multistart(x_data, y_data, mode="5peaks", region="first_order",
                   n_starts=6, seed=0):
    """Best of ``n_starts`` whole-model fits from randomised starting points."""
    return _run_best_of(_run_single, x_data, y_data, mode, region, n_starts, seed)


def fit_staged_multistart(x_data, y_data, mode="5peaks", region="first_order",
                          n_starts=6, seed=0):
    """Best of ``n_starts`` staged cascades from randomised starting points."""
    return _run_best_of(_run_staged, x_data, y_data, mode, region, n_starts, seed)


_PROCEDURES = {
    "single": fit_single,
    "staged": fit_staged,
    "multistart": fit_multistart,
    "staged_multistart": fit_staged_multistart,
}


def fit_model(x_data, y_data, mode="5peaks", region="first_order",
              procedure="staged_multistart", n_starts=6, seed=0):
    """
    Fit the peaks to the Raman spectrum.

    Parameters
    ----------
    x_data, y_data : numpy.ndarray
        Preprocessed wavenumber and intensity.
    mode, region : str, optional
        Which band combination to fit, and from which region of the TOML.
    procedure : str, optional
        How the optimiser is started. The model is the same in every case; what
        changes is how much confidence you can have that the answer does not depend
        on where the fit began.

        ``'single'``
            One fit from the initial values in the TOML. Fastest, and what earlier
            versions of this package did.
        ``'staged'``
            Fit G and D first, then add the remaining bands one at a time, each stage
            warm-started from the previous one.
        ``'multistart'``
            Best of ``n_starts`` single fits from randomised starts.
        ``'staged_multistart'`` (default)
            Best of ``n_starts`` staged cascades from randomised starts.
    n_starts : int, optional
        Number of starting points for the multi-start procedures. Ignored by
        ``'single'`` and ``'staged'``.
    seed : int, optional
        Seed for the randomised starts, so a given spectrum always gives the same
        answer.

    Returns
    -------
    result : lmfit.model.ModelResult
        With three extra attributes:

        ``params_at_bound``
            Parameters left resting on a bound. Should be empty; anything here means
            the fit was stopped by the box rather than by the data.
        ``basin_spread``
            Relative spread of I_D/I_G across the starts that reached within 1% of the
            best chi-square. Near zero means every start found the same optimum. NaN
            when only one start was run.
        ``n_basins``
            How many distinct ratios the starts reached.
    """
    if procedure not in _PROCEDURES:
        raise ValueError(
            f"Unknown procedure '{procedure}'; choose one of {sorted(_PROCEDURES)}."
        )
    if procedure in ("single", "staged"):
        return _PROCEDURES[procedure](x_data, y_data, mode, region)
    return _PROCEDURES[procedure](x_data, y_data, mode, region,
                                  n_starts=n_starts, seed=seed)


def get_id_ig(result):
    """Compute the I_D/I_G ratio from the fitted integrated band areas"""
    p = result.params
    return float(p['D_amplitude'].value / p['G_amplitude'].value)


def _is_per_spectrum(name):
    """Band areas and the background may differ between measurements; shapes may not."""
    return name.endswith("_amplitude") or name.startswith(BACKGROUND_PREFIX)


def _spectrum_name(name, index):
    return f"{name}_s{index}" if _is_per_spectrum(name) else name


def _spectrum_params(model, params, index):
    """The parameters of one measurement, as a normal lmfit Parameters object."""
    single = model.make_params()
    for name in model.param_names:
        single[name].set(value=params[_spectrum_name(name, index)].value)
    single.update_constraints()
    return single


def fit_sample(spectra, mode="5peaks", region="first_order", titles=None,
               plot=False, display_plot=True):
    """
    Fit every measurement of one sample at once, with the band shapes shared.

    The measurements are the same material at different spots, so what may legitimately
    differ between them is how much of each band is present, not where the bands sit or
    how wide they are. Sharing the centres and widths across the measurements says that,
    and constrains the shape parameters with several times as much data, so they cannot
    wander from spot to spot and drag the band areas with them.

    Parameters
    ----------
    spectra : sequence of (x_data, y_data)
        Preprocessed measurements of one sample.
    mode, region : str, optional
        As for :func:`fit_model`.
    titles : sequence of str, optional
        A name per measurement, used when plotting.
    plot : bool, optional
        If True, also plot every measurement — see :func:`plot_sample_fit`.
    display_plot : bool, optional
        If True (default), show the figure when ``plot`` is True.

    Returns
    -------
    result : lmfit.minimizer.MinimizerResult
        The joint fit, or ``(fig, axes, result, summary)`` when ``plot=True``. Its
        parameters carry a ``_s<index>`` suffix where they belong to a single
        measurement, and no suffix where they are shared. It also carries the
        measurements and the model it was fitted with, so :func:`plot_sample_fit` can
        redraw it later.
    summary : dict
        ``id_ig``
            The sample's I_D/I_G, from the band areas summed over the measurements.
        ``per_spectrum``
            I_D/I_G for each measurement on its own, from the shared-shape fit. Their
            spread is the spot-to-spot variation left once the shapes are pinned down.
        ``shared``
            ``{band: {'center': ..., 'fwhm': ...}}`` for the shared band shapes.
    """
    if not spectra:
        raise ValueError("fit_sample needs at least one spectrum.")
    peaks = get_mode_peaks(mode, region)
    if not {"D", "G"} <= set(peaks):
        raise ValueError(f"Mode '{mode}' fits {peaks}; a sample ratio needs both a D "
                         "and a G band.")
    model, template = build_lmfit_model(mode, region)

    params = Parameters()
    for name in model.param_names:
        source = template[name]
        names = ([_spectrum_name(name, index) for index in range(len(spectra))]
                 if _is_per_spectrum(name) else [name])
        for param_name in names:
            params.add(param_name, value=source.value, min=source.min,
                       max=source.max, vary=source.vary)

    def residual(params):
        return np.concatenate([
            model.eval(x=x_data, **{name: params[_spectrum_name(name, index)].value
                                    for name in model.param_names}) - y_data
            for index, (x_data, y_data) in enumerate(spectra)
        ])

    result = minimize(residual, params, method='least_squares', **_FIT_KWS)

    per_spectrum = [_spectrum_params(model, result.params, index)
                    for index in range(len(spectra))]
    areas = {peak: [p[f"{peak}_amplitude"].value for p in per_spectrum]
             for peak in peaks}
    summary = {
        "id_ig": float(sum(areas["D"]) / sum(areas["G"])),
        "per_spectrum": [float(d / g) for d, g in zip(areas["D"], areas["G"])],
        "shared": {peak: {"center": per_spectrum[0][f"{peak}_center"].value,
                          "fwhm": per_spectrum[0][f"{peak}_fwhm"].value}
                   for peak in peaks},
    }

    # Keep what a later plot needs: a MinimizerResult knows nothing about the data it
    # was fitted to, and the wrappers do not hand the preprocessed spectra back.
    result.spectra = spectra
    result.model = model
    result.peaks = peaks
    result.titles = titles
    result.id_ig = summary["id_ig"]

    if plot:
        fig, axes = plot_sample_fit(result, display_plot=display_plot)
        return fig, axes, result, summary
    return result, summary


def peak_fit_sample_from_data(
    measurements,
    baseline='iasls',
    window_length=21,
    polyorder=3,
    region="first_order",
    mode="5peaks",
    plot=False,
    display_plot=True,
):
    '''Preprocess and jointly fit every measurement of one sample.

    ``measurements`` is a sequence of ``(wavenumber, intensity)`` array pairs.
    Returns ``(result, summary)`` as :func:`fit_sample` does, or
    ``(fig, axes, result, summary)`` when ``plot=True``.'''
    spectra = [preprocess(wavenumber=wavenumber, intensity=intensity, baseline=baseline,
                          region=region, window_length=window_length, polyorder=polyorder)
               for wavenumber, intensity in measurements]
    return fit_sample(spectra, mode, region, plot=plot, display_plot=display_plot)


def peak_fit_sample_from_files(
    file_paths,
    baseline='iasls',
    window_length=21,
    polyorder=3,
    region="first_order",
    mode="5peaks",
    plot=False,
    display_plot=True,
):
    '''Preprocess and jointly fit every measurement file of one sample.

    Returns ``(result, summary)`` as :func:`fit_sample` does, or
    ``(fig, axes, result, summary)`` when ``plot=True``. Plots are titled with each
    file's name.'''
    spectra = [preprocess(file_path=file_path, baseline=baseline, region=region,
                          window_length=window_length, polyorder=polyorder)
               for file_path in file_paths]
    titles = [os.path.basename(file_path) for file_path in file_paths]
    return fit_sample(spectra, mode, region, titles=titles, plot=plot,
                      display_plot=display_plot)


def _draw_fit(ax, x_data, y_data, comps, best_fit, peaks, id_ig, rsquared, title):
    """Draw one fitted spectrum onto an existing axis."""
    textstr = f"$I_D/I_G$ = {id_ig:.3f} \n R$^2$ = {rsquared:.4f}"
    props = dict(boxstyle='round', facecolor='gainsboro', alpha=0.5)
    for peak in peaks:
        ax.plot(x_data, comps[f'{peak}_'], linestyle='--', label=peak)
    if BACKGROUND_PREFIX in comps:
        ax.plot(x_data, comps[BACKGROUND_PREFIX], linestyle=':', c='grey',
                label="background")
    ax.scatter(x_data, y_data, c='k', label="raw data", s=1)
    ax.plot(x_data, best_fit, label="fit", linestyle='-', c='r')
    ax.set_title(title)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)


def _plot_fit(x_data, y_data, result, region, mode, title, display_plot=True):
    """Plot the fitted peaks, the composite fit and the raw data. Returns (fig, ax)."""
    fig, ax = plt.subplots()
    _draw_fit(ax, x_data, y_data, result.eval_components(x=x_data), result.best_fit,
              get_mode_peaks(mode, region), get_id_ig(result), result.rsquared, title)
    ax.set_xlabel(_X_LABEL, fontsize=14)
    ax.set_ylabel(_Y_LABEL, fontsize=14)
    ax.legend()
    if display_plot:
        plt.show()
    return fig, ax


def _spectrum_fit(result, index):
    """Components, composite curve, ratio and R-squared for one measurement."""
    x_data, y_data = result.spectra[index]
    params = _spectrum_params(result.model, result.params, index)
    comps = result.model.eval_components(params=params, x=x_data)
    best_fit = result.model.eval(params=params, x=x_data)
    rsquared = 1 - (np.sum((y_data - best_fit) ** 2)
                    / np.sum((y_data - y_data.mean()) ** 2))
    id_ig = params['D_amplitude'].value / params['G_amplitude'].value
    return x_data, y_data, comps, best_fit, id_ig, rsquared


def plot_sample_fit(result, index=None, titles=None, display_plot=True):
    """
    Plot a sample fit: every measurement as its own panel, or one of them alone.

    Takes only the result of :func:`fit_sample`, which carries the measurements and the
    model it was fitted with, so a fit can be plotted at any point after it was made.

    Parameters
    ----------
    result : lmfit.minimizer.MinimizerResult
        As returned by :func:`fit_sample` or either ``peak_fit_sample_*`` wrapper.
    index : int, optional
        Which measurement to draw. The default, None, draws all of them in a grid.
    titles : sequence of str, optional
        One title per measurement. Defaults to the names the fit was given, or
        ``measurement 1..n``.
    display_plot : bool, optional
        If True (default), call ``plt.show()``.

    Returns
    -------
    (fig, axes) for the whole sample, where ``axes`` is a 2-D array, or (fig, ax) for a
    single ``index``.
    """
    if not hasattr(result, "spectra"):
        raise ValueError("plot_sample_fit needs a result from fit_sample; this one does "
                         "not carry the measurements it was fitted to.")
    n_spectra = len(result.spectra)
    labels = (titles or result.titles
              or [f"measurement {i + 1}" for i in range(n_spectra)])

    if index is not None:
        if not -n_spectra <= index < n_spectra:
            raise IndexError(f"index {index} is out of range for a sample with "
                             f"{n_spectra} measurement(s).")
        index %= n_spectra          # the parameters are named with positive indices
        fig, ax = plt.subplots()
        x_data, y_data, comps, best_fit, id_ig, rsquared = _spectrum_fit(result, index)
        _draw_fit(ax, x_data, y_data, comps, best_fit, result.peaks, id_ig, rsquared,
                  labels[index])
        ax.set_xlabel(_X_LABEL, fontsize=14)
        ax.set_ylabel(_Y_LABEL, fontsize=14)
        ax.legend()
        if display_plot:
            plt.show()
        return fig, ax

    n_cols = min(n_spectra, 2)
    n_rows = int(np.ceil(n_spectra / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, squeeze=False,
                             figsize=(5 * n_cols, 3.6 * n_rows))
    for position, ax in enumerate(axes.ravel()):
        if position >= n_spectra:
            ax.axis("off")
            continue
        x_data, y_data, comps, best_fit, id_ig, rsquared = _spectrum_fit(result, position)
        _draw_fit(ax, x_data, y_data, comps, best_fit, result.peaks, id_ig, rsquared,
                  labels[position])
    # One legend is enough: every panel draws the same bands. It goes top right, out of
    # the way of the ratio box each panel puts top left.
    axes.ravel()[0].legend(fontsize=8, ncol=2, loc='upper right')
    fig.suptitle(f"sample $I_D/I_G$ = {result.id_ig:.3f}")
    fig.supxlabel(_X_LABEL)
    fig.supylabel(_Y_LABEL)
    fig.tight_layout()
    if display_plot:
        plt.show()
    return fig, axes


def peak_fit_from_file(
    file_path,
    baseline='iasls',
    window_length=21,
    polyorder=3,
    region="first_order",
    mode="5peaks",
    procedure="staged_multistart",
    n_starts=6,
    seed=0,
    plot=False,
    display_plot=True
):
    '''Fit peaks to a Raman spectrum from a file using RamanSPy for preprocessing.

    See :func:`fit_model` for ``procedure``, ``n_starts`` and ``seed``.

    Returns the lmfit result, or (fig, ax, result) when ``plot=True``.'''
    x_data, y_data = preprocess(file_path=file_path, baseline=baseline, region=region,
                                window_length=window_length, polyorder=polyorder)
    result = fit_model(x_data, y_data, mode, region,
                       procedure=procedure, n_starts=n_starts, seed=seed)
    if plot:
        fig, ax = _plot_fit(x_data, y_data, result, region, mode,
                            os.path.basename(file_path), display_plot)
        return fig, ax, result
    return result


def peak_fit_from_data(
    wavenumber,
    intensity,
    baseline='iasls',
    window_length=21,
    polyorder=3,
    region="first_order",
    mode="5peaks",
    procedure="staged_multistart",
    n_starts=6,
    seed=0,
    plot=False,
    display_plot=True
):
    '''Fit peaks to a Raman spectrum from data arrays using RamanSPy for preprocessing.

    See :func:`fit_model` for ``procedure``, ``n_starts`` and ``seed``.

    Returns the lmfit result, or (fig, ax, result) when ``plot=True``.'''
    x_data, y_data = preprocess(wavenumber=wavenumber, intensity=intensity, baseline=baseline,
                                region=region, window_length=window_length, polyorder=polyorder)
    result = fit_model(x_data, y_data, mode, region,
                       procedure=procedure, n_starts=n_starts, seed=seed)
    if plot:
        fig, ax = _plot_fit(x_data, y_data, result, region, mode,
                            'Raman spectrum', display_plot)
        return fig, ax, result
    return result


def get_ratio_from_file(
        file_path,
        baseline='iasls',
        window_length=21,
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
