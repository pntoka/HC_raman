# HC_raman

Python package for processing Raman spectroscopy data of **hard carbon** samples and
extracting the **I_D/I_G ratio** — a standard measure of structural disorder in carbon
materials.

Give it a raw Renishaw Raman file (or raw wavenumber/intensity arrays) and it will:

1. **Preprocess** the spectrum (despike → Savitzky-Golay denoise → crop → baseline
   correction → normalise → crop to the fit region), using
   [RamanSPy](https://ramanspy.readthedocs.io/).
2. **Fit peaks** in the first-order region with a configurable multi-peak
   Lorentzian/Gaussian model plus a background term
   ([lmfit](https://lmfit.github.io/lmfit-py/)). The band scheme follows
   [Sadezky et al., *Carbon* 43 (2005) 1731](https://doi.org/10.1016/j.carbon.2005.02.018);
   the config layout comes from [MyPyDavid/raman-fitting](https://github.com/MyPyDavid/raman-fitting).
3. **Compute I_D/I_G** — from the fitted band areas, for a single spectrum or for a
   whole sample at once, or with a simpler conventional band-intensity method.

Data on the quality of a spectrum (signal-to-noise and background
magnitude) can alsoe be generated (see [Data quality](#data-quality)).

## Installation

First, clone this repository:

```bash
git clone https://github.com/pntoka/HC_raman.git
cd HC_raman
```

In the cloned directory run the pip install commands.

```bash
pip install .
# or, for development:
pip install -e .
```

Requires Python ≥ 3.11. Dependencies: `numpy`, `lmfit`, `rosettasciio`, `ramanspy`,
`matplotlib`.

## Three ways to get I_D/I_G

| Method | Function | How the ratio is computed |
|--------|----------|---------------------------|
| **Sample fitting** (recommended) | `peak_fit_sample_from_data`, `peak_fit_sample_from_files` | Fits every measurement of one sample at once with the band shapes shared; ratio = summed `D_area / G_area`. |
| **Peak fitting** | `peak_fit_from_file`, `peak_fit_from_data` | Fits one spectrum; ratio = fitted `D_area / G_area` via `get_id_ig`. |
| **Conventional** | `get_ratio_from_file` | Max intensity in the D band (1300–1390 cm⁻¹) / max intensity in the G band (1500–1670 cm⁻¹). No fitting. |

Sample fitting is recommended because a single spectrum does not pin down the G band
area well enough for a per-spectrum ratio to be stable. Measurements of one sample are
the same material at different spots, so only the band *areas* should differ between
them and not the positions and widths. Sharing these parameters constrains the fit several times as much data, and gives one representative ratio per sample.

## Usage

```python
from hc_raman import (
    peak_fit_sample_from_data,
    peak_fit_sample_from_files,
    peak_fit_from_data,
    peak_fit_from_file,
    get_ratio_from_file,
    get_id_ig,
    params_at_bound,
)

# 1) One ratio per sample, from all of its measurements at once
measurements = [(wavenumber_1, intensity_1), (wavenumber_2, intensity_2), ...]
result, summary = peak_fit_sample_from_data(measurements, mode="3peaks")
summary["id_ig"]        # the sample's I_D/I_G
summary["per_spectrum"] # I_D/I_G of each measurement; their spread is spot-to-spot variation
summary["shared"]       # {'G': {'center': ..., 'fwhm': ...}, 'D': {...}, ...}

# ...or straight from files
result, summary = peak_fit_sample_from_files(["spot1.wdf", "spot2.wdf"])

# 2) A single spectrum, from a file or from arrays already in memory
result = peak_fit_from_file("sample.wdf", baseline="iasls", mode="5peaks")
result = peak_fit_from_data(wavenumber, intensity, mode="5peaks")
print("ID/IG =", get_id_ig(result))
print("stuck parameters:", params_at_bound(result))   # should be empty

# ...with a plot of the fit (returns the figure objects too)
fig, ax, result = peak_fit_from_file("sample.wdf", mode="5peaks", plot=True)

# 3) Conventional D/G ratio (no peak fitting)
id_ig = get_ratio_from_file("sample.wdf", baseline="iasls")
```

When `plot=True`, the single-spectrum fitting functions return `(fig, ax, result)` and
the conventional function returns `(fig, ax, ratio)`. Pass `display_plot=False` to build
the figure without calling `plt.show()` (useful for saving figures in a script).

## Checking a fit

Overlapping broad bands make it easy for a fit to look excellent and still be
meaningless: R² stays above 0.98 whatever the bands do, because D and G carry almost all
the variance. Check these instead. Single-spectrum results carry all three as
attributes; `params_at_bound` is also a function, usable on any result including a
sample fit.

| Check | Meaning |
|-------|---------|
| `result.params_at_bound` | Parameters left sitting on a bound. This should be empty and any parameter that is here was stopped by the box rather than by the data, so that band is not describing what it is named for. |
| `result.basin_spread` | Spread of I_D/I_G across the starts that reached within 1% of the best chi-square. Near zero means every start found the same optimum; `NaN` when only one start ran. |
| `result.n_basins` | Distinct ratios the starts reached. Above 1 means several equally good solutions exist and the ratio depends on where the fit began. |

## Configuration

### Peak models (`mode`)

Defined in [`hc_raman/spectrum_config/peaks_config.toml`](hc_raman/spectrum_config/peaks_config.toml):

| `mode` | Peaks |
|--------|-------|
| `1peak` | G |
| `2peaks` | G + D |
| `3peaks` | G + D + D3 |
| `4peaks` | G + D + D3 + D4 |
| `5peaks` *(default)* | G + D + D2 + D3 + D4 |
| `6peaks` | G + D + D2 + D3 + D4 + D5 |

Each peak has a line shape (`Lorentzian` or `Gaussian`) and
initial/bounded values for center, sigma and amplitude. To adjust peak positions/bounds
or add new peaks, edit the TOML — no code changes needed.

Two things to know before changing the first-order bands:

- **D3 is Gaussian**, following Sadezky et al.; the others are Lorentzian.
- **D2 and D4 have a fixed center and width, and D3 a fixed width** (`vary = false`).
  Data of this quality does not determine them, so letting them float costs
  reproducibility without buying information. Their *amplitudes* are still fitted.

More bands raise R² but widen the spread of I_D/I_G between repeat measurements, so
`5peaks` is not automatically the better choice.

### Background

The `[first_order.background]` section of the same TOML adds an explicit background to
the model under the `bkg_` prefix, with `type` set to `linear` (the default), `constant`
or `none`, and bounds for each of its parameters.

A baseline-corrected spectrum does not reach zero inside the fit window. Without a
background term that residue is absorbed by whichever band is broadest. This is in practice D4, which then ends up pinned against its bounds acting as a sloping background rather than a band. The bounds in the TOML assume the normalised spectrum `preprocess` produces, whose tallest band is 1.0.

### Fitting procedure (`procedure`)

How the optimiser is started. The model is identical in every case; what changes is how
much confidence you can have that the answer does not depend on where the fit began.

| `procedure` | What it does |
|-------------|--------------|
| `single` | One fit from the initial values in the TOML. Fastest. |
| `staged` | Fits G and D first, then adds the remaining bands one at a time, each stage warm-started from the previous one. |
| `multistart` | Best of `n_starts` single fits from randomised starts. |
| `staged_multistart` *(default)* | Best of `n_starts` staged cascades from randomised starts. |

`n_starts` (default 6) and `seed` (default 0) control the randomised starts; the seed is
fixed, so a given spectrum always returns the same answer. With the shipped configuration
all four agree, and the multi-start ones cost roughly 10–15× the time for the same number —
what they add is the `basin_spread` and `n_basins` diagnostics, which catch the case
where loosened bounds or an extra band make the fit ambiguous again. Use
`procedure="single"` for speed once a configuration is known to be sound.

### Baseline (`baseline`)

One of `iasls`, `airpls`, `iarpls`, `asls` (all from RamanSPy). The peak-fitting entry
points default to `iasls`.

### Region (`region`)

Named wavenumber ranges from
[`hc_raman/spectrum_config/spectrum_regions.toml`](hc_raman/spectrum_config/spectrum_regions.toml),
e.g. `first_order` (900–1900 cm⁻¹, the default for fitting), `second_order`, `full`, etc.
Peak fitting is currently configured for the `first_order` region.

`preprocess` crops to `baseline_window` (800–2150 cm⁻¹) *before* correcting the baseline,
so the algorithm is not dragged by the second-order bands above ~2200 cm⁻¹ that no model
accounts for, then crops to the fit region. `g_band` (1500–1620), `d_band` (1300–1390)
and `mid` (1850–2150, signal-free) are used by `get_quality_metrics`.

## Data quality

`get_quality_metrics` measures how good a *measurement* is, as opposed to how well the
model fitted it. Use it to screen a batch before trusting the ratios it produces.

```python
from hc_raman import get_quality_metrics

metrics = get_quality_metrics("sample.wdf", baseline="iasls")
print(metrics)
# {'snr': 46.6, 'signal': 861.4, 'noise': 18.2,
#  'baseline_ratio': 0.61, 'baseline_tilt': 0.21, 'n_spike_points': 2}

# from arrays you already have — any instrument or file format, either axis direction
metrics = get_quality_metrics(wavenumber=wavenumber, intensity=intensity)

# inspect the recovered baseline and the regions the metrics were taken from
fig, ax, metrics = get_quality_metrics("sample.wdf", plot=True)
```

| Metric | Meaning |
|--------|---------|
| `snr` | `signal / noise`. The headline quality number. |
| `signal` | Height of the G band above the baseline. |
| `noise` | Spread of the spectrum in a signal-free window (1850–2150 cm⁻¹ by default), after removing a linear trend. |
| `baseline_ratio` | Mean baseline over the fit region divided by `signal` — how much of the measured intensity is background rather than Raman scattering. |
| `baseline_tilt` | Peak-to-peak baseline over the fit region divided by `signal` — how much the background varies under the bands, i.e. how much the baseline algorithm had to invent. |
| `n_spike_points` | Points altered by despiking. One cosmic ray typically alters one to three points. |

`snr`, `baseline_ratio` and `baseline_tilt` are dimensionless and so can be compared
across instruments and units. `signal` and `noise` are in whatever intensity units you
supplied, so compare them only within one instrument and one set of acquisition
settings.

The noise is measured *before* Savitzky-Golay denoising (smoothing would remove the
noise being measured) and without normalisation, so `baseline_ratio` includes any
constant offset such as a detector dark level. The windows are named regions from
`spectrum_regions.toml` and can be changed with `signal_region=` and `noise_region=` —
useful if your acquisition does not reach 1850–2150 cm⁻¹, which raises a `ValueError`
naming the region and your spectrum's range.

Pass the same `baseline=` you intend to fit with, so the baseline metrics describe the
curve the fit will actually see.

## API reference

- `preprocess(file_path=None, wavenumber=None, intensity=None, baseline='iasls', ...)` —
  full preprocessing pipeline (baseline corrected, normalised, cropped) used by the
  fitting functions.
- `conv_preprocess(...)` — preprocessing for the conventional ratio (no normalisation,
  optional crop).
- `get_quality_metrics(file_path=None, wavenumber=None, intensity=None, ...)` — measurement
  quality metrics (signal-to-noise, background magnitude) as a dict.
- `peak_fit_sample_from_data(measurements, ...)` /
  `peak_fit_sample_from_files(file_paths, ...)` — preprocess + fit a whole sample;
  return `(result, summary)`.
- `fit_sample(spectra, mode, region)` — the sample fit on already preprocessed spectra.
- `peak_fit_from_file(...)` / `peak_fit_from_data(...)` — preprocess + fit one spectrum;
  return an lmfit result.
- `get_ratio_from_file(...)` — conventional I_D/I_G without fitting.
- `get_id_ig(result)` — extract I_D/I_G from a fitted result.
- `params_at_bound(result)` — parameters left resting on a bound.
- `fit_model(x, y, mode, region, procedure, n_starts, seed)` /
  `build_lmfit_model(mode, region)` — lower-level fitting helpers.
