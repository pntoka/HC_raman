# HC_raman

Python package for processing Raman spectroscopy data of **hard carbon** samples and
extracting the **I_D/I_G ratio** — a standard measure of structural disorder in carbon
materials.

Give it a raw Renishaw Raman file (or raw wavenumber/intensity arrays) and it will:

1. **Preprocess** the spectrum (despike → Savitzky-Golay denoise → normalise → baseline
   correction → crop to a region of interest), using
   [RamanSPy](https://ramanspy.readthedocs.io/).
2. **Fit peaks** in the first-order region with a configurable multi-peak
   Lorentzian/Gaussian model ([lmfit](https://lmfit.github.io/lmfit-py/)). The configuration for the peak fitting comes from [MyPyDavid/raman-fitting](https://github.com/MyPyDavid/raman-fitting)
3. **Compute I_D/I_G** — either from the fitted peak areas, or with a simpler
   conventional band-intensity method.

It can also **measure the quality** of a spectrum (signal-to-noise and background
magnitude) before you trust the ratio it produces — see [Data quality](#data-quality).

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

## Two ways to get I_D/I_G

| Method | Function | How the ratio is computed |
|--------|----------|---------------------------|
| **Peak fitting** (recommended) | `peak_fit_from_file`, `peak_fit_from_data` | Fits a multi-peak model; ratio = fitted `D_area / G_area` via `get_id_ig`. |
| **Conventional** | `get_ratio_from_file` | Max intensity in the D band (1300–1390 cm⁻¹) / max intensity in the G band (1500–1670 cm⁻¹). No fitting. |

## Usage

```python
from hc_raman import (
    peak_fit_from_file,
    peak_fit_from_data,
    get_ratio_from_file,
    get_id_ig,
)

# 1) Peak fitting straight from a raw Renishaw file
result = peak_fit_from_file("sample.wdf", baseline="iasls", mode="5peaks", region="first_order")
print("ID/IG =", get_id_ig(result))

# ...with a plot of the fit (returns the figure objects too)
fig, ax, result = peak_fit_from_file("sample.wdf", mode="5peaks", plot=True)

# 2) Peak fitting from arrays you already have in memory
result = peak_fit_from_data(wavenumber, intensity, mode="6peaks")
print("ID/IG =", get_id_ig(result))

# 3) Conventional D/G ratio (no peak fitting)
id_ig = get_ratio_from_file("sample.wdf", baseline="iasls")
print("ID/IG (conventional) =", id_ig)
```

When `plot=True`, the fitting functions return `(fig, ax, result)` and the conventional
function returns `(fig, ax, ratio)`. Pass `display_plot=False` to build the figure
without calling `plt.show()` (useful for saving figures in a script).

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
settings — which is what makes them useful for spotting an exposure or laser power
problem across a batch.

The noise is measured *before* Savitzky-Golay denoising (smoothing would remove the
noise being measured) and without normalisation, so `baseline_ratio` includes any
constant offset such as a detector dark level. The windows are named regions from
`spectrum_regions.toml` and can be changed with `signal_region=` and `noise_region=` —
useful if your acquisition does not reach 1850–2150 cm⁻¹, which raises a `ValueError`
naming the region and your spectrum's range.

Pass the same `baseline=` you intend to fit with, so the baseline metrics describe the
curve the fit will actually see.

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

Each peak has a line shape (Lorentzian or Gaussian) and initial/bounded values for
center, sigma and amplitude. To adjust peak positions/bounds or add new peaks, edit the
TOML — no code changes needed.

### Baseline (`baseline`)

One of `iasls`, `airpls`, `iarpls` (all from RamanSPy). The peak-fitting entry points
default to `iasls`.

### Region (`region`)

Named wavenumber ranges from
[`hc_raman/spectrum_config/spectrum_regions.toml`](hc_raman/spectrum_config/spectrum_regions.toml),
e.g. `first_order` (900–2000 cm⁻¹, the default for fitting), `second_order`, `full`, etc.
Peak fitting is currently configured for the `first_order` region. `g_band` (1500–1620),
`d_band` (1300–1390) and `mid` (1850–2150, signal-free) are used by
`get_quality_metrics`.

## API reference

- `preprocess(file_path=None, wavenumber=None, intensity=None, baseline='iarpls', ...)` —
  full preprocessing pipeline (normalised + cropped) used by the fitting functions.
- `conv_preprocess(...)` — preprocessing for the conventional ratio (no normalisation,
  optional crop).
- `get_quality_metrics(file_path=None, wavenumber=None, intensity=None, ...)` — measurement
  quality metrics (signal-to-noise, background magnitude) as a dict.
- `peak_fit_from_file(...)` / `peak_fit_from_data(...)` — preprocess + fit; return an
  lmfit result.
- `get_ratio_from_file(...)` — conventional I_D/I_G without fitting.
- `get_id_ig(result)` — extract I_D/I_G from a fitted result.
- `fit_model(x, y, mode, region)` / `build_lmfit_model(mode, region)` — lower-level
  fitting helpers.
