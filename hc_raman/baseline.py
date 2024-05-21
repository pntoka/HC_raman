from pybaselines.whittaker import airpls, iasls
import torch
import torch.nn as nn
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
    baseline = iasls(x_data, y_data, lam=lam, p=p)
    return baseline[0]


def interpolate_data(x_data, y_data, size=2200):
    """
    Interpolate the data to a new size.

    Parameters
    ----------
    x_data : numpy.ndarray
        Wavenumber values.
    y_data : numpy.ndarray
        Data to be interpolated.
    new_x_data : numpy.ndarray
        New wavenumber values.
    size : int, optional
        Size of the interpolated data (default is 2200).

    Returns
    -------
    interpolated_data : numpy.ndarray
        Interpolated data.
    """
    # Interpolate the data
    new_x_data = np.linspace(x_data.min(), x_data.max(), size)
    f = interp1d(x_data, y_data, kind="cubic")
    interpolated_data = f(new_x_data)
    return new_x_data, interpolated_data


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


def load_nn_model():
    model = torch.load(os.path.join(os.path.dirname(__file__), "NNmodel.pkl"))
    return model


def comp(a, b):
    if a == 1 and b >= 0.4:
        return 1
    elif a == 0 and b < 0.4:
        return 1
    else:
        return 0


def poly_baseline(x, p, intensity, b):
    y = (x / len(x)) ** p + b
    return y * intensity / max(y)


def poly_decrease_baseline(x, p, intensity, b):
    y = -((x / len(x)) ** p) + 1 + b
    return y * intensity / max(y)


def gaussian_baseline(x, mean, sd, intensity, b):
    y = np.exp(-((x - mean) ** 2) / (2 * sd**2)) / (sd * np.sqrt(2 * np.pi)) + b
    return y * intensity / max(y)


def pg_baseline(x, p, in1, mean, sd, in2, b):
    y1 = (x / len(x)) ** p
    y2 = np.exp(-((x - mean) ** 2) / (2 * sd**2)) / (sd * np.sqrt(2 * np.pi))
    return y1 / max(y1) * in1 + y2 / max(y2) * in2 + b


def pd_baseline(x, p1, in1, p2, in2, b):
    y1 = (x / len(x)) ** p1
    y2 = -((x / len(x)) ** p2) + 1
    return y1 / max(y1) * in1 + y2 / max(y2) * in2 + b


def dg_baseline(x, p, in1, mean, sd, in2, b):
    y1 = -((x / len(x)) ** p) + 1
    y2 = np.exp(-((x - mean) ** 2) / (2 * sd**2)) / (sd * np.sqrt(2 * np.pi))
    return y1 / max(y1) * in1 + y2 / max(y2) * in2 + b


def pdg_baseline(x, p1, in1, p2, in2, mean, sd, in3, b):
    y1 = (x / len(x)) ** p1
    y2 = -((x / len(x)) ** p2) + 1
    y3 = np.exp(-((x - mean) ** 2) / (2 * sd**2)) / (sd * np.sqrt(2 * np.pi))
    return y1 / max(y1) * in1 + y2 / max(y2) * in2 + y3 / max(y3) * in3 + b


def polynomial_baseline(x, a, b, c, d, e):
    y = a * x**4 + b * x**3 + c * x**2 + d * x + e
    return y


def polynomial_baseline2(x, b, c, d, e, f, g):
    y = b * x**5 + c * x**4 + d * x**3 + e * x**2 + f * x + g
    return y


def mix_max(sp, baseline):
    new_baseline = []
    for i in range(len(sp)):
        new_baseline.append(max(sp[i], baseline[i]))
    return new_baseline


def mix_min(sp, baseline):
    new_baseline = []
    for i in range(len(sp)):
        new_baseline.append(min(sp[i], baseline[i]))
    return new_baseline


def iterative_fitting(sp, ite):
    x = np.linspace(1, 2200, 2200)
    tempb = sp
    torch_tempb = torch.Tensor(tempb.reshape(1, 1, 2200))
    i = 0
    mynet = load_nn_model()
    while i < ite:
        tadvice = mynet(torch_tempb)
        # print(tadvice)
        if tadvice[0][0] > 0.5 and tadvice[0][1] > 0.5 and tadvice[0][2] > 0.5:
            p, c = curve_fit(
                pdg_baseline,
                x,
                tempb,
                bounds=(
                    [0, 0, 0, 0, 300, 200, 0, -0.3],
                    [3, 2, 3, 2, 1900, 600, 2, 0.3],
                ),
            )
            fitted_baseline = pdg_baseline(
                x, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]
            )
        elif tadvice[0][0] > 0.5 and tadvice[0][1] > 0.5:
            p, c = curve_fit(
                pd_baseline, x, tempb, bounds=([0, 0, 0, 0, -0.3], [2, 3, 2, 3, 0.3])
            )
            fitted_baseline = pd_baseline(x, p[0], p[1], p[2], p[3], p[4])
        elif tadvice[0][0] > 0.5 and tadvice[0][2] > 0.5:
            p, c = curve_fit(
                pg_baseline,
                x,
                tempb,
                bounds=([0, 0, 300, 200, 0, -0.3], [3, 2, 1900, 600, 2, 0.3]),
            )
            fitted_baseline = pg_baseline(x, p[0], p[1], p[2], p[3], p[4], p[5])
        elif tadvice[0][1] > 0.5 and tadvice[0][2] > 0.5:
            p, c = curve_fit(
                dg_baseline,
                x,
                tempb,
                bounds=([0, 0, 300, 200, 0, -0.3], [3, 2, 1900, 500, 2, 0.3]),
            )
            fitted_baseline = dg_baseline(x, p[0], p[1], p[2], p[3], p[4], p[5])
        elif tadvice[0][0] > 0.5:
            p, c = curve_fit(
                poly_baseline, x, tempb, bounds=([0, 0, -0.3], [3, 2, 0.3])
            )
            fitted_baseline = poly_baseline(x, p[0], p[1], p[2])
        elif tadvice[0][1] > 0.5:
            p, c = curve_fit(
                poly_decrease_baseline, x, tempb, bounds=([0, 0, -0.3], [3, 1, 0.3])
            )
            fitted_baseline = poly_decrease_baseline(x, p[0], p[1], p[2])
        elif tadvice[0][2] > 0.5:
            p, c = curve_fit(
                gaussian_baseline,
                x,
                tempb,
                bounds=([500, 200, 0, -0.3], [1700, 500, 1, 0.3]),
            )
            fitted_baseline = gaussian_baseline(x, p[0], p[1], p[2], p[3])
        if max(tadvice[0]) < 0.5:
            if tadvice[0][0] < 0.05 and tadvice[0][1] < 0.05 and tadvice[0][2] > 0.05:
                p, c = curve_fit(
                    gaussian_baseline,
                    x,
                    tempb,
                    bounds=([500, 200, 0, -0.3], [1700, 500, 1, 0.3]),
                )
                fitted_baseline = gaussian_baseline(x, p[0], p[1], p[2], p[3])
            elif tadvice[0][0] < 0.05 and tadvice[0][2] < 0.05 and tadvice[0][1] > 0.05:
                p, c = curve_fit(poly_decrease_baseline, x, tempb)
                fitted_baseline = poly_decrease_baseline(x, p[0], p[1], p[2])
            elif tadvice[0][1] < 0.05 and tadvice[0][2] < 0.05 and tadvice[0][0] > 0.05:
                p, c = curve_fit(poly_baseline, x, tempb)
                fitted_baseline = poly_baseline(x, p[0], p[1], p[2])
            elif tadvice[0][0] < 0.05 and tadvice[0][1] > 0.05 and tadvice[0][2] > 0.05:
                p, c = curve_fit(
                    dg_baseline,
                    x,
                    tempb,
                    bounds=([0, 0, 500, 200, 0, -0.3], [3, 0.5, 1700, 500, 0.5, 0.3]),
                )
                fitted_baseline = dg_baseline(
                    x, p[0], p[1], p[2], p[3], p[4], p[5], p[6]
                )
            elif tadvice[0][1] < 0.05 and tadvice[0][2] > 0.05 and tadvice[0][0] > 0.05:
                p, c = curve_fit(
                    pg_baseline,
                    x,
                    tempb,
                    bounds=([0, 0, 500, 200, 0, -0.3], [3, 0.5, 1700, 500, 0.5, 0.3]),
                )
                fitted_baseline = pg_baseline(x, p[0], p[1], p[2], p[3], p[4], p[5])
            elif tadvice[0][0] > 0.05 and tadvice[0][1] > 0.05 and tadvice[0][2] < 0.05:
                p, c = curve_fit(
                    pd_baseline,
                    x,
                    tempb,
                    bounds=([0, 0, 0, 0, -0.3], [3, 0.5, 3, 0.5, 0.3]),
                )
                fitted_baseline = pd_baseline(x, p[0], p[1], p[2], p[3], p[4], p[5])
            elif tadvice[0][0] < 0.05 and tadvice[0][1] < 0.05 and tadvice[0][2] < 0.05:
                fitted_baseline = np.zeros(2200)
        tempb = mix_min(fitted_baseline, tempb)
        torch_tempb = torch.Tensor(np.asfarray(tempb).reshape(1, 2200, 1)).permute(
            0, 2, 1
        )
        i += 1
        if max(tadvice[0]) < 0.01:
            # print(tadvice)
            break
    return tempb


def nn_baseline(x_data, y_data, size=2200, iter=10):
    """
    Apply the neural network baseline correction.

    Parameters
    ----------
    x_data : numpy.ndarray
        Wavenumber values.
    y_data : numpy.ndarray
        Data to be corrected.
    iter : int, optional
        Number of iterations (default is 20).

    Returns
    -------
    baseline : numpy.ndarray
        Baseline corrected data.
    """
    # Apply the neural network baseline correction
    new_x_data, new_y_data = interpolate_data(x_data, y_data, size=2200)
    baseline = iterative_fitting(new_y_data, iter)
    return baseline
