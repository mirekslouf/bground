Module: bground.ffunc
---------------------
Backbround subtraction by fitting the background with a function.
"""

import numpy as np
from scipy.optimize import curve_fit

def trim_data(data, xrange=None, baseline_tol=10, messages=False):
    trimmed = None
    trim_level_used = None

    imax = np.argmax(data[1])
    data = data[:,imax:]

    trim_levels = [
        ("low",    dict(min_len=12, pct=10)),
        ("medium", dict(min_len=8,  pct=30)),
        ("hard",   dict(min_len=4,  pct=50))]

    if xrange is not None:
        xmin,xmax = xrange
        data = data[:,(xmin<=data[0])&(data[0]<=xmax)]
        trimmed = True
    else:
        for name, params in trim_levels:
            pct = params['pct']
            min_len = params['min_len']

            threshold = np.percentile(data[1], 95) * pct / 100
            mask = data[1] > threshold
            indices = np.where(mask)[0]

            if len(indices) >= min_len:
                start = max(indices[0] - baseline_tol, 0)
                end   = min(indices[-1] + baseline_tol, len(data[1])-1)
                data  = data[:,start:end+1]
                trim_level_used = name
                trimmed = True
                break

    if trimmed is None:
        raise ValueError("Trimming failed, try stronger parameters.")
    else:
        return(data)

def subtract_bacgkground(data, iter_max=20, threshold_factor=1.2, window=10, margin=1e-6):
    x = data[0]
    data1_masked = data[1].copy()

    def exp_func(x, a, b, c):
        return a * np.exp(-b * x) + c

    N_check = min(30, len(data[1]))
    dy = np.diff(data[1][:N_check])
    start_anchor_idx = np.argmin(dy) + 1

    for _ in range(iter_max):
        popt, _ = curve_fit(
            exp_func, x[start_anchor_idx:], data1_masked[start_anchor_idx:],
            p0=[data[1][start_anchor_idx]-data[1][-1], 0.01, data[1][-1]], maxfev=20000
        )
        baseline_fit = exp_func(x, *popt)
        baseline_fit = np.minimum(baseline_fit, data[1])

        for i in range(1, len(baseline_fit)):
            if baseline_fit[i] > baseline_fit[i-1]:
                baseline_fit[i] = baseline_fit[i-1]

        mask = data[1] > baseline_fit * threshold_factor
        data1_masked[mask] = baseline_fit[mask]

    data1_subtracted = data[1] - baseline_fit

    i = 0
    while i < len(data[1]) - window:
        if all(data[1][i+j] > baseline_fit[i+j] + margin for j in range(window)):
            break
        data1_subtracted[i] = 0
        i += 1

    N_plateau_start = 5
    for i in range(N_plateau_start-1, -1, -1):
        if all(data[1][i+j] > baseline_fit[i+j] for j in range(window) if i+j < len(data[1])):
            data1_subtracted[:i+1] = 0
            break

    return np.vstack([x, data1_subtracted])
