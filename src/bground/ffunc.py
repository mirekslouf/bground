# =============================================================================
"""
Module: bground.ffunc
---------------------
Backbround subtraction by fitting the background with a function.
"""
# =============================================================================

import numpy as np
from scipy.optimize import curve_fit

# =============================================================================
def trim_data(data, xrange=None, baseline_tol=10, messages=False):
    """
    Trim XYdata using multiple rounds.

    Parameters
    ----------
    data : 2D numpy.array
        XYdata = dimensional array with two rows.
        Typically read from a file with two columns.
        The first row/column = X-data.
        The second row/column = Y-data = intensity/signal.
        Background will be subtracted from this row.
    xrange : tuple or None
        Manual trimming range (Xmin, Xmax). Overrides auto-trimming.
    baseline_tol : int
        Number of points to extend trimming window before and after detected signal region.
    messages : bool
        If True, prints which trim level was applied.

    Returns
    -------
    data : 2D numpy.array
        XYdata with trimmed regions cut off.
    """

    trimmed = None
    trim_level_used = None

    # (1) Left cut (specific for diffraction patterns) ------------------------
    # Start from the first global maximum.
    imax = np.argmax(data[1])
    data = data[:, imax:]

    # (2) Define levels of trimming -------------------------------------------
    # Trial-and-error parameters, from the lowest to the hardest.
    trim_levels = [
        ("low",    dict(min_len=12, pct=10)),
        ("medium", dict(min_len=8,  pct=30)),
        ("hard",   dict(min_len=4,  pct=50))
    ]

    # (3) Start spectrum trimming ---------------------------------------------
    # (3a) Manual trimming - user defined xrange
    if xrange is not None:
        xmin, xmax = xrange
        data = data[:, (xmin <= data[0]) & (data[0] <= xmax)]
        trimmed = True

    # (3b) Auto-trimming - estimate and remove uninteresting regions
    else:
        for name, params in trim_levels:
            pct = params['pct']
            min_len = params['min_len']

            # Threshold based on 95th percentile scaled by pct.
            threshold = np.percentile(data[1], 95) * pct / 100
            mask = data[1] > threshold
            indices = np.where(mask)[0]

            if len(indices) >= min_len:
                start = max(indices[0] - baseline_tol, 0)
                end   = min(indices[-1] + baseline_tol, len(data[1]) - 1)
                data = data[:, start:end + 1]
                trim_level_used = name
                if messages:
                    print(f'Trim level used: {trim_level_used}')
                trimmed = True
                break

    # (4) Return trimmed data -------------------------------------------------
    if trimmed is None:
        raise ValueError("Trimming failed, try stronger parameters.")
    else:
        return data


# =============================================================================
def subtract_background(data, iter_max=20, threshold_factor=1.2,
                        margin=1e-6, remove_fake_peak=True):
    """
    Background subtraction using iterative exponential fitting.

    Parameters
    ----------
    data : 2D numpy.array
        XYdata = dimensional array with two rows.
        First row = X-data, second row = Y-data.
    iter_max : int
        Maximum number of refinement iterations.
    threshold_factor : float
        Factor controlling masking threshold relative to fitted baseline.
    margin : float
        Threshold for detecting a fake initial peak.
    remove_fake_peak : bool
        If True, removes artificial initial peak if present.

    Returns
    -------
    data : 2D numpy.array
        XYdata with background subtracted and initial fake-peak removal.
    """

    data_masked = data.copy()
    data_masked[1] = data[1].copy()

    def exp_func(data_x, a, b, c):
        return a * np.exp(-b * data_x) + c

    # Stable initial fit region -----------------------------------------------
    start_idx = np.argmin(np.diff(data[1][:min(30, len(data[1]))])) + 1

    # Iterative fit & masking ------------------------------------------------
    for _ in range(iter_max):
        popt, _ = curve_fit(
            exp_func,
            data[0][start_idx:], data_masked[1][start_idx:],
            p0=[data[1][start_idx] - data[1][-1], 0.01, data[1][-1]],
            maxfev=20000
        )
        baseline = exp_func(data[0], *popt)
        baseline = np.minimum.accumulate(np.minimum(baseline, data[1]))
        mask = data[1] > baseline * threshold_factor
        data_masked[1][mask] = baseline[mask]

    data_subtracted = data[1] - baseline

    # Suppression of fake initial peak ----------------------------------------
    if remove_fake_peak:

        def suppress_initial_peak(data_in, margin):
            data_copy = data_in.copy()
            i = 0
            while i < len(data_copy[1]) and data_copy[1][i] <= margin:
                i += 1
            start = i
            while i < len(data_copy[1]) and data_copy[1][i] > margin:
                i += 1
            end = i
            if start == 0 and end < len(data_copy[1]):
                data_copy[1][start:end] = 0
            return data_copy

        data_masked = suppress_initial_peak(np.vstack([data[0], data_subtracted]), margin)
        data_subtracted = data_masked[1]

    return np.vstack([data[0], data_subtracted])