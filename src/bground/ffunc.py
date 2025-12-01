"""
Module: bground.ffunc
---------------------
Backbround subtraction by fitting the background with a function.
"""

import numpy as np
from scipy.optimize import curve_fit

# =============================================================================
def trim_data(data, xrange=None, baseline_tol=10, messages=False):
    """
    Trim XYdata using multiple rounds.

    Parameters
    ----------
    data : 2D numpy.array
        XYdata = dimensional array with two rows;
        the array is typically read from a file with two columns.
        The first row/column = X-data.
        The second row/column = Y-data = intensity/signal
        from which the bacgkround should be subtracted.
    baseline_tol : int
        Number of points to extend trimming window before and after signal

    Returns
    -------
    data : 2D-numpy.array
        XYdata with trimmed = uninteresting regiouns cut off.
    """

    trimmed = None
    trim_level_used = None

    # (1) Left cut (specific for diffraction patterns) ------------------------
    # (this step should be made optional => additional argument
    # (start from the first global maximum
    
    imax = np.argmax(data[1])
    data = data[:,imax:]
    
    # (2) Define levels of trimming -------------------------------------------
    # (trial-and-error parametrs, from the lowest to the hardest
    # (the following code applies all trimming levels)
    
    trim_levels = [
        ("low",    dict(min_len=12, pct=10)),
        ("medium", dict(min_len=8,  pct=30)),
        ("hard",   dict(min_len=4,  pct=50))]

    # (3) Start spectrum trimming ---------------------------------------------
    # (test all above-defined levels
    
    # (3a) Manual trimming - user defined xrange
    if xrange is not None:
        xmin,xmax = xrange
        data = data[:,(xmin<=data[0])&(data[0]<=xmax)]
        trimmed = True
    # (3b) Auto-trimming - estimate and remove uninteresting regons
    else:
        for name, params in trim_levels:
            pct = params['pct']
            min_len = params['min_len']
        
            # Threshold based on 95th percentile scaled by pct
            threshold = np.percentile(data[1], 95) * pct / 100
            mask = data[1] > threshold
            indices = np.where(mask)[0]
        
            if len(indices) >= min_len:
                start = max(indices[0] - baseline_tol, 0)
                end   = min(indices[-1] + baseline_tol, len(data[1])-1)
                data  = data[:,start:end+1]
                trim_level_used = name
                if messages:
                    print(f'Trim level used: {trim_level_used}')
                trimmed = True
                break
  
    # (4) Return trimmed data -------------------------------------------------
    
    if trimmed is None:
        raise ValueError("Trimming failed, try stronger parameters.")
        return(None)
    else:
        return(data)

def subtract_background(data, iter_max=20, threshold_factor=1.2, window=10, margin=1e-6):
    """
    Background subtraction by fitting.

    Parameters
    ----------
    data : 2D numpy.array
        XYdata = [x, y] array
    iter_max : int
        Maximum number of iterations for masking
    threshold_factor : float
        Factor to detect peaks above baseline
    window : int
        Window for first stable peak detection
    margin : float
        Small offset for numerical stability

    Returns
    -------
    data_subtracted : 2D numpy.array
        XYdata with baseline subtracted
    """
    x = data[0]
    data1_masked = data[1].copy()
    
    def exp_func(x, a, b, c):
        return a * np.exp(-b * x) + c

    # --- detect steep drop at start ---
    N_check = min(30, len(data[1]))
    dy = np.diff(data[1][:N_check])
    start_anchor_idx = np.argmin(dy) + 1

    # --- iterative fit ---
    for _ in range(iter_max):
        popt, _ = curve_fit(
            exp_func, x[start_anchor_idx:], data1_masked[start_anchor_idx:],
            p0=[data[1][start_anchor_idx]-data[1][-1], 0.01, data[1][-1]], maxfev=20000
        )
        baseline_fit = exp_func(x, *popt)
        baseline_fit = np.minimum(baseline_fit, data[1])

        # ensure monotonic baseline
        for i in range(1, len(baseline_fit)):
            if baseline_fit[i] > baseline_fit[i-1]:
                baseline_fit[i] = baseline_fit[i-1]

        mask = data[1] > baseline_fit * threshold_factor
        data1_masked[mask] = baseline_fit[mask]

    # --- subtract baseline ---
    data1_subtracted = data[1] - baseline_fit

    # --- zero before first stable peak ---
    i = 0
    while i < len(data[1]) - window:
        if all(data[1][i+j] > baseline_fit[i+j] + margin for j in range(window)):
            break
        data1_subtracted[i] = 0
        i += 1

    # --- plateau start mask ---
    N_plateau_start = 5
    for i in range(N_plateau_start-1, -1, -1):
        if all(data[1][i+j] > baseline_fit[i+j] for j in range(window) if i+j < len(data[1])):
            data1_subtracted[:i+1] = 0
            break

    return np.vstack([x, data1_subtracted])
