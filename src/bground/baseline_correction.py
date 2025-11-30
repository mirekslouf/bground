"""
Module: baseline_correction
---------------------
Automatic baseline correction for trimmed 1D SAED spectra.

* Automatic method using iterative exponential fitting.
* Post-processing removes noise and ensures corrected spectrum
  is flat up to the first real peak.

Key features:
---------------
* Detects initial sharp drop to anchor the baseline fit.
* Iterative masking of false (baseline) peaks to avoid fitting artifacts.
* Ensures monotonic decreasing baseline.
* Post-processing can enforce flat baseline at spectrum edges.
* Visualization included for raw, baseline, and corrected spectra.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

# =============================================================================
# Utility function: Exponential fit function
def exp_func(x, a, b, c):
    """
    Exponential function: a * exp(-b*x) + c
    Used for iterative baseline fitting of SAED spectra.
    
    Parameters
    ----------
    x : np.ndarray
        X-axis values (scattering vector or diffraction angle)
    a : float
        Initial amplitude of exponential decay
    b : float
        Decay rate
    c : float
        Offset (asymptotic baseline)
    
    Returns
    -------
    np.ndarray
        Evaluated exponential function
    """
    return a * np.exp(-b * x) + c

# =============================================================================
# Main function: Fit baseline
def fit_baseline(filename, desktop=True, iter_max=20, threshold_factor=1.02,
                 N_check=30, N_plateau_start=20, N_plateau_end=20, plot=True):
    """
    Fit baseline for SAED spectrum stored in TXT file.
    
    Parameters
    ----------
    filename : str
        Name of TXT file containing two columns [X, Y] with spectrum data.
    desktop : bool, optional
        If True, looks for file on Desktop. Default = True.
    iter_max : int, optional
        Number of iterations for iterative peak masking. Default = 20.
    threshold_factor : float, optional
        Factor to detect peaks above baseline during masking. Default = 1.02.
    N_check : int, optional
        Number of initial points to check for sharp drop to anchor baseline.
    N_plateau_start : int, optional
        Number of points at start to enforce flat baseline.
    N_plateau_end : int, optional
        Number of points at end to enforce flat baseline.
    plot : bool, optional
        If True, show matplotlib plots of raw, baseline, and corrected spectrum.
    
    Returns
    -------
    x : np.ndarray
        X-axis values
    y_corr : np.ndarray
        Baseline-corrected spectrum
    baseline_fit : np.ndarray
        Fitted baseline
    """
    # --- Load data ---
    ?

    # --- Detect sharp initial drop ---
    dy = np.diff(y[:N_check])
    start_anchor_idx = np.argmin(dy) + 1

    # --- Iterative exponential fit with peak masking ---
    y_masked = y.copy()
    for _ in range(iter_max):
        popt, _ = curve_fit(
            exp_func, x[start_anchor_idx:], y_masked[start_anchor_idx:],
            p0=[y[start_anchor_idx]-y[-1], 0.01, y[-1]], maxfev=20000
        )
        baseline_fit = exp_func(x, *popt)

        # Prevent baseline exceeding raw signal
        baseline_fit = np.minimum(baseline_fit, y)

        # Ensure monotonic decreasing baseline
        for i in range(1, len(baseline_fit)):
            if baseline_fit[i] > baseline_fit[i-1]:
                baseline_fit[i] = baseline_fit[i-1]

        # Mask peaks above threshold
        mask = y > baseline_fit * threshold_factor
        y_masked[mask] = baseline_fit[mask]

    # --- Baseline correction ---
    y_corr = y - baseline_fit

    # --- Post-processing: plateau at start up to first real peak ---
    first_real_idx_start = np.argmax(y_corr > 0)
    y_corr[:first_real_idx_start] = 0

    # --- Post-processing: plateau at end ---
    last_real_idx_end = np.where(y_corr > 0)[0][-1]
    y_corr[last_real_idx_end+1:] = 0

    # --- Optional plotting ---
    if plot:
        plt.figure(figsize=(10,5))
        plt.plot(x, y, label='Raw Spectrum')
        plt.plot(x, baseline_fit, label='Fitted Baseline')
        plt.xlabel("X")
        plt.ylabel("Intensity")
        plt.legend()
        plt.show()

        plt.figure(figsize=(10,5))
        plt.plot(x, y_corr, label='Corrected Spectrum')
        plt.xlabel("X")
        plt.ylabel("Corrected Intensity")
        plt.legend()
        plt.show()

    return x, y_corr, baseline_fit
