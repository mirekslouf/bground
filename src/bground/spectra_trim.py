import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# --- Load data ---

# Handle 1D vs 2D data
if data.ndim == 1:
    data = data[:, None]

x = data[:, 0]
signal = data[:, 1]

# =============================================================================
# --- Left cut: start from global maximum ---
imax = np.argmax(signal)
x_cut = x[imax:]
signal_cut = signal[imax:]

# =============================================================================
# --- Trimming rounds ---
# Ordered from low to hard
rounds = [
    ("low",    dict(k=6, min_len=12, pct=10, morph=5, varf=2.0, vmin=12)),
    ("medium",  dict(k=4, min_len=8,  pct=30, morph=9, varf=1.5, vmin=8)),
    ("hard", dict(k=2, min_len=4,  pct=50, morph=15, varf=1.2, vmin=4)),
]

# =============================================================================
def trim_spectrum(signal, x, rounds, baseline_tol=10):
    """
    Trim spectrum automatically using multiple rounds.

    Parameters
    ----------
    signal : np.ndarray
        Intensity values of the spectrum
    x : np.ndarray
        X-axis values
    rounds : list
        List of tuples (name, params dict) defining rounds of trimming
    baseline_tol : int
        Number of points to extend trimming window before and after signal

    Returns
    -------
    x_trimmed : np.ndarray
        X-axis of trimmed spectrum
    trimmed : np.ndarray
        Trimmed intensity values
    used_round : str
        Name of the trimming round used
    start, end : int
        Start and end indices of trimmed spectrum within input signal
    """
    trimmed = None
    used_round = None

    for name, params in rounds:
        pct = params['pct']
        min_len = params['min_len']

        # Threshold based on 95th percentile scaled by pct
        threshold = np.percentile(signal, 95) * pct / 100
        mask = signal > threshold
        indices = np.where(mask)[0]

        if len(indices) >= min_len:
            start = max(indices[0] - baseline_tol, 0)
            end   = min(indices[-1] + baseline_tol, len(signal)-1)
            trimmed = signal[start:end+1]
            x_trimmed = x[start:end+1]
            used_round = name
            break

    if trimmed is None:
        raise ValueError("Trimming failed, try stronger parameters.")

    return x_trimmed, trimmed, used_round, start, end

# =============================================================================
# --- Run trimming ---
x_trim, sig_trim, used, start_idx, end_idx = trim_spectrum(signal_cut, x_cut, rounds, baseline_tol=10)
print(f"Trimming level: {used} | original length: {len(signal_cut)} → trimmed: {len(sig_trim)}")

# =============================================================================
# --- Save trimmed spectrum ---
?
