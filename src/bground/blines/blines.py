'''
Module: bground.blines
----------------------
Backbround subtraction using baseline fitting.

* Automatic background subtraction method.
* Algorithm from https://pypi.org/project/pybaselines
* Input: TXT file with two columns: X-coords, Y-coords.
* Output: TXT file with four columns: X, Y=Iraw, Ibkg, I=(Ibkg-Iraw)

Alternative input/output:
* ELD = ediff.io.Profile object
* Profile object at input  (2 cols): ELD.Pixels, ELD.Iraw
* Profile object at output (4 cols): ELD.Pixels, ELD.Iraw, ELD.Ibkg, ELD.I
'''

import numpy as np
import matplotlib.pyplot as plt

def select_xrange(x: np.ndarray, y: np.ndarray, xrange: tuple[int, int]):
    '''Select data from a range on the x axis.

    Parameters
    ----------
    x : np.ndarray
        X axis of the data.
    y : np.ndarray
        Y axis of the data.
    xrange : tuple[int, int]
        A range on the x axis to be selected.
        The range is inclusive.

    Returns
    -------
    tuple
        Arrays x and y in the specified xrange.
    '''
    start, end = xrange
    mask = (x >= start) & (x <= end)
    return x[mask], y[mask]

def subtract_baseline(x: np.ndarray, y: np.ndarray, baseline: np.ndarray, xrange: tuple[int, int]):
    '''Subtract baseline from XY-data.

    Parameters
    ----------
    x : np.ndarray
        X axis of the data.
    y : np.ndarray
        Y axis of the data.
    baseline : np.ndarray
        Baseline to be subtracted.
    xrange : tuple[int, int]
        A range on the x axis, where the baseline is defined.
        The range is inclusive.

    Returns
    -------
    np.ndarray
        Array with 4 rows `[X, Iraw, Ibkg, I = Iraw - Ibkg]`.
    '''


    start, end = xrange

    bground_mask = (x >= start) & (x <= end)
    baseline_full = np.zeros_like(y)
    baseline_full[bground_mask] = baseline

    return np.stack((
        x,
        y,
        baseline_full,   
        np.where(bground_mask, y - baseline_full, 0)
    ))