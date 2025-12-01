'''
Module: bground.ffunc
---------------------
Backbround subtraction by fitting the background with a function.
'''

import numpy as np


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
        ("low",    dict(min_len=12, pct=10, morph=5, varf=2.0, vmin=12)),
        ("medium", dict(min_len=8,  pct=30, morph=9, varf=1.5, vmin=8)),
        ("hard",   dict(min_len=4,  pct=50, morph=15, varf=1.2, vmin=4))]

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


def subtract_bacgkground(data):
    '''
    Toto by mela byt kombinace dvou funkci (exp + fitting).
    Vstup: data (po pripadnem trimmingu) a PO background subtraction.
    '''
    pass
