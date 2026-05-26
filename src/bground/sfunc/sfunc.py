'''
Module: bground.ffunc
---------------------
Backbround subtraction using simple/ab-initio fitting function.

* Automatic background subtraction method.
* Using pure NumPy and SciPy functions and tools.
* Input: TXT file with two columns: X-coords, Y-coords.
* Output: TXT file with four columns: X, Y=Iraw, Ibkg, I=(Ibkg-Iraw)

Alternative input/output:

* ELD = ediff.io.Profile object
* Profile object at input  (2 cols): ELD.Pixels, ELD.Iraw
* Profile object at output (4 cols): ELD.Pixels, ELD.Iraw, ELD.Ibkg, ELD.I
'''

import numpy as np
import scipy as scp
import skimage as ski


def rolling_ball(bsObj, **kwargs):
    '''
    Subtract background from an array using *rolling ball* algorithm.

    Parameters
    ----------
    bsObj : bground.api.SimpleFunction object
        The object for bckground subtraction.
        It contains all necessary parameters for background subtraction.
        Namely, bsObj.data contains XYdata = array with [X, Y=Intensity].
    radius : int, optional, default is 20
        Radius of the rolling ball.

    Returns
    -------
    None 
        The background is subtracted and stored
        in bsObj.data and bsObj.background.
    '''
    
    # (0) Get parameters
    xrange = kwargs.get('xrange') or None
    radius = kwargs.get('radius') or 70
    
    # (1) Prepare variables for calculation
    # (xmin,xmax = x-range, in which we will calculate the background
    if xrange is not None:
        xmin,xmax = xrange
    else:
        xmin,xmax = bsObj.pars.xlim
    # (data = just convenience to have a shorter name
    data = bsObj.data
    
    # (2) Prepare XY-data for bkg calculation
    # (only the data in the selected range
    bkg_range = (xmin <= data[0]) & (data[0] <= xmax)
    Xbkg = data[0, bkg_range]
    Ybkg = data[1, bkg_range]
    
    # (3) Calculate background
    # (here: rolling_ball algorithm from skimage package
    Ybkg = ski.restoration.rolling_ball(Ybkg, radius=radius)
    
    # (4) Save the calculated background => update bsObj.background.curve
    bsObj.background.curve.X = Xbkg
    bsObj.background.curve.Y = Ybkg
    
    # (5) Prepare the data => we need array with 4 rows
    # (the 1st two rows = X,Y-data from initialization
    # (the last two rows should contain zeros (either add them or zero them) 
    # -----
    # Get number of rows in current {data} array
    number_of_rows = data.shape[0]
    # Ensure that we have {data} with 4 rows: [X,Y,0,0]
    if number_of_rows == 2:
        # If we have a new array with 2 rows, add two rows filled with zeros.
        data = np.vstack(
            [data, np.zeros((2, data.shape[1]), dtype=data.dtype)])
    else:
        # If we have an existing array with 4 rows, zero the last two rows.
        data[2] = 0
        data[3] = 0

    # (6) Subtract background
    # (more precisely: save bkg to data[2] + bkg-subtracted data to data[3]
    # -----
    # Save background to data[2]
    data[2, bkg_range] = bsObj.background.curve.Y
    # Subtract background from raw data and save it to data[3]
    data[3] = np.where(bkg_range, data[1]-data[2], 0)
    
    # (7) Save the complete data to sbObj
    bsObj.data = data
