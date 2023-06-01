'''
Module bground.bfunc
--------------------
Functions for background calculation.

* The functions are usually called from bground.iplot module.
* The bground.iplot module defines interactive plot and its events.
* This module defines functions that perform the real background calculation.
'''

import numpy as np
from scipy import interpolate

def sort_bkg_points(bkg):
    '''
    Sort background points according to their X-coordinate.
    The background points are inserted as the whole bkg object.
    
    Parameters
    ----------
    bkg : bground.bdata.bkg object
        A bkg object containing unsorted list of background points.

    Returns
    -------
    None; the output is bkg object with sorted background points.
    '''
    # Sorting is based on the trick found on www
    # GoogleSearch: python sort two 1D arrays
    # https://stackoverflow.com/q/9007877
    X,Y = (bkg.points.X, bkg.points.Y)
    x,y = zip( *sorted( zip(X,Y) ) )
    bkg.points.X = list(x)
    bkg.points.Y = list(y)
    
def calculate_background(data,bkg):
    '''
    Calculate background
    = calculate interpolated background curve;
    the calculated background curve is saved within bkg object.
    
    Parameters
    ----------
    bkg : bground.bdata.bkg object
        Object containing the following items:
            * basename = string, basename of output file(s)
            * points = 3-column list: [PointType, X-coord, Y-coord]
            * itype = type of interpolation for the calculation of bkground
    
    Returns
    -------
    None; the result is the updated bkg object.
        * bkg.X = calculated X-coordinates of the WHOLE background
        * bkg.Y = calculated Y-coordinates of the WHOLE background
    '''
    # (1) Prepare background points = X,Y coordinates for interpolation
    X,Y = (bkg.points.X,bkg.points.Y)
    # (2) Interpolate background points = calculcate background curve
    try:
        # Interpolation = calculation of interpolation function F.
        # (F = interpolation object/function
        # (with which we easily calculate the interpolated data - see below
        F = interpolate.interp1d(X,Y, kind=bkg.itype)
        Xmin = bkg.points.X[0]
        Xmax = bkg.points.X[-1]
        Xnew = data[0,(Xmin<=data[0])&(data[0]<=Xmax)]
        Ynew = F(Xnew)
        bkg.curve.X = Xnew
        bkg.curve.Y = Ynew
    except Exception as err:
        # Exceptions: interpolation can fail for whatever reason
        # In such a case we print error and return an empty array
        print(err)
        print(type(err))
        return(np.array([]))

def subtract_background(data, bkg):
    '''
    Subtract background
    = subtract interpolated background curve from original data;
    the data with subtracted bkgr are added as new column to data variable.  
    
    Parameters
    ----------
    data : 2D numpy array
        The array contains two colums [X,Intensity].
    bkg : bdata.bkg object
        The object contains several items,
        namely interpolated background curve.

    Returns
    -------
    data : 2D numpy array
        The array with 3 columns [X,Intensity,BackgroundCorrectedIntensity].
    '''
    # (1) Add one more column to data variable.
    data = np.insert(data,2,data[1],0)
    # (2) Get Xmin and Xmax of background curve.
    Xmin = bkg.curve.X[0]
    Xmax = bkg.curve.X[-1]
    # (3) Define range in which the backgrou5nd is subtracted.
    bkg_range = (Xmin<=data[0]) & (data[0]<=Xmax)
    # (4) Zero intensities below Xmin & above Xmax.
    data[2] = np.where(bkg_range,data[1],0)
    # (5) Subtract background from intensities between Xmin and Xmax
    data[2,bkg_range] = data[2,bkg_range] - bkg.curve.Y
    # (6) Set possible negative intensities after bkgr subtraction to zero
    data[2,data[2]<0] = 0
    # (7) Return modified data array
    # (the last column the array contains background-corrected intensities
    return(data)
