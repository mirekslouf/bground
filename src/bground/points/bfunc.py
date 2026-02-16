'''
Module: bground.bfunc
---------------------
Functions for the background calculation.

* The functions are usually called indirectly, from bground.iplot module.
* The bground.iplot module defines an interactive plot, which is used as a GUI.

Technical notes:

* All functions in this module manipulate with bground.bdata.bkg object.
    - The bkg object contains all info needed for background subtraction.
* The last function works also with data = 2D-numpy array object.
    - The 2D-numpy array with two rows: [X,Y], where Y = Iraw = raw intensity
    - The last function calculates a 3-row array: [X, Iraw, Ibkg, I=Iraw-Ibkg]
'''

import numpy as np
from scipy import interpolate


def load_bkg_points(bkg_object):
    '''
    Load backround points to {bkg_object}.
    
    * {bkg_object} must be known/pre-defined
    * {bkg_object.bname} contains name of input/outpu files
    * {bkg_object.bname}+'.bkg' is the name of file with bkg points to read

    Parameters
    ----------
    bkg_object : bground.points.bdata.XYbackground object
        This object stores bname, btype, XYpoints, and XYcurve.
        
        * bname = name of the input/output file with bkg points
        * btype = type of the backround (linear, quadratic, cubic)
        * XYpoints = simple object with background points (X-list and Y-list).

    Returns
    -------
    bkg_object : bground.points.bdata.XYbackground object
        The object contains the following items: 
        (i) name of the input file, 
        (ii) type of backround subtraction, and
        (iii) the X,Y background points and type of the.
    '''
    # Prepare name bkg-file = file with background points
    bkg_file = str(bkg_object.bname) + '.bkg'
        
    # Go through {bkg_file} and save data into {bkg_object}
    with open(bkg_file, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines (continue if the line is empty)
            if not line: continue
            # Process comments (and continue when processed)
            if line.startswith("#"):
                if line.startswith("# Background correction type"):
                    # Read background type if given
                    bkg_object.btype = line.split(":")[-1].strip()
                continue
            # Read data lines
            parts = line.split()
            # Special case 1 - ensure compatibility with old format
            # (3 parts = probably the old format: index, X, Y => ignore 1st
            if len(parts) == 3: parts = parts[1:]
            # Special case 2 - ensure compatibility with old format
            # (we have read ['X','Y] = header of the old format => skip
            if parts == ['X','Y']: continue
            # Final pre-check => does the line contain two values now?
            if len(parts) != 2:
                raise TypeError(f'Strange line in bkg-file: {line}')
            # Correct data line (hopefully) => read x-coord and y-coord
            try:
                x, y = map(float, parts)
                bkg_object.points.X.append(x)
                bkg_object.points.Y.append(y)
            except ValueError:
                print('Cannot convert X,Y-coords to float!')
                raise
            except Exception:
                print('Something went wrong!')
                raise
            
    # Return {bkg_object}
    # (at the end of this function, the bkg_object should contain
    # ( - bkg_object.bname = input filename without extension
    # ( - bkg_object.btype = background type (read here) OR default='linear'
    # ( - bkg_object.points.X, bkg_object.points.Y = X,Y-coords of bkg points
    # (later we can calculate also:
    # ( - bkg_object.curve => to finish background subtraction procedure
    return bkg_object


def save_bkg_points(bkg_object):
    '''
    Save bacgkround points to the output text file.
    
    * The background points are stored in {bkg_object}.
    * The background points output file will be {bkg_object.bname}+'.bkg'.

    Parameters
    ----------
    bkg_object : bground.bdata.XYbackground object
        Object that stores background-related data,
        including bacgkround points.

    Returns
    -------
    None
        The background points are stored in {bkg_object.bname}+'.bkg' file.
    '''
    # Prepare name bkg-file = file with background points
    bkg_file = str(bkg_object.bname) + '.bkg'

    # Go through the data in the bkg_object and write them to file
    with open(bkg_file, 'w') as fh:
        fh.writelines('# Background points\n')
        fh.writelines('# 2 columns: [X-coords, Y-coords]\n')
        fh.writelines(f'# Background correction type: {bkg_object.btype}\n')
        X = bkg_object.points.X
        Y = bkg_object.points.Y
        for x,y in zip(X,Y): print(f'{x:10.1f}{y:10.1f}', file=fh)
    

def sort_bkg_points(bkg_object):
    '''
    Sort background points according to their X-coordinate.
    
    Parameters
    ----------
    bkg_object : bground.bdata.bkg object
        A bkg-object containing (among other things)
        a list of background points, which are probably unsorted.

    Returns
    -------
    None
        The result is the updated bkg object.
        The updated object contains bkg.points sorted according to
        their X-coordinate.
    '''
    # Sorting is based on the trick found on www
    # GoogleSearch: python sort two 1D arrays
    # https://stackoverflow.com/q/9007877
    X,Y = (bkg_object.points.X, bkg_object.points.Y)
    x,y = zip( *sorted( zip(X,Y) ) )
    bkg_object.points.X = list(x)
    bkg_object.points.Y = list(y)

    
def calculate_baseline(data, bkg):
    '''
    Calculate background
    = calculate interpolated background curve;
    the calculated background curve is saved within bkg object.
    
    Parameters
    ----------
    data : 2D numpy array
        The array contains two colums [X,Intensity].
    bkg : bground.points.bdata.XYbackground object
        Object containing the following items:
            
        * bname = string, basename of output file(s)
        * btype = a type of interpolation for the calculation of the bkground
        * points = 2-column list: [X-coord, Y-coord]
        * curve = baseline (to be (re)calculated when calling this function)
        
    Returns
    -------
    None
        The result is the updated bkg object,
        which should contain the following (re)calculated items:
        
        * bkg.curve.X = calculated X-coordinates of the whole background
        * bkg.curve.Y = calculated Y-coordinates of the WHOLE background
    '''
    # (1) Prepare background points = X,Y coordinates for interpolation
    X,Y = (bkg.points.X,bkg.points.Y)
    # (2) Interpolate background points = calculcate background curve
    try:
        # Interpolation = calculation of interpolation function F.
        # (F = interpolation object/function
        # (with which we easily calculate the interpolated data - see below
        F = interpolate.interp1d(X,Y, kind=bkg.btype)
        Xmin = bkg.points.X[0]
        Xmax = bkg.points.X[-1]
        Xnew = data[0,(Xmin<=data[0])&(data[0]<=Xmax)]
        Ynew = F(Xnew)
        bkg.curve.X = Xnew
        bkg.curve.Y = Ynew
    except Exception as err:
        # Exceptions: interpolation can fail for whatever reason
        # In such a case we print the error and continue ...
        print(err)
        print(type(err))


def calculate_bkg_data(data, bkg):
    '''
    Final calculation/update of background data = numpy array with XY-data.
    
    * Input background data = 2 rows: `X, Y = Iraw = raw intensity`
    * Output background data = 4 rows: `X, Iraw, Ibkg, I = (Iraw - Ibkg)`
    
    Parameters
    ----------
    data : 2D numpy array
        The array with two rows [X,Y] as described above.
    bkg : bground.points.bdata.XYbackground object
        The object contains several items,
        including the interpolated background curve.

    Returns
    -------
    data : 2D numpy array
        The array with 4 rows `[X, Iraw, Ibkg, I = Iraw - Ibkg]`.
    '''
    # (0) Recalculate baseline
    # (in most cases, baseline is calculated when we call this function
    # (BUT calculation might be omitted or have just read the background points
    calculate_baseline(data, bkg)
    # (1) Modify the data variable if needed.
    number_of_rows = data.shape[0]
    # If number_of_rows == 2 then add two new rows.
    # => the func is called for the 1st time and data have just 2 original rows
    if number_of_rows == 2:      
        data = np.insert(data,2,[data[1],data[1]],0)
    # If number_of_rows != 2  then rewrite two additional rows with data[1].
    # => func is called for 2nd, 3rd ... time and data already have 4 rows
    else:
        data[2],data[3] = data[1],data[1]   
    # (2) Get Xmin and Xmax of background/baseline curve.
    Xmin = bkg.points.X[0]
    Xmax = bkg.points.X[-1]
    # (3) Set range in which the background/baseline is defined.
    # (in this package, we define background only inside the selected range
    bkg_range = (Xmin<=data[0]) & (data[0]<=Xmax)
    # (4) Define 2nd data row = Ibkg = baseline.
    # (the baseline will contain zeros outside bkg_range
    # (a) Zero intensities outside bkg_range.
    data[2] = np.where(bkg_range,data[2],0)
    # (b) Baseline intentities inside bkg_range
    data[2,bkg_range] = bkg.curve.Y
    # (5) Define 3rd data row  = I = Iraw - Ibkg = net intensity.
    # (the net intensity will contain zeros outsice bkg_range
    # (a) Zero intensities outside bkg_range.
    data[3] = np.where(bkg_range,data[3],0)
    # (b) Net intensities inside bkg_range = raw_intensities - bkg_intensities
    data[3,bkg_range] = data[3,bkg_range] - bkg.curve.Y
    # (c) Set possible negative intensities after bkgr subtraction to zero
    data[3,data[3]<0] = 0
    # (b) Return modified data array
    # (the two rows of the modified array contain baseline and net intensity
    return(data)


def save_bkg_data(data, bkg, out_file):
    file_header = (
        'XY-data with background subtraction\n' +
        '4 columns: [X, Y=Iraw, Ibkg, I=(Iraw-Ibkg)]\n'+
        f'Background correction type: {bkg.btype}')
    np.savetxt(
        out_file, 
        np.transpose(data),
        fmt=('%8.3f','%11.3e','%11.3e', '%11.3e'),
        header=file_header)
    