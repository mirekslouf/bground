'''
Module: bground.points.bfunc
----------------------------
Functions for the background calculation.

* The functions are usually called indirectly, via bground.api.
* The functions manimulate with background data of {iplot} object.
* The {iplot} object can come from two classes:
    - bground.api.InteractivePlot = classical interactive plot
    - bground.api.RestoreFromPoints
      = restore bkg from previous interactive plot
'''

import numpy as np
from scipy import interpolate


def load_bkg_points(iplot, bkg_file=None):
    '''
    Load backround points to {iplot.background} object.
    
    * {iplot} must be known/pre-defined and contain {iplot.background} object
    * {iplot.background.bname} contains name of input/output files
    * {iplot.background..bname}+'.bp' = name of file with bkg points to read

    Parameters
    ----------
    iplot : bground.api.InteractivePlot object
        From this objec, we take the iplot.background sub-object.
        The iplot.background sub-object stores
        bname, btype, XYpoints, and XYcurve.

    bkg_file : optional, str or PathLike object, default is None
        If {bkt_file} argument is given,
        the background points will be loaded from the specified {bkg_file}
        instead of the standard {iplot.background.bname}+'.bp' file.
        
    Returns
    -------
    iplot.background object
        The object is described above.
        In addition, the data are loaded into iplot.backround,
        i.e. the original iplot object is updated with the loaded background.
    '''
    
    # Prepare name bkg-file = file with background points
    if bkg_file is None:
        # bkg_file argument not given (default)
        # => we suppose that the filename is defined inside bkg_object
        bkg_file = str(iplot.background.bname) + '.bp'
    else:
        # bkg_file argument was specified (either str or PathLike object)
        # => convert to str and suppose that it the full name of bkg_file 
        bkg_file = str(bkg_file)
        
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
                    iplot.background.btype = line.split(":")[-1].strip()
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
                iplot.background.points.X.append(x)
                iplot.background.points.Y.append(y)
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
    return iplot.background


def save_bkg_points(iplot):
    '''
    Save bacgkround points to the output text file.
    
    * The bkg points are stored in {iplot.background} sub-object.
    * The bkg points output file will be {iplot.background.bname}+'.bp'.

    Parameters
    ----------
    iplot : bground.api.InteractivePlot object
        From this objec, we take the iplot.background sub-object.
        The iplot.background sub-object stores
        bname, btype, XYpoints, and XYcurve.

    Returns
    -------
    None
        The background points are stored in
        {iplot.background.bname}+'.bp' file.
    '''
  
    # Prepare variables
    # (bkg_type = background type, short name for convenience
    # (bkg_file = name of bkg-file = file with background points
    bkg_type = iplot.background.btype
    bkg_file = str(iplot.background.bname) + '.bp'

    # Go through the data in the bkg_object and write them to file
    with open(bkg_file, 'w') as fh:
        fh.writelines('# Background points\n')
        fh.writelines('# 2 columns: [X-coords, Y-coords]\n')
        fh.writelines(f'# Background correction type: {bkg_type}\n')
        X = iplot.background.points.X
        Y = iplot.background.points.Y
        for x,y in zip(X,Y): print(f'{x:10.1f}{y:10.1f}', file=fh)

    
def calculate_baseline(iplot):
    '''
    Calculate background
    = calculate interpolated background curve;
    the calculated background curve is saved within iplot.background object.
    
    Parameters
    ----------
    iplot : api.bground.InteractivePlot object
        From this object, we use two sub-objects
        in this function: data (original XY data)
        and background (background object = bground.bdata.XYbackground).
        
    Returns
    -------
    None
        The result is the updated iplot object,
        specifically the iplot.background sub-object,
        which should contain the following (re)calculated items:
        
        * iplot.background.curve.X = X-coordinates of the whole bkg curve
        * iplot.background.curve.Y = X-coordinates of the whole bkg curve
    '''
    # (1) Prepare background points = X,Y coordinates for interpolation
    X,Y = (iplot.background.points.X, iplot.background.points.Y)
    # (2) Interpolate background points = calculcate background curve
    try:
        # Interpolation = calculation of interpolation function F.
        # (F = interpolation object/function
        # (with which we easily calculate the interpolated data - see below
        F = interpolate.interp1d(X,Y, kind=iplot.background.btype)
        # Prepare X-range = X-limits for the future background curve
        Xmin = iplot.background.points.X[0]
        Xmax = iplot.background.points.X[-1]
        # Prepare X-data = X-coordinates in range (Xmin...Xmax)
        Xdata = iplot.data
        Xdata = Xdata[0, (Xmin<=Xdata[0]) & (Xdata[0]<=Xmax)]
        # Calculate Y-data = Y-coordinates of the background curve
        Ydata = F(Xdata)
        # Save the calculated background curve in iplot.background object
        iplot.background.curve.X = Xdata
        iplot.background.curve.Y = Ydata
    except Exception as err:
        # Exceptions: interpolation can fail for whatever reason
        # In such a case we print the error and continue ...
        print(type(err))
        print(err)
        

def calculate_bkg_data(iplot):
    '''
    Final calculation/update of background data = numpy array with XY-data.
    
    Parameters
    ----------
    iplot : api.bground.InteractivePlot object
        From this object, we use two sub-objects
        in this function: data (original 2-col data => convert to 4-col)
        and background (background object with interploated background curve).

    Returns
    -------
    iplot.data : iplot sub-object, np.ndarray
        The array with 4 rows `[X, Iraw, Ibkg, I = Iraw - Ibkg]`.
        Morover, the data are saved in the iplot.data sub-object anyway.
    
    Technical notes
    ---------------
    The results of the calculation will be saved in two objects:
    iplot.data + iplot.background.
    
    * The original iplot.data object contains two columns:
      `[X, Iraw]`
    * The recalculated iplot.data object will contain four columns:
      `[X, Iraw, Ibkg, I = Iraw-Ibkg]`
    * Where X = X-coordinate
      and Iraw, Ibkg a I = raw, background and net/final/corrected intensity.
    '''
    # (0) Recalculate baseline
    # (in most cases, baseline is calculated when we call this function
    # (BUT calculation might be omitted or have just read the background points
    calculate_baseline(iplot)
    # (1) Modify the data variable if needed.
    number_of_rows = iplot.data.shape[0]
    # If number_of_rows == 2 then add two new rows.
    # => the func is called for the 1st time and data have just 2 original rows
    data = iplot.data  # shorter name, used below multiple times 
    if number_of_rows == 2:      
        data = np.insert(data,2,[data[1],data[1]],0)
    # If number_of_rows != 2  then rewrite two additional rows with data[1].
    # => func is called for 2nd, 3rd ... time and data already have 4 rows
    else:
        data[2],data[3] = data[1],data[1]   
    # (2) Get Xmin and Xmax of background/baseline curve.
    Xmin = iplot.background.points.X[0]
    Xmax = iplot.background.points.X[-1]
    # (3) Set range in which the background/baseline is defined.
    # (in this package, we define background only inside the selected range
    bkg_range = (Xmin<=data[0]) & (data[0]<=Xmax)
    # (4) Define 2nd data row = Ibkg = baseline.
    # (the baseline will contain zeros outside bkg_range
    # (a) Zero intensities outside bkg_range.
    data[2] = np.where(bkg_range,data[2],0)
    # (b) Baseline intentities inside bkg_range
    data[2,bkg_range] = iplot.background.curve.Y
    # (5) Define 3rd data row  = I = Iraw - Ibkg = net intensity.
    # (the net intensity will contain zeros outsice bkg_range
    # (a) Zero intensities outside bkg_range.
    data[3] = np.where(bkg_range,data[3],0)
    # (b) Net intensities inside bkg_range = raw_intensities - bkg_intensities
    data[3,bkg_range] = data[3,bkg_range] - iplot.background.curve.Y
    # (c) Set possible negative intensities after bkgr subtraction to zero
    data[3,data[3]<0] = 0
    # (b) Return modified data array
    # (the two rows of the modified array contain baseline and net intensity
    iplot.data = data
    return(iplot.data)


def save_bkg_data(iplot):
    '''
    Save background data to file.
    '''
    file_header = (
        'XY-data with background subtraction\n' +
        '4 columns: [X, Y=Iraw, Ibkg, I=(Iraw-Ibkg)]\n'+
        f'Background correction type: {iplot.background.btype}')
    np.savetxt(
        iplot.pars.bkg_file, 
        np.transpose(iplot.data),
        fmt=('%8.3f','%11.3e','%11.3e', '%11.3e'),
        header=file_header)
