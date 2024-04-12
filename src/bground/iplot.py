'''
Module bground.iplot
--------------------
The module defines functions that can create an interactive plot.

* The interactive plot is defined at two levels.
* Level 1 = a general interactive plot = a plot linked with keypress events.
* Level 2 = specific functions for the individual keypress events.
* We define just keypress events, while mouse events = matplotlib defaults.
* The default matplotlib mouse events are very good - no reason to change them.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from bground import bdata, bfunc
import warnings; warnings.filterwarnings("ignore")


# Level 1: Define plot with events -------------------------------------------- 

def interactive_plot(data, bkg, ppar):
    '''
    Create plot from input data.
    
    This is a plot window, which will be made interactive later.
    In the rest of the program, the plot will be the program interface.

    Parameters
    ----------
    data : 2D numpy array
        Data for plotting; columns [X,Y].
    xlabel : str
        Label of X-axis.
    ylabel : str, default is 'Intensity'
        Label of Y-axis.
    xlim : list or tuple (containing two values)
        Lower and upper limit of X in the plot; the default is [0,300].
    ylim : list or tuple (containing two values)
        Lower and upper limit of Y in the plot; the default is [0,300].
    
    Returns
    -------
    fig,ax : maptplotlib.pyplot objects
        The figure and axis of the interactive plot which shows XY-data.
    '''
    
    # (0) Initialize
    plt.close('all')  # Close all previous plots - to avoid mess in Jupyter
    initialize_plot_parameters()
    
    # (1) Prepare the plot: fig,ax including window title
    # (num argument below can take both integers and strings
    fig,ax = plt.subplots(num='Background definition')

    # (2) Read XY data and create the plot
    # Get XY data from the function argument data
    X,Y = (data[0],data[1])
    # Plot XY data
    ax.plot(X,Y, 'b-')
    # Set the remaining plot parameters
    ax.set_xlim(ppar.xlim)
    ax.set_ylim(ppar.ylim)
    ax.set_xlabel(ppar.xlabel)
    ax.set_ylabel(ppar.ylabel)
    
    # (3) Connect the plot with keypress events
    # = link fig.canvas events to a callback function.
    #   Here: events = key_press_events, callback function = on_keypress.
    #   The callback function links the events to further user-defind funcs.
    # * Note1: all is based on standard matplotlib function canvas.mpl_connect
    # * Note2: the individual functions are defined below.
    # ! Trick: we need an event with multiple arguments => lambda function
    fig.canvas.mpl_connect('key_press_event',
        lambda event: on_keypress(event, fig, ax, data, bkg, ppar))
    
    # (4) Optimize the plot layout
    plt.tight_layout()
    
    # (5) Return fig,ax
    # (This is necessary, among others, fof Jupyter + %matplotlib widget
    return(fig,ax)


# Level 2: Callback function for all events -------------------------------------------- 

def on_keypress(event, fig, ax, data, bkg, ppar):
    '''
    Definition of events for a plot.
    Master callback procedure, which defines all keypress events.
    '''
    # Step 3 in defining interactive plot
    # = defining individual functions for specific pressed keys.
    # -----
    # Read pressed key and mouse coordinates
    key = event.key
    xm,ym = event.xdata,event.ydata
    # Mouse outside graph area - just print warning!
    if xm == None or ym == None:
        print(f'Key [{key}] mouse outside plot area - no action!')
    # Mouse inside graph area, run corresponding function.
    else:
        print(f'Key [{key:s}] mouse [{xm:.1f},{ym:.1f}]', end=' ')
        # Functions run by means try-except
        # Reason: to ignore nonsense actions...
        # ...such as delete/draw points if no points are defined
        try:
            if   key == '0': print_help()
            elif key == '1': add_bkg_point(plt,data,bkg,xm,ym)
            elif key == '2': delete_bkg_point_close_to_mouse(plt,bkg,xm,ym)
            elif key == '3': replot_with_bkg_points(plt,data,bkg)
            elif key == '4': replot_with_bkg(plt,data,bkg,'linear')
            elif key == '5': replot_with_bkg(plt,data,bkg,'quadratic')
            elif key == '6': replot_with_bkg(plt,data,bkg,'cubic')
            elif key == 'b': save_bkg_points(bkg)
            elif key == 'c': load_bkg_points(plt,data,bkg)
            elif key == 't': subtract_bkg_and_save(plt, data, bkg, ppar)
            else           : print()
        except Exception:
            pass


# Level 3: Functions for individual events ------------------------------------

def print_help(ppar):
    '''
    Function for pressed key = 0:
    Print help for all pre-defined keys to console window.
    '''
    print()
    print('==========================================================')
    print('BGROUND :: Interactive background removal :: Brief help')
    print('----------------------------------------------------------')
    print('0 = print this help')
    print('1 = add background point')
    print('2 = delete background point - closest to mouse')
    print('3 = re-draw plot with background points')
    print('4 = re-draw plot with linear spline background')
    print('5 = re-draw plot with quadratic spline background')
    print('6 = re-draw plot with cubic spline background')
    print('------')
    print('b = background points :: save to BKG-file') 
    print('c = background points :: load from BKG-file')
    print('f(BKG-file = {ppar.outfile}' + '.bkg')
    print('------')
    print('s = save current image as PNG (default matplotlib shortcut')
    print('t = subtract current background & save data to TXT-file')
    print('f(TXT-file = {ppar.outfile}')
    print('------')
    print('All standard matplotlib tools and shortcuts work as well.')
    print('See: https://matplotlib.org/stable/users/interactive.html')
    print('===========================================================')
       

def add_bkg_point(plt, data, bkg, xm, ym):
    '''
    Function for keypress = '1'.
    Add background point to at current mouse position.
    More precisely: add background point at the XY-point,
    whose X-coordinate is the closest to the mouse X-coordinate.
    '''
    idx = find_nearest(data[0],xm)
    xm,ym = (data[0,idx],data[1,idx])
    bkg.points.add_point(xm,ym)
    plt.plot(xm,ym,'r+')
    plt.draw()
    print('background point added.')
    

def delete_bkg_point_close_to_mouse(plt, bkg, xm, ym):
    '''
    Function for keypress = '2'.
    Remove background point (the point closest to the mouse position).
    More precisely: remove background point,
    whose X-coordinate is the closest to the mouse X-coordinate.
    
    '''
    # a) Sort bkg points (sorted array is necessary for the next step)
    bfunc.sort_bkg_points(bkg)
    # b) Find index of background point closest to the mouse X-position
    idx = find_nearest(np.array(bkg.points.X), xm)
    # c) Remove element with given index from X,Y-lists (save coordinates)
    xr = bkg.points.X.pop(idx)
    yr = bkg.points.Y.pop(idx)
    # d) Redraw removed element with background color
    plt.plot(xr,yr, 'w+')
    # e) Redraw plot
    plt.draw()
    # f) Print message to stdout.
    print('background point deleted.')


def replot_with_bkg_points(plt, data, bkg, message_to_stdout=True):
    '''
    Function for keypress = '3'.
    Re-draw plot with backround points.
    '''
    clear_plot()
    plt.plot(data[0],data[1],'b-')
    plt.plot(bkg.points.X,bkg.points.Y,'r+')
    plt.draw()
    # Print message to stdout if requested
    # (This is requested by default, but it can be omitted
    # (If called from other functions, we do not want additional messages
    if message_to_stdout:
        print('backround points re-drawn.')

def replot_with_bkg(plt, data, bkg, itype):
    '''
    Function for keypress = '4,5,6'.
    Re-draw plot with backround points and background curve.
    * Type of the curve is given by parameter itype.
    * For key = 4/5/6 the function called with itype = linear/quadratic/cubic.
    '''
    bfunc.sort_bkg_points(bkg)
    bkg.itype = itype
    bfunc.calculate_background(data, bkg)
    clear_plot()
    plt.plot(data[0],data[1],'b-')
    plt.plot(bkg.points.X,bkg.points.Y,'r+')
    plt.plot(bkg.curve.X,bkg.curve.Y,'r:')
    plt.draw()
    if bkg.itype == 'linear':
        print('linear background displayed.')
    elif bkg.itype == 'quadratic':
        print('quadratic background displayed.')
    elif bkg.itype == 'cubic':
        print('cubic background displayed.')

def save_bkg_points(bkg):
    '''
    Function for keypress = 'b'.
    Save background points to file.
    (basename of output file is saved in bkg.basename).
    '''
    bfunc.sort_bkg_points(bkg)
    output_filename = bkg.basename + '.bkg'
    df = bkg_to_df(bkg)
    with open(output_filename, 'w') as f:
        f.write(df.to_string())
        print(f'background points saved to: [{output_filename}].')


def load_bkg_points(plt, data, bkg):
    '''
    Function for keypress = 'c'.
    Load background points from previously saved file
    (basename of input file with background points saved in bkg.basename).
    '''
    # a) get input file with previously saved background points
    # (the filename is fixed to [output_file_name].bkg
    # (reason: inserting a name during an interactive plot session is a hassle
    # (solution: manual renaming of the BKG-file before running this program
    input_filename = bkg.basename + '.bkg'
    # b) read input file to DataFrame
    df = pd.read_csv(input_filename, sep='\s+')
    # c) initialize bkg object by means of above-read DataFrame
    bkg.points = bdata.XYpoints(X = list(df.X), Y = list(df.Y))
    bkg.itype='linear'
    # d) print message & replot with currently loaded background
    print(f'background points read from: [{input_filename}].')
    replot_with_bkg_points(plt, data, bkg, message_to_stdout=False)
    

def subtract_bkg_and_save(plt, data, bkg, ppar):
    '''
    Function for keypress 't'.
    This is the final function which:
        a) Recalculates recently defined background
        b) Calculates background-corrected data = subtracts bkg from data
        c) Saves the results to TXT-file with 3 cols [X, Y, bkg-corrected-Y]
    '''
    # Subtract recently defined background and save results
    # (a) Recalculate background
    bfunc.calculate_background(data,bkg)
    # (b) Subtract background
    data = bfunc.subtract_background(data,bkg)
    # (c) Save background-corrected data to TXT-file
    # (we will use ppar object properties for this
    # (ppar.output_file = output file name, ppar.xlabel = label of X-data...
    my_file_header = (
        f'Columns: {ppar.xlabel}, {ppar.ylabel}, ' +
        f'background-corrected-{ppar.ylabel}\n' +
        f'Background correction type: {bkg.itype}')
    np.savetxt(
        ppar.output_file, np.transpose(data), fmt=('%8.3f','%11.3e','%11.3e'),
        header=my_file_header)
    print(f'backround-corrected data saved to: [{ppar.output_file}].')
    

# Auxiliary functions (to levels 1,2) .........................................

def initialize_plot_parameters():
    '''
    Initialize parameters for plotting.
    '''
    plt.rcParams.update({
        'figure.figsize' : (6,4),
        'figure.dpi' : 100,
        'font.size' : 12})


def print_ultrabrief_help():
    '''
    Print ultra-brief help in console window before activating the plot.
    '''
    print('(0) A window named {Background definition} with XY-data will open.')
    print('(1) Click on this window to activate it.')
    print('(2) Use default mouse events and icons to adjust plot.')
    print('(3) Use user-defined keys/events to define the background.')
    

# Auxiliary functions (to level 3 = individual events) ........................

def find_nearest(arr, value):
    '''
    Find index of the element with nearest value in 1D-array.
    
    Parameters
    ----------
    arr : 1D numpy array
        The array, in which we search the element with closest value.
        Important prerequisite: the array must be sorted.
    value : float
        The value, for which we search the closest element.

    Returns
    -------
    idx : int
        Index of the element with the closest value.
    '''
    # Find index of the element with nearest value in 1D-array.
    # Important prerequisite: the array must be sorted.
    # https://stackoverflow.com/q/2566412
    # 1) Key step = np.searchsorted
    idx = np.searchsorted(arr, value, side="left")
    # 2) finalization = consider special cases and return final value
    if idx > 0 and (
            idx == len(arr) or abs(value-arr[idx-1]) < abs(value-arr[idx])):
        return(idx-1)
    else:
        return(idx)

def clear_plot():
    '''
    Auxilliary function: clear plot before re-drawing.
    Note: the functions keeps current labels and XY-limits of the plot.
    '''
    my_xlabel = plt.gca().get_xlabel()
    my_ylabel = plt.gca().get_ylabel()
    my_xlim = plt.xlim()
    my_ylim = plt.ylim()
    plt.cla()
    plt.xlabel(my_xlabel)
    plt.ylabel(my_ylabel)
    plt.xlim(my_xlim)
    plt.ylim(my_ylim)

def bkg_to_df(bkg):
    '''
    Convert current background points to dataframe.
    Reason: df can be used to print/save background points nicely.
    '''
    # Convert bkg to DataFrame to get nicely formated output
    # (our trick: df.to_string & then print/save to file as string
    # (more straightforward: df.to_csv('something.txt', sep='\t')
    # (BUT the output with to_string has better-aligned columns
    df = pd.DataFrame(
        np.transpose([bkg.points.X, bkg.points.Y]), columns=['X','Y'])
    return(df)
