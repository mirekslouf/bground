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

# Level 1: Define interactive plot -------------------------------------------- 

def create_plot(data, xlabel, ylabel='Intensity', xlim=[0,300], ylim=[0,300]):
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
    plt : maptplotlib.pyplot object
        The line plot showing XY data.
    '''
    # Step 1 in defining an interactive plot
    # = preparation of a standard plot, which will be made interactive later
    initialize_plot_parameters()
    X,Y = (data[0],data[1])
    plt.gcf().canvas.set_window_title('Background definition')
    plt.plot(X,Y, 'b-')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    return(plt)

def define_key_bindings(plt, data, bkgr, outfile):
    '''
    Define key bindings for plot = link plot with a callback function,
    which links selected [keypress_events] with further functions.
    
    Parameters
    ----------
    plt : matplotlib.pyplot object
        Prepared interactive XY-plot,
        for which we want to define new [key_press_events].
    data : 2D numpy array
        XY-data (which are shown in plt),
        which must be sent to functions after some [key_press_events].
    bkgr : bdata.bkg object
        Empty background object,
        which must be sent to functions after some [key_press_events].
    outfile : string
        Name of the output file,
        which must be sent to functions after some [key_press_events].
    
    Returns
    -------
    plt : maptplotlib.pyplot object
        The line plot showing XY data with defined key-bindings.
    '''
    # Step 2 in defining interactive plot
    # = link plt to a callback function (here: on_keypress),
    #   which links selected [keypress_events] to further functions. 
    # * Note1: all is based on standard matplotlib function canvas.mpl_connect
    # * Note2: the individual functions are defined below in on_keypress func.
    # ! Trick: we need an event with multiple arguments => lambda function
    plt.gcf().canvas.mpl_connect('key_press_event',
        lambda event: on_keypress(event, plt, data, bkgr, outfile))
    # return plt object, in which key bindings were defined
    return(plt)

def on_keypress(event, plt, data, bkg, outfile):
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
        print(f'Pressed {key:s}, mouse outside plot area - no action!')
    # Mouse inside graph area, run corresponding function.
    else:
        print(f'Pressed {key:s}, mouse coordinates {xm:.2f},{ym:2f}.')
        # Functions run by means try-except
        # Reason: to ignore nonsense actions...
        # ...such as delete/draw points if no points are defined
        try:
            if   key == '0': print_help(outfile)
            elif key == '1': add_bkg_point(plt,data,bkg,xm,ym)
            elif key == '2': delete_bkg_point(plt,bkg)
            elif key == '3': delete_bkg_point_close_to_mouse(plt,bkg,xm,ym)
            elif key == '4': replot_with_bkg_points(plt,data,bkg)
            elif key == '5': replot_with_bkg(plt,data,bkg,'linear')
            elif key == '6': replot_with_bkg(plt,data,bkg,'quadratic')
            elif key == '7': replot_with_bkg(plt,data,bkg,'cubic')
            elif key == 'a': print_bkg_points(bkg)
            elif key == 'b': save_bkg_points(bkg)
            elif key == 'c': load_bkg_points(plt,data,bkg)
            elif key == 't': subtract_bkg_and_save(plt, outfile, data, bkg)
        except Exception:
            pass

# Auxiliary functions (to Level 1 = defining ineractive plot) .................

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
    print('Activate interactive plot window and press:')
    print('0 = to print brief help in notebook/console window')
    print('1 = to draw background point at current mouse position')
    print('2 = to delete background point...')
    print('---')

# Level 2: Functions for individual events ------------------------------------

def print_help(outfile):
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
    print('2 = delete background point - last one')
    print('3 = delete background point - closest to mouse')
    print('4 = re-draw plot with background points')
    print('5 = re-draw plot with linear spline background')
    print('6 = re-draw plot with quadratic spline background')
    print('7 = re-draw plot with cubic spline background')
    print('------')
    print('a = background points :: print')
    print('b = background points :: save to BKG-file') 
    print('c = background points :: load from BKG-file')
    print('(BKG-file = %s' % (outfile+'.bkg'))
    print('------')
    print('s = save current image as PNG (default matplotlib shortcut')
    print('t = subtract current background & save data to TXT-file')
    print('(TXT-file = %s' % outfile)
    print('------')
    print('All standard matplotlib tools and shortcuts work as well.')
    print('See: https://matplotlib.org/stable/users/interactive.html')
    print('===========================================================')
       
def add_bkg_point(plt, data, bkg, xm, ym):
    '''
    Function for pressed key = 1:
    Add background point to at current mouse position.
    More precisely: add background point at the XY-point,
    whose X-coordinate is the closest to the mouse X-coordinate.
    '''
    idx = find_nearest(data[0],xm)
    xm,ym = (data[0,idx],data[1,idx])
    bkg.points.add_point(xm,ym)
    plt.plot(xm,ym,'r+')
    plt.draw()
    
def delete_bkg_point(plt, bkg):
    '''
    Function for pressed key = 2:
    Remove background point (the last inserted).
    '''
    xr = bkg.points.X.pop()
    yr = bkg.points.Y.pop()
    plt.plot(xr,yr, 'w+')
    plt.draw()

def delete_bkg_point_close_to_mouse(plt, bkg, xm, ym):
    '''
    Function for pressed key = 3:
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

def replot_with_bkg_points(plt, data, bkg):
    '''
    Function for pressed key = 4:
    Re-draw plot with backround points.
    '''
    clear_plot()
    plt.plot(data[0],data[1],'b-')
    plt.plot(bkg.points.X,bkg.points.Y,'r+')
    plt.draw()

def replot_with_bkg(plt, data, bkg, itype):
    '''
    Function for pressed keys = 5,6,7:
    Re-draw plot with backround points and background curve.
    * Type of the curve is given by parameter itype.
    * For key = 5/6/7. the function called with itype = linear/quadratic/cubic.
    '''
    bfunc.sort_bkg_points(bkg)
    bkg.itype = itype
    bfunc.calculate_background(data, bkg)
    clear_plot()
    plt.plot(data[0],data[1],'b-')
    plt.plot(bkg.points.X,bkg.points.Y,'r+')
    plt.plot(bkg.curve.X,bkg.curve.Y,'r:')
    plt.draw()
    
def subtract_bkg_and_save(plt, outfile, data, bkg):
    '''
    Function for pressed key = 8:
    This is the final function which:
        a) Recalculates recently defined background
        b) Calculates background-corrected data = subtracts bkg from data
        c) Saves the results to output file = TXT with 3 cols [X,Int,CInt]
    '''
    print('Subtract recently defined background and save results.')
    print('a) Recalculating background...')
    bfunc.calculate_background(data,bkg)
    print('b) Subtracting background...')
    data = bfunc.subtract_background(data,bkg)
    print('c) Saving background to TXT-file...')
    np.savetxt(
        outfile, np.transpose(data), fmt=('%8.3f','%11.3e','%11.3e'),
        header='Columns: X, Intensity, Background-corrected intensity')
    print('File with backround-corrected intensities saved:')
    print('[%s]' % outfile)
    print('-----')

def print_bkg_points(bkg):
    '''
    Function for pressed key = a:
    Print background points in the console window.
    '''
    bfunc.sort_bkg_points(bkg)
    df = bkg_to_df(bkg)
    print('Current background points:')
    print(df.to_string())
    print('-----')

def save_bkg_points(bkg):
    '''
    Function for pressed key = b:
    Save background points to file.
    (basename of output file is saved in bkg.basename).
    '''
    bfunc.sort_bkg_points(bkg)
    output_filename = bkg.basename + '.bkg'
    df = bkg_to_df(bkg)
    with open(output_filename, 'w') as f:
        f.write(df.to_string())
        print('Background points saved to:')
        print('[%s]' % output_filename)
        print('-----')

def load_bkg_points(plt, data, bkg):
    '''
    Function for pressed key = c:
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
    print(df)
    # c) initialize bkg object by means of above-read DataFrame
    bkg.points = bdata.XYpoints(X = list(df.X), Y = list(df.Y))
    bkg.itype='linear'
    # d) print message & replot with currently loaded background
    print('Background points read from file:')
    print('[%s]' % input_filename)
    print('-----')
    replot_with_bkg_points(plt, data, bkg)
    
# Auxiliary functions (to level 2 = individual events) ........................

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
