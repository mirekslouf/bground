'''
Module: bground.api
-------------------

Definition of API for (application programming interface) of BGROUND package.

* The API can be employed as a simple UI within Spyder and/or Jupyter scripts.
* The main purpose of the API - easy access to functions for bkgr subtraction.

Simple example how to get help and run a background subtraction method:
    
>>> # Standard import
>>> # (alternative: import ediff.bkg as bkg
>>> import bground.api as bkg
>>> 
>>> # Simple help system
>>> # (note: universal access by means of bground.api.Help class
>>> bkg.Help.intro()
>>> bkg.Help.more_help()
>>> bkg.Help.InteractivePlot()
>>>
>>> # Run selected background subtraction method
>>> # (note: universal access by means of bground.api.Run class
>>> bkg.Run.InteractivePlot()

More help to the individual background subtraction methods:

* bground.api.InteractivePlot
  = semi-automatatic method, universal, finished  
* bground.api.FitFunction
  = automatic method, simple fitting, TODO - Edvard
* bground.api.BaseLines
  = automatic method, advanced fitting, TOFINISH - Jakub
* bground.api.WaveletMethod
  = automatic method, wavelet-based fitting, TODO - Edvard
'''


# Bground components
import bground.help
# {points} sub-package
import bground.points.bdata 
import bground.points.bfunc
import bground.points.iplot

import bground.blines.blines

# Reading and analyzing input data
import numpy as np
import pandas as pd
from pathlib import Path

# Plotting
import matplotlib
import matplotlib.pyplot as plt


def set_plot_parameters(
        size=(10,5), dpi=100, fontsize=8, my_defaults=True, my_rcParams=None):
    '''
    Set global plot parameters (mostly for plotting in Jupyter).

    Parameters
    ----------
    size : tuple of two floats, optional, the default is (8,6)
        Size of the figure (width, height) in [cm].
    dpi : int, optional, the defalut is 100
        DPI of the figure.
    fontsize : int, optional, the default is 8
        Size of the font used in figure labels etc.
    my_defaults : bool, optional, default is True
        If True, some reasonable additional defaults are set,
        namely line widths and formats.
    my_rcParams : dict, optional, default is None
        Dictionary in plt.rcParams format
        containing any other allowed global plot parameters.

    Returns
    -------
    None
        The result is a modification of the global plt.rcParams variable.
    '''
    # (1) Basic arguments -----------------------------------------------------
    if size:  # Figure size
        # Convert size in [cm] to required size in [inch]
        size = (size[0]/2.54, size[1]/2.54)
        plt.rcParams.update({'figure.figsize' : size})
    if dpi:  # Figure dpi
        plt.rcParams.update({'figure.dpi' : dpi})
    if fontsize:  # Global font size
        plt.rcParams.update({'font.size' : fontsize})
    # (2) Additional default parameters ---------------------------------------
    if my_defaults:  # Default rcParams if not forbidden by my_defaults=False
        plt.rcParams.update({
            'lines.linewidth'    : 0.8,
            'axes.linewidth'     : 0.6,
            'xtick.major.width'  : 0.6,
            'ytick.major.width'  : 0.6,
            'grid.linewidth'     : 0.6,
            'grid.linestyle'     : ':'})
    # (3) Further user-defined parameter in rcParams format -------------------
    if my_rcParams:  # Other possible rcParams in the form of dictionary
        plt.rcParams.update(my_rcParams)


class InputData:
    '''
    Class defining {InputData} for {InteractivePlot}.
    
    * Input data can be: file, np.array, pd.DataFrame, ediff.io.Profile.
    * The usage of InputData class is shown in the example above.
    * The rest of the documentation => detailed comments in the source code.
    '''
    
    def __init__(self, input_data, **kwargs):
        # Initialization of InputData object.
        # No docstring: just class description above + comments below.
        
        # Call read_input_data method with **kwargs:
        # The method reads input data and returns:
        #  - self.name = name of the data, filename or variable name
        #  - self.data = numpy array with two rows ~ X-data, Y-data
        # The input_data may be:
        #  - text file containing several columns
        #    (then **kwargs are passed to pd.read_csv
        #  - numpy array
        #    (user responsibility: array contains just two rows ~ X,Y
        #  - pandas DataFrame
        #    (user responsibility: DataFrame contains just two cols ~ X,Y)
        #  - data saved in Profile object from ediff package
        #    (program responsibility: df['pixel'] ~ X, df['Iraw'] ~ Y
        self.name, self.data, self.profile = \
            self.read_input_data(input_data, **kwargs)
        
        
    def read_input_data(self, input_data, **kwargs): 
        # Read input data with XY-data.
        # No docstring: just class description above + comments below.
        # * The optional arguments (comment, sep, usecols)
        #   are passed to pandas.read_csv if the input_data ~ filename.
    
        if isinstance(input_data, (str, Path)):
            # The input is a file = filename given be str or PathLike var.
            # => read the file using pd.read_csv
            # => user's responsibility: modify optional args as needed
            #    the optional arguments are: comment, sep, usecols
            #    these arguments are passed to pd.read_csv
            # (a) set name = filename
            name = str(input_data)
            # (b) get **kwargs for pd.read_csv OR set reasonable defaults
            # ... comments may be present
            my_comment  = kwargs.get('comment') or '#'
            # ... header should be None if there are no column headers!
            my_header   = kwargs.get('header') or 'infer'
            # ... skiprows can be defined for completness and flexibility
            my_skiprows = kwargs.get('skiprows') or 0
            # ... separator can be defined, defalt is any whitespace
            my_sep      = kwargs.get('sep') or r'\s+'
            # ... usecols may be defined, default is the first two columns
            my_usecols  = kwargs.get('usecols') or [0,1]
            # ... columns may be re-defined OR default column names are used
            my_columns  = kwargs.get('columns') or None
            # (c) read the file with pd.read_csv and the **kwargs from above 
            df = pd.read_csv(input_data, 
                comment=my_comment, header=my_header, skiprows=my_skiprows, 
                sep=my_sep, usecols=my_usecols)
            if my_columns is not None: df.columns = my_columns
            # (d) convert pd.DataFrame to np.ndarray
            #     and transpose the array (columns should be the 1st index)
            data = np.transpose(np.array(df))
            # (e) set that the data are NOT the special ed.io.Profile
            profile = None
        elif isinstance(input_data, np.ndarray):
            # The input is a numpy array.
            # => just assign array to data
            # => user's responsibility: ensure that the numpy.array
            #    contains just two rows: X-values and Y-values
            #    (for example, pass something like: arr[[0,2]]
            name = 'np.ndarray'
            data = input_data
            profile = None
        elif type(input_data) == pd.DataFrame:
            # The input is pandas.DataFrame
            # => convert DataFrame to numpy.array
            # => user's responsibility: ensure that the pandas.Dataframe
            #    contains just two columns: X-values and Y-values
            #    (for example, pass something like: df[['Pixel','Iraw']]
            # * we use type(input_data)==pd.DataFrame instead of isinstance
            #   we need to do this due the the next option = {ediff.io.Profile}
            #   => {ediff.io.Profile} is-instance of pd.DataFrame
            #   => we would think that {Profiles} are DataFrames 
            name = 'pd.DataFrame'
            data = np.transpose(np.array(input_data))
            profile = None
        elif input_data.__class__.__name__ == 'Diffractogram1D':
            # The input is ediff.io.Profile.
            # (special case, input from our own, connected package ediff
            # => convert Profile to numpy.array
            # => the Profile has more-or-less fixed structure
            #    the array is created from Profile columns {Pixel} a {Iraw}
            # * we use var.__class__.__name__ instead of type
            #   we DO NOT KNOW ediff.io.Profile here - it is not imported
            #   we could import this, but then bground would depend on ediff
            name = 'ediff.io.Diffractogram1D'
            data = np.transpose(np.array(input_data[['Pixels','Iraw']]))
            profile = input_data
        else:
            raise TypeError('Unknown type of input data!')
         
        # Return the result = 2xN numpy array with XY-data.
        # (i.e. data[0] = X-data/values, data[1] = Y-data/values)
        return(name, data, profile)


class OutputParams:
    '''
    The name of output file(s) + plot parameters + output verbosity.
    
    * Output file name(s) = file(s) with the background calculation results.
    * The plot parameters are X,Y-axis labels + X,Y-axes ranges/limits.
    
    The usage of OutputParams class is shown in the example above.
    
    * The rest of the documentation => detailed comments in the source code.
    '''
    
    def __init__(self, output_file, 
                 xlabel=None, ylabel=None, xlim=None, ylim=None, 
                 messages=True):
        # Initialization of PlotParams object.
        # No docstring: just class description above + comments below.
        
        # Brief description of PlotParams object:
        #  * The object defines the output file(s) name
        #    and interactive plot parameters.
        #  * The output file will store the background-corrected data.
        #  * The interactive plot is a simple user interface,
        #    by means of which the background is defined and calculated.
        #    The plot parameters adjust the plot as described below.
        
        self.output_file = output_file  # Name of the output file(s)
        self.xlabel = xlabel            # x-axis label of the interactive plot
        self.ylabel = ylabel            # y-axis label of the interactive plot
        
        if isinstance(xlim,list): self.xlim = xlim  # xlim = [xmin,xmax]
        else: self.xlim = [0,xlim]                  # ...or just xmax
        
        if isinstance(ylim,list): self.ylim = ylim  # ylim = [ymin,ymax]
        else: self.ylim = [0,ylim]                  # ...or just ymax
        
        self.messages = messages        # Printing of short messages to stdout
        
        # Note: messages argument determines,
        # if we print a short messages on stdout when the plot is interactive.
        # The argument can be overrident by InteractivePlot object;
        # the reasons are explained below in InteractivPlot object definition.


class InteractivePlot:
    '''
    InteractivePlot method of backround subtraction.

    * When running the method, a new window with interactive plot is opened.
    * The user defines background points using mouse and keyboard shortcuts.
    * The program does the rest - subtracts bkgr and shows/saves the results.
            
    Example 1 :: classical, step-by-step way
        
    >>> # Standard import (alternative: import ediff.bkg as bkg)
    >>> import bground.api as bkg
    >>>
    >>> # Define I/O files
    >>> IN  = 'ed1_raw.txt'
    >>> OUT = 'ed2_bcorr.txt'
    >>>
    >>> # Define input data and output/plot parameters
    >>> DATA = bkg.InputData(IN, usecols=[0,1], unpack=True)
    >>> PPAR = bkg.PlotParams(OUT,'Pix','Intensity',xlim=[0,200],ylim=[0,180])
    >>>
    >>> # Initialize and run InteractivePlot subtraction method
    >>> # (a new window with interactive plot is opened
    >>> # (user defines background points with the mouse and keyboard
    >>> # (ouput files are saved automatically at the end of processing 
    >>> SMET = bkg.WaveletMethod(DATA, PPAR)
    >>> SMET.run()
    
    Example 2 :: simplified, one-step approach
    
    >>> # Standard import (alternative: import ediff.bkg as bkg)
    >>> import bground.api as bkg
    >>>
    >>> # Define I/O files
    >>> IN  = 'ed1_raw.txt'
    >>> OUT = 'ed2_bcorr.txt'
    >>>
    >>> # Run InteractivePlot subtraction method with a single function
    >>> # (the function initializes all objects and runs the method
    >>> bkg.Run.InteractivePlot(IN_FILE, OUT_FILE, 
    >>>     xlabel='Pixels', ylabel='Intensity', xlim=300, ylim=300)
    '''

    
    def __init__(self, DATA, PARS, CLI=False):
        # Initialization of InteractivePlot object.
        # No docstring: just class description above + comments below.
        
        # Brief description:
        # * This object defines the interactive plot parameters.
        # * The interactive plot is a simple user interface,
        #   by means of which the background is defined and calculated.
        # * InteractivePlot object takes three arguments/properties:
        #   - DATA = InputData object above = input data
        #   - PPAR = PlotParameters object above = plot params + output file
        #   - CLI = set CLI=Trure if the program runs in command line interface 

        # Basic properties
        # (DATA = InputData object
        # (PPAR = PlotParameters object
        # (these two objects should be defined BEFORE InteractivePlot object
        self.data = DATA
        self.pars = PARS
        
        # Additional property - empty XYbackground object
        # (this object is defined as a semi-empty object here
        # (the only argument we supply is the name of the output file
        self.background = \
            bground.points.bdata.XYbackground(self.pars.output_file)
            
        # Initialize specific interactive backend
        # in case Python runs in CLI = Command Line Interface,
        # i.e. if the program runs outside Spyder or Jupyter environments
        if CLI == True:
            matplotlib.use('QtAgg')

        
    def run(self):
        '''
        Run the interactive plot.
        
        Technical notes
        ---------------
        * This method is very simple - it just runs the interactive plot.
        * No parameters => everything is defined in InteractivePlot object.
        '''

        # Run interactive plot.
        # No docstring: just class description above + comments below.
        
        # Brief description:
        # * This method runs the interactive plot.
        # * In addition, it does a few minor things.
        # * It takes no parameters - everything is in InteractivePlot object.
        
        # Clear background points from possible previous runs.
        # (Possible issue in Jupyter, when re-running cell with the command.
        self.background.points.X = []
        self.background.points.Y = []
        
        # Run the interactive plot
        # (the return values seem to be necessary for current Jupyter interface
        fig,ax = bground.points.iplot.interactive_plot(
            self.data.data, self.background, self.pars, self.data.profile)
        
        # The plot appears in a new window
        # (if we use the recommended %matplotlib qt).
        # In addition to this, we print a brief help to stdout
        # (CLI in standard python, Console in Spyder, output cell in Jupyter).
        bground.points.iplot.print_brief_help(self.pars)
        
        # Optimize the layout of the returned figure.
        # (this works in all three supported interfaces: CLI, Spyder, Jupyter)
        fig.tight_layout()
        
        # Show the plot
        plt.show()


    def plot_data_before_processing(
            self, title='Raw data before processing', grid=True):
        '''
        Plot raw XY-data BEFORE any processing.

        Parameters
        ----------
        title : str, optional
            Title of the plot.
            The default can be changed to None (no title) or any other string.
        grid : bool, optional, default is True
            If true, show grid in the plot.

        Returns
        -------
        None
            The result is the plot shown on the screen.
            
        Notes and limitations
        ---------------------
        * This is a supplementary method of bground.ui.InteractivePlot object.
        * The method can be used BEFORE the interactive plot is run.
        * *The method has a limited number of parameters*  
          as most of them are taken from calling InteractivePlot object.
        * *It can be used in Jupyter*   
          to visualize the background definition
          in the notebook after the interactive plot is closed
          and after we switch to non-interactive plots (%matplotlib inline).
        * *It will not work in CLI or Spyder*   
          because we cannot switch from interactive to non-interactive plots
          within one script.
          Moreover, there tend to be some hard-to-debug confusions
          connected with the fact that it is not so clear, what is the
          current active plot.
        '''        

        # Close all previous plots.
        # (necessary to avoid confusions about current plot in Jupyter
        plt.close('all')
        # Get XY-data
        X,Y = self.data.data
        # PLot XY-data
        plt.plot(X,Y, 'b-')
        if title is not None:
            plt.title(title)
        # ...add xy-labels and limits
        plt.xlabel(self.ppar.xlabel)
        plt.ylabel(self.ppar.ylabel)
        plt.xlim(self.ppar.xlim)
        plt.ylim (self.ppar.ylim)
        # ...add grid
        if grid is not None:
            plt.grid()
        # Show the final plot
        plt.tight_layout()
        plt.show()


    def plot_data_with_bkgr_definition(
            self, title='Data with background definition', grid=True):
        '''
        Plot XY-data and background AFTER the interactive plot is closed.

        Parameters
        ----------
        title : str, optional
            Title of the plot.
            The default can be changed to None (no title) or any other string.
        grid : bool, optional, default is True
            If true, show grid in the plot.

        Returns
        -------
        None
            The result is the plot shown on the screen.
            
        Notes and limitations
        ---------------------
        * This is a supplementary method of bground.ui.InteractivePlot object.
        * The method can be used AFTER the interactive plot is closed.
        * *The method has a limited number of parameters*  
          as most of them are taken from calling InteractivePlot object.
        * *It can be used in Jupyter*   
          to visualize the background definition
          in the notebook after the interactive plot is closed
          and after we switch to non-interactive plots (%matplotlib inline).
        * *It will not work in CLI or Spyder*   
          because we cannot switch from interactive to non-interactive plots
          within one script.
          Moreover, there tend to be some hard-to-debug confusions
          connected with the fact that it is not so clear, what is the
          current active plot.
        '''
        
        # Close all previous plots.
        # (necessary to avoid confusions about current plot in Jupyter
        plt.close('all')
        # Get XY-data
        X,Y = self.data.data
        # Get background points
        Xp = self.background.points.X
        Yp = self.background.points.Y
        # Get background interpolation curve
        Xc = self.background.curve.X
        Yc = self.background.curve.Y
        # PLot XY-data
        plt.plot(X,Y, 'b-')
        # ...add background points
        plt.plot(Xp,Yp, 'r+')
        # ...background interpolation curve
        plt.plot(Xc,Yc, 'r--')
        # ...add title
        if title is not None:
            plt.title(title)
        # ...add xy-labels and limits
        plt.xlabel(self.ppar.xlabel)
        plt.ylabel(self.ppar.ylabel)
        plt.xlim(self.ppar.xlim)
        plt.ylim (self.ppar.ylim)
        # ...add grid
        if grid is not None:
            plt.grid()
        # Show the final plot
        plt.tight_layout()
        plt.show()

        
    def plot_data_after_bkgr_subtraction(
            self, title='Data after background subtraction', 
            xlim=None, ylim=None, grid=True):
        '''
        Show background-corrected XY-data AFTER the interactive plot is closed.

        Parameters
        ----------
        title : str, optional
            Title of the plot.
            The default can be changed to None (no title) or any other string.
        xlim : tuple or list with two values, default is None
            X-axis limits [xmin,xmax].
            If the default value is unchanged,
            the limits are taken from self.ppar.xlim.
        ylim : tuple or list with two values, default is None
            Y-axis limits [ymin,ymax].
            If the default value is unchanged,
            the limits are taken from self.ppar.xlim.
        grid : bool, optional, default is True
            If the argument is True, a grid is added to the plot.

        Returns
        -------
        None
            The result is the plot shown on the screen.

        Notes and limitations
        ---------------------
        * The same limitations as in the case of
          bground.ui.InteractivePlot.show_data_after_background_definition.
        '''
        
        # Close all previous plots.
        # (necessary to avoid confusions about current plot in Jupyter
        plt.close('all')
        # Get XY-data
        data = self.data.data
        # Get background object
        bkgr = self.background
        # Re-perform background subtraction
        data_corr = bground.points.bfunc.subtract_background(data, bkgr)
        X,Y = data_corr[0],data_corr[2]
        # Plot background-corrected XY-data
        plt.plot(X,Y, 'b-')
        # ...add title
        if title is not None:
            plt.title(title)
        # ...add xy-labels
        plt.xlabel(self.ppar.xlabel)
        plt.ylabel(self.ppar.ylabel)
        # ...add xy-limits
        if xlim is None: xlim = self.ppar.xlim
        if ylim is None: ylim = self.ppar.ylim
        plt.xlim(xlim)
        plt.ylim(ylim)
        # ...add grid
        if grid is not None:
            plt.grid()
        # Show the final plot
        plt.tight_layout()
        plt.show()
    

class RestoreFromPoints:
    '''
    TODO: This is an empty class; the method is under developlment ...
    '''
    pass


class FitFunction:
    '''
    TODO: This is an empty class; the method is under developlment ...
    '''


class BaseLines:
    '''
    BaseLines method of backround subtraction.

    This method uses algorithms from pybaselines for automatic baseline 
    detection. The result is saved to a file.

    Example

    >>> # Standard import
    >>> import bground.api as bkg
    >>> 
    >>> # (1) Define input and output file
    >>> IN  = 'ed1_raw.txt'  # input file,  2cols: X, Yraw
    >>> OUT = 'ed2_bkg.txt'  # output file, 4cols: X, Yraw, Ybkg, Y=Yraw-Ybkg
    >>>
    >>> # (2) Call the method, subtract background, and save results.
    >>> BMET = bkg.BaseLines(
    >>>     IN, OUT, method='method', xrange=(30,250), **kwargs)
    >>> BMET.run()

    '''
    
    def __init__(self, in_file, out_file, method = "asls", xrange=(30,250), 
                 **kwargs):
        '''
        Initialize BaseLines.

        Parameters
        ----------
        in_file : str
            input file
        out_file : str
            output file
        method : str, optional, by default "asls"
            Algorithm for pybaselines for baseline detection.
            Currently supported algorithms:
                "asls",
                "imodpoly"
            
            Please refer to 
            https://pybaselines.readthedocs.io/en/latest/algorithms/index.html
            for more details
            
        xrange : tuple, optional, by default (30,250)
            A range on the x axis, where the baseline should be subtracted..
            The range is inclusive.
        '''
        self.data = InputData(in_file).data
        self.out_file = out_file
        self.kwargs = kwargs
        self.xrange = xrange
        self.method = method
        self.x, self.y = self.data
        
        # necessary for save_bkg_data function
        self.background = bground.points.bdata.XYbackground(
            self.out_file,
            btype="pybaselines " + method
        )

    def run(self):
        '''Run the background subtraction.
            The result is saved to the file specified in the constructor.
        '''

        result = bground.blines.blines.calculate_baseline(
            self.x, self.y, self.method, self.xrange, **self.kwargs)
        bground.points.bfunc.save_bkg_data(
            result, self.background, self.out_file)

    

class WaveletMethod:
    '''
    TODO: This is an empty class; the method is under development ...
    '''
    

class Run:
    '''
    This class runs the background subtraction methods.
    
    * The class contains a function for each bkg subtraction method.
    * The function initializes and runs selected method in one step.
    * This is convenient in our simple API and for the OO-use in EDIFF package.
    '''


    def InteractivePlot(
            in_data, out_file='bground.txt', 
            comment='#', skiprows=0, header='infer', sep=r'\s+', usecols=[0,1], 
            xlabel=None, ylabel=None, xlim=None, ylim=None, messages=True):
        '''
        Run bground.api.InteractivePlot method with a single function/command.
        '''
        DATA = InputData(in_data, sep=sep, usecols=usecols,
            comment=comment, skiprows=skiprows, header=header)
        PARS = OutputParams(out_file, xlabel, ylabel, xlim, ylim, messages)
        BMET = InteractivePlot(DATA, PARS, CLI=False)
        BMET.run()
        
    
    def RestoreFromPoints():
        '''
        Run *api.RestoreFromPoints* method with a single function/command.
        
        * TODO: Mirek (straightforward, almost done)
        '''
        pass
    
    
    def FitFunction():
        '''
        Blah...
        '''
        pass


    def BaseLines():
        '''
        Blah...
        '''
        pass
        
    
    def WaveletMethod():
        '''
        Blah...
        '''
        pass
    
    
class Help():
    '''
    Class providing simple access to help functions.
    
    * The help functions just print basic information on stdout.
    * The help funcs are in bground.help module and can be called from here.
    
    >>> import bground.api as bkg
    >>> bkg.Help.intro()
    >>> bkg.Help.more_help()
    >>> bkg.Help.InteractivePlot()
    '''
  
    
    def intro():
        '''
        Help :: general :: brief introduction 
        '''
        bground.help.GeneralHelp.brief_intro()
        
    
    def more_help():
        '''
        Help :: general :: where to get more information
        '''
        bground.help.GeneralHelp.more_help()


    def InteractivePlot():    
        '''
        Help :: InteractivePlot method of background subtraction
        '''
        bground.help.InteractivePlot.how_it_works()
            
    def InteractivePlot_shortcuts(out_file='bground.txt'):
        '''
        Help :: InteractivePlot :: keyboard shortcuts
        
        * The optional {out_file} argument can be ingored when calling help.
        * It is used internally/programmatically
          when the name is known from the context.
        '''
        # The optional {out_file} argument = name of the output file.
        # Typically, this arg is used InteractivePlot is running
        # and the name of the output file is known from context.
        
        
        bground.help.InteractivePlot.keyboard_shortcuts(out_file)
    
    
    def RestoreFromPoints():    
        '''
        Help :: RestoreFromPoints method of background subtraction
        '''
        bground.help.RestoreFromPoints.how_it_works()
    
    
    def FitFunction():
        '''
        Help :: FitFunction method of background subtraction
        '''
        bground.help.FitFunctions.how_it_works()


    def BaseLines():
        '''
        Help :: BaseLines method of background subtraction
        '''
        bground.help.BaseLines.how_it_works()


    def WaveletMethod():
        '''
        Help :: WaveletMethod method of background subtraction
        '''
        bground.help.Wavelet.how_it_works()
