'''
Module: bground.api
-------------------

Definition of API (Application Programming Interface) of the BGROUND package.

* The API can be used as a simple UI within Spyder or Jupyter.
* The main purpose of the API: easy access to functions for bkgr subtraction.

Minimal example of running a background subtraction method:
    
>>> import bground.api as bkg
>>> IN_FILE  = r'input_data.txt'
>>> BKG_FILE = r'processed_data.txt'
>>> bkg.Run.InteractivePlot(IN_FILE, BKG_FILE)

More details to all individual background subtraction methods:

* bground.api.InteractivePlot
  = semi-automatatic method, universal, finished
* bground.api.RestoreFromPoints
  = special case of the previous, restore bkg from saved bkg points  
* bground.api.FitFunction
  = automatic method, fit bkg with a simple function, TODO - Edvard
* bground.api.BaseLines
  = automatic method, fit bkg with PyBaseLines funcs
* bground.api.WaveletMethod
  = automatic method, fit bkg with wavelet-based funcs, TODO - Adriana
'''


# Bground components
import bground.help
# {points} sub-package
import bground.points.bdata 
import bground.points.bfunc
import bground.points.ifunc

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
    Class defining {InputData} for background subtraction.
    
    * Input data can be:
      filename, np.array, pd.DataFrame, or ediff.io.Diffractogram1D object.
    * In any case, the input data should contain two columns:
      `[X, Y = Iraw = raw_intensity]`.
    '''
    
    def __init__(self, input_data, **kwargs):
        '''
        Initialize {InputData} object that defines input for bkg subtraction.

        Parameters
        ----------
        input_data : str or np.array or pd.DataFrame or ediff.io.Diffraction1D
            The input_data are converted to
            self.name, self.data and self.diff1D sub-objects.
            Self.name = name of the input file (or description of the source).
            Self.data = np.array with 2 rows: `[X, Y = Iraw = raw intensity]`.
            Self.diff1D = None or ediff.io.Diffraction1D object
            if it was given as an argument.
        kwargs : keyword arguments for pd.read_csv, optional
            The keyword arguments are relevant only if {input_data}
            is str/PathLike object defining the input data file.
            The keyword arguments are passed to pd.read_csv function
            to read the file as a simple DataFrame, which is immediately
            converted to a simple np.array = to self.data sub-object
            described above.

        Returns
        -------
        bground.api.InputData
            The object is fully initialized
            and ready to use in background subtraction methods.
            It contains the self.name,
            self.data and (optionally) self.diff1D sub-objects described above.
            
        Technical notes
        ---------------
        * If {input_data} is a file, we read it to np.array with
          pd.read_csv instead of np.loadtxt, as it is more flexible/powerful.
        * The {kwargs} are pre-set reasonably enough to read most of common
          files, but user can adjuste them if needed - of course.
        * If {kwargs} need to be adjusted, see the comments in the source
          code and pandas.read_csv online documentation.
        '''
        
        # Initialization of InputData object.
        # * Two pieces of input:
        #   input_data (file or array or DataFrame or ediff.io.Diffraction1D)
        #   kwargs (to read the file using pd.read_csv function
        # * The input is saved in two/three properties:
        #   self.name = name of the input file (or str describing data source)
        #   self.data = the XY-data itself = array with 2 rows [X,Y=Intensity]
        #   self.diff1D = extra object if ediff.io.Diffracton1D was the input
        #                 this is set to None if the input was "common data"
        self.name, self.data, self.diff1D = \
            InputData.read_input_data(input_data, **kwargs)
        
    
    @staticmethod
    def read_input_data(input_data, **kwargs):
        '''
        Read {input_data} using optional {kwargs}
        return the results to be saved in {InputData} object properties.
    
        * intrinsic method of {InputData} object
        * more info => look at the detailed comments in the func source code
        '''
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
            diff1D = None
        elif isinstance(input_data, np.ndarray):
            # The input is a numpy array.
            # => just assign array to data
            # => user's responsibility: ensure that the numpy.array
            #    contains just two rows: X-values and Y-values
            #    (for example, pass something like: arr[[0,2]]
            name = 'np.ndarray'
            data = input_data
            diff1D = None
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
            diff1D = None
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
            diff1D = input_data
        else:
            raise TypeError('Unknown type of input data!')
         
        # Return the result = 2xN numpy array with XY-data.
        # (i.e. data[0] = X-data/values, data[1] = Y-data/values)
        return(name, data, diff1D)


class BkgParams:
    '''
    Class defining {BkgParams} = background parameters
    for background subtraction.
    
    * BkgParams, group 1 = name of the output/background file(s).
    * BkgParams, group 2 = x,y-labels and x,y-limits of the interactive plot
    * BkgParams, group 3 = messages argument;
      if True the program prints brief messages on stdout. 
    '''
    
    def __init__(self, bkg_file, 
                 xlabel=None, ylabel=None, xlim=None, ylim=None, 
                 messages=True):
        '''
        Initialize {BkgParams} object that defines properties of background.

        Parameters
        ----------
        bkg_file : str or PathLike object
            Name of the background/output file(s).
            There can be three different output files:
            data file (basic output; {bkg_file}.txt),
            background points (if bkg points are defined; {bkg_file}.txt.bp),
            background plot (if user requests this; {bkg_file}.txt.png).
            The main output ({bkg_file}.txt) has 4 columns:
            `[X, Y = Iraw, Ibkg, I = (Iraw - Ibkg)]`.
        xlabel : str, optional, default is None
            Label of X-axis (in the background plots).
        ylabel : str, optional, default is None
            Label of X-axis (in the background plots).
        xlim : float or tuple of two floats, optional, default is None
            X-range (in the background plots).
            If xlim = 300, set Xrange/xlim to (0,300).
            If xlim = (50,300), set Xrange/xlim to (50,300).
        ylim : float or tuple of two floats, optional, default is None
            Y-range (in the background plots).
            If ylim = 300, set Yrange/ylim to (0,300).
            If ylim = (50,300), set Yrange/ylim to (50,300).
        messages : bool, optional, default is True
            If True, the program shows short messages on stdout as it runs.

        Returns
        -------
        bground.api.BkgParams object
            The object is fully initialized and ready to be used.
            In fact the object is simple
            - just a collectoin of properties, no methods.
        '''
        # Initialization of BkgParams object => 3 groups of params:
        #  1) self.bkg_file = name of background file(s
        #  2) self.xlabel/ylabel/xlim/ylim = params of bkg plots
        #  3) self.messages = if True, print short messages on stdout
        #----
        
        # (1) Set the name of background file(s), which can be:
        # * some.txt     = main output, 4 cols = [X, Iraw, Ibkg, I = Iraw-Ibkg]
        # * some.txt.bp  = saved background points (for possible recalculation)
        # * some.txt.png = saved plot (from interactive matplotlib interface)
        #----
        # If {bkg_file} is a PathLike object, convert to string
        bkg_file = str(bkg_file)
        # If {bkg_file} ends with '.bp' or '.txt', remove the extension
        # (this may be a copy paste error
        # (OR intentional in the inherited method RestoreFromBackground,
        # (where the input bkg_file usually HAS the extension '.txt.bp'
        bkg_file = bkg_file.lower()
        bkg_file = bkg_file.removesuffix('.bp')
        bkg_file = bkg_file.removesuffix('.png')
        # Iif the out_file name does not have .txt extension, add
        if not(bkg_file.lower().endswith('.txt')):
            bkg_file = bkg_file + '.txt'
        # Save the final name to self.out_file
        self.bkg_file = bkg_file

        # (2) Set the interactive plot parameters
        # (x,y-labels
        self.xlabel = xlabel            # x-axis label of the interactive plot
        self.ylabel = ylabel            # y-axis label of the interactive plot
        # (x,y-limits
        if isinstance(xlim,list): self.xlim = xlim  # xlim = [xmin,xmax]
        else: self.xlim = [0,xlim]                  # ...or just xmax
        if isinstance(ylim,list): self.ylim = ylim  # ylim = [ymin,ymax]
        else: self.ylim = [0,ylim]                  # ...or just ymax
        
        # (3) Additional parameter - if we want brief messages to stdout
        # (we usually want the messages for InteractivePlot => default is True
        self.messages = messages
        

class InteractivePlot:
    '''
    Class defining {InteractivePlot} method of backround subtraction.
    
    * When running the method, a new window with interactive plot is opened.
    * The user defines background points using mouse and keyboard shortcuts.
    * The program does the rest - it subtracts bkg + shows/saves the results.
    
    The method can be run in two ways:
        
    * Classical, step-by-step approach - see the example below
    * Modern, single-funtion approach - see bground.api.Run.InteractivePlot
    
    Example - running InteractivePlot in classical way:
        
    >>> # Standard import
    >>> # (alternative: import ediff as ed => bkg is available as ed.bkg
    >>> import bground.api as bkg
    >>>
    >>> # Define I/O files
    >>> # (input file  = XYdata, 2 cols: [X, Y = Iraw = Raw Intensity]
    >>> # (output file = XYdata, 4 cols: [X, Iraw, Inet, I = (Iraw - Inet)]
    >>> IN  = 'data1_raw_intensity.txt'
    >>> OUT = 'data2_bkg_subtracted.txt'
    >>>
    >>> # Define auxiliary objects
    >>> DATA = bkg.InputData(IN, usecols=[0,1], unpack=True)
    >>> PPAR = bkg.BkgParams(OUT,'Pix','Intensity',xlim=[0,200],ylim=[0,180])
    >>>
    >>> # Initialize and run InteractivePlot subtraction method
    >>> # (the method is initialized using the auxiliary objects and run
    >>> # (SMET.data + SMET.background sub-objects will contain the results
    >>> SMET = bkg.InteractivePlot(DATA, PPAR)
    >>> SMET.run()
    '''
    
    
    def __init__(self, DATA, PARS, CLI=False):
        '''
        Initialize {InteractivePlot} object.

        Parameters
        ----------
        DATA : bground.api.InputData object
            The object contains description of input data.
            The {InputData} object should be defined
            either before initializing the {InteractivePlot},
            or intrinsically if we use bground.api.Run.InteractivePlot.
        PARS : bground.api.BkgParams object
            The object contains description of input data.
            The {BkgData} object should be defined
            either before initializing the {InteractivePlot},
            or intrinsically if we use bground.api.Run.InteractivePlot.
        CLI : bool, optional, default is False
            If we run the {InteractivePLot} from CLI interface,
            this arbument should be set to True so that the plot stayed
            in the screen.

        Returns
        -------
        bground.api.InteractivePlot object
            The object should be ready to use.
            The principal object method is InteractivePlot.run.
        '''
        
        # Save properties from DATA object
        self.name = DATA.name
        self.data = DATA.data
        self.diff1D = DATA.diff1D
        
        # Save properties from PARS object
        self.pars = PARS
        
        # Additional property - empty XYbackground object
        # (this object is defined as a semi-empty object here
        # (the only argument we supply is the name of the background file
        self.background = \
            bground.points.bdata.XYbackground(self.pars.bkg_file)
            
        # Initialize plotting - by means of composition
        self.plots = Plots(self)
        
        # Initialize specific interactive backend
        # in case Python runs in CLI = Command Line Interface,
        # i.e. if the program runs outside Spyder or Jupyter environments
        if CLI == True:
            matplotlib.use('QtAgg')

        
    def run(self):
        '''
        Run {InteractivePlot} method of background subtraction.
        
        * The method (i) clears variables
          and (ii) shows the plot on the screen.
        * The plot is waiting for the user to define
          background points with keyboard and mouse. 
        * A brief help is printed on stdout
          when the interactive plot is opened.
        * No params; everything defined during InteractivePlot initialization.
        '''

        # (0) Clear background points from possible previous runs.
        # * Issue in Jupyter, when re-running the cell with this command
        self.background.points.X = []
        self.background.points.Y = []
        
        # (1) Run the interactive plot
        # * the plot appears in a new window
        #   if we use the recommended %matplotlib qt
        # * in addition, a brief help is printed to to stdout
        #   = Screen in Python, Console in Spyder, output cell in Jupyter
        # * the return values seem to be necessary for current Jupyter version
        fig,ax = bground.points.ifunc.interactive_plot(self)
        
        # (2) Optimize the layout of the returned figure and show the plot.
        # * fig.tight_layout() functin works
        #   in all three supported interfaces: CLI, Spyder, Jupyter
        fig.tight_layout()
        plt.show()
    

class RestoreFromPoints(InteractivePlot):
    '''
    RestoreFromPoints background subtraction method.
    
    * This is a specific case (and subclass) of InteractivePlot method.
    * It inherits the initialization and visualization from InteractivePlot.
    * But it runs non-interactively, just reading bkgpoints + calculating bkg.
    
    The method can be run in two ways:
        
    * Classical, step-by-step approach - see the example below
    * Modern, single-funtion approach - see bground.api.Run.RestoreFromPoints
    
    Example - running RestoreFromPoints in classical way:
        
    >>> # Standard import
    >>> # (alternative: import ediff as ed => bkg is available as ed.bkg
    >>> import bground.api as bkg
    >>>
    >>> # Define I/O files
    >>> # (input file  = XYdata, 2 cols: [X, Y = Iraw = Raw Intensity]
    >>> # (background file = TXT.BP file from a previous run of InteractivePlot
    >>> IN  = 'data1_raw_intensity.txt'
    >>> BKG = 'data2_bkg_subtracted.txt.bp'
    >>>
    >>> # Define auxiliary objects
    >>> DATA = bkg.InputData(IN, usecols=[0,1], unpack=True)
    >>> PPAR = bkg.BkgParams(BKG,'Pix','Intensity',xlim=[0,200],ylim=[0,180])
    >>>
    >>> # Initialize and run RestoreFromPoints subtraction method
    >>> # (the method is initialized using the auxiliary objects and run
    >>> # (SMET.data + SMET.background sub-objects will contain the results
    >>> SMET = bkg.RestoreFromPointsPlot(DATA, PPAR)
    >>> SMET.run()
    '''
    
    def __init__(self, DATA, PARS, CLI=False):
        '''
        Initialize {RestoreFromPoints} object.

        Parameters
        ----------
        DATA : bground.api.InputData object
            The object contains description of input data.
            The {InputData} object should be defined
            either before initializing the {InteractivePlot},
            or intrinsically if we use bground.api.Run.InteractivePlot.
        PARS : bground.api.BkgParams object
            The object contains description of input data.
            The {BkgData} object should be defined
            either before initializing the {InteractivePlot},
            or intrinsically if we use bground.api.Run.InteractivePlot.
        CLI : bool, optional, default is False
            If we run the {InteractivePLot} from CLI interface,
            this arbument should be set to True so that the plot stayed
            in the screen.

        Returns
        -------
        bground.api.RestoreFromPoints object
            The object should be ready to use.
            The principal object method is InteractivePlot.run.
        '''
        
        # The same initialization like in the superclass.
        # (if __init__ contains just this command, it can be ommited
        # (here: just for historical reasons; we used to have additional args
        super().__init__(DATA, PARS, CLI)
        
        # Note: it is possible to initialize this function
        # with all possible names: 'some', 'some.txt', 'some.txt.bp'
        # and the initialization procedure of BkgParams calls resolves this.
        
        # Note: CLI=True if program runs from pure Python in command line
        # => in super().__init__ we set the same matplotlib backend
        # => we get the same interface in InteractivePlot and RestoreFromPoints
        # => for compatibility with methods show_data_with_bkg_definition ...
 

    def run(self):
        '''
        Run {RestoreFromPoints} method of background subtraction.
        
        * The method reads a TXT.BP file with background points
          and recalculates background and saves the results.
        * The TXT.BP file usually comes from
          a previous run of InteractivePlot method.
        * No params; everything defined during InteractivePlot initialization.
        '''

        # (0) Read background points
        # (self.background comes from initialization
        # (it contains the name of the file with bkg points
        # (this filename is in PARS and saved in self.background.bname
        self.background = bground.points.bfunc.load_bkg_points(self)
        
        # (1) Calculate background-corrected data AND save them to self.data
        # (self.data = the RestoreFromPoints object ALWAYS contains the output
        self.data = bground.points.bfunc.calculate_bkg_data(self)
        
        # (2) Save the calculated data ALSO in self.diff1D => if it was defined
        # (self.diff1D IF ediff.io.Diffractogram1D used for the initialization 
        if self.diff1D is not None:
            self.diff1D['Ibkg'] = self.data[2]
            self.diff1D['I']    = self.data[3]

        # (3) Save the calculated data ALSO to out_file => if it was defined
        # (save data to file IF {out_file} argument was given
        bground.points.bfunc.save_bkg_data(self)
       


class FitFunction:
    '''
    TODO: Edvard Sidoryk ...
    '''


class BaseLines:
    '''
    BaseLines method of backround subtraction.

    This method uses algorithms from pybaselines for automatic baseline 
    detection. The result is saved to a file.

    Example 1 :: method with default parameters

    >>> # Standard import
    >>> import bground.api as bkg
    >>> 
    >>> # (1) Define input and output file
    >>> IN  = 'ed1_raw.txt'  # input file,  2cols: X, Yraw
    >>> OUT = 'ed2_bkg.txt'  # output file, 4cols: X, Yraw, Ybkg, Y=Yraw-Ybkg
    >>>
    >>> # (2) Call the method, subtract background, and save results.
    >>> BMET = bkg.BaseLines(IN, OUT, method='peak_filling', xrange=(30,250))
    >>> BMET.run()


    Example 2 :: specify method parameters

    >>> # Standard import
    >>> import bground.api as bkg
    >>> 
    >>> # (1) Define input and output file
    >>> IN  = 'ed1_raw.txt'  # input file,  2cols: X, Yraw
    >>> OUT = 'ed2_bkg.txt'  # output file, 4cols: X, Yraw, Ybkg, Y=Yraw-Ybkg
    >>>
    >>> # (2) Call the method, subtract background, and save results.
    >>> # Add parameter decreasing=True for the snip algorithm.
    >>> # The kwargs (parameters for the method) must be specified last.
    >>> BMET = bkg.BaseLines(
    >>>         IN, OUT, method='snip', xrange=(30,250), decreasing=True)
    >>> BMET.run()

    '''
    
    def __init__(self, input_data, bkg_params, method = "peak_filling", 
                 xrange=(30,250), **kwargs):
        '''
        Initialize BaseLines.

        Parameters
        ----------
        input_data : str or InputData
            InputData directly or any type supported by InputData.
        bkg_params : str or BkgParams
            BkgParams or output file as a string.
        method : str, optional, by default "peak_filling"
            Algorithm from pybaselines for baseline detection.     
        xrange : tuple, optional, by default (30,250)
            A range on the x axis, where the baseline should be subtracted.
            The range is inclusive.

        Notes
        -----
        * `kwargs` are passed to the algorithm.

        * Recommended methods (and parameters in kwargs):
            * `peak_filling` (with parameter `half_window=1`),
            * `snip` (with parameter `decreasing=True`),
            * `pspline_iasls` (with parameter `lam=5`), 
            * `irsqr` (with parameter `lam=1000`),
            * `rubberband` (with parameter `lam=2`)
        

        * Please refer to 
        https://pybaselines.readthedocs.io/en/latest/api/Baseline.html
        for more methods and their details.
        '''
        if isinstance(input_data, InputData):
            self.data = input_data.data
        else:
            self.data = InputData(input_data).data

        if isinstance(bkg_params, BkgParams):
            self.pars = bkg_params
        else:
            self.pars = BkgParams(bkg_params)

        self.kwargs = kwargs
        self.xrange = xrange
        self.method = method
        self.x, self.y = self.data
        
        # necessary for save_bkg_data function
        self.background = bground.points.bdata.XYbackground(
            self.pars.bkg_file,
            btype="pybaselines " + method
        )

        self.plots = Plots(self)

    def run(self):
        '''Run the background subtraction.
            The result is saved to the file specified in the constructor.
        '''

        self.data = bground.blines.blines.calculate_baseline(
            self.x, self.y, self.method, self.xrange, **self.kwargs)
        self.background.curve.X, self.background.curve.Y = \
            bground.blines.blines.select_xrange(
                self.x, self.data[2], self.xrange)
        bground.points.bfunc.save_bkg_data(self)

    

class WaveletMethod:
    '''
    TODO: Adriana Vasquez Pelayo ...
    '''
    

class Run:
    '''
    This class runs the background subtraction methods.
    
    * The class contains functions for each bkg subtraction method.
    * Each function initializes and runs the selected method in one step.
    * Behind the scenes, the initializes bground.api.InputData and bground.api.
    * This is convenient in our simple API and for the OO-use in EDIFF package.
    
    List of the available methods:
    
    * Run.InteractivePlot   = run {InteractivePlot} bkg subtraction method
    * Run.RestoreFromPoints = run {RestoreFromPoints} bkg subtration method
    * Run.FitFunction       = run {FitFunction} bkg subtraction method
    * Run.BaseLines         = run {BaseLines} bkg subtraction method(s)
    * Run.WaveletMethod     = run {WaveletMethod} bkg subtraction method(s)
    
    How does it work?
    
    * Behind the scenes, each Run-function in this class initializes
      bground.api.InputData and bground.api.BkgParams.
    * In the next step, the Run-function defines and runs selected method,
      using all parameters for {InputData}, {BkgParams}, and the method itself.
    * Important is that all parameters are set in one place,
      within single Run-function without, which further simplifies the API.
    '''

    def InteractivePlot(
            in_data, bkg_file, 
            comment='#', skiprows=0, header='infer', sep=r'\s+', usecols=[0,1], 
            xlabel=None, ylabel=None, xlim=None, ylim=None, messages=True,
            CLI=False):
        '''
        Run bground.api.InteractivePlot method with a single function/command.
               
        Parameters
        ----------
        in_data : filename or np.ndarray or pd.DataFrame or ELD profile
            Input XYdata, containing 2 columns:
            `[X, Y = Iraw = raw intensity]`.
        bkg_file : filename
            Name of the background/output file(s).
            There can be three different output files:
            data file (basic output; {bkg_file}.txt),
            background points (if bkg points are defined; {bkg_file}.txt.bp),
            background plot (if user requests this; {bkg_file}.txt.png).
            The main output ({bkg_file}.txt) has 4 columns:
            `[X, Y = Iraw, Ibkg, I = (Iraw - Ibkg)]`.
        comment, skiprows, header, sep, usecols: params for pd.read_csv func
             Parameters that are passed to pd.read_csv function
             if the {in_data} is an XYfile with two columns.
             See bground.api.InputData docs for more details.
         xlabel, ylabel, xlim, ylim, messages, CLI: params for plotting
             Parameters that are used when plotting the XYdata
             in the form of matplotlib interactive graph.
             See bground.api.BkgParams docs for more details.
        
        Returns
        -------
        bground.api.InteractivePlot
            The object is used to run the InteractivePlot method,
            but it also saves and plots the original and bkg-corrected data.
        
        Example
        -------
        
        >>> # Standard import
        >>> # (alternative: import ediff as ed => bkg is available as ed.bkg
        >>> import bground.api as bkg
        >>>
        >>> # Define I/O files
        >>> # (input file  = XYdata, 2 cols: [X, Y = Iraw = Raw Intensity]
        >>> # (output file = XYdata, 4 cols: [X, Iraw, Inet, I = (Iraw - Inet)]
        >>> IN  = 'data1_raw_intensity.txt'
        >>> OUT = 'data2_bkg_subtracted.txt'
        >>>
        >>> # Run InteractivePlot subtraction method with a single function
        >>> # (the func initializes all objects and runs the method
        >>> # (SMET.data + SMET.background sub-objects will contain the results
        >>> SMET = bkg.Run.InteractivePlot(IN, OUT, 
        >>>     xlabel='Pixels', ylabel='Intensity', xlim=300, ylim=300)
        '''
        
        # (1) Define objects with input and output data
        DATA = InputData(in_data, sep=sep, usecols=usecols,
            comment=comment, skiprows=skiprows, header=header)
        PARS = BkgParams(bkg_file, xlabel, ylabel, xlim, ylim, messages)
        
        # (2) Define the method
        # (including optional argument CLI if it runs from pure CLI python
        BMET = InteractivePlot(DATA, PARS, CLI=CLI)
        
        # (3) Run the method
        BMET.run()
        
        # (4) Return the final InteractivePlot object
        # (the data are auto-saved to bkg_file(s)
        # (BUT returning the object is useful to see/show/plot the data
        return(BMET)
        
    
    def RestoreFromPoints(in_data, bkg_points, out_file=None,
            comment='#', skiprows=0, header='infer', sep=r'\s+', usecols=[0,1], 
            xlabel=None, ylabel=None, xlim=None, ylim=None, messages=False,
            CLI=False):
        '''
        Run bground.api.RestoreFromPoints method with a single func/command.

        Parameters
        ----------
        in_data : filename or np.ndarray or pd.DataFrame or ELD profile
            Input XYdata, containing 2 columns:
            `[X, Y = Iraw = raw intensity]`.
        bkg_points : filename
            Name of the file with background points (usually a file
            with TXT.BP extension from previous InteractivePlot run).
            This TXT.BP background-points-file is loaded
            and TXT data-file with 4 cols (`[X, Iraw, Inet, I=Iraw-Inet]`)
            is recalculated - this is the core of the RestoreFromPoints method.
        comment, skiprows, header, sep, usecols: params for pd.read_csv func
            Parameters that are passed to pd.read_csv function
            if the {in_data} is an XYfile with two columns.
            See bground.api.InputData docs for more details.
         xlabel, ylabel, xlim, ylim, messages, CLI: params for plotting
            Parameters that are used when plotting the XYdata
            in the form of matplotlib interactive graph.
            See bground.api.BkgParams docs for more details.
        
        Returns
        -------
        bground.api.RestoreFromPoints
            The object is used to run the RestoreFromPoints method,
            but it also saves and plots the original and bkg-corrected data.
        
        Example
        -------
        
        >>> # Standard import
        >>> # (alternative: import ediff as ed => bkg is available as ed.bkg
        >>> import bground.api as bkg
        >>> 
        >>> # Define I/O files
        >>> # (input file = XYdata, 2 cols: [X, Y = Iraw = Raw Intensity]
        >>> # (bkg points = TXT.BP file from a previous run of InteractivePlot
        >>> IN  = r'../_DATA/tbf3_sum_hsd_i300.txt'
        >>> BKG = r'test2_simple.txt.bp'
        >>> 
        >>> # Run RestoreFromPoints using a single command
        >>> # (the method is initialized using the auxiliary objects and run
        >>> # (SMET.data + SMET.background sub-objects will contain the results
        >>> SMET = bkg.Run.RestoreFromPoints(
        >>>         IN, BKG,
        >>>         xlabel='Pix', ylabel='Intensity',xlim=[0,200],ylim=[0,180])
        '''
        
        # (1) Define objects with input and output data
        DATA = InputData(in_data, sep=sep, usecols=usecols,
            comment=comment, skiprows=skiprows, header=header)
        PARS = BkgParams(bkg_points, xlabel, ylabel, xlim, ylim, messages)
        
        # (2) Define the method
        # (including optional {out_file} to specify 
        BMET = RestoreFromPoints(DATA, PARS, CLI)
        
        # (3) Run the method
        BMET.run()
        
        # (4) Return the final RestoreFromPoints object
        # (the data are auto-saved to bkg_file(s)
        # (BUT returning the object is useful to see/show/plot the data
        return(BMET)
    
    
    def FitFunction():
        '''
        Blah...
        '''
        pass


    def BaseLines(
            in_file, out_file, method="peak_filling", 
            xrange=(30,250), **kwargs):
        '''
        Run bground.api.BaseLines method with a single function/command.
        '''
        BMET = BaseLines(in_file, out_file, method, xrange, **kwargs)
        BMET.run()
        
    
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


class Plots:
    '''
    Class defining plotting for all background correction methods.
    
    * The class returns Plots object.
    * The {Plots} object is designed as a component
      for arbitrary background subtraction class/method.
    
    Example
    
    >>> import bground.api as bkg
    >>> IN_FILE  = r'input_data.txt'
    >>> BKG_FILE = r'processed_data.txt'
    >>> SMET = bkg.Run.InteractivePlot(IN_FILE, BKG_FILE)
    >>> SMET.plots.plot_original()
    >>> SMET.plots.plot_without_bkg() 
    
    How is it done?
    
    * Using true *composition* (not just aggregation).
    * The component class (Plots) holds the reference to the parent class.
    * The parent/composite class can be any background subtraction class,
      such as: InteractivePlot, RestoreFromPoints, ...
    * The parent class (for example InteractivePlot)
      saves the reference to Plots during initialization
      by means of the following command: `self.plots = Plots(self)`. 
    '''
    
        
    def __init__(self, parent):
        '''
        Initialize {Plots} object.

        * The {Plots} object is designed as a component
          for arbitrary background subtraction class/method.
        * The method is typically not initialized directly,
          but within the composite/parent class that defines
          a bacgkround subtraction method.
    
        Parameters
        ----------
        parent : class defining a background subtraction method
            An arbitrary class in this module
            that defines a backround subtraction method, such as:
            InteractivePlot, RestoreFromPoints ...

        Returns
        -------
        Plots object
            The object has the only property: `self._parent = parent`.
            This is the whole trick that makes this class a true component.
        '''
        self._parent = parent   # aggregated object

    
    def plot_original(self, title='Raw data before processing', grid=True):
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
        '''        
        # Close all previous plots.
        # (necessary to avoid confusions about current plot in Jupyter
        plt.close('all')
        # Get XY-data
        X,Y = self._parent.data[0:2]
        # PLot XY-data
        plt.plot(X,Y, 'b-')
        if title is not None: plt.title(title)
        # ...add xy-labels and limits
        plt.xlabel(self._parent.pars.xlabel)
        plt.ylabel(self._parent.pars.ylabel)
        plt.xlim(self._parent.pars.xlim)
        plt.ylim (self._parent.pars.ylim)
        # ...add grid
        if grid is not None: plt.grid()
        # Show the final plot
        plt.tight_layout()
        plt.show()

    
    def plot_with_bkg(       
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
        '''
        
        # (0) Initialize, prepare parameters for plotting
        # Close all previous plots.
        # (necessary to avoid confusions about current plot in Jupyter
        plt.close('all')
        # Verify, if the background points are defined
        # (some methods use bkg points, some not
        # (below we use and plot the bkg points only if they are defined
        background_points_defined = len(self._parent.background.points.X) > 0
        # Get XY-data
        X,Y = self._parent.data[0:2]
        # Get background points => if they are defined!
        if background_points_defined:
            Xp = self._parent.background.points.X
            Yp = self._parent.background.points.Y
        # Get background interpolation curve
        Xc = self._parent.background.curve.X
        Yc = self._parent.background.curve.Y
        
        # (1) Plot data
        # ... XY-data
        plt.plot(X,Y, 'b-')
        # ... background interpolation curve
        plt.plot(Xc,Yc, 'r--')
        # ... background points => if they are defined!
        if background_points_defined: plt.plot(Xp,Yp, 'r+')
        
        # (2) Finalize the plot
        # ...add title
        if title is not None: plt.title(title)
        # ...add xy-labels and limits
        plt.xlabel(self._parent.pars.xlabel)
        plt.ylabel(self._parent.pars.ylabel)
        plt.xlim(self._parent.pars.xlim)
        plt.ylim (self._parent.pars.ylim)
        # ...add grid
        if grid is not None: plt.grid()
        
        # (3) Show the finalized plot
        plt.tight_layout()
        plt.show()

    
    def plot_without_bkg(
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
        '''
        
        # Close all previous plots.
        # (necessary to avoid confusions about current plot in Jupyter
        plt.close('all')
        
        # Re-perform background subtraction
        # (just to be sure, it is quite fast
        data_without_bkg = \
            bground.points.bfunc.calculate_bkg_data(self._parent)
        X,Y = data_without_bkg[0],data_without_bkg[3]
        
        # Plot background-corrected XY-data
        plt.plot(X,Y, 'b-')
        # ...add title
        if title is not None:
            plt.title(title)
        # ...add xy-labels
        plt.xlabel(self._parent.pars.xlabel)
        plt.ylabel(self._parent.pars.ylabel)
        # ...add xy-limits
        if xlim is None: xlim = self._parent.pars.xlim
        if ylim is None: ylim = self._parent.pars.ylim
        plt.xlim(xlim)
        plt.ylim(ylim)
        # ...add grid
        if grid is True: plt.grid()
        # Show the final plot
        plt.tight_layout()
        plt.show()
