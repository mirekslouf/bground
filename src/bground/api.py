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
* bground.api.SimpleFuncs
  = automatic methods, fit bkg with a simple funcs, TODO - Edvard (+ Adri)
* bground.api.BaseLines
  = automatic method, fit bkg with PyBaseLines funcs, TODO - Jakub
* bground.api.Wavelets
  = automatic method, fit bkg with wavelet-based funcs, TODO - Adri
'''


# {points} sub-package
import bground.points.bdata 
import bground.points.bfunc
import bground.points.ifunc
# sub-packages for fully automated bkg subtraction
import bground.sfunc.sfunc
import bground.blines.blines

# Reading and analyzing input data
import numpy as np
import pandas as pd
from pathlib import Path

# Plotting
import matplotlib
import matplotlib.pyplot as plt

# Automatic baseline detection
from pybaselines import Baseline

# Inteligent dedent (in Help class)
import re


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
    
    def __init__(self, bkg_file=None, 
                 xlabel=None, ylabel=None, xlim=None, ylim=None, 
                 saveTXT=True, messages=True):
        '''
        Initialize {BkgParams} object that defines properties of background.

        Parameters
        ----------
        bkg_file : str or PathLike object
            Name of the background/output file(s).
            There can be three different output files:
            data file (XY-data, basic output; {bkg_file}.txt),
            background points (if bkg points are defined; {bkg_file}.txt.bp),
            background plot (if user requests this; {bkg_file}.txt.png).
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
        saveTXT : bool, optional, default is True
            If True (default), save also in TXT and TXT.BP files.
            If False, the results are saved only within this object.
            The False in suitable for certain automated processing paths.            
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
        #  1) self.bkg_file = name of background file(s)
        #  2) self.xlabel/ylabel/xlim/ylim = params of bkg plots
        #  3) self.saveTXT  = if True (default), save results in TXT-files
        #     self.messages = if True (default), print short messages on stdout
        #------
        
        # (1) Set the name of background/output file(s), which can be:
        # * some.txt     = main output, 4 cols = [X, Iraw, Ibkg, I=(Iraw-Ibkg)]
        # * some.txt.bp  = saved background points (for possible recalculation)
        # * some.txt.png = saved plot (from interactive matplotlib interface)
        #------
        if bkg_file is None:
            # In some cases, we may not want to save output to TXT-files.
            # => then we save the data just in the calling object:
            # => specifically in: self.data, self.background, and self.diff1D.
            self.bkg_file = None
        else:
            # In classical usage, we save the data also in output files, 
            # If {bkg_file} is a PathLike object, convert to string
            bkg_file = str(bkg_file)
            # If {bkg_file} ends with '.bp' or '.txt', remove the extension
            # (this may be a copy paste error
            # (OR intentional in the inherited method RestoreFromBackground,
            # (where the input bkg_file usually HAS the extension '.txt.bp'
            bkg_file = bkg_file.lower()
            bkg_file = bkg_file.removesuffix('.bp')
            bkg_file = bkg_file.removesuffix('.png')
            # Ii the out_file name does not have .txt extension, add
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
        
        # (3) Additional parameters
        # saveTXT - if {True} save results also TXT and TXT.BP files
        # messages - if {True} print brief messages to stdout
        self.saveTXT = saveTXT
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
    >>> IN  = 'data1_raw_intensity.txt'
    >>> BKG = 'data2_bkg_subtracted.txt.bp'
    >>>
    >>> # Define auxiliary objects
    >>> DATA = bkg.InputData(IN, usecols=[0,1], unpack=True)
    >>> PPAR = bkg.BkgParams(BKG,'Pix','Intensity',xlim=[0,200],ylim=[0,180])
    >>>
    >>> # Initialize and run RestoreFromPoints subtraction method
    >>> # (SMET.data + SMET.background sub-objects will contain the results
    >>> SMET = bkg.RestoreFromPoints(DATA, PPAR)
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
        # (save data to file IF {saveTXT} is True
        if self.pars.saveTXT is True:
            bground.points.bfunc.save_bkg_data(self)
       

class SimpleFuncs:
    '''
    SimpleFunc background subtraction method(s).
    
    * This is a specific case (and subclass) of InteractivePlot method.
    * It inherits the initialization and visualization from InteractivePlot.
    * But it runs non-interactively, just reading bkgpoints + calculating bkg.
    
    The method can be run in two ways:
        
    * Classical, step-by-step approach - see the example below
    * Modern, single-funtion approach - see bground.api.Run.SimpleFunc
    
    Example - running SimpleFunc in classical way:
        
    >>> # Standard import
    >>> # (alternative: import ediff as ed => bkg is available as ed.bkg
    >>> import bground.api as bkg
    >>>
    >>> # Define I/O files
    >>> IN  = 'data1_raw_intensity.txt'
    >>> BKG = 'data2_bkg_corrected.txt'
    >>>
    >>> # Define auxiliary objects
    >>> DATA = bkg.InputData(IN, usecols=[0,1], unpack=True)
    >>> BPAR = bkg.BkgParams(BKG,'Pixel','Intensity',xlim=[0,200],ylim=[0,280])
    >>>
    >>> # Initialize and run SimpleFunc bkg subtraction method
    >>> # here: selected SimpleFunc method = RollingBall (+ its params)
    >>> SMET = bkg.SimpleFuncs(DATA, PPAR)
    >>> SMET.run(algorithm='RollingBall', xrange=(40,250), radius=70)

    TODO: Evard ...
    
    * Add simple tailored NumPy/SciPy based function
    * This class should not be modified => add func to sfunc.sfunc
    '''
    
    def __init__(self, DATA, PARS):
        '''
        Initialize {SimpleFunc} object.
        
        * The {SimpleFunc} object can run simple bkg subtraction methods.

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

        Returns
        -------
        bground.api.SimpleFunc object
            The object should be ready to use.
            The principal object method is SimpleFunc.run(method=...).
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
        

    def run(self, algorithm='RollingBall', **kwargs):
        '''
        Run specific background subtraction {method} from {SimpleFunc} class.
        
        * Default SimpleFunc method is 'RollingBall'.
        * Optional {kwargs} are passed to the selected method.
        
        Parameters
        ----------
        method : str, optional, default is 'RollingBall'
            Name of the background subtraction method.
            Implemented methods: 'RollingBall'.
        kwargs : keyword arguments
            Arbitrary keyword arguments.
            The arguments are passed to selected bkg subtraction {method}.
            Example: for 'RollingBall' method, we should specify {radius} arg.
        
        Summary of kwargs
        -----------------
        xrange : tuple/list of two floats
            Argument relevant to all functions/algorithms.
            If None, then the func/algorihtm will use the whole range
            as defined in calling object = self.pars.xlim.
            If specified (recommended), then the algorithm/func will use
            the selected sub-range where are the relevant diffraction peaks.
        radius: integer
            Argument relevant to 'RollingBall' algorithm.
            Radius of the rolling ball.
        
        Returns
        -------
        None
            The results are saved in self object,
            specifically in self.background (bkg curve)
            and self.data (bgk-corrected data).
            The bkg-corrected data are also saved in a TXT-file
            if saveTXT=False argument was not used in initialization.
        '''

        # (1) Run the selected bkg subtraction method
        # (the results are ALWAYS auto-saved in self.background and self.data
        if algorithm == 'RollingBall':
            bground.sfunc.sfunc.rolling_ball(self, **kwargs)
        else:
            raise ValueError('Uknown background subtraction method!')
        
        # (2) Save the calculated data ALSO in self.diff1D => if it was defined
        # (self.diff1D IF ediff.io.Diffractogram1D used for the initialization 
        if self.diff1D is not None:
            self.diff1D['Ibkg'] = self.data[2]
            self.diff1D['I']    = self.data[3]

        # (3) Save the calculated data ALSO to out_file => if it was defined
        # (save data to file IF {saveTXT} is True
        # (we can use the existing func from bground.points sub-package
        if self.pars.saveTXT is True:
            bground.points.bfunc.save_bkg_data(self)


class BaseLines:
    '''
    TODO: Jakub David ...
    
    * Move "real" code to bground.blines.
    * Keep lines < 80 chars, add docstrings.
    * Add testing examples to GDrive.
    '''

    
    def __init__(self, in_file, out_file, method = "asls", xrange=(30,250),  **kwargs):
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

        x_xrange, y_xrange = \
            bground.blines.blines.select_xrange(self.x, self.y, self.xrange)
        baseline_fitter = Baseline(x_data=x_xrange)

        if self.method == "asls":
            baseline, _ = baseline_fitter.asls(y_xrange, **self.kwargs)
        elif self.method == "imodpoly":
            baseline, _ = baseline_fitter.imodpoly(y_xrange, **self.kwargs)
        else:
            raise ValueError(f"unknown method '{str(self.method)}'")

        new_data = \
            bground.blines.blines.subtract_baseline(
                self.x, self.y, baseline, self.xrange)
        bground.points.bfunc.save_bkg_data(
            new_data, self.background, self.out_file)

    

class Wavelets:
    '''
    TODO: Adri ...
    
    * Implement a simple wavelet method - template: SimpleFuncs - RollingBall
    * Then implement additional method(s) according to {Cotret_2017.pdf}
    '''        
    

class Run:
    '''
    The class with funcs to run each bkg subtraction method in one step.
    
    * Funcs in this class initialize and run
      a selected method in one step.
    * They set bground.api.InputData and bground.api.BkgParams
      and run the method.
    * All parameters are set in one place,
      which is convenient for the modern/simple OO-processing.
    
    List of the available methods:
    
    * Run.InteractivePlot   = run {InteractivePlot} bkg subtraction method
    * Run.RestoreFromPoints = run {RestoreFromPoints} bkg subtration method
    * Run.SimpleFuncs       = run {SimpleFuncs} bkg subtraction method(s)
    * Run.BaseLines         = run {BaseLines} bkg subtraction method(s)
    * Run.Wavelets          = run {Wavelets} bkg subtraction method(s)    
    '''

    def InteractivePlot(
            in_data, bkg_file=None,
            comment='#', skiprows=0, header='infer', sep=r'\s+', usecols=[0,1], 
            xlabel=None, ylabel=None, xlim=None, ylim=None, 
            saveTXT=True, messages=True, CLI=False):
        '''
        Run bground.api.InteractivePlot method with a single function/command.
               
        Parameters
        ----------
        in_data : filename or np.ndarray or pd.DataFrame or ELD profile
            Input XYdata, containing 2 columns:
            `[X, Y = Iraw = raw intensity]`.
        bkg_file : filename, optional, default is None
            Name of the background/output file(s).
            There can be three different output files:
            data file ({bkg_file}.txt),
            background points ({bkg_file}.txt.bp),
            background plot ({bkg_file}.txt.png).
            If {bkg_file} is not specified (None) and {saveTXT}=False,
            then the TXT-files are not created and the results are
            saved only within the InteractivePlot object
            (in self.data, self.background, and possibly self.diff1D).
            The self.diff1D sub-object is saved only if the input is
            a ediff.io.Diffractogram1D object.
        comment, skiprows, header, sep, usecols: params for pd.read_csv func
             Parameters that are passed to pd.read_csv function
             if the {in_data} is an XYfile with two columns.
             See bground.api.InputData docs for more details.
         xlabel, ylabel, xlim, ylim: params for plotting
             Parameters that are used when plotting the XYdata
             in the form of matplotlib interactive graph.
             See bground.api.BkgParams docs for more details.
        saveTXT : bool, optional, default is True
            If True (default), save results in both TXT-file and self.diff1D.
            It can be switched to False in certain automated processing paths.            
        messages : bool, optional, default is True
            If True (default), print short messages on stdout.
        CLI : bool, optional, default is False
            Must be set to True when program runs from pure Python in CLI.
            It ensures that the interactive plot will not close immediately.
        
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
        PARS = BkgParams(bkg_file, 
            xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim,
            messages=messages, saveTXT=saveTXT)
        
        # (2) Define the method
        # (including optional argument CLI if it runs from pure CLI python
        SMET = InteractivePlot(DATA, PARS, CLI=CLI)
        
        # (3) Run the method
        SMET.run()
        
        # (4) Return the final InteractivePlot object
        # (the data are auto-saved to bkg_file(s)
        # (BUT returning the object is useful to see/show/plot the data
        return(SMET)
        
    
    def RestoreFromPoints(in_data, bkg_file, saveTXT=True,
            comment='#', skiprows=0, header='infer', sep=r'\s+', usecols=[0,1], 
            xlabel=None, ylabel=None, xlim=None, ylim=None, messages=False):
        '''
        Run bground.api.RestoreFromPoints method with a single func/command.

        Parameters
        ----------
        in_data : filename or np.ndarray or pd.DataFrame or ELD profile
            Input XYdata, containing 2 columns:
            `[X, Y = Iraw = raw intensity]`.
        bkg_file : filename
            Name of the file with background points.
            According to bground package convention,
            the filename should be `something.txt` or `something.txt.bp`.
            TXT-files contain the XY-data
            and TXT.BP-files contain background points.
            If {bkg_file} is a TXT file, the extension is converted to TXT.BP.
        saveTXT : bool, optional, default is True
            If True (default), save results in both TXT-file and self.diff1D.
            It can be switched to False in certain automated processing paths.            
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
        >>> IN  = r'orig_data.txt'
        >>> BKG = r'bkg-corrected_data.txt'
        >>> 
        >>> # Run RestoreFromPoints using a single command
        >>> SMET = bkg.Run.RestoreFromPoints(IN, BKG,
        >>>         xlabel='Pix', ylabel='Intensity',xlim=[0,200],ylim=180)
        '''
        
        # (1) Define objects with input and output data
        DATA = InputData(in_data, sep=sep, usecols=usecols,
            comment=comment, skiprows=skiprows, header=header)
        PARS = BkgParams(bkg_file, 
             xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim,
             saveTXT=saveTXT, messages=messages)
        
        # (2) Define the method
        # (including optional {out_file} to specify 
        SMET = RestoreFromPoints(DATA, PARS, CLI=False)
        
        # (3) Run the method
        SMET.run()
        
        # (4) Return the final RestoreFromPoints object
        # (the data are auto-saved to bkg_file(s)
        # (BUT returning the object is useful to see/show/plot the data
        return(SMET)
    
    
    def SimpleFuncs(in_data, bkg_file=None, saveTXT=True,
            comment='#', skiprows=0, header='infer', sep=r'\s+', usecols=[0,1], 
            xlabel=None, ylabel=None, xlim=None, ylim=None, messages=False,
            algorithm='RollingBall', **kwargs):
        '''
        Run bground.api.RestoreFromPoints method with a single func/command.
        
        TODO: docs => Adri
        '''
        
        # (1) Define objects with input and output data
        DATA = InputData(in_data, sep=sep, usecols=usecols,
            comment=comment, skiprows=skiprows, header=header)
        PARS = BkgParams(bkg_file, 
             xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim,
             saveTXT=saveTXT, messages=messages)
        
        # (2) Define the method
        # (including optional {out_file} to specify
        SMET = SimpleFuncs(DATA, PARS)
        
        # (3) Run the method
        SMET.run(algorithm, **kwargs)
        
        # (4) Return the final RestoreFromPoints object
        # (the data are auto-saved to bkg_file(s)
        # (BUT returning the object is useful to see/show/plot the data
        return(SMET)


    def BaseLines():
        '''
        TODO: Jakub ... 
        '''
        pass
        
    
    def Wavelets():
        '''
        TODO: Adri ...
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
    
    
    def intelligent_dedent(text: str) -> str:
        '''
        Intelligent dedentation - print multi-line string reasonably.

        Parameters
        ----------
        text : str
            A multi-line string to print.
            The initial Python indentation is ignored (Python dedentation).
            The internal string indentation is preserved (internal indentation).

        Returns
        -------
        str
            Processed string after inteligent dedentation.
            
        Technical notes
        ---------------
        * This function is used in printed help.
        * The printed help messages are typically multiline texts.
        '''
        
        # The following algorithm created with AI/ChGPT.
        # The code was slightly simplified - extra commands removed.
        # Original comments were improved + additional comments were added.
        
        # Split {text} into lines and ignore empty leading/trailing ones
        # (note #1: line.strip() returns falsi if it is an empty line
        # (note #2: the algorithm can delete 
        lines = text.splitlines()
        # Remove possible empty line(s) at the beginning
        while lines and not lines[0].strip():
            lines.pop(0)
        # Remove possible empty line(s) at the end
        while lines and not lines[-1].strip():
            lines.pop()
        # Return empty string if the multiline string contained just empty lines
        if not lines:
            return ""

        # Find minimum indentation (tabs or spaces) among non-empty lines.
        # (note #1: line.strip() returns false if it is an empty line
        # (note #2: re.match(r'^[\t]*',line).groupt(0) = initial tabs = \t chars
        # (... line with two tabs contains \t\t at the beginning = 2 chars
        # (... the same is done for spaces => we use r'^[ \t]*' - space is there
        indent_levels = [
            len(re.match(r'^[ \t]*', line).group(0))
            for line in lines if line.strip()
        ]
        min_indent = min(indent_levels)

        # Remove {min_indent} characters from the start of each line
        # (this is where some unnecessary/extra commands were removed
        dedented = [ line[min_indent:] for line in lines ]
        return("\n".join(dedented))
  
    
    def intro():
        '''
        BGROUND printed help :: Brief introduction
        '''
        
        help_text = '''
        =====================================================================
        BGROUND :: (semi)automated background subtraction for XY-data
        ---------------------------------------------------------------------
        * input data = XY-data (two columns)
            - text file or object with two (or more) columns
            - one of the columns = X-data, some other column = Y-data
            - allowed types of input (user specified input types and columns)
              text file, np.array or pd.DataFrame or ediff.io.Profile
        * output data = XY-data (four columns)
            - text file(s) and/or ediff.io.Profile object with four columns
            - cols in the text file: X, Y=Iraw, Ibkg, I=(Iraw-Ibkg)
            - cols in ELD = Profile: ELD.Pixels, ELD.Iraw, ELD.Ibkg, ELD.I 
        * semi-automated background subtraction:
            - computer opens an interactive plot
            - user defines background points with a mouse and keyboard
            - computer calculates the background + saves it as/when requested
        * fully-automated background subtraction:
            - user selects a method for background subtraction
            - computer calculates tha background and saves the output data
            - the output data are saved to file(s) and/or ediff.io.Profile
        * ediff.io.Profile
            - a specific type of input/output data
            - an object comming from our super-package ediff
            - the object is a 1D-profile from powder electron diffractogram 
        =====================================================================
        '''
        
        print(Help.intelligent_dedent(help_text))
        
    
    def more_help():
        '''
        BGROUND printed help :: Where to find additional help
        '''
        
        help_text = '''
        ==================================================================
        BGROUND package :: where to find more help
        ------------------------------------------------------------------
        Documentation + examples in www:
        * GitHub pages : https://mirekslouf.github.io/bground
        * GitHub docum : https://mirekslouf.github.io/bground/docs
        ------------------------------------------------------------------
        Additional help to the individual bkgr subtraction methods:
        >>> import bground.api as bkg
        >>> bkg.InteractivePlot.Help.how_it_works()
        ------------------------------------------------------------------
        Alternative access to help functions within ediff package
        >>> import ediff as ed
        >>> ed.bkg.InteractivePlot.Help.how_it_works()
        ==================================================================
        '''
        
        print(Help.intelligent_dedent(help_text))


    def InteractivePlot():
        '''
        BGROUND printed help :: InteractivePlot
        '''
        
        help_text = '''
        ===============================================================
        BGROUND :: InteractivePlot :: How it works
        ---------------------------------------------------------------
        * BGROUND opens Matplotlib interactive plot
        * the user defines backround points with mouse and keyboard
        * mouse actions/events are Matplotlib UI defaults
        * keyboard shortcuts/actions/events are defined by the program
          - keys for background definition: 1,2,3,4,5,6
          - keys for saving the results   : a,b,t,s,u
          - basic help is printed when the interactive plot opens
        --------------------------------------------------------------
        Complete help on keyboard shortcuts with detailed explanation:
        >>> import bground.api as bkg
        >>> bkg.InteractivePlot.Help.keyboard_shortcuts()
        --------------------------------------------------------------
        Alternative access to help from EDIFF package:
        >>> import ediff as ed
        >>> ed.bkg.InteractivePlot.Help.keyboard_shortcuts()
        ===============================================================
        '''
        
        print(Help.intelligent_dedent(help_text))
    
    
    def InteractivePlot_shortcuts( output_file='some_file' ):
        '''
        BGROUND printed help :: InteractivePlot :: Keyboard shortcuts
        '''
        
        # (1) Define output file names
        # (objective: all should have correct extensions
        # (but we want to avoid double TXT extension for the main TXT file
        TXTfile = output_file
        BKGfile = output_file + '.bp'
        PNGfile = output_file + '.png'
        if not(TXTfile.lower().endswith('.txt')): TXTfile = TXTfile + '.txt'
        
        # (2) Print help including the above defined output file names
        
        help_text = f'''
        ============================================================
        BGROUND :: Interactive plot :: Keyboard shortcuts
        ------------------------------------------------------------
        1 = add a background point (at the mouse cursor position)
        2 = delete a background point (close to the mouse cursor)
        3 = show the plot with all background points
        4 = show the plot with linear spline background
        5 = show the plot with quadratic spline background
        6 = show the plot with cubic spline background
        ------------------------------------------------------------
        a = background points :: load the previously saved
        b = background points :: save to BKG-file'
        (BKG-file = {BKGfile}
        --------
        s = save current plot to PNG-file:
        (PNG-file = {PNGfile}
        (note: Matplotlib UI shortcut; optional output
        --------
        t = subtract current background & save data to TXT-file
        (TXT-file = {TXTfile}
        (note: this is a universal output, applicable in all cases
        --------
        u = subtract current background & update ediff.io.Profile
        (ediff.io.Profile = (alternative) object with i/o data
        (note: the object is from ediff package; ignore if not used
        ------------------------------------------------------------
        Standard Matplotlib UI tools and shortcuts work as well.
        See: https://matplotlib.org/stable/users/interactive.html
        ============================================================
        '''
        
        print(Help.intelligent_dedent(help_text))
        
        
    def RestoreFromPoints():
        '''
        BGROUND printed help :: RestoreFromPoints
        '''
        
        # TODO: brief text explanation
        # (analogy of the same method in InterativePlot class above
        print("Not finished yet ...")


    def SimpleFuncs():
        '''
        BGROUND printed help :: RestoreFromPoints
        '''
        
        # TODO: Edvard
        print("Not finished yet ...")
        print("TODO: Edvard ...")
        
        
    def Baselines():
        '''
        BGROUND printed help :: RestoreFromPoints
        '''
        
        # TODO: Jakub
        print("TODO:  Jakub ...")        


    def Wavelets():
        '''
        BGROUND printed help :: RestoreFromPoints
        '''
        
        # TODO: 
        print("TODO: Adriana ...")
        
        
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

    
    @staticmethod
    def set_plot_params(size=(10,5), dpi=100, fontsize=8, 
                        my_defaults=True, my_rcParams=None):
        '''
        Set global plot parameters.
        
        * This is a static method.
        * Mostly for plotting in Jupyter.
        * It may be used internally in bkg subtraction methods,
          but InteractivePlot uses its own settings of global plot parameters.

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
        
        # Re-subtract background from points => if the points exist!
        # (just to be sure, it is quite fast
        # (only for methods that use background.points
        if len(self._parent.background.points.X) > 0:
            # If there are background points defined, recalculate.
            # (not necessary to save result => auto-saved in self._parent.data
            bground.points.bfunc.calculate_bkg_data(self._parent)
        
        # Define X,Y (X-coordinate, Intensity after bkg subtraction)
        # (the data have been recalculated above
        # (OR they should be present from automated bkg subtraction methods
        X,Y = self._parent.data[0], self._parent.data[3]
        
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
