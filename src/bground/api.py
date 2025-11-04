'''
Module: bground.api
-------------------

Definition of API for (application programming interface) of BGROUND package.

* The API can be employed as a simple UI within Spyder and/or Jupyter scripts.
* The main purpose of the API - easy access to functions for bkgr subtraction.

Short example:
        
>>> # Semi-automated background subtraction with InteractivePlot method
>>> # (in Spyder, use %matplotlib qt before running this script
>>>
>>> # Import API of BGROUND package
>>> import bground.api as bkg
>>>
>>> # Define I/O files
>>> INPUT  = 'ed1_raw.txt'
>>> OUTPUT = 'ed2_bcorr.txt'
>>>
>>> # Define data, plot parameters and background subtraction method
>>> DATA = bkg.InputData(INPUT, usecols=[0,1], unpack=True)
>>> PPAR = bkg.PlotParams(OUTPUT,'Pixel','Intensity',xlim=[0,200],ylim=[0,180])
>>> SMET = bkg.InteractivePlot(DATA, PPAR, CLI=False)
>>>
>>> # Run the InteractivePlot method
>>> # (a new window with an interactive plot will be opened
>>> # (follow the instructions in stdout to define and subtract bkgr
>>> SMET.run()

More examples for the individual bkgr subtraction methods:

* Semi-automated background subtraction: bground.api.InteractivePlot
* Automated background subtraction: bground.api.WaveletMethod
'''


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import bground.bdata, bground.bfunc, bground.help
import bground.iplot, bground.wvlet


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
    The input data = input file.
    
    * InputData class is a simple wrapper to `numpy.loadtxt` function.
    * All arguments from the object initialization go to this function.
    * The exception is the auto-set of `unpack=True` if *unpack* not defined.
    
    The usage of InputData class is shown in the example above.
    
    * The rest of the documentation => detailed comments in the source code.
    '''
    
    def __init__(self, input_file, **kwargs):
        # Initialization of InputData object.
        # No docstring: just class description above + comments below.
        
        # Brief description:
        # * The purpose of this object = to define and read the input XY-data.
        # * Basically, it is just a wrapper around the numpy.loadtxt function.
        
        # Name of the input file, which contains XY-data
        self.input_file = input_file
        # Call read_input_File method, which calls numpy.loadtxt function
        # with all keyword arguments (**kwargs) that were given by the user.
        self.data = self.read_input_file(input_file, **kwargs)

    
    @staticmethod
    def read_input_file(input_file, **kwargs):
        # Read input file with XY-data.
        # No docstring: just class description above + comments below.
        
        # Brief description:
        # * This function is a wrapper around numpy.loadtxt.
        # * The exception is the auto-set of unpack=True if unpack not defined.
        
        # (1) If unpack argument was not given in **kwargs,
        # set unpack=True (we expect data in columns, not in rows).
        if not 'unpack' in kwargs.keys(): kwargs.update({'unpack':True})
        
        # (2) Load data using np.loadtxt.
        # (This method is basically a wrapper of np.loadtxt,
        # (i.e. all arguments given to this function transfer to np.loadtxt
        # => more help on all optional aguments: GoogleSearch numpy.loadtxt
        data = np.loadtxt(input_file, **kwargs)
        
        # (3) Return the result = 2xN numpy array with XY-data.
        # (i.e. data[0] = X-data/values, data[1] = Y-data/values)
        return(data)



class PlotParams:
    '''
    The interactive plot parameters + name of output file(s).
    
    * PlotParams class defines plot parameters and output file name.
    * The plot parameters are X,Y-axis labels + X,Y-axes ranges/limits.
    * The output file(s) contain the background, bkgr points, and bkgr plot.
    * The file(s) are saved by the user during the interactive bkgr processing.
    
    The usage of PlotParams class is shown in the example above.
    
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
        self.xlim = xlim                # [xmin,xmax] range of x-axis
        self.ylim = ylim                # [ymin,ymax] range of y-axis
        self.messages = messages        # Printing of short messages to stdout
        
        # Note: messages argument determines,
        # if we print a short messages on stdout when the plot is interactive.
        # The argument can be overrident by InteractivePlot object;
        # the reasons are explained below in InteractivPlot object definition.


class InteractivePlot:
    '''
    InteractivePlot method of backround subtraction.

    * When running the method, a new window with interactive plot is opened.
    * The user defines background points using keyboard shortcuts and mouse.
    * The program does the rest - subtracts bkgr and shows/saves the results.
        
    Example:
        
    >>> # Semi-automated background subtraction with InteractivePlot method
    >>> # (before running in Spyder, turn on interactive plots: %matplotlib qt
    >>> # (after finishing, switch back to non-interactive: %matplotlib inline
    >>>
    >>> # Import API of BGROUND package
    >>> import bground.api as bkg
    >>>
    >>> # Define I/O files
    >>> IN  = 'ed1_raw.txt'
    >>> OUT = 'ed2_bcorr.txt'
    >>>
    >>> # Define data, plot parameters and background subtraction method
    >>> DATA = bkg.InputData(IN, usecols=[0,1], unpack=True)
    >>> PPAR = bkg.PlotParams(OUT,'Pix','Intensity',xlim=[0,200],ylim=[0,180])
    >>> SMET = bkg.InteractivePlot(DATA, PPAR, CLI=False)
    >>>
    >>> # Run the InteractivePlot method
    >>> # (a new window with an interactive plot will be opened
    >>> # (follow the instructions in stdout to define and subtract bkgr
    >>> # (ouput files will be saved at the end the interactive processing 
    >>> SMET.run()
    
    Note - arguments when initializing InteractivePlot method:
        
    * The first two arguments (DATA, PPAR) are the two classes defined above.
    * The third argument (CLI) should be True for command-line interfaces/runs.
    * The fourth argument (messages) determines if to print messages to stdout.

    '''

    
    def __init__(self, DATA, PPAR, CLI=False, messages=True):
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
        self.ppar = PPAR
        
        # Additional property - empty XYbackground object
        # (this object is defined as a semi-empty object here
        # (the only argument we supply is the name of the output file
        self.background = bground.bdata.XYbackground(self.ppar.output_file)
            
        # Initialize specific interactive backend
        # in case Python runs in CLI = Command Line Interface,
        # i.e. if the program runs outside Spyder or Jupyter environments
        if CLI == True:
            matplotlib.use('QtAgg')

        # Messages property
        # (If messages=True, short messages are printed on stdout
        # (during the interactive plot processing after each keypress event.
        # (The property is saved to
        # (both InteractivePlot object and PlotParams object.
        # (  => PlotParams.messages are transferred to InteractivePlot.run()
        # (  => Therefore, the info about messages is readily available.
        self.messages = messages
        self.ppar.messages = messages
        
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
        fig,ax = bground.iplot.interactive_plot(
            self.data.data, self.background, self.ppar)
        
        # The plot appears in a new window
        # (if we use the recommended %matplotlib qt).
        # In addition to this, we print a brief help to stdout
        # (CLI in standard python, Console in Spyder, output cell in Jupyter).
        bground.iplot.print_brief_help(self.ppar)
        
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
        data_corr = bground.bfunc.subtract_background(data, bkgr)
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

    
    class Help():
        '''
        Help functions to InteractivePlot method of background subtraction.
        '''
        
    
        def intro():
            '''
            BGROUND printed help :: Brief introduction
            
            Returns
            -------
            None
                The result is the help text printed on stdout.
            '''
            bground.help.GeneralHelp.brief_intro()
            
        
        def more_help():
            '''
            BGROUND printed help :: Where to find more help
            
            Returns
            -------
            None
                The result is the help text printed on stdout.
            '''
            bground.help.GeneralHelp.more_help()
    
        
        def how_it_works():
            '''
            BGROUND printed help :: InteractivePlot :: How it works
    
            Returns
            -------
            None
                The result is the help text printed on stdout.
            '''
            bground.help.InteractivePlot.how_it_works()
    
            
        def keyboard_shortcuts(output_file='output_file.txt'):
            '''
            BGROUND printed help :: Interactive plot :: Keyboard shortcuts
    
            Parameters
            ----------
            output_file : str, optional
                Name of real (or fictive) output file.
                The name is used just in the help text.
                It is possible to keep the default = 'output_file.txt'
    
            Returns
            -------
            None
                The result is the help text printed on stdout.
            '''
            bground.help.InteractivePlot.keyboard_shortcuts(output_file)


class WaveletMethod:
    '''
    WaveletMethod of backround subtraction.
   
    * Define the input parameters and run the method.
    * The method is fully automated - it subtracts bkgr and saves the results.
        
    Example:
        
    >>> # Automated background subtraction with WaveletMethod
    >>>
    >>> # Import API of BGROUND package
    >>> import bground.api as bkg
    >>>
    >>> # Define I/O files
    >>> IN  = 'ed1_raw.txt'
    >>> OUT = 'ed2_bcorr.txt'
    >>>
    >>> # Define data, plot parameters and background subtraction method
    >>> DATA = bkg.InputData(IN, usecols=[0,1], unpack=True)
    >>> PPAR = bkg.PlotParams(OUT,'Pix','Intensity',xlim=[0,200],ylim=[0,180])
    >>> SMET = bkg.WaveletMethod(DATA, PPAR, ...)
    >>> 
    >>> # Run the WaveletMethod
    >>> # (ouput files will be saved automatically at the end of processing 
    >>> SMET.run()
    '''
    

    class Help():
        '''
        Help functions to WaveletMethod of background subtraction.
        '''

    
        def intro():
            '''
            BGROUND printed help :: Brief introduction
            
            Returns
            -------
            None
                The result is the help text printed on stdout.
            '''
            bground.help.GeneralHelp.brief_intro()
            
        
        def more_help():
            '''
            BGROUND printed help :: Where to find more help
            
            Returns
            -------
            None
                The result is the help text printed on stdout.
            '''
            bground.help.GeneralHelp.more_help()
        
            
        def how_it_works():
            '''
            BGROUND printed help :: WaveletMethod :: How it works
    
            Returns
            -------
            None
                The result is the help text printed on stdout.
            '''
            bground.help.WaveletMethod.how_it_works()
            
            
