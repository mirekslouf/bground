'''
Module bground.ui
-----------------
The module defines simple user interface for program bground.

* The user interface is OO-oriented, simple and intuitive.
* The user interface can be used easily in both Spyder and Jupyter.

>>> # Simple usage of BGROUND package
>>> # ! Before running in Spyder switch to interactive plots: %matplotlib qt
>>> # ! After finishing, switch back to non-interactive: %matplotlib inline
>>>
>>> # Import user interface of background package
>>> import bground.ui as bkg
>>>
>>> # Define I/O files
>>> ED_FILE1 = 'ed1_raw.txt'
>>> ED_FILE2 = 'ed2_bcorr.txt'
>>>
>>> # Define data, plot parameters and interactive plot
>>> DATA = bkg.InputData(ED_FILE1, usecols=[0,1], unpack=True)
>>> PPAR = bkg.PlotParams('Pixel', 'Intensity', xlim=[0,200], ylim=[0,180])
>>> IPLOT = bkg.InteractivePlot(DATA, PPAR, ED_FILE2, CLI=False)
>>>
>>> # Run the interactive plot
>>> # (a new window with interactive plot will be opened
>>> # (basic help will be printed; more help = press 0 in the plot window
>>> IPLOT.run()
'''

import numpy as np
import matplotlib
import bground.bdata
import bground.iplot



class InputData:

    
    def __init__(self, input_file, **kwargs):
        self.input_file = input_file
        self.data = self.read_input_file(input_file, **kwargs)

    
    @staticmethod
    def read_input_file(input_file, **kwargs):
        
        # (1) If unpack argument was not given in **kwargs,
        #     set unpack=True (we expect data in columns, not in rows).
        if not 'unpack' in kwargs.keys(): kwargs.update({'unpack':True})
        
        # (2) Load data using np.loadtxt.
        # (This method is basically a wrapper of np.loadtxt,
        # (i.e. all arguments given to this function transfer to np.loadtxt
        data = np.loadtxt(input_file, **kwargs)
        
        # (3) Return the result = 2xN numpy array with XY-data.
        return(data)



class PlotParams:

    
    def __init__(self, output_file, 
                 xlabel=None, ylabel=None, xlim=None, ylim=None):
        self.output_file = output_file
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.xlim = xlim
        self.ylim = ylim



class InteractivePlot:

    
    def __init__(self, DATA, PPAR, CLI=False):
        # Basic parameters
        self.data = DATA
        self.ppar = PPAR
        # Additional parameters
        self.background = bground.bdata.bkg(self.ppar.output_file, 
            bground.bdata.XYpoints([],[]), bground.bdata.XYcurve([],[]))
        # Initialize specific interactive backend
        # (in case Python runs in CLI = command line interface, outside Spyder
        if CLI == True:
            matplotlib.use('QtAgg')

        
    def run(self):
        fig,ax = bground.iplot.interactive_plot(
            self.data.data, self.background, self.ppar)
        bground.iplot.print_ultrabrief_help()
        fig.tight_layout()
