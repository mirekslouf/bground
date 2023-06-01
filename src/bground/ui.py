'''
Module bground.ui
-----------------
The module defines simple user interface for program bground.

* The user interface is OO-oriented, simple and intuitive.
* The user interface can be used easily in both Spyder and Jupyter.

>>> # Simple usage of BGROUND package
>>> # ! Before running, switch to interactive plots: %matplotlib qt 
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
        if 'unpack' in kwargs.keys(): kwargs.update({'unpack':True})
        data = np.loadtxt(input_file, **kwargs)
        return(data)


class PlotParams:
    
    def __init__(self, xlabel=None, ylabel=None, xlim=None, ylim=None):
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.xlim = xlim
        self.ylim = ylim


class InteractivePlot:
    
    def __init__(self, DATA, PPAR, OUT_FILE, CLI=False):
        # Basic parameters
        self.data = DATA
        self.ppar = PPAR
        self.out_file = OUT_FILE
        # Additional parameters
        self.background = bground.bdata.bkg(OUT_FILE, 
            bground.bdata.XYpoints([],[]), bground.bdata.XYcurve([],[]))
        # Initialize specific interactive backend
        # (if Python runs in CLI = command line interface, outside Spyder
        if CLI == True:
            matplotlib.use('QtAgg')
        
    def run(self):
        plt = bground.iplot.create_plot(
            self.data.data,
            xlabel = self.ppar.xlabel,
            ylabel = self.ppar.ylabel,
            xlim = self.ppar.xlim,
            ylim = self.ppar.ylim)
        plt = bground.iplot.define_key_bindings(
            plt, self.data.data, self.background, self.out_file)
        bground.iplot.print_ultrabrief_help()
        plt.show()
