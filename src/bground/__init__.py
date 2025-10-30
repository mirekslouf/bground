# Package initialization file.

'''
Package: BGROUND
----------------
Semi-automatic background subtraction.

* The package can subtract background in 1D-data = X,Y-data.
* The X,Y-data are typically saved in a file containing two (or more) columns.

Sub-modules of bground package:

* bground.api = package API, which can be used as a simple UI = user interface
* bground.bdata = classes/data structures for background definition
* bground.bfunc = functions and utilities for final background removal
* bground.help = supplementary functions, which can print help to the package
* bground.iplot = semi-automatic background subtraction with an interactive plot
* bground.wvlet = automatic background subtraction using wavelet transform

Usage of bground package:

* See the initial example at the top of bground.ui documentation.
'''

__version__ = '1.1'


# Obligatory acknowledgement -- the development was co-funded by TACR.
#  TACR requires that the acknowledgement is printed when we run the program.
#  Nevertheless, Python packages run within other programs, not directly.
# The following code ensures that the acknowledgement is printed when:
#  (1) You run this file: __init__.py
#  (2) You run the package from command line: python -m bground
# Technical notes:
#  To get item (2) above, we define __main__.py (next to __init__.py).
#  The usage of __main__.py is not very common, but still quite standard.

def acknowledgement():
    print('BGROUND package - semi-automatic background subtraction.')
    print('------')
    print('The development of the package was co-funded by')
    print('the Technology agency of the Czech Republic,')
    print('program NCK, project TN02000020.')
    
if __name__ == '__main__':
    acknowledgement()
