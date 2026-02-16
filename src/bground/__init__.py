# Package initialization file.

'''
Package: BGROUND
----------------
Semi-automatic background subtraction.

* The package can subtract background in 1D-data = XY-data.
* The XY-data are typically saved in a file containing two (or more) columns.

Sub-packages of bground package:

* bground.api = package API, which can be used as a simple UI = user interface
* bground.points = interactive method: background defined by points on XY-curve
* bground.blines = automatic method: fit background curve/line by various funcs 

Usage of bground package:

* See the initial examples at the top of each sub-package.
'''

__version__ = '1.1.5'


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
