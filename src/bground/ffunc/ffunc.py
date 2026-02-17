'''
Module: bground.ffunc
---------------------
Backbround subtraction using simple/ab-initio fitting function.

* Automatic background subtraction method.
* Using pure NumPy and SciPy functions and tools.
* Input: TXT file with two columns: X-coords, Y-coords.
* Output: TXT file with four columns: X, Y=Iraw, Ibkg, I=(Ibkg-Iraw)

Alternative input/output:

* ELD = ediff.io.Profile object
* Profile object at input  (2 cols): ELD.Pixels, ELD.Iraw
* Profile object at output (4 cols): ELD.Pixels, ELD.Iraw, ELD.Ibkg, ELD.I
'''

import numpy as np
import scipy as scp
import matplotlib.pyplot as plt
