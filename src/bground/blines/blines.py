'''
Module: bground.blines
----------------------
Backbround subtraction using baseline fitting.

* Automatic background subtraction method.
* Algorithm from https://pypi.org/project/pybaselines
* Input: TXT file with two columns: X-coords, Y-coords.
* Output: TXT file with four columns: X, Y=Iraw, Ibkg, I=(Ibkg-Iraw)

Alternative input/output:
* ELD = ediff.io.Profile object
* Profile object at input  (2 cols): ELD.Pixels, ELD.Iraw
* Profile object at output (4 cols): ELD.Pixels, ELD.Iraw, ELD.Ibkg, ELD.I
'''

import numpy as np
import matplotlib.pyplot as plt
