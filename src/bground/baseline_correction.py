'''
Module: bground.wvlet
---------------------
Backbround subtraction using wavelet transformation.

* Automatic background subtraction method.
* Algorithm from {Cotret 2017} = https://doi.org/10.1063/1.4972518
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from bground import bdata, bfunc
