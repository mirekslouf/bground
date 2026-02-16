'''
Subpackage: bground.blines
--------------------------

Define background by means of estimated/fitted baseline.

* The baseline functions are taken from {idiff} package.
* The {idiff.bkg1d} module wraps the external pybaselines package.
* From user's point of view, the usage is simple, as shown below.

>>> # Simple usage of bground.blines module to subtract background from XYdata
>>> import bground.api as bkg
>>> 
>>> # (1) Define input and output file
>>> IN_FILE  = 'ed1_raw.txt'  # input file,  2cols: X, Yraw
>>> OUT_FILE = 'ed2_bkg.txt'  # output file, 4cols: X, Yraw, Ybkg, Y=Yraw-Ybkg
>>>
>>> # (2) Call selected method to remove background + save results in OUT_FILE
>>> api.blines(IN_FILE, OUT_FILE, method='some_method', xrange=(30,250))
'''
