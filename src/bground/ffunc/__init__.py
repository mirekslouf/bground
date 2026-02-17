'''
Subpackage: bground.ffunc
-------------------------

Define background by means of simple background fitting function.

* The simple background fitting functin is defined in this module.
* From user's point of view, the usage is simple, as shown below.

>>> # Simple usage of bground.ffunc module to subtract background from XYdata
>>> import bground.api as bkg
>>> 
>>> # (1) Define input and output file
>>> IN_FILE  = 'ed1_raw.txt'  # input file,  2cols: X, Yraw
>>> OUT_FILE = 'ed2_bkg.txt'  # output file, 4cols: X, Yraw, Ybkg, Y=Yraw-Ybkg
>>>
>>> # (2) Call selected method to remove background + save results in OUT_FILE
>>> BMET = api.FitFunction(IN_FILE, OUT_FILE, xrange=(30,250), **kwargs)
>>> BMET.run()
'''
