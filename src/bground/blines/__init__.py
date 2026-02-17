'''
Subpackage: bground.blines
--------------------------

Define background by means of estimated/fitted baseline.

* The baseline functions are taken pybaselines package.
* From user's point of view, the usage is simple, as shown below.

>>> # Simple usage of bground.blines module to subtract background from XYdata
>>> import bground.api as bkg
>>> 
>>> # (1) Define input and output file
>>> IN_FILE  = 'ed1_raw.txt'  # input file,  2cols: X, Yraw
>>> OUT_FILE = 'ed2_bkg.txt'  # output file, 4cols: X, Yraw, Ybkg, Y=Yraw-Ybkg
>>>
>>> # (2) Call the method, subtract background, and save results in OUT_FILE.
>>> BMET = api.BaseLines(
>>>     IN_FILE, OUT_FILE, method='method', xrange=(30,250), **kwargs)
>>> BMET.run()
'''
