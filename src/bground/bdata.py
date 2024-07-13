'''
Module: bground.bdata
---------------------
The module defines three simple clasess.
The classes keep data for the backround definition.

1. Class XYpoints = coordinates of the user-defined bakground points.
2. Class XYcurve = two numpy arrays defining the whole calculated bkg curve.
3. Class bkg = complete info about the user-defined background, containing:
    - XYpoints object = the user-defined coordinates of bkg point
    - XYcurve object = the calculated background curve
    - a few other properties (name of file for saving bkg, type of bkg)

Technical notes:

* The first two classes (XYpoints, XYcurve) are used just inside the 3rd one.
* The 3rd class (bkg) is used in module iplot = the interactive bkg definition.
* For a common user, the classes are behind the sceenes, completely invisible.
'''

class XYpoints:
    '''
    XYpoints = object containing two lists X,Y.
    The lists X,Y contain X,Y coordinates of background points.
    This simple object is used in the following bkg object below.
    '''
    
    def __init__(self, X=[], Y=[]):
        '''
        Initialize XYpoints object.

        Parameters
        ----------
        X : list, optional, the default is []
            X-coordinates of user-defined background points
        Y : list, optional, the default is []
            Y-coordinates of user-defined background points

        Returns
        -------
        New XYobject.
        '''
        self.X = X
        self.Y = Y
        
    def add_point(self,Xcoord,Ycoord):
        '''
        Add one background point to XYpoints object.
        

        Parameters
        ----------
        Xcoord : float
            X-coordinate of a background point.
        Ycoord : float
            Y-coordinate of a background point.

        Returns
        -------
        None; just the Xcoord,Ycoord are added to XYpoints object.
        '''
        self.X.append(Xcoord)
        self.Y.append(Ycoord)

class XYcurve:
    '''
    XYcurve = object containing two 1D numpy arrays X,Y.
    Two arrays X,Y contain all X,Y points defining the calculated bkg curve.
    This simple object is used in the following bkg object below.
    '''
    
    def __init__(self, X=[], Y=[]):
        '''
        Initialize XYpoints object.
        
        Parameters
        ----------
        X : TYPE, optional
            DESCRIPTION. The default is [].
        Y : TYPE, optional
            DESCRIPTION. The default is [].

        Returns
        -------
        None.
        '''
        self.X = X
        self.Y = Y

class XYbackground:
    '''
    User-defined background.
    '''
    
    def __init__(self, basename, 
                 points = XYpoints([],[]), 
                 curve = XYcurve([],[]),
                 btype = 'linear'):
        '''
        Initialize background.

        Parameters
        ----------
        basename : str
            Basename of output file = filename without extension.
            The extension will be added automatically according to context.
        points : bdata.XYpoints object
            Coordinates of user-defined backround points.
        curve  : bdata.XYcurve object
            Backround curve = X,Y of all points of the calculated background.
        btype : string; default is 'linear' 
            Background interpolation type
            = interpolation during backround calculation.
            Implemented interpolation types: 'linear', 'quadratic', 'cubic'.
            
        Returns
        -------
        None; the result is the initialized object: bdata.bkg.
        
        Technical notes
        ---------------
        * In function definition, we use XYpoints([],[]) and XYcurve([],[]).
        * The empty arrays should eliminate possible non-zero values
          from possible previous run in Spyder.
        * Nevertheless, in current version this is not sufficient
          and the background in the main program must be initialized
          with empty objects XYpoints and XYcurve as well.
        * At the moment, I regard this as a Python mystery.
        '''
        self.basename = basename
        self.points   = points
        self.curve    = curve 
        self.btype    = btype
