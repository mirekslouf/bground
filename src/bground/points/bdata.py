'''
Module: bground.points.bdata
----------------------------
The module defines three simple clasess.
The classes keep data for the backround definition.

1. Class XYpoints = coordinates of the user-defined bakground points.
2. Class XYcurve = two numpy arrays defining the whole calculated bkg curve.
3. Class XYbackground = complete user-defined background, containing:
    - XYpoints object = the user-defined coordinates of bkg point
    - XYcurve object = the calculated background curve
    - a few other properties (name of file for saving bkg, type of bkg)

Technical notes:

* The first two classes (XYpoints, XYcurve) are used just inside the 3rd one.
* The 3rd class (XYbackground) is used in bground.api.InteractivePlot object.
* For a common user, all classes are behind the sceenes, completely invisible.
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
        bground.bdata.XYpoints
            Object containing X and Y coords of the backround points.
            After the initialization, the object is usually empty;
            the real values are added later.
       '''
        self.X = X
        self.Y = Y

        
    def add_point(self, Xcoord, Ycoord):
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
        None
            The XYpoints object is updated
            => Xcoord,Ycoord are appended at the end.
        '''
        self.X.append(Xcoord)
        self.Y.append(Ycoord)

        
    def sort_acc_to_X(self):
        '''
        Sort bacground points according to X-coordinate.

        Returns
        -------
        None
            The XYpoints object is updated
            => the bkg points are sorted.
        '''
        
        # Sort the two lists => this is a trick found in wwww 
        # https://stackoverflow.com/q/9007877
        (self.X, self.Y) = zip( *sorted( zip(self.X, self.Y) ) )
        # One more step is needed because zip returns tuples and we need lists
        self.X = list(self.X)
        self.Y = list(self.Y)
        
        
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
        X : list or np.array, optional, default is []
            X-coordinates of the whole background curve. 
        Y : list or np.array, optional, default is []
            X-coordinates of the whole background curve.

        Returns
        -------
        bground.bdata.XYcurve
            Object containing X and Y coords of the whole background curve.
            After the initialization, the object is usually empty;
            the real values are added later.
        '''
        self.X = X
        self.Y = Y


class XYbackground:
    '''
    Class defining {XYbackground} objects.
    
    The {XYbackground} object collects/saves the following properties:
        
    * self.bname  = of the bkg/output file(s) to which we will save the result
    * self.btype  = type of background - linear, quadratic or cubic spline
    * self.points = XYpoints object = X,Y-coordinates of the bkg points
    * self.curve  = XYcurve object = X,Y-coordinates of the whole bkg curve
    '''
    
    def __init__(self, bname, 
                 btype = 'linear',
                 points = XYpoints([],[]), 
                 curve = XYcurve([],[])):
        '''
        Initialize XYbackground object.

        Parameters
        ----------
        bname : str
            {bname} = filename of output file *without extension*.
            The extension will be added automatically according to context.
        btype : string; default is 'linear' 
            Background interpolation type
            = interpolation during backround calculation.
            Implemented interpolation types: 'linear', 'quadratic', 'cubic'.
        points : bdata.XYpoints object
            Coordinates of user-defined backround points.
        curve  : bdata.XYcurve object
            Backround curve = X,Y of all points of the calculated background.
            
        Returns
        -------
        bground.bdata XYbackground
            The object is initialized and ready to be used.
            Typically, the initialized XYbackground object is semi-empty,
            containing just self.bname and self.btype, while the self.points
            and self.curve sub-objects are empty.
        
        Technical notes
        ---------------
        * During the initialization, the default values
          for points and curve sub-objects are empty lists/arrays
          = XYpoints([],[]) and XYcurve([],[].
        * The empty arrays should eliminate possible non-zero values
          from possible previous runs in Spyder or Jupyter.
        * Nevertheless, in some environments this may not be sufficient
          and the background in the main program must be initialized
          with empty objects XYpoints and XYcurve explicitly.
        * Some of our functions re-initialize the object with empty XYpoints
          and XYcurve to be on the safe side; this is a bit mysterious feature.
        '''
        self.bname  = bname
        self.btype  = btype
        self.points = points
        self.curve  = curve 
        