'''
Module: bground.help
--------------------
Help functions for bground package.

* This module is a collecton if simple help functions.
* The functions just print a brief textual information on stdout.
* The funcs are grouped into classes; each bkgr subtraction method ~ class.
* Three functions - intro, more_help, how_it_works - available for all methods.

>>> # Typical access to printed help from bground
>>> # (the help accessed via bground.api
>>> import bground.api as bkg
>>> bkg.InteractivePlot.Help.intro()
>>> bkg.InteractivePlot.Help.more_help()
>>> bkg.InteractivePlot.Help.how_it_works()

>>> # Typical access to printed help from ediff
>>> # (bground is usually used within ediff package
>>> # (bground.api is auto-imported as ediff.bkg or ed.bkg
>>> import ediff as ed
>>> ed.bkg.InteractivePlot.Help.intro()
>>> ed.bkg.InteractivePlot.Help.more_help()
>>> ed.bkg.InteractivePlot.Help.how_it_works()
'''


# At the beginning of this module, we define intelligent_detent method.
# It performs intelligent de-dentation of multiline Python strings.
# This method is used in all help functions below.

import re


def intelligent_dedent(text: str) -> str:
    '''
    Intelligent dedentation - print multi-line string reasonably.

    Parameters
    ----------
    text : str
        A multi-line string to print.
        The initial Python indentation is ignored (Python dedentation).
        The internal string indentation is preserved (internal indentation).

    Returns
    -------
    str
        Processed string after inteligent dedentation.
        
    Technical notes
    ---------------
    * This function is used in printed help.
    * The printed help messages are typically multiline texts.
    '''
    
    # The following algorithm created with AI/ChGPT.
    # The code was slightly simplified - extra commands removed.
    # Original comments were improved + additional comments were added.
    
    # Split {text} into lines and ignore empty leading/trailing ones
    # (note #1: line.strip() returns falsi if it is an empty line
    # (note #2: the algorithm can delete 
    lines = text.splitlines()
    # Remove possible empty line(s) at the beginning
    while lines and not lines[0].strip():
        lines.pop(0)
    # Remove possible empty line(s) at the end
    while lines and not lines[-1].strip():
        lines.pop()
    # Return empty string if the multiline string contained just empty lines
    if not lines:
        return ""

    # Find minimum indentation (tabs or spaces) among non-empty lines.
    # (note #1: line.strip() returns false if it is an empty line
    # (note #2: re.match(r'^[\t]*',line).groupt(0) = initial tabs = \t chars
    # (... line with two tabs contains \t\t at the beginning = 2 chars
    # (... the same is done for spaces => we use r'^[ \t]*' - space is there
    indent_levels = [
        len(re.match(r'^[ \t]*', line).group(0))
        for line in lines if line.strip()
    ]
    min_indent = min(indent_levels)

    # Remove {min_indent} characters from the start of each line
    # (this is where some unnecessary/extra commands were removed
    dedented = [ line[min_indent:] for line in lines ]
    return("\n".join(dedented))


class GeneralHelp:
    '''
    Class with help functions.
    
    * The functions print simple textual help.
    * Here: general help to whole bground package.
    '''
    
    
    def intro():
        '''
        BGROUND printed help :: Brief introduction
        '''
        
        help_text = '''
        =====================================================================
        BGROUND :: (semi)automated background subtraction for XY-data
        ---------------------------------------------------------------------
        * input data = XY-data (two columns)
            - text file or object with two (or more) columns
            - one of the columns = X-data, some other column = Y-data
            - allowed types of input (user specified input types and columns)
              text file, np.array or pd.DataFrame or ediff.io.Profile
        * output data = XY-data (four columns)
            - text file(s) and/or ediff.io.Profile object with four columns
            - cols in the text file: X, Y=Iraw, Ibkg, I=(Iraw-Ibkg)
            - cols in ELD = Profile: ELD.Pixels, ELD.Iraw, ELD.Ibkg, ELD.I 
        * semi-automated background subtraction:
            - computer opens an interactive plot
            - user defines background points with a mouse and keyboard
            - computer calculates the background + saves it as/when requested
        * fully-automated background subtraction:
            - user selects a method for background subtraction
            - computer calculates tha background and saves the output data
            - the output data are saved to file(s) and/or ediff.io.Profile
        * ediff.io.Profile
            - a specific type of input/output data
            - an object comming from our super-package ediff
            - the object is a 1D-profile from powder electron diffractogram 
        =====================================================================
        '''
        
        print(intelligent_dedent(help_text))
        
    
    def more_help():
        '''
        BGROUND printed help :: Where to find additional help
        '''
        
        help_text = '''
        ==================================================================
        BGROUND package :: where to find more help
        ------------------------------------------------------------------
        Documentation + examples in www:
        * GitHub pages : https://mirekslouf.github.io/bground
        * GitHub docum : https://mirekslouf.github.io/bground/docs
        ------------------------------------------------------------------
        Additional help to the individual bkgr subtraction methods:
        >>> import bground.api as bkg
        >>> bkg.InteractivePlot.Help.how_it_works()
        ------------------------------------------------------------------
        Alternative access to help functions within ediff package
        >>> import ediff as ed
        >>> ed.bkg.InteractivePlot.Help.how_it_works()
        ==================================================================
        '''
        
        print(intelligent_dedent(help_text))


class InteractivePlot:
    '''
    Class with help functions.
    
    * The functions print simple textual help.
    * Here: help to bground.iplot = InteractivePlot bkgr subtraction method.
    '''
    
    
    def how_it_works():
        '''
        BGROUND printed help :: InteractivePlot :: How it works
        '''
        
        help_text = '''
        ===============================================================
        BGROUND :: InteractivePlot :: How it works
        ---------------------------------------------------------------
        * BGROUND opens Matplotlib interactive plot
        * the user defines backround points with mouse and keyboard
        * mouse actions/events are Matplotlib UI defaults
        * keyboard shortcuts/actions/events are defined by the program
          - keys for background definition: 1,2,3,4,5,6
          - keys for saving the results   : a,b,t,s,u
          - basic help is printed when the interactive plot opens
        --------------------------------------------------------------
        Complete help on keyboard shortcuts with detailed explanation:
        >>> import bground.api as bkg
        >>> bkg.InteractivePlot.Help.keyboard_shortcuts()
        --------------------------------------------------------------
        Alternative access to help from EDIFF package:
        >>> import ediff as ed
        >>> ed.bkg.InteractivePlot.Help.keyboard_shortcuts()
        ===============================================================
        '''
        
        print(intelligent_dedent(help_text))
    
    
    def keyboard_shortcuts( output_file='some_file' ):
        '''
        BGROUND printed help :: InteractivePlot :: Keyboard shortcuts
        '''
        
        # (1) Define output file names
        # (objective: all should have correct extensions
        # (but we want to avoid double TXT extension for the main TXT file
        TXTfile = output_file
        BKGfile = output_file + '.bkg'
        PNGfile = output_file + '.png'
        if not(TXTfile.lower().endswith('.txt')): TXTfile = TXTfile + '.txt'
        
        # (2) Print help including the above defined output file names
        
        help_text = f'''
        ============================================================
        BGROUND :: Interactive plot :: Keyboard shortcuts
        ------------------------------------------------------------
        1 = add a background point (at the mouse cursor position)
        2 = delete a background point (close to the mouse cursor)
        3 = show the plot with all background points
        4 = show the plot with linear spline background
        5 = show the plot with quadratic spline background
        6 = show the plot with cubic spline background
        ------------------------------------------------------------
        a = background points :: load the previously saved
        b = background points :: save to BKG-file'
        (BKG-file = {BKGfile}
        --------
        s = save current plot to PNG-file:
        (PNG-file = {PNGfile}
        (note: Matplotlib UI shortcut; optional output
        --------
        t = subtract current background & save data to TXT-file
        (TXT-file = {TXTfile}
        (note: this is a universal output, applicable in all cases
        --------
        u = subtract current background & update ediff.io.Profile
        (ediff.io.Profile = (alternative) object with i/o data
        (note: the object is from ediff package; ignore if not used
        ------------------------------------------------------------
        Standard Matplotlib UI tools and shortcuts work as well.
        See: https://matplotlib.org/stable/users/interactive.html
        ============================================================
        '''
        
        print(intelligent_dedent(help_text))
        
        
class WaveletMethod:
    '''
    Class with help functions.
    
    * The functions print simple textual help.
    * Here: help to bground.wvlet = WaveletMethod of background subtraction.
    '''
     
    
    def how_it_works():
        '''
        BGROUND printed help :: WaveletMethod :: How it works
        '''
        
        # TODO: brief text explanation
        # (analogy of the same method in InterativePlot class above
        print("Not implemented yet ...")

