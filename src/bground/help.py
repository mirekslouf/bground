'''
Module: bground.help
--------------------
Help functions for bground package.

* This module is a collecton if simple textual help functions.
* The functions are usually called within bground.ui module = from GUI.
* Nevertheless, they can be also called directly, without any magic:

>>> # A direct access to the BGROUND help functions
>>> import bground as bkg
>>> bkg.help.print_general_description()
>>> bkg.help.print_how_it_works()
>>> bkg.help.print_all_keyboard_shortcuts()
>>> bkg.help.print_info_about_additional_help_on_www()

'''

def print_general_description():
    '''
    Print help - BGROUND package :: General description
    '''
    print('=============================================================')
    print('BGROUND package :: General description')
    print('-------------------------------------------------------------')
    print('* BGROUND = semi-automatic removal of background in XY-data')
    print('* XY-data = usually a file with two (or more) columns')
    print('  one of the columns = X-data, some other column = Y-data')
    print('* semi-automatic removal = user defines background points')
    print('  and computer does the rest')
    print('=============================================================')
    

def print_how_it_works():
    '''
    Print help - BGROUND package :: How does it work?'
    '''
    print('=============================================================')
    print('BGROUND package :: How it works?')
    print('-------------------------------------------------------------')
    print('* BGROUND opens Matplotlib interactive plot')
    print('* the user defines backround points with mouse and keyboard')
    print('* mouse actions/events are Matplotlib UI defaults')
    print('* keyboard actions/events are defined by the program')
    print('  - keys for background definition: 1,2,3,4,5,6')
    print('  - keys for saving the results   : a,b,t,s')
    print('  - basic help is printed when the interactive plot opens')
    print('  - more details: bground.help.print_all_keyboard_shortcuts')
    print('=============================================================')
    

def print_all_keyboard_shortcuts(output_file='some_file'):
    '''
    Print help - BGROUND :: Interactive plot :: Keyboard shortcuts
    '''
    
    # (1) Define output file names
    # (objective: all should have correct extensions
    # (but we want to avoid double TXT extension for the main TXT file
    TXTfile = output_file
    BKGfile = output_file + '.bkg'
    PNGfile = output_file + '.png'
    if not(TXTfile.lower().endswith('.txt')): TXTfile = TXTfile + '.txt'
    
    # (2) Print help including the above defined output file names
    print('============================================================')
    print('BGROUND :: Interactive plot :: Keyboard shortcuts')
    print('------------------------------------------------------------')
    print('1 = add a background point (at the mouse cursor position)')
    print('2 = delete a background point (close to the mouse cursor)')
    print('3 = show the plot with all background points')
    print('4 = show the plot with linear spline background')
    print('5 = show the plot with quadratic spline background')
    print('6 = show the plot with cubic spline background')
    print('------------------------------------------------------------')
    print('a = background points :: load the previously saved')
    print('b = background points :: save to BKG-file') 
    print(f'(BKG-file = {BKGfile}')
    print('--------')
    print('t = subtract current background & save data to TXT-file')
    print(f'(TXT-file = {TXTfile}')
    print('--------')
    print('s = save current plot to PNG-file:')
    print(f'(PNG-file = {PNGfile}')
    print('(note: Matplotlib UI shortcut; filename just recommened')
    print('------------------------------------------------------------')
    print('Standard Matplotlib UI tools and shortcuts work as well.')
    print('See: https://matplotlib.org/stable/users/interactive.html')
    print('============================================================')


def print_info_about_additional_help_on_www():
    '''
    Print help - BGROUND package :: Additional help on www
    '''
    print('=============================================================')
    print('BGROUND package :: Additional help on www')
    print('-------------------------------------------------------------')
    print('* PyPI    : https://pypi.org/project/bground')
    print('* GitHub  : https://github.com/mirekslouf/bground')
    print('  - pages : https://mirekslouf.github.io/bground')
    print('  - docum : https://mirekslouf.github.io/bground/docs')
    print('=============================================================')
