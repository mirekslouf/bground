'''
Module: bground.help
--------------------
Help functions for bground package.

* The functions are collected in this package.
* The functions are connected to the InteractivePlot object.
* The functions are usually called by means of the InteractivePlot object. 
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
    Print help - BGROUND package :: How it works?'
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
    

def print_all_keyboard_shortcuts(output_file='ouput_file.txt'):
    '''
    Print help - BGROUND :: Interactive plot :: Keyboard shortcuts
    '''
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
    print(f'(BKG-file = {output_file}' + '.bkg')
    print('--------')
    print('t = subtract current background & save data to TXT-file')
    print(f'(TXT-file = {output_file}')
    print('--------')
    print('s = save current plot to PNG-file:')
    print(f'(PNG-file = {output_file}' + '.png')
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
