# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""
Spyder Editor

Test some different plotting functionalities
Loads data and views them using functions written with PyQtGraph

"""
import sys
from .plot_functions import BEPSwindow
import pyqtgraph as pg

if __name__ == '__main__':
    '''
    Need to load the file from PySPM that should be visualized.
    Then need to extract the parameters from the file, such as the type 
    of measurement, number of cycles and the Vdc vector.
    We then need to feed all of this to the visualizer
    We also need to add another selection bar, that determines what we are 
    plotting i.e. A, phi, omega, or Q.
    '''

    app = pg.QtGui.QApplication([])

    h5_path = None

    '''
    Make window, cross-hairs, widgets, ROI, etc.
    '''
    win = BEPSwindow()
    win.setup(h5_path)
    win.showMaximized()
    win.setSignals()

    sys.exit(app.exec_())
