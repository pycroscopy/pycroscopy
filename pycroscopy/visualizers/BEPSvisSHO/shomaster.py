# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""
Spyder Editor

Test some different plotting functionalities
Loads data and views them using functions written with PyQtGraph

"""
import sys
from plotFunctions import BEPSwindow
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
    BEPS
    
    I'm hardcoding paths here for easy of use.  If no path given a FileDialog 
    window will pop up to allow for selection.
    '''
    # h5_path = '../data/Raw_Data/BEPS/New_Data_Format/Real_Data/Sangmo/in_out_relax_5x5_0003/newdataformat/in_out_relax_5x5_0003.h5'
    # h5_path = '../data/Raw_Data/BEPS/New_Data_Format/Real_Data/Suhas/Ar3_BEPS_2VAC_20VDC_0003/newdataformat/Ar3_BEPS_2VAC_20VDC_0003.h5'
    # h5_path = r'C:\Users\cq6\git\PySPM\PySPM_project\data\Raw_Data\BEPS\New_Data_Format\Real_Data\Sangmo\BEPS_250x250nm2_0002\newdataformat\BEPS_250x250nm2_0002.h5'
    # h5_path = '../data/Raw_Data/BEPS/New_Data_Format/Real_Data/Sangmo/BEPS_800x800nm2_0003/newdataformat/BEPS_800x800nm2_0003.h5'
    # h5_path = '../data/Raw_Data/BEPS/New_Data_Format/Real_Data/Numan/spot1_50by50_0009/newdataformat/spot1_50by50_0009.h5'
    # h5_path = '../data/Raw_Data/BEPS/New_Data_Format/Real_Data/Suhas/HOPG_0_RH_chirp_excitation_0002/newdataformat/HOPG_0_RH_chirp_excitation_0002.h5'
    # h5_path = '../data/Raw_Data/BEPS/Old_Data_Format/Real_Data/LingLong/NonlafterFORC_2um_3Vac_50p_2cyc_0001/NonlafterFORC_2um_3Vac_50p_2cyc_0001.h5'
    # h5_path = r'../data/Raw_Data/BEPS/Old_Data_Format/Real_Data/Marius/1d_5x5mum_BEPS_0003_d/1d_5x5mum_BEPS_0003.h5'
    h5_path = r'C:\Users\cq6\git\PySPM\PySPM_project\data\Raw_Data\BEPS\New_Data_Format\Real_Data\Daehee Seol\20170224_MoTe2_S3_F4_d33_0001\newdataformat\20170224_MoTe2_S3_F4_d33_0001.h5'
    
    '''
    Make window, cross-hairs, widgets, ROI, etc.
    '''
    win = BEPSwindow()
    win.setup(h5_path)
    win.showMaximized()
    win.setSignals()
    
    sys.exit(app.exec_())
    
    

