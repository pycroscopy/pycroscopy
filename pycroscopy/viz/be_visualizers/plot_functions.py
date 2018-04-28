"""
Created on Apr 21, 2016

@author: Chris Smith -- csmith55@utk.edu
"""
import numpy as np
from .io_funcs import readData
import pyqtgraph as pg
from pyqtgraph import QtGui


class BEPSwindow(QtGui.QMainWindow):
    """
    Window object that will handle all the plotting
    """

    def __init__(self, **kwargs):
        """
        Create the initial window
        """
        super(BEPSwindow, self).__init__()
        winTitle = kwargs.get('winTitle', 'BEPS Visualization')
        self.setWindowTitle(winTitle)
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        pg.setConfigOption('antialias', True)

        '''
        Define some custom colormaps from VisIt
        '''
        self.hot_desaturated = {'ticks': [(1, (255, 76, 76, 255)),
                                          (0.857, (107, 0, 0, 255)),
                                          (0.714, (255, 96, 0, 255)),
                                          (0.571, (255, 255, 0, 255)),
                                          (0.429, (0, 127, 0, 255)),
                                          (0.285, (0, 255, 255, 255)),
                                          (0.143, (0, 0, 91, 255)),
                                          (0, (71, 71, 219, 255))],
                                'mode': 'rgb'}
        self.hot_cold = {'ticks': [(1, (255, 255, 0, 255)),
                                   (0.55, (255, 0, 0, 255)),
                                   (0.5, (0, 0, 127, 255)),
                                   (0.45, (0, 0, 255, 255)),
                                   (0, (0, 255, 255, 255))],
                         'mode': 'rgb'}
        self.difference = {'ticks': [(1, (255, 0, 0, 255)),
                                     (0.5, (255, 255, 255, 255)),
                                     (0, (0, 0, 255, 255))],
                           'mode': 'rgb'}
        '''
        Colormaps from Matlab
        '''
        self.jet = {'ticks': [(1, (128, 0, 0, 255)),
                              (0.875, (255, 0, 0, 255)),
                              (0.625, (255, 255, 0, 255)),
                              (0.375, (0, 255, 255, 255)),
                              (0.125, (0, 0, 255, 255)),
                              (0, (0, 0, 143, 255))],
                    'mode': 'rgb'}

    #     def sliceViewFunc(self,**kwargs):
    #         '''
    #         Function to slice N-d data to (N-1)-d.
    #         Makes a QWindow, fills it up with 2 ImageView objects, sets an roi to slice.
    #         Input:
    #             winTitle: string, Optional
    #                     Title of QWindow, default = 'Window'.None
    #             plotTitle: string, Optional
    #                     Title of Roi plot window, default = 'Plot'.
    #         Output:  QtGui.QMainWindow, PyQtGraph.ImageView,  PyQtGraph.ImageView, PyQtGraph.ROI
    #         '''
    #
    #         #Make QtWindow, populate it with ImageView and add roi.
    #
    #         win.setWindowTitle('pyqtgraph example: DataSlicing')
    #         cw = QtGui.QWidget()
    #         win.setCentralWidget(cw)
    #         l = QtGui.QGridLayout()
    #         cw.setLayout(l)
    #         imv1 = pg.ImageView()
    #         imv2 = pg.ImageView()
    #         l.addWidget(imv1, 0, 0)
    #         l.addWidget(imv2, 0, 1)
    #         roi = pg.LineSegmentROI([[10, 64], [120,64]], pen='y')
    #         imv1.addItem(roi)
    #
    #         # Customize plot window
    #         pltWdgt = imv1.getRoiPlot()
    #         plt = pltWdgt.getPlotItem()
    #         plt.setLabels(title = plotTitle)
    #         plt.showGrid(x = True, y = True, alpha = 1)
    #         pltdataitem = plt.listDataItems()[0]
    #         pltdataitem.setPen(color = 'r', width =1.)
    #         pltdataitem.setSymbol('o')
    #
    #         return win, imv1, imv2, roi

    # %%

    def setup(self, h5_path=None):
        """
        Call the readData functions from ioFuncs to setup the
        arrays for later uses and get the proper parameters
        """
        if not h5_path:
            h5_path = pg.FileDialog.getOpenFileName(caption='Select H5 file',
                                                    filter="H5 file (*.h5)")
            h5_path = str(h5_path)
        data_guess, data_results, xvec, xvec_labs, data_parts, ndims, data_main, freq_vec = readData(h5_path)

        num_cycles = data_guess.shape[1]

        y_labs = np.array([['Amplitude', 'V'], ['Frequency', 'Hz'], ['Q Factor', ''], ['Phase', 'rad']])

        # The following should ALL be derived from the file in question via PySPM!!
        plot_elements = [['Amplitude [V]', np.abs], ['Phase [rad]', np.angle]]

        initialPars = {'Number of Cycles': num_cycles,
                       'Type of Measurement:': 'IV',
                       'xlabel': xvec_labs,
                       'ylabel': y_labs,
                       'Plot Elements': plot_elements,
                       'Pieces': data_parts,
                       'NDims': ndims}
        '''
        Set up the plots using the read data
        '''
        self.__setupPlots(initialPars)
        '''
        Set the initial data
        '''
        self.__setInitialData(data_guess, data_results, xvec, data_main, freq_vec)

    def __setupPlots(self, initialPars, **kwargs):
        """
        This function sets up the GUI layout, adding PyQT widgets
        Fills the layout with objects, sets an roi to crosshair.
        
        Input: 
            initialParameters -- Dictionary,  
                                 Dictionary containing parameters of 
                                 data file, needed for GUI setup.
            winTitle -- string, Optional 
                        Title of QWindow, default = 'Window'.None
            
        Output:  
            None
            
        Shared Objects:
            imv1 -- QtGui.ImageView
                    Item which is used to plot the map of the SHO values at 
                    a given step
            imv2 -- QtGui.PlotWidget
                    Item which is used to plot the current loop selected by 
                    the CrossHairsRoi from imv1
            imv3 -- QtGui.PlotWidget
                    Item which is used to plot the DC offset or AC Amplitude 
                    vs. step number
            roiPt -- pg.CrossHairROI
                    ROI object within imv1 which determines the position used 
                    to generate the loop in imv2
            roi1 -- pg.ROI
                    Standard ROI object within imv1
            posLine -- pt.InfiniteLine
                    Object in imv3 which is used to denote the current UDVS 
                    step.  It's position is sinced with the frame number in 
                    imv1
            cycle_list -- QtGui.ComboBox
                    Object which allows the user to select which cycle to 
                    display
            plot_list -- QtGui.ComboBox
                    Object which allows the used to select which of the SHO 
                    parameters to plot
            roi_list -- QtGui.ComboBox
                    Object which allows the user to change the x-variable in 
                    imv2 from Voltage/Current to UDVS step
            part_list -- QtGuiComboBox
                    Object which allows the user to select between in or 
                    out-of-field for DC and forward or reverse for AC
            xlabel -- Numpy Array
                    Array holding (Name,Unit) pairs of options for the x-axis 
                    of imv2
            ylabel -- Numpy Array
                    Array holding (Name,Unit) pairs of options for the y-axis 
                    of imv2
            ndims -- Int
                    Number of spacial dimensions in data
        """
        #         Get the relevant parameters to enable plotting, labels etc.
        num_of_cycles = initialPars['Number of Cycles']
        ylabel = initialPars['ylabel']  # This has to change!
        xlabel = initialPars['xlabel']
        plot_elements_list = initialPars['Plot Elements']
        data_part_list = initialPars['Pieces']
        Ndims = initialPars['NDims']

        roi_elements_list = xlabel[:, 0]
        '''
        Setup the layout for the window
        '''
        cw = QtGui.QWidget()  # Add the plotting widget
        self.setCentralWidget(cw)  # What does this do?
        l = QtGui.QGridLayout()  # Use a layout
        cw.setLayout(l)

        '''
        Create the Image and plot widgets
        '''
        if Ndims == 1:
            imv1, imv2, imv3 = self.__setupOneD(xlabel, ylabel)
        else:
            imv1, imv2, imv3 = self.__setupTwoD(xlabel, ylabel)

        '''
        Create combo boxes
        '''
        cycle_list = QtGui.QComboBox()  # Make a combo box for selecting cycle number
        plot_list = QtGui.QComboBox()  # Make a combo box for selecting what variable to plot
        roi_list = QtGui.QComboBox()  # Choose between plotting versus voltage or step number
        part_list = QtGui.QComboBox()  # Choose the field or direction

        func_list = []
        '''
        Now populate them
        '''
        for i in xrange(num_of_cycles):
            cycle_list.addItem("Cycle " + str(i))

        for plot_element in plot_elements_list:
            plot_list.addItem(plot_element[0])
            func_list.append(plot_element[1])

        for roi_element in roi_elements_list:
            roi_list.addItem(roi_element)

        for part in data_part_list:
            part_list.addItem(part)

        glab = QtGui.QLabel('SHO Guess Parameters')
        g0lab = QtGui.QLabel('Amp: {}'.format('-'))
        g1lab = QtGui.QLabel('w0: {}'.format('-'))
        g2lab = QtGui.QLabel('Q: {}'.format('-'))
        g3lab = QtGui.QLabel('Phase: {}'.format('-'))

        rlab = QtGui.QLabel('SHO Result Parameters')
        r0lab = QtGui.QLabel('Amp: {}'.format('-'))
        r1lab = QtGui.QLabel('w0: {}'.format('-'))
        r2lab = QtGui.QLabel('Q: {}'.format('-'))
        r3lab = QtGui.QLabel('Phase: {}'.format('-'))

        '''
        Add the widgets to the layout
        
        addWidget(WidgetName, Row,Col, RowSpan,ColSpan) 
        '''
        l.addWidget(imv1, 0, 0, 13, 5)  # Add them at these positions
        l.addWidget(imv2, 0, 5, 13, 5)
        l.addWidget(imv3, 14, 0, 2, 10)
        l.addWidget(cycle_list, 0, 10)
        l.addWidget(plot_list, 1, 10)
        l.addWidget(roi_list, 2, 10)
        l.addWidget(part_list, 3, 10)
        l.addWidget(glab, 4, 10)
        l.addWidget(g0lab, 5, 10)
        l.addWidget(g1lab, 6, 10)
        l.addWidget(g2lab, 7, 10)
        l.addWidget(g3lab, 8, 10)
        l.addWidget(rlab, 9, 10)
        l.addWidget(r0lab, 10, 10)
        l.addWidget(r1lab, 11, 10)
        l.addWidget(r2lab, 12, 10)
        l.addWidget(r3lab, 13, 10)

        '''
        Customize the Voltage/Current vs time plot
        '''
        imv3.setMaximumHeight(150)  # Don't want this to be too big
        imv3.getPlotItem().setLabel('left', text=ylabel[0, 0], units=ylabel[0, 1])
        imv3.getPlotItem().setLabel('bottom', text=xlabel[1, 0], units=xlabel[1, 1])
        imv3.getViewBox().setMouseEnabled(x=False, y=False)
        posLine = pg.InfiniteLine(angle=90, movable=True, pen='g')
        imv3.addItem(posLine)
        posLine.setValue(0)
        posLine.setZValue(100)

        '''
        Share variables we'll need for later
        '''
        self.ndims = Ndims
        self.imv1 = imv1
        self.imv2 = imv2
        self.imv3 = imv3
        self.posLine = posLine
        self.cycle_list = cycle_list
        self.plot_list = plot_list
        self.roi_list = roi_list
        self.part_list = part_list
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.func_list = func_list
        self.func = func_list[0]
        self.glabs = [g0lab, g1lab, g2lab, g3lab]
        self.rlabs = [r0lab, r1lab, r2lab, r3lab]

        return

    #     def rawSpectViewFunc(self,initialPars, **kwargs):
    #         """
    #         This function sets up the GUI layout, adding PyQT widgets
    #         Makes a QWindow, fills it with ImageView objects, sets an roi to line.
    #         Allows viewing of raw spectrograms
    #         Input:
    #             initialParameters Dictionary,
    #                     Dictionary containing parameters of data file, needed for GUI setup.
    #             winTitle: string, Optional
    #                     Title of QWindow, default = 'Window'.None
    #
    #         Output:  QtGui.QMainWindow, PyQtGraph.ImageView,  PyQtGraph.ImageView, PyQtGraph.ROI
    #
    #         """
    #
    #         #Get the relevant parameters to enable plotting, labels etc.
    #         #Make QtWindow, populate it with ImageView and add roi.
    #         win = QtGui.QMainWindow()
    #         cw = QtGui.QWidget() #Add the plotting widget
    #         win.setCentralWidget(cw) #What does this do?
    #         l = QtGui.QGridLayout() #Use a layout
    #         cw.setLayout(l)
    #         imv1 = pg.ImageView() #Make an ImageView widget.
    #
    #         return win, imv1

    @staticmethod
    def __setupOneD(xlabel, ylabel):
        """
        Creates the needed widgets for plotting 1D data

        Parameters
        ----------
        xlabel : list of str
            list of labels and units to use for x-axis of plots
        ylabel : list of str
            list of labels and units to use for y-axis of plots

        Returns
        -------
        imv1 : pyqtgraph.PlotWidget
            PlotWidget in which the value vs Position will be plotted
        imv2 : pyqtgraph.PlotWidget
            PlotWidget in which the loops will be plotted
        imv3 : pyqtgraph.PlotWidget
            PlotWidget in which the timeline will be plotted

        """

        imv1 = pg.PlotWidget()
        imv2 = pg.PlotWidget()
        imv3 = pg.PlotWidget()

        '''
        Setup all the labeling
        '''
        imv1.setTitle('Plot of {} vs Frequency Bin'.format(ylabel[0, 0]))
        imv1.setLabel('left', text=ylabel[0, 0], units=ylabel[0, 1])
        imv1.setLabel('bottom', text='Frequency Bin')

        '''
        Customize Spectra graph labels and plotting    
        '''
        imv2.setLabel('left', text=ylabel[0, 0], units=ylabel[0, 1])
        imv2.setLabel('bottom', text=xlabel[0, 0], units=xlabel[0, 1])

        return imv1, imv2, imv3

    def __setupTwoD(self, xlabel, ylabel):
        """
        Creates the needed widgets for plotting 1D data

        Inputs:
            xlabel -- list of labels and units to use for x-axis of plots
            ylabel -- list of labels and units to use for y-axis of plots

        Outputs:
            imv1 -- ImageView Widget in which the map will be plotted
            imv2 -- PlotWidget in which the loops will be plotted
            imv3 -- PlotWidget in which the timeline will be plotted

        Shared:
            plt1 -- PlotItem associated with the map to show position axes
                    and the title
            roi1 -- The crosshairs roi object that determines the position for
                    which the loops are plotted in imv2
            roiplt1 -- The box roi object associated with imv1
            posline -- the
        """
        plt1 = pg.PlotItem()
        imv1 = pg.ImageView(view=plt1)
        imv2 = pg.PlotWidget()
        imv3 = pg.PlotWidget()

        '''
        Prevent map from being to small and add the crosshair ROI
        '''
        imv1.setMinimumSize(500, 500)
        plt1.setTitle('Map of {} vs Position'.format(ylabel[0, 0]))
        roiPt = pg.CrosshairROI([0, 0], size=2, pen='b')  # Add a cross-hair ROI item
        imv1.addItem(roiPt)
        grad1 = imv1.ui.histogram.gradient
        #         grad1.loadPreset('thermal') #choose the default colormap from presets
        grad1.restoreState(self.jet)  # Set default colormap defined in __init__
        imv1.getView().setMouseEnabled(x=False, y=False)
        imv1.ui.menuBtn.hide()

        '''
        Customize Spectra graph labels and plotting    
        '''
        imv2.setTitle('Loop for Position {}'.format(roiPt.pos()))
        imv2.setLabel('left', text=ylabel[0, 0], units=ylabel[0, 1])
        imv2.setLabel('bottom', text=xlabel[0, 0], units=xlabel[0, 1])

        '''
        Customize ROI plot window
        '''
        roiplt1 = imv1.getRoiPlot().getPlotItem()
        roiplt1.setLabel('left', text=ylabel[0, 0], units=ylabel[0, 1])
        roiplt1.setLabel('bottom', text=xlabel[0, 0], units=xlabel[0, 1])

        '''
        Share variables unique to 2D mode
        '''
        self.plt1 = plt1
        self.roi1 = roiplt1
        self.roiPt = roiPt

        return imv1, imv2, imv3

    def __setInitialData(self, data_guess, data_results, xvec, data_main, freq_vec):
        """
        Tell the window what data it should be plotting from
        """
        points = np.arange(len(xvec[0, 0, 0, :]) + 1)
        self.num_bins = len(freq_vec)

        if self.ndims == 1:
            self.__setDataOneD(data_guess, data_results, xvec, data_main, freq_vec, points)
        else:
            self.__setDataTwoD(data_guess, data_results, xvec, data_main, freq_vec, points)

        '''
        Initialize all shared selectors
        '''
        self.data_guess = data_guess
        self.data_results = data_results
        self.data_main = data_main
        self.freq_vec = freq_vec
        self.xvec = xvec
        self.points = points
        self.sel_frame = self.imv1.currentIndex
        self.sel_cycle = self.cycle_list.currentIndex()
        self.sel_plot = self.plot_list.currentIndex()
        self.plot_name = str(self.plot_list.currentText())
        self.sel_roi = self.roi_list.currentIndex()
        self.sel_part = self.part_list.currentIndex()

        return

    def __setDataOneD(self, data_guess, data_results, xvec, data_main, freq_vec, points):
        """
        Sets the initial data for the case of one spacial dimension
        Inputs:
            data_guess -- 6D numpy array holding BEPS data.
                        Indices are:
                        [Field, SHO, Cycle #, UDVS Step, X-Pos, Y-Pos]

                    Field:  "In-field" or "Out-of-field" for DC
                            "Forward" or "Reverse" for AC
                    SHO:    SHO parameter to be plotted
                            Amplitude, Resonance Frequency, Q-factor, Phase,
                            or R^2 value
                    Cycle:  Which cycle should be plotted
                    UDVS:   The UDVS step of the current plot
                    X-Pos:  X position from dataset
                    Y-Pos:  Y position from dataset

                        These indices should be present for all datasets
                        even if they only have a length of 1

             xvec -- 4D numpy array containing the values which will be plotted
                     against
                     Indices are
                     [Variable, Field, Cycle #, UDVS Step]

                     As with datamat, all indices must be present no matter
                     the shape of the actual data

                Variable:   The variable to be plotted.  DC voltage or
                            AC amplitude depending on the mode
                Field:      Same as is datamat
                Cycle #:    Same as in datamat
                UDVS Step:  Same as in datamat

            points -- numpy arrange of UDVS step numbers.  One extra step is
                      added at the end to allow for plotting in step mode
        """

        '''
        Plot Initial data
        '''
        positions = np.arange(data_guess.shape[3])
        self.plot1 = self.imv1.plot(x=positions,
                                    y=data_guess[0, 0, 0, :, 0],
                                    pen='k')
        self.imv1.currentIndex = 0
        roiPt = pg.InfiniteLine(angle=90, movable=True, pen='r')
        self.imv1.addItem(roiPt)
        self.roiPt = roiPt
        self.roiPt.setPos(0)
        self.point_roi = self.__getROIpos()
        self.roiPt.setBounds((positions[0], positions[-1]))

        self.row = int(self.point_roi[0])
        self.col = int(self.point_roi[1])

        self.main_plot = pg.PlotDataItem(x=np.arange(self.num_bins),
                                         y=self.func(data_main[0, 0, :, 0, 0]),
                                         pen='g')
        self.guess = self.__calc_sho(data_guess[0, 0, 0, 0, 0], freq_vec)
        self.guess_plot = pg.PlotDataItem(x=np.arange(self.num_bins),
                                          y=self.func(self.guess),
                                          pen='k')

        self.imv2.addItem(self.main_plot)
        self.imv2.addItem(self.guess_plot)

        if data_results is not None:
            self.results = self.__calc_sho(data_results[0, 0, 0, 0, 0], freq_vec)
            self.results_plot = pg.PlotDataItem(x=np.arange(self.num_bins),
                                                y=self.func(self.results),
                                                pen='r')
            self.imv2.addItem(self.results_plot)
        else:
            self.results = None
            self.results_plot = None

        self.imv2.setTitle('Loop for Position {}'.format(self.point_roi))

        self.imv3.plot(points, xvec[0, 0, 0, :],
                       stepMode=True,
                       pen='k')
        self.posLine.setBounds([points[0], points[-2]])

    def __setDataTwoD(self, data_mat, data_results, xvec, data_main, freq_vec, points):
        """
        Sets the initial data for the case of two spacial dimensions

        Inputs:
            data_guess -- 6D numpy array holding BEPS data.
                        Indices are:
                        [Field, SHO, Cycle #, UDVS Step, X-Pos, Y-Pos]

                    Field:  "In-field" or "Out-of-field" for DC
                            "Forward" or "Reverse" for AC
                    SHO:    SHO parameter to be plotted
                            Amplitude, Resonance Frequency, Q-factor, Phase,
                            or R^2 value
                    Cycle:  Which cycle should be plotted
                    UDVS:   The UDVS step of the current plot
                    X-Pos:  X position from dataset
                    Y-Pos:  Y position from dataset

                        These indices should be present for all datasets
                        even if they only have a length of 1

             xvec -- 4D numpy array containing the values which will be plotted
                     against
                     Indices are
                     [Variable, Field, Cycle #, UDVS Step]

                     As with datamat, all indices must be present no matter
                     the shape of the actual data

                Variable:   The variable to be plotted.  DC voltage or
                            AC amplitude depending on the mode
                Field:      Same as is datamat
                Cycle #:    Same as in datamat
                UDVS Step:  Same as in datamat

            points -- numpy arrange of UDVS step numbers.  One extra step is
                      added at the end to allow for plotting in step mode
        """

        '''
        Plot Initial data
        '''
        self.imv1.setImage(data_mat['Amplitude [V]'][0, 0, :, :, :],
                           xvals=xvec[0, 0, 0, :])
        self.plt1.getViewBox().autoRange(padding=0.1)
        '''
        Prevent the selection of a point from outside the domain
        '''
        self.roiPt.maxBounds = self.imv1.getImageItem().boundingRect()
        roisize = self.roiPt.size()
        self.roiPt.maxBounds.adjust(0, 0, roisize[0] / 2, roisize[1] / 2)

        self.point_roi = self.__getROIpos()  # Get position

        self.row = int(self.point_roi[0])
        self.col = int(self.point_roi[1])

        self.main_plot = pg.PlotDataItem(x=np.arange(self.num_bins),
                                         y=self.func(data_main[0, 0, 0, :, 0, 0]),
                                         pen='g')
        self.guess = self.__calc_sho(data_mat[0, 0, 0, 0, 0], freq_vec)
        self.guess_plot = pg.PlotDataItem(x=np.arange(self.num_bins),
                                          y=self.func(self.guess),
                                          pen='k')

        self.imv2.addItem(self.main_plot)
        self.imv2.addItem(self.guess_plot)

        if data_results is not None:
            self.results = self.__calc_sho(data_results[0, 0, 0, 0, 0], freq_vec)
            self.results_plot = pg.PlotDataItem(x=np.arange(self.num_bins),
                                                y=self.func(self.results),
                                                pen='r')
            self.imv2.addItem(self.results_plot)
        else:
            self.results = None
            self.results_plot = None

        self.imv3.plot(points, xvec[0, 0, 0, :],
                       stepMode=True,
                       pen='k')
        self.posLine.setBounds([points[0],
                                points[-2]])

        self.point_roi = self.__getROIpos()

    def __getROIpos(self):
        """
        Returns the current position of the ROI selector
        """
        roipos = self.roiPt.pos()
        if self.ndims == 1:
            for i in xrange(1, len(roipos)):
                roipos[i] = 0
        return roipos

    def setSignals(self):
        """
        Define the signals to be watched and how they connect to
        update functions
        """
        if self.ndims > 1:
            self.roiPt.sigRegionChanged.connect(self.updateROICross)  # Link roi changes to update function
            self.imv1.timeLine.sigPositionChanged.connect(self.updateTimeMap)
        else:
            self.roiPt.sigPositionChanged.connect(self.updateROICross)  # Link roi changes to update function
        self.posLine.sigPositionChanged.connect(self.updateTimeLine)
        self.cycle_list.activated[str].connect(self.updateCycleList)  # Link cycle changes to update function
        self.plot_list.activated[str].connect(self.updatePlotList)  # Link plot variable changes to update function
        self.roi_list.activated[str].connect(self.updateROIList)  # Link ROI variable changes to update function
        self.part_list.activated[str].connect(self.updatePartList)
        self.guess_plot.sigPlotChanged.connect(self.updateParms)

    #     Update functions for interactivity
    def updateCycleList(self):
        """
        Save the current frame
        """
        current_frame = self.sel_frame

        '''
        Get the new cycle
        '''
        self.sel_cycle = self.cycle_list.currentIndex()

        '''
        Replot everything
        '''
        if self.ndims > 1:
            self.imv1.setImage(self.data_guess[self.plot_name][self.sel_part, self.sel_cycle, :, :, :],
                               xvals=self.xvec[self.sel_roi,
                                     self.sel_part,
                                     self.sel_cycle, :],
                               autoHistogramRange=True)

        else:
            self.plot1.setData(y=self.data_guess[self.plot_name][self.sel_part,
                                 self.sel_cycle,
                                 self.sel_frame, :, 0])

        self.main_plot.setData(y=self.func(self.data_main[self.sel_part,
                                           self.sel_cycle,
                                           self.sel_frame, :,
                                           self.row,
                                           self.col]))
        self.guess = self.__calc_sho(self.data_guess[self.sel_part,
                                                     self.sel_cycle,
                                                     self.sel_frame,
                                                     self.row,
                                                     self.col],
                                     self.freq_vec)
        self.guess_plot.setData(y=self.func(self.guess))

        if self.results is not None:
            self.results = self.__calc_sho(self.data_results[self.sel_part,
                                                             self.sel_cycle,
                                                             self.sel_frame,
                                                             self.row,
                                                             self.col],
                                           self.freq_vec)
            self.results_plot.setData(y=self.func(self.results))

        '''
        Set the selected frame to the current frame
        '''
        self.sel_frame = current_frame
        if self.ndims > 1:
            self.imv1.setCurrentIndex(self.sel_frame)
        self.posLine.setPos(self.sel_frame)

    def updatePlotList(self):
        """
        Save the current frame
        """
        current_frame = self.sel_frame

        '''
        Get the new plot selection and update Map
        '''
        self.sel_plot = self.plot_list.currentIndex()
        self.plot_name = str(self.plot_list.currentText())
        self.func = self.func_list[self.sel_plot]

        if self.ndims > 1:
            self.imv1.setImage(self.data_guess[self.plot_name][self.sel_part,
                               self.sel_cycle, :, :, :],
                               xvals=self.xvec[self.sel_roi,
                                     self.sel_part,
                                     self.sel_cycle, :],
                               autoHistogramRange=True)
            self.plt1.setTitle('Map of {} vs Position'.format(self.ylabel[self.sel_plot, 0]))

            self.roi1.setLabel('left',
                               text=self.ylabel[self.sel_plot, 0],
                               units=self.ylabel[self.sel_plot, 1])
        else:
            self.plot1.setData(y=self.data_guess[self.plot_name][self.sel_part,
                                 self.sel_cycle,
                                 self.sel_frame, :, 0])
            self.imv1.setTitle('Plot of {} vs {}'.format(self.ylabel[self.sel_plot, 0],
                                                         self.xlabel[self.sel_roi, 0]))

        self.main_plot.setData(y=self.func(self.data_main[self.sel_part,
                                           self.sel_cycle,
                                           self.sel_frame, :,
                                           self.row,
                                           self.col]))
        self.guess = self.__calc_sho(self.data_guess[self.sel_part,
                                                     self.sel_cycle,
                                                     self.sel_frame,
                                                     self.row,
                                                     self.col],
                                     self.freq_vec)
        self.guess_plot.setData(y=self.func(self.guess))
        self.imv2.setLabel('left',
                           text=self.ylabel[self.sel_plot, 0],
                           units=self.ylabel[self.sel_plot, 1])

        if self.results is not None:
            self.results = self.__calc_sho(self.data_results[self.sel_part,
                                                             self.sel_cycle,
                                                             self.sel_frame,
                                                             self.row,
                                                             self.col],
                                           self.freq_vec)
            self.results_plot.setData(y=self.func(self.results))

        '''
        Set the selected frame to the current frame
        '''
        self.sel_frame = current_frame
        if self.ndims > 1:
            self.imv1.setCurrentIndex(self.sel_frame)
        self.posLine.setPos(self.sel_frame)

    def updateROIList(self):
        """
        Save the current frame
        """
        current_frame = self.sel_frame

        '''
        Get the new choice of ROI variable from the list and update plots
        '''
        self.sel_roi = self.roi_list.currentIndex()

        if self.ndims > 1:
            self.imv1.setImage(self.data_guess[self.plot_name][self.sel_part,
                               self.sel_cycle, :, :, :],
                               xvals=self.xvec[self.sel_roi,
                                     self.sel_part,
                                     self.sel_cycle, :],
                               autoHistogramRange=True)

            self.roi1.setLabel('bottom',
                               text=self.xlabel[self.sel_roi, 0],
                               units=self.xlabel[self.sel_roi, 1])

        self.main_plot.setData(y=self.func(self.data_main[self.sel_part,
                                           self.sel_cycle,
                                           self.sel_frame, :,
                                           self.row,
                                           self.col]))
        self.guess = self.__calc_sho(self.data_guess[self.sel_part,
                                                     self.sel_cycle,
                                                     self.sel_frame,
                                                     self.row,
                                                     self.col],
                                     self.freq_vec)
        self.guess_plot.setData(y=self.func(self.guess))

        if self.results is not None:
            self.results = self.__calc_sho(self.data_results[self.sel_part,
                                                             self.sel_cycle,
                                                             self.sel_frame,
                                                             self.row,
                                                             self.col],
                                           self.freq_vec)
            self.results_plot.setData(y=self.func(self.results))

        self.imv2.setLabel('bottom',
                           text=self.xlabel[self.sel_roi, 0],
                           units=self.xlabel[self.sel_roi, 1])

        '''
        Set the selected frame to the current frame
        '''
        self.sel_frame = current_frame
        if self.ndims > 1:
            self.imv1.setCurrentIndex(self.sel_frame)
        self.posLine.setPos(self.sel_frame)

    def updatePartList(self):
        """
        Save the current frame
        """
        current_frame = self.sel_frame

        '''
        Get the new part variable and update plots
        
        The part variable is the Field for DC and the direction for AC
        '''
        self.sel_part = self.part_list.currentIndex()

        if self.ndims > 1:
            self.imv1.setImage(self.data_guess[self.plot_name][self.sel_part,
                               self.sel_cycle, :, :, :],
                               xvals=self.xvec[self.sel_roi,
                                     self.sel_part,
                                     self.sel_cycle, :],
                               autoHistogramRange=True)
        else:
            self.plot1.setData(y=self.data_guess[self.plot_name][self.sel_part,
                                 self.sel_cycle,
                                 self.sel_frame, :, 0])

        self.main_plot.setData(y=self.func(self.data_main[self.sel_part,
                                           self.sel_cycle,
                                           self.sel_frame, :,
                                           self.row,
                                           self.col]))
        self.guess = self.__calc_sho(self.data_guess[self.sel_part,
                                                     self.sel_cycle,
                                                     self.sel_frame,
                                                     self.row,
                                                     self.col],
                                     self.freq_vec)
        self.guess_plot.setData(y=self.func(self.guess))

        if self.results is not None:
            self.results = self.__calc_sho(self.data_results[self.sel_part,
                                                             self.sel_cycle,
                                                             self.sel_frame,
                                                             self.row,
                                                             self.col],
                                           self.freq_vec)
            self.results_plot.setData(y=self.func(self.results))

        '''
        Set the selected frame to the current frame
        '''
        if self.ndims > 1:
            self.sel_frame = current_frame
            self.imv1.setCurrentIndex(self.sel_frame)
        self.posLine.setPos(self.sel_frame)

    def updateROICross(self):
        """
        Get the new roi position and update the loop plot accordingly
        """
        self.point_roi = self.__getROIpos()

        self.row = int(self.point_roi[0])
        self.col = int(self.point_roi[1])

        self.main_plot.setData(x=np.arange(self.num_bins),
                               y=self.func(self.data_main[self.sel_part,
                                           self.sel_cycle,
                                           self.sel_frame, :,
                                           self.row,
                                           self.col]))
        self.guess = self.__calc_sho(self.data_guess[self.sel_part,
                                                     self.sel_cycle,
                                                     self.sel_frame,
                                                     self.row,
                                                     self.col],
                                     self.freq_vec)
        self.guess_plot.setData(y=self.func(self.guess))

        if self.results is not None:
            self.results = self.__calc_sho(self.data_results[self.sel_part,
                                                             self.sel_cycle,
                                                             self.sel_frame,
                                                             self.row,
                                                             self.col],
                                           self.freq_vec)
            self.results_plot.setData(y=self.func(self.results))

        self.imv2.setTitle('SHO for Position {}'.format(self.point_roi))

    def updateTimeMap(self):
        """
        Get the new frame index from the map and update timeline
        """
        self.sel_frame = self.imv1.currentIndex

        self.posLine.setPos(self.sel_frame)

    def updateTimeLine(self):
        """
        Get the new frame index from the timeline and update map
        """
        self.sel_frame = int(np.floor(self.posLine.value()))

        if self.ndims == 1:
            self.plot1.setData(y=self.data_guess[self.plot_name][0, 0, self.sel_frame, :, 0])
        else:
            self.imv1.setCurrentIndex(self.sel_frame)

        self.main_plot.setData(x=np.arange(self.num_bins),
                               y=self.func(self.data_main[self.sel_part,
                                           self.sel_cycle,
                                           self.sel_frame, :,
                                           self.row,
                                           self.col]))
        self.guess = self.__calc_sho(self.data_guess[self.sel_part,
                                                     self.sel_cycle,
                                                     self.sel_frame,
                                                     self.row,
                                                     self.col],
                                     self.freq_vec)
        self.guess_plot.setData(y=self.func(self.guess))

        if self.results is not None:
            self.results = self.__calc_sho(self.data_results[self.sel_part,
                                                             self.sel_cycle,
                                                             self.sel_frame,
                                                             self.row,
                                                             self.col],
                                           self.freq_vec)
            self.results_plot.setData(y=self.func(self.results))

    def updateParms(self):
        """
        update the values of the parameters printed to the sidebar
        """
        gparms = self.data_guess[self.sel_part,
                                 self.sel_cycle,
                                 self.sel_frame,
                                 self.row,
                                 self.col]
        self.glabs[0].setText('Amp: {}'.format(gparms['Amplitude [V]']))
        self.glabs[1].setText('w0: {}'.format(gparms['Frequency [Hz]']))
        self.glabs[2].setText('Q: {}'.format(gparms['Quality Factor']))
        self.glabs[3].setText('Phase: {}'.format(gparms['Phase [rad]']))

        if self.data_results is not None:
            rparms = self.data_results[self.sel_part,
                                       self.sel_cycle,
                                       self.sel_frame,
                                       self.row,
                                       self.col]
            self.rlabs[0].setText('Amp: {}'.format(rparms['Amplitude [V]']))
            self.rlabs[1].setText('w0: {}'.format(rparms['Frequency [Hz]']))
            self.rlabs[2].setText('Q: {}'.format(rparms['Quality Factor']))
            self.rlabs[3].setText('Phase: {}'.format(rparms['Phase [rad]']))

    @staticmethod
    def __calc_sho(p, xvec):
        """
        Calculates the SHO for the given parameters and frequency vector

        Parameters
        ----------
        p
        xvec

        Returns
        -------

        """
        from pycroscopy.analysis.utils.be_sho import SHOfunc

        p = [p[name] for name in p.dtype.names]

        return SHOfunc(p, xvec)
