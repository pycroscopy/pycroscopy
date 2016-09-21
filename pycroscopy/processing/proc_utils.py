'''
Created on Mar 1, 2016

@author: Chris Smith -- cmsith55@utk.edu
'''
import numpy as np
import numpy_groupies as npg
def buildHistogram(x_hist, data_mat, N_x_bins, N_y_bins, weighting_vec=1, min_resp=None, max_resp=None, func=None, debug=False, *args, **kwargs):
        '''
        Creates histogram for a single block of pixels
        
        Parameters:
        -------------
        x_hist : 1D numpy array
            bins for x-axis of 2d histogram
        data_mat : numpy array
            data to be binned for y-axis of 2d histogram
        weighting_vec : 1D numpy array or float
            weights. If setting all to one value, can be a scalar
        N_x_bins : integer
            number of bins in the x-direction
        N_y_bins : integer 
            number of bins in the y-direction
        min_resp : float
            minimum value for y binning
        max_resp : float
            maximum value for y binning
        func : function
            function to be used to bin data_vec.  All functions should take as input data_vec.
            Arguments should be passed properly to func.  This has not been heavily tested.
                
        Output:
        --------------
        pixel_hist : 2D numpy array
            contains the histogram of the input data
        
            
        
        Apply func to input data, convert to 1D array, and normalize
        '''
        if debug: print 'min_resp',min_resp,'max_resp',max_resp
        y_hist = data_mat
        if func is not None:
            y_hist = func(y_hist, *args, **kwargs)
        y_hist = np.squeeze(np.reshape(y_hist,(data_mat.size,1)))
        y_hist = np.clip(y_hist, min_resp, max_resp, y_hist)
        y_hist = y_hist-min_resp
        y_hist = y_hist/(max_resp-min_resp)
        
        '''
        Descritize y_hist
        '''
        y_hist = np.rint(y_hist*(N_y_bins-1))
        if debug: print 'ymin',min(y_hist),'ymax',max(y_hist)
        
        '''
        Combine x_hist and y_hist into one matrix
        '''
        if debug:
            print np.shape(x_hist)
            print np.shape(y_hist)
        
        try:
            group_idx = np.zeros((2,x_hist.size), dtype = np.int32)
            group_idx[0,:] = x_hist
            group_idx[1,:] = y_hist
        except:
            raise
        
        '''
        Aggregate matrix for histogram of current chunk
        '''
        if debug:
            print np.shape(group_idx)
            print np.shape(weighting_vec)
            print N_x_bins,N_y_bins
            
        try:
            pixel_hist = npg.aggregate_np(group_idx, weighting_vec, func='sum', size = (N_x_bins,N_y_bins), dtype = np.int32)
        except:
            raise

        return pixel_hist
    