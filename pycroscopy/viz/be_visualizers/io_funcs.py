"""
Created on Apr 20, 2016

@author: Chris Smith -- csmith55@utk.edu
"""

import sys
import numpy as np
import h5py
sys.path.append('../')
from scipy.io import loadmat


def loadDataFunc(filePath, **kwargs):
    """
    Function to load the N-D data from a .mat file
    Output:  N-D numpy data array, Nx2 x-vector array
            array indices are (Step, #rows, #cols, cycle#)
    """
    data = loadmat(filePath)
    data_mat = data['loop_mat']
    data_mat = data_mat[:, :, :, :]
    data_mat[np.where(data_mat > 1E-1)] = 0  # Presumably, this will not be requried.
    data_mat[np.where(data_mat < -1E-1)] = 0  # Otherwise we will have to write smoothing/cleaning functions
    data_mat = np.transpose(data_mat, (3, 0, 1, 2))
    xvec = data['VV'].ravel()

    return data_mat, xvec


def readData(h5_path, dset_name='SHO_Fit_Guess'):
    """
    Reads the hdf5 data file and calls appropriate reader based on data type
    Input:
        h5_path -- the absolute file path to the hdf5 file to be read in.
        dset_name -- the name of the main dataset

    Outputs:
        data_mat -- the transformed data read to be plotted
        xvec -- numpy array containing the possible plot data of the slice viewer
        xvec_labs -- numpy array of labels and units for the xvec array

    """

    h5_file = h5py.File(h5_path, 'r')

    exp_type = h5_file.attrs['data_type']

    h5_group = h5_file['Measurement_000']

    data_type = h5_group.attrs['VS_mode']
    if data_type == 'DC modulation mode':
        return readDCData(h5_group)
    elif data_type == 'AC modulation mode with time reversal':
        return readACData(h5_group)


# return readDataGen(h5_group['SHO_Fit_Results'])


def readDCData(h5_group):
    """
    Reads the data for DC modulation experiments

    Inputs:
        h5_group -- hdf5 group holding the SHO_Fit Data for the chosen
                    Measurement group

    Outputs:
        data_guess -- the transformed data to be plotted
        xvec -- numpy array containing the possible plot data of the slice viewer
        xvec_labs -- numpy array of labels and units for the xvec array
    """
    h5_chan = h5_group['Channel_000']
    h5_main = h5_chan['Raw_Data']
    h5_file = h5_main.file
    h5_specv = h5_file[h5_main.attrs['Spectroscopic_Values']]
    h5_bins = h5_file[h5_main.attrs['Bin_Frequencies']]

    h5_shogroup = h5_chan['Raw_Data-SHO_Fit_000']
    h5_guess = h5_shogroup['Guess']

    h5_data_labels = h5_specv.attrs['labels']
    h5_sho_specv = h5_file[h5_guess.attrs['Spectroscopic_Values']]
    h5_indices = h5_file[h5_guess.attrs['Spectroscopic_Indices']]
    h5_pos = h5_file[h5_main.attrs['Position_Indices']]

    num_rows, num_cols = __getPos(h5_pos)
    num_pos = num_rows * num_cols

    ndims = len(np.where(np.array(h5_pos.shape) > 1)[0])

    num_cycles = h5_group.attrs['VS_number_of_cycles']
    num_steps = len(np.unique(h5_indices[:, 1]))
    num_bins = len(np.unique(h5_bins))

    data_xvec = h5_sho_specv[h5_sho_specv.attrs['DC_Offset']].flatten()
    field_type = h5_group.attrs['VS_measure_in_field_loops']

    '''
    Separate data into cycles
    '''
    data_guess = np.reshape(h5_guess, [num_pos, num_cycles, -1])
    data_guess = np.transpose(data_guess, [1, 2, 0])
    data_xvec = np.reshape(data_xvec, [num_cycles, -1])
    data_main = np.reshape(h5_main, [num_pos, num_cycles, -1])
    data_main = np.transpose(data_main, [1, 2, 0])

    pt_xvec = np.tile(np.arange(data_xvec.shape[1]), [num_cycles, 1])

    if field_type == 'in and out-of-field':
        '''
    Seperate out in-field and out-of-field values
        '''
        data_guess = np.array([data_guess[:, slice(0, None, 2), :], data_guess[:, slice(1, None, 2), :]])
        data_guess = np.reshape(data_guess, (2, num_cycles, -1, num_rows, num_cols))
        data_xvec = np.array([data_xvec[:, slice(0, None, 2)], data_xvec[:, slice(1, None, 2)]])
        pt_xvec = np.array([pt_xvec[:, slice(0, None, 2)], pt_xvec[:, slice(1, None, 2)]])

        data_main = np.reshape(data_main, (num_cycles, -1, num_bins, num_rows, num_cols))
        data_main = np.array([data_main[:, slice(0, None, 2), :, :, :], data_main[:, slice(1, None, 2), :, :, :]])

        data_parts = ['In-field', 'Out-of-field']
    elif field_type == 'in-field':
        data_guess = np.reshape(data_guess, (1, num_cycles, -1, num_rows, num_cols))
        data_xvec = data_xvec.reshape(1, num_cycles, -1)
        pt_xvec = pt_xvec.reshape(1, num_cycles, -1)

        data_main = np.reshape(data_main, (1, num_cycles, -1, num_bins, num_rows, num_cols))

        data_parts = ['In-field']

    elif field_type == 'out-of-field':
        data_guess = np.reshape(data_guess, (1, num_cycles, -1, num_rows, num_cols))
        data_xvec = data_xvec.reshape(1, num_cycles, -1)
        pt_xvec = pt_xvec.reshape(1, num_cycles, -1)

        data_main = np.reshape(data_main, (1, num_cycles, -1, num_bins, num_rows, num_cols))

        data_parts = ['Out-of-field']
    else:
        raise ValueError('Unknown field type in data.  Cannot parse')

    '''
    Repeat steps for reading guess on the results if present
    '''

    try:
        h5_results = h5_shogroup['Fit']
        data_results = np.reshape(h5_results, [num_pos, num_cycles, -1])
        data_results = np.transpose(data_results, [1, 2, 0])

        if field_type == 'in and out-of-field':
            data_results = np.array([data_results[:, slice(0, None, 2), :],
                                     data_results[:, slice(1, None, 2), :]])
            data_results = np.reshape(data_results, (2, num_cycles, -1, num_rows, num_cols))
        elif field_type == 'in-field':
            data_results = np.reshape(data_results, (1, num_cycles, -1, num_rows, num_cols))
        elif field_type == 'out-of-field':
            data_results = np.reshape(data_results, (1, num_cycles, -1, num_rows, num_cols))

    except KeyError:
        data_results = None

    xvec = np.array([data_xvec, pt_xvec])
    xvec_labs = np.array([['Voltage', 'V'], ['UDVS Step', '']])
    data_labs = h5_data_labels
    data_units = list()
    for label in data_labs:
        if label == 'Amplitude':
            data_units.append('V')
        elif label == 'Frequency':
            data_units.append('Hz')
        elif label == 'Quality Factor':
            data_units.append('')
        elif label == 'Phase':
            data_units.append('rad')


            #     xaxis = {'XData':data_xvec,
            #              'XData Name':'Voltage',
            #              'XData Unit':'V',
            #              'XStep':pt_xvec,
            #              'XStep Name':'UDVS Step',
            #              'XStep Unit':''}

            #     yaxis = dict([(data_labs[i],data_units[i]) for i in xrange(4)])
            #     elements = {'Field':data_parts,
            #                 'Parameters':yaxis,
            #                 'Cycles':num_cycles,
            #                 'Steps':num_steps,
            #                 'X positions':num_rows,
            #                 'Y positions':num_cols}
            #
            #     data_pack = {'Guess':data_guess,
            #                  'Fit':data_results,
            #                  'Raw':data_main,
            #                  'Data Elements':elements,
            #                  'X-axis':xaxis,
            #                  'Y-axis':yaxis,
            #                  'Num Cycles':num_cycles,
            #                  'Groups':data_parts,
            #                  'Dims':ndims,
            #                  'Bins':h5_bins.value}

    return data_guess, data_results, xvec, xvec_labs, data_parts, ndims, data_main, np.unique(h5_bins.value)


def readACData(h5_group):
    """
    Reads the data for AC modulation experiments

    Inputs:
        h5_group -- hdf5 group holding the SHO_Fit Data for the chosen
                    Measurement group

    Outputs:
        data_guess -- the transformed data to be plotted
        xvec -- numpy array containing the possible plot data of the slice viewer
        xvec_labs -- numpy array of labels and units for the xvec array
    """
    h5_chan = h5_group['Channel_000']
    h5_main = h5_chan['Raw_Data']
    h5_specv = h5_chan['Spectroscopic_Values']
    h5_bins = h5_chan['Bin_Frequencies']

    h5_shogroup = h5_chan['Raw_Data-SHO_Fit_000']
    h5_guess = h5_shogroup['Guess']

    h5_data_labels = h5_specv.attrs['labels']
    h5_sho_specv = h5_shogroup['Spectroscopic_Values']
    h5_indices = h5_shogroup['Spectroscopic_Indices']
    h5_pos = h5_chan['Position_Indices']

    num_rows, num_cols = __getPos(h5_pos)
    num_pos = num_rows * num_cols

    ndims = len(np.where(np.array(h5_pos.shape) > 1)[0])

    num_cycles = h5_group.attrs['VS_number_of_cycles']
    num_steps = len(np.unique(h5_indices[:, 1]))
    num_bins = len(np.unique(h5_bins))

    data_xvec = h5_sho_specv[h5_sho_specv.attrs['AC_Amplitude']].flatten()
    direction = h5_specv[h5_specv.attrs['Direction']].flatten()
    direction = h5_sho_specv[h5_sho_specv.attrs['Direction']].flatten()

    '''
    Separate data into cycles
    '''
    data_guess = np.reshape(h5_guess, [num_pos, num_cycles, -1])
    data_guess = np.transpose(data_guess, [1, 2, 0])
    data_xvec = np.reshape(data_xvec, [num_cycles, -1])
    data_main = np.reshape(h5_main, [num_pos, num_cycles, -1])
    data_main = np.transpose(data_main, [1, 2, 0])

    pt_xvec = np.tile(np.arange(data_xvec.shape[1]), [num_cycles, 1])

    '''
    Seperate out forward and reverse values
    '''
    for_dir = np.where(direction == 1.0)[0]
    rev_dir = np.where(direction == -1.0)[0]

    data_guess = np.array([data_guess[:, for_dir, :], data_guess[:, rev_dir, :]])
    data_guess = np.reshape(data_guess, (2, num_cycles, -1, num_rows, num_cols))

    data_main = np.reshape(data_main, (num_cycles, -1, num_bins, num_rows, num_cols))
    data_main = np.array([data_main[:, for_dir, :, :, :], data_main[:, rev_dir, :, :, :]])

    data_xvec = np.array([data_xvec[:, for_dir], data_xvec[:, rev_dir]])
    pt_xvec = np.array([pt_xvec[:, for_dir], pt_xvec[:, rev_dir]])
    data_parts = ['Forward', 'Reverse']

    #     '''
    #     Get the mean and standard deviation of each variable
    #     Then use these to set upper and lower bounds
    #     '''
    #     for var in xrange(data_guess.shape[0]):
    #         for cycle in xrange(data_guess.shape[1]):
    #             mean_data, std_data = getGoodLims(data_guess[var,cycle,:,:])
    #             max_data = mean_data+3*std_data
    #             min_data = mean_data-3*std_data
    #             np.clip(data_guess[var,cycle,:,:],min_data,max_data,data_guess[var,cycle,:,:])

    xvec = np.array([data_xvec, pt_xvec])
    xvec_labs = np.array([['AC Current', 'A'], ['UDVS Step', '']])

    '''
    Repeat steps for reading guess on the results if present
    '''

    try:
        h5_results = h5_shogroup['Fit']
        data_results = np.reshape(h5_results, [num_pos, num_cycles, -1])
        data_results = np.transpose(data_results, [1, 2, 0])

        data_results = np.array([data_results[:, for_dir, :], data_results[:, rev_dir, :]])
        data_results = np.reshape(data_results, (2, num_cycles, -1, num_rows, num_cols))

    except KeyError:
        data_results = None
    except:
        raise

    data_labs = h5_data_labels
    data_units = list()
    for label in data_labs:
        if label == 'Amplitude':
            data_units.append('V')
        elif label == 'Frequency':
            data_units.append('Hz')
        elif label == 'Quality Factor':
            data_units.append('')
        elif label == 'Phase':
            data_units.append('rad')


            #     xaxis = {'XData':data_xvec,
            #              'XData Name':'Voltage',
            #              'XData Unit':'V',
            #              'XStep':pt_xvec,
            #              'XStep Name':'UDVS Step',
            #              'XStep Unit':''}

            #     yaxis = dict([(data_labs[i],data_units[i]) for i in xrange(4)])
            #     elements = {'Field':data_parts,
            #                 'Parameters':yaxis,
            #                 'Cycles':num_cycles,
            #                 'Steps':num_steps,
            #                 'X positions':num_rows,
            #                 'Y positions':num_cols}
            #
            #     data_pack = {'Guess':data_guess,
            #                  'Fit':data_results,
            #                  'Raw':data_main,
            #                  'Data Elements':elements,
            #                  'X-axis':xaxis,
            #                  'Y-axis':yaxis,
            #                  'Num Cycles':num_cycles,
            #                  'Groups':data_parts,
            #                  'Dims':ndims,
            #                  'Bins':h5_bins.value}

    return data_guess, data_results, xvec, xvec_labs, data_parts, ndims, data_main, np.unique(h5_bins.value)


def getSpectralData(point, data_mat):
    """
    This function accepts a tuple (x,y) and extracts the spectra
    from matrix (data) at that point, and returns an Nx2 vector
    
    Inputs:
        point -- (x,y) position to be retrieved from data_mat
        data_mat -- matrix to retrieve data from
        
    Outputs:
        
    """
    spec_ydata = data_mat[:, int(np.round(point[0])), int(np.round(point[1]))]
    return spec_ydata.ravel()


def __getPos(h5_pos):
    """
    Return the number of rows and columns in the dataset
    """
    num_rows = len(np.unique(h5_pos[:, 0]))
    try:
        num_cols = len(np.unique(h5_pos[:, 1]))
    except ValueError:
        num_cols = 1

    return num_rows, num_cols


def __findDataset(h5_file, ds_name):
    """
    Uses visit() to find all datasets with the desired name
    """
    print('Finding all instances of', ds_name)
    ds = []

    def __findName(name, obj):
        if name.split('/')[-1] == ds_name and isinstance(obj, h5py.Dataset):
            ds.append([name, obj])
        return

    h5_file.visititems(__findName)

    return ds
