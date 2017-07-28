
"""
============
Load Dataset
============

Load a dataset from the hdf5 file.  For this example, we will be loading the Raw_Data dataset.

"""

# Code source: pycroscopy
# Liscense: MIT

import h5py
import pycroscopy as px

h5_path = px.uiGetFile(caption='Select .h5 file', filter='HDF5 file (*.h5)')

# Load the dataset with h5py
h5_file1 = h5py.File(h5_path, 'r')

h5_raw1 = h5_file1['Measurement_000\Channel_000\Raw_Data']

# Load the dataset with pycroscopy
h5_file2 = px.ioHDF5(h5_path)

h5_raw2 = h5_file2['Measurement_000\Channel_000\Raw_Data']
