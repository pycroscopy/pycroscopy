"""
========================================================================================================
Tutorial 3: Handling Multidimensional datasets
========================================================================================================

**Suhas Somnath**
8/8/2017

This set of tutorials will serve as examples for developing end-to-end workflows for and using pycroscopy.

**In this example, we will learn how to slice multidimensional datasets.**

Introduction
============

In pycroscopy, all position dimensions of a dataset are collapsed into the first dimension and all other
(spectroscopic) dimensions are collapsed to the second dimension to form a two dimensional matrix. The ancillary
matricies, namely the spectroscopic indices and values matrix as well as the position indicies and values matrices
will be essential for reshaping the data back to its original N dimensional form and for slicing multidimensional
datasets

We highly recommend reading

"""

# Ensure python 3 compatibility:
from __future__ import division, print_function, absolute_import, unicode_literals

# The package for accessing files in directories, etc.:
import os
import wget

# The mathematical computation package:
import numpy as np

# The package used for creating and manipulating HDF5 files:
import h5py

# Packages for plotting:
import matplotlib.pyplot as plt

# basic interactive widgets:
from ipywidgets import interact

# Finally import pycroscopy for certain scientific analysis:
import pycroscopy as px

#########################################################################
# Load the dataset
# ================
#
# For this example, we will be working with a Band Excitation Polarization Switching (BEPS) First Order Reversal
# Curve (FORC) dataset acquired from advanced atomic force microscopes. In the much simpler Band Excitation (BE)
# imaging datasets, a single spectra is acquired at each location in a two dimensional grid of spatial locations.
# Thus, BE imaging datasets have two position dimensions (X, Y) and one spectroscopic dimension (frequency - against
# which the spectra is recorded). The BEPS-FORC dataset used in this example has a spectra for each combination of
# three other paramaters (DC offset, Field, bias waveform type {FORC}). Thus, this dataset has three new spectral
# dimensions in addition to the spectra itself. Hence, this dataet becomes a 2+4 = 6 dimensional dataset

# download the raw data file from Github:
h5_path = 'temp_3.h5'
url = 'https://raw.githubusercontent.com/pycroscopy/pycroscopy/cades_dev/data/BEPS_small.h5'
if os.path.exists(h5_path):
    os.remove(h5_path)
_ = wget.download(url, h5_path, bar=None)

# Open the file in read-only mode
h5_file = h5py.File(h5_path, mode='r')

print('Datasets and datagroups within the file:\n------------------------------------')
px.hdf_utils.print_tree(h5_file)

#########################################################################

h5_meas_grp = h5_file['Measurement_000']
h5_main = h5_meas_grp['Channel_000/Raw_Data']
print('\nThe main dataset:\n------------------------------------')
print(h5_main)

#########################################################################
# The main dataset clearly does not provide the multidimensional information about the data that will be necessary to
# slice the data. For that we need the ancillary datasets that support this main dataset

# pycroscopy has a convenient function to access datasets linked to a given dataset:
h5_spec_ind = px.hdf_utils.getAuxData(h5_main, 'Spectroscopic_Indices')[0]
h5_spec_val = px.hdf_utils.getAuxData(h5_main, 'Spectroscopic_Values')[0]
h5_pos_ind = px.hdf_utils.getAuxData(h5_main, 'Position_Indices')[0]
h5_pos_val = px.hdf_utils.getAuxData(h5_main, 'Position_Values')[0]

#########################################################################
# Understanding the ancillary datasets:
# =====================================
#
# The position datasets are shaped as [spatial points, dimension] while the spectroscopic datasets are shaped as
# [dimension, spectral points]. Clearly the first axis of the position dataset and the second axis of the spectroscopic
# datasets match the correponding sizes of the main dataset.
#
# Again, the sum of the position and spectroscopic dimensions results in the 6 dimensions originally described above.
#
# Essentially, there is a unique combination of position and spectroscopic parameters for each cell in the two
# dimensionam main dataset. The interactive widgets below illustrate this point. The first slider represents the
# position dimension while the second represents the spectroscopic dimension. Each position index can be decoded
# to a set of X and Y indices and values while each spectroscopic index can be decoded into a set of frequency,
# dc offset, field, and forc parameters

print('Main Datasets of shape:', h5_main.shape)
print('Position Datasets of shape:', h5_pos_ind.shape)
print('Spectroscopic Datasets of shape:', h5_spec_ind.shape)

spec_labels = px.hdf_utils.get_formatted_labels(h5_spec_ind)
pos_labels = px.hdf_utils.get_formatted_labels(h5_pos_ind)

def myfun(pos_index, spec_index):
    for dim_ind, dim_name in enumerate(pos_labels):
        print(dim_name,':',h5_pos_ind[pos_index, dim_ind])
    for dim_ind, dim_name in enumerate(spec_labels):
        print(dim_name,':',h5_spec_ind[dim_ind, spec_index])
interact(myfun, pos_index=(0,h5_main.shape[0]-1, 1), spec_index=(0,h5_main.shape[1]-1, 1));

#########################################################################
# Visualizing the ancillary datasets
# ==================================
#
# The plots below show how the position and spectrocopic dimensions vary. Due to the high dimensionality of the
# spectroscopic dimensions, the variation of each dimension has been plotted separately.
#
# How we interpret these plots:
# =============================
#
# **Positions**: For each Y index, the X index ramps up from 0 to 4 and repeats. Essentially, this means that for
# a given Y index, there were multiple measurments (different values of X)
#
# **Spectroscopic**: The plot for `FORC` shows that the next fastest dimension - `DC offset` was varied 6 times.
# Correspondingly, the plot for `DC offset` plot shows that this dimension ramps up from 0 to a little less than
# 40 for each `FORC` index. This trend is the same for the faster varying dimensions - `Frequency` and `Field`.

fig_1, axes = plt.subplots(ncols=2, figsize=(10, 5))
px.plot_utils.plot_line_family(axes[0], np.arange(h5_pos_ind.shape[0]), h5_pos_ind[()].T,
                               line_names=pos_labels)
axes[0].set_xlabel('Position points')
axes[0].set_ylabel('Index')
axes[0].set_title('Position Indices')
axes[0].legend()
px.plot_utils.plot_line_family(axes[1], np.arange(h5_spec_ind.shape[1]), h5_spec_ind,
                               line_names=spec_labels)
axes[1].set_xlabel('Spectroscopic points')
axes[1].set_title('Spectroscopic Indices')
axes[1].legend()

fig_2, axes = plt.subplots(ncols=3, nrows=2, figsize=(15, 10))
for dim_ind, axis, dim_label, dim_array in zip([0, 2], [axes.flat[0], axes.flat[3]],
                                               [spec_labels[0], spec_labels[2]],
                                               [h5_spec_ind[0, :500], h5_spec_ind[2, :500]]):
    axis.plot(np.arange(dim_array.size), dim_array)
    axis.set_xlabel('Spectroscopic points')
    axis.set_ylabel('Index')
    axis.set_title('Dim ' + str(dim_ind) + ' - ' + dim_label)

rhs_axes = [axes.flat[ind] for ind in [1, 2, 4, 5]]
for dim_ind, axis, dim_label, dim_array in zip(range(h5_spec_ind.shape[0]), rhs_axes, spec_labels, h5_spec_ind):
    axis.plot(np.arange(dim_array.size), dim_array)
    axis.set_xlabel('Spectroscopic points')
    axis.set_ylabel('Index')
    axis.set_title('Dim ' + str(dim_ind) + ' - ' + dim_label)

#########################################################################

# A similar version of this function is available in pycroscopy.io.hdf_utils.get_formatted_labels
def describe_dimensions(h5_aux):
    for name, unit in zip(px.hdf_utils.get_attr(h5_aux, 'labels'),
                            px.hdf_utils.get_attr(h5_aux, 'units')):
        print(name, '[', unit, ']')

print('Position dimension names and units:')
describe_dimensions(h5_pos_ind)

print('\nSpectrocopic dimension names and units:')
describe_dimensions(h5_spec_ind)

#########################################################################
# Now lets find the size in each dimension
# ========================================
# For now lets assume that data is sampled at each position and at each spectral step
# In other words lets assume that data was not sampled over a random subset of points within a grid of points

# The function below has been implemented as pycroscopy.io.hdf_utils.get_dimensionality
def get_dim_sizes(ind_dset, is_position=False):
    # ind_dset here is expected to be of the shape [dimension, points] like the spectroscopic indices
    if is_position:
        # Position dimensions will be turned from [points, dimension] to [dimension, points]
        ind_dset = np.transpose(ind_dset[()])

    dim_size = []
    for col in ind_dset:
        # For each dimension array, we will find the number of unique elements in it
        dim_size.append(len(np.unique(col)))
    return dim_size


pos_dim_sizes = get_dim_sizes(h5_pos_ind, is_position=True)
spec_dim_sizes = get_dim_sizes(h5_spec_ind)

print('Positions:', pos_dim_sizes, '\nSpectroscopic:', spec_dim_sizes)

#########################################################################
# Slicing the Main dataset
# ========================
#
# Let's assume that we are interested in visualizing the spectrograms at the first field of the second cycle at
# position - row:3 and column 2. There are two ways of accessing the data:
# 1. The easier method - reshape the data to N dimensions and slice the dataset
#     * This approach, while trivial, may not be suitable for large datasets which may or may not fit in memory
# 2. The harder method - find the spectroscopic and position indices of interest and slice the 2D dataset
#
# Approach 1 - N-dimensional form
# -------------------------------
# We will use convenient pycroscopy function that safely reshapes the data to its N dimensional form with a single
# line. Note that while this approach appears simple on the surface, there are a fair number of lines of code that
# make up this function.

ds_nd, success, labels = px.hdf_utils.reshape_to_Ndims(h5_main, get_labels=True)
print('Shape of the N-dimensional dataset:', ds_nd.shape)
print(labels)

#########################################################################
# Now that we have the data in its original N dimensional form, we can easily slice the dataset:

spectrogram = ds_nd[2,3, :, 1, :, 0]
# Now the spectrogram is of order (frequency x DC_Offset).
spectrogram = spectrogram.T
# Now the spectrogram is of order (DC_Offset x frequency)
fig, axis = plt. subplots()
axis.imshow(np.abs(spectrogram), origin='lower')
axis.set_xlabel('Frequency Index')
axis.set_ylabel('DC Offset Index')
axis.set_title('Spectrogram Amplitude');

#########################################################################
# Approach 2 - slicing the 2D matrix
# ----------------------------------
#
# This approach is more hands-on and requires that we be very careful with the indexing and slicing. Nonetheless,
# the process is actually fairly intuitive. We rely entirely upon the spectroscopic and position ancillary datasets
# to find the indices for slicing the dataset. Unlike the main dataset, the ancillary datasets are very small and
# can be stored easily in memory. Once the slicing indices are calculated, we __only read the desired portion of
# `main` data to memory__. Thus the amount of data loaded into memory is only the amount that we absolutely need.
# __This is the only approach that can be applied to slice very large datasets without ovwhelming memory overheads__.
# The comments for each line explain the entire process comprehensively

# Get only the spectroscopic dimension names:
spec_dim_names = px.hdf_utils.get_attr(h5_spec_ind, 'labels')

# Find the row in the spectroscopic indices that corresponds to the dimensions we want to slice:
cycle_row_ind = np.where(spec_dim_names == 'Cycle')[0][0]
# Find the row correspoding to field in the same way:
field_row_ind = np.where(spec_dim_names == 'Field')[0][0]

# Find all the spectral indices corresponding to the second cycle:
desired_cycle = h5_spec_ind[cycle_row_ind] == 1

# Do the same to find the spectral indicies for the first field:
desired_field = h5_spec_ind[field_row_ind] == 0

# Now find the indices where the cycle = 1 and the field = 0 using a logical AND statement:
spec_slice = np.logical_and(desired_cycle, desired_field)

# We will use the same approach to find the position indices
# corresponding to the row index of 3 and colum index of 2:
pos_dim_names = px.hdf_utils.get_attr(h5_pos_ind,'labels')

x_col_ind = np.where(pos_dim_names == 'X')[0][0]
y_col_ind = np.where(pos_dim_names == 'Y')[0][0]

desired_x = h5_pos_ind[:, x_col_ind] == 2
desired_y = h5_pos_ind[:, y_col_ind] == 3

pos_slice = np.logical_and(desired_x, desired_y)

# Now use the spectroscopic and position slice arrays to slice the 2D dataset:
data_vec = h5_main[pos_slice, :][:, spec_slice]
print('Sliced data is of shape:', data_vec.shape)

#########################################################################
# Note that the sliced data is effectively one dimensional since the spectroscopic dimensions were flattened to a
# single dimension.
#
# Now that we have the data we are interested in, all we need to do is reshape the vector to the expected 2D
# spectrogram shape. We still have to be careful about the order of the indices for reshaping the vector to the
# 2D matrix. Note that in python, we specify the slower axis before the faster axis in the reshape command.

# Reshape this dataset to the 2D spectrogram that we desire:

# For this we need to find the size of the data in the DC_offset and Frequency dimensions:
dc_dim_ind = np.where(spec_dim_names == 'DC_Offset')[0][0]
# Find the row correspoding to field in the same way:
freq_dim_ind = np.where(spec_dim_names == 'Frequency')[0][0]

dc_dim_size = spec_dim_sizes[dc_dim_ind]
freq_dim_size = spec_dim_sizes[freq_dim_ind]

# Since we know that the DC offset varies slower than the frequency, we reshape the
# the data vector by (dc_dim_size, freq_dim_size)
print('We need to reshape the vector by the tuple:', (dc_dim_size, freq_dim_size))

#########################################################################
# The dimensions in the ancillary datasets may or may not be arranged from fastest to slowest even though that is
# part of the requirements. We can still account for this. In the event that we don't know the order in which to
# reshape the data vector because we don't know which dimension varies faster than the other(s), we would need to
# sort the dimensions by how fast their indices change. Fortuantely, pycroscopy has a function called `px.hdf_utils.
# get_sort_order` that does just this. Knowing the sort order, we can easily reshape correctly in an automated manner.
# We will do this below

# Sort the spectroscopic dimensions by how fast their indices changes (fastest --> slowest)
spec_sort_order = px.hdf_utils.get_sort_order(h5_spec_ind)
print('Spectroscopic dimensions arranged as is:\n',
      spec_dim_names)
print('Dimension indices arranged from fastest to slowest:',
      spec_sort_order)
print('Dimension namess now arranged from fastest to slowest:\n',
      spec_dim_names[spec_sort_order])

if spec_sort_order[dc_dim_ind] > spec_sort_order[freq_dim_ind]:
    spectrogram_shape = (dc_dim_size, freq_dim_size)
else:
    spectrogram_shape = (freq_dim_size, dc_dim_size)

print('We need to reshape the vector by the tuple:', spectrogram_shape)

# Reshaping from 1D to 2D:
spectrogram2 = np.reshape(np.squeeze(data_vec), spectrogram_shape)

#########################################################################
# Now that the spectrogram is indeed two dimensional, we can visualize it

# Now the spectrogram is of order (DC_Offset x frequency)
fig, axis = plt. subplots()
axis.imshow(np.abs(spectrogram2), origin='lower')
axis.set_xlabel('Frequency Index')
axis.set_ylabel('DC Offset Index')
axis.set_title('Spectrogram Amplitude');

#########################################################################

# Close and delete the h5_file
h5_file.close()
os.remove(h5_path)
