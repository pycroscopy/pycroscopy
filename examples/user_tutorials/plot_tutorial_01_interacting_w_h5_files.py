"""
================================================
Tutorial 1: Interacting with Pycroscopy H5 files
================================================
Suhas Somnath

11/11/2017

This set of tutorials will serve as examples for using and developing end-to-end workflows for pycroscopy.

In this example, we will learn how to interact with pycroscopy formatted h5 files.

Introduction
============
We highly recommend reading about the pycroscopy data format - available in the docs.

Pycroscopy uses a data-centric approach to data analysis and processing meaning that results from all data analysis and
processing are written to the same h5 file that contains the recorded measurements. The Hierarchical Data Format (HDF5)
allows data to be stored in multiple datasets in a tree-like manner. However, certain rules and considerations have
been made in pycroscopy to ensure consistent and easy access to any data. pycroscopy.hdf_utils contains a lot of
utility functions that simplify access to data and this tutorial provides an overview of many of the these functions
"""

import os
# Warning package in case something goes wrong
from warnings import warn
# Package for downloading online files:
try:
    # This package is not part of anaconda and may need to be installed.
    import wget
except ImportError:
    warn('wget not found.  Will install with pip.')
    import pip
    pip.main(['install', 'wget'])
    import wget
import h5py
import numpy as np
import matplotlib.pyplot as plt
try:
    import pycroscopy as px
except ImportError:
    warn('pycroscopy not found.  Will install with pip.')
    import pip
    pip.main(['install', 'pycroscopy'])
    import pycroscopy as px

################################################################################################

# Downloading the example file from the pycroscopy Github project
url = 'https://raw.githubusercontent.com/pycroscopy/pycroscopy/master/data/BEPS_small.h5'
h5_path = 'temp.h5'
_ = wget.download(url, h5_path)

print('Working on:\n' + h5_path)

################################################################################################
# Pycroscopy uses the h5py python package to access the HDF5 files and its contents.
# Conventionally, the h5py package is used to create, read, write, and modify h5 files.

# Open the file in read-only mode
h5_f = h5py.File(h5_path, mode='r')

# We can also use the ioHDF5 class from Pycroscopy to open the file.  Note that you do need to close the
# file in h5py before opening it again.
h5_f.close()
hdf = px.ioHDF5(h5_path)
h5_f = hdf.file

# Here, h5_f is an active handle to the open file

################################################################################################
# Inspect the contents of this h5 data file
# =========================================
#
# The file contents are stored in a tree structure, just like files on a contemporary computer. The file contains
# datagroups (similar to file folders) and datasets (similar to spreadsheets).
# There are several datasets in the file and these store:
#
# * The actual measurement collected from the experiment
# * Spatial location on the sample where each measurement was collected
# * Information to support and explain the spectral data collected at each location
# * Since pycroscopy stores results from processing and analyses performed on the data in the same file, these
#   datasets and datagroups are present as well
# * Any other relevant ancillary information
#
# Soon after opening any file, it is often of interest to list the contents of the file. While one can use the open
# source software HDFViewer developed by the HDF organization, pycroscopy.hdf_utils also has a simply utility to
# quickly visualize all the datasets and datagroups within the file within python.

print('Contents of the H5 file:')
px.hdf_utils.print_tree(h5_f)

################################################################################################
# Accessing datasets and datagroups
# ==================================
#
# There are numerous ways to access datasets and datagroups in H5 files. First we show the methods using native h5py
# functionality.
#
# Datasets and datagroups can be accessed by specifying the path, just like a web page or a file in a directory

# Selecting a datagroup by specifying the absolute path:
h5_meas_group = h5_f['Measurement_000']
print('h5_meas_group:', h5_meas_group)
print('h5_chan_group:', h5_f['Measurement_000/Channel_000'])

# Selecting a dataset by specifying the absolute path:
h5_fft = h5_f['Measurement_000/Channel_000/Bin_FFT']
print('h5_fft:', h5_fft)

# Selecting the same dataset using the relative path.
# First we get "Channel_000" from h5_meas_group:
h5_group = h5_meas_group['Channel_000']

# Now we access Bin_FFT from within h5_group:
h5_fft = h5_group['Bin_FFT']
print('h5_fft:', h5_fft)

################################################################################################
# The datagroup "Channel_000" contains several "members", where these members could be datasets like "Bin_FFT" or
# datagroups like "Channel_000"
#
# The output above shows that the "Bin_FFT" dataset is a one dimensional dataset, and has complex value (a +bi)
# entries at each element in the 1D array.
# This dataset is contained in a datagroup called "Channel_000" which itself is contained in a datagroup called
# "Measurement_000"
#
# And here's two methods using pycroscopy.hdf_utils

# Specific match of dataset name:
udvs_dsets_1 = px.hdf_utils.getDataSet(h5_f, 'UDVS')
for item in udvs_dsets_1:
    print(item)

# This function returns all datasets that match even a portion of the name
udvs_dsets_2 = px.hdf_utils.findDataset(h5_f, 'UDVS')
for item in udvs_dsets_2:
    print(item)

################################################################################################
# Pycroscopy hdf5 files contain three kinds of datasets:
#
# * Main datasets that contain data recorded / computed at multiple spatial locations.
# * Ancillary datasets that support a main dataset
# * Other datasets
#
# For more information, please refer to the documentation on the pycroscopy data format.
#
# We can check which datasets within h5_group are Main datasets using a handy hdf_utils function:

for dset_name in h5_group:
    print(px.hdf_utils.checkIfMain(h5_group[dset_name]), ':\t', dset_name)

################################################################################################
# The data of interest is almost always contained within Main Datasets. Thus, while all three kinds of datasets can
# be accessed using the methods shown above, we have a function in hdf_utils that allows us to only list the main
# datasets within the file / group:

main_dsets = px.hdf_utils.get_all_main(h5_f)
for dset in main_dsets:
    print(dset.name, dset.shape)

################################################################################################
# The datasets above show that the file contains three main datasets. Two of these datasets are contained in a folder
# called Raw_Data-SHO_Fit_000 meaning that they are results of an operation called SHO_Fit performed on the main
# dataset Raw_Data. The first of the three main datasets is indeed the Raw_Data dataset from which the latter
# two datasets (Fit and Guess) were derived.
#
# Pycroscopy allows the same operation, such as 'SHO_Fit', to be performed on the same dataset (Raw_Data), multiple
# times. Each time the operation is performed, a new datagroup is created to hold the new results. Often, we may
# want to perform a few operations such as:
#
# * Find the (source / main) dataset from which certain results were derived
# * Check if a particular operation was performed on a main dataset
# * Find all datagroups corresponding to a particular operation (e.g. - SHO_Fit) being applied to a main dataset
#
# hdf_utils has a few handy functions that simply many of these use cases:

# First get the dataset corresponding to Raw_Data
h5_raw = h5_f['/Measurement_000/Channel_000/Raw_Data']

print('Instances of operation "{}" applied to dataset named "{}":'.format('SHO_Fit', h5_raw.name))
h5_sho_group_list = px.hdf_utils.findH5group(h5_raw, 'SHO_Fit')
print(h5_sho_group_list)

################################################################################################
# As expected, the SHO_Fit operation was performed on Raw_Data only once, which is why findH5group returned only one
# datagroup - SHO_Fit_000.
#
# Often one may want to check if a certain operation was performed on a dataset with the very same parameters to
# avoid recomputing the results. hdf_utils has a function for this too:

print('Parameters already used for computing SHO_Fit on Raw_Data in the file:')
print(px.hdf_utils.get_attributes(h5_f['/Measurement_000/Channel_000/Raw_Data-SHO_Fit_000']))
print('\nChecking to see if SHO Fits have been computed on the raw dataset:')
print('Using pycroscopy')
print(px.hdf_utils.check_for_old(h5_raw, 'SHO_Fit',
                                 new_parms={'SHO_fit_method': 'pycroscopy BESHO'}))
print('Using BEAM')
print(px.hdf_utils.check_for_old(h5_raw, 'SHO_Fit',
                                 new_parms={'SHO_fit_method': 'BEAM BESHO'}))

################################################################################################
# Clearly, while findH5group returned any and all groups corresponding to SHO_Fit being applied to Raw_Data,
# check_for_old only returned the group(s) where the operation was performed using the same parameters.
#
# Let's consider the inverse scenario where we are interested in finding the source dataset from which the known
# result was derived:

h5_sho_group = h5_sho_group_list[0]
print('Datagroup containing the SHO fits:')
print(h5_sho_group)
print('\nDataset on which the SHO Fit was computed:')
h5_source_dset = px.hdf_utils.get_source_dataset(h5_sho_group)
print(h5_source_dset)

################################################################################################
# Accessing Attributes:
# =====================
#
# HDF5 datasets and datagroups can also store metadata such as experimental parameters. These metadata can be text,
# numbers, small lists of numbers or text etc. These metadata can be very important for understanding the datasets
# and guide the analysis routines.
#
# h5py offers a basic method for accessing attributes attached to datasets and datagroups. However, more complicated
# operations such as accessing multiple attributes or accessing the original string value of string attributes can
# be problematic in python 3. pycroscopy.hdf_utils has a few functions that simplifies the process of accessing
# attributes

# Listing all attributes using get_attributes:
attr_dict = px.hdf_utils.get_attributes(h5_meas_group, attr_names=None)
for att_name in attr_dict:
    print('{} : {}'.format(att_name, attr_dict[att_name]))

################################################################################################

# accessing specific attributes only:
print(px.hdf_utils.get_attributes(h5_meas_group, attr_names=['VS_mode', 'BE_phase_content']))

################################################################################################
# Comparing the number value of attributes is not a problem using h5py:

# via the standard h5py library:
print(h5_meas_group.attrs['VS_amplitude_[V]'])
print(h5_meas_group.attrs['VS_amplitude_[V]'] == 8)

################################################################################################
# However, accessing string valued attributes and using them for comparison is a problem using the standard h5py
# library

print(h5_meas_group.attrs['VS_measure_in_field_loops'])

# comparing the (byte)string value of attributes is a problem with python 3:
h5_meas_group.attrs['VS_measure_in_field_loops'] == 'in and out-of-field'

################################################################################################
# the get_attr function in hdf_utils handles such string complications by itself:

str_val = px.hdf_utils.get_attr(h5_meas_group, 'VS_measure_in_field_loops')
print(str_val == 'in and out-of-field')

################################################################################################
# Main Datasets via PycroDataset
# ==============================
#
# For this example, we will be working with a Band Excitation Polarization Switching (BEPS) dataset acquired from
# advanced atomic force microscopes. In the much simpler Band Excitation (BE) imaging datasets, a single spectra is
# acquired at each location in a two dimensional grid of spatial locations. Thus, BE imaging datasets have two
# position dimensions (X, Y) and one spectroscopic dimension (frequency - against which the spectra is recorded).
# The BEPS dataset used in this example has a spectra for each combination of three other parameters (DC offset,
# Field, and Cycle). Thus, this dataset has three new spectral dimensions in addition to the spectra itself. Hence,
# this dataset becomes a 2+4 = 6 dimensional dataset
#
# In pycroscopy, all spatial dimensions are collapsed to a single dimension and similarly, all spectroscopic
# dimensions are also collapsed to a single dimension. Thus, the data is stored as a two-dimensional (N x P)
# matrix with N spatial locations each with P spectroscopic datapoints.
#
# This general and intuitive format allows imaging data from any instrument, measurement scheme, size, or
# dimensionality to be represented in the same way. Such an instrument independent data format enables a single
# set of analysis and processing functions to be reused for multiple image formats or modalities.
#
# Main datasets can be thought of as substantially more capable and information-packed than standard datasets
# since they have (or are linked to) all the necessary information to describe a measured dataset. The additional
# information contained / linked by Main datasets includes:
#
# * the recorded physical quantity
# * units of the data
# * names of the position and spectroscopic dimensions
# * dimensionality of the data in its original N dimensional form etc.
#
# While it is most certainly possible to access this information via the native h5py functionality, it can become
# tedious very quickly.  Pycroscopy's PycroDataset class makes such necessary information and any necessary
# functionality easily accessible.
#
# PycroDataset objects are still h5py.Dataset objects underneath, like all datasets accessed above, but add an
# additional layer of functionality to simplify data operations. Let's compare the information we can get via the
# standard h5py library with that from PycroDataset to see the additional layer of functionality. The PycroDataset
# makes the spectral and positional dimensions, sizes immediately apparent among other things.

# Accessing the raw data
pycro_main = main_dsets[0]
print('Dataset as observed via h5py:')
print()
print('\nDataset as seen via a PycroDataset object:')
print(pycro_main)
# Showing that the PycroDataset is still just a h5py.Dataset object underneath:
print()
print(isinstance(pycro_main, h5py.Dataset))
print(pycro_main == h5_raw)

################################################################################################
# Main datasets are often linked to supporting datasets in addition to the mandatory ancillary datasets.  The main
# dataset contains attributes which are references to these datasets

for att_name in pycro_main.attrs:
    print(att_name, pycro_main.attrs[att_name])

################################################################################################
# These datasets can be accessed easily via a handy hdf_utils function:

print(px.hdf_utils.getAuxData(pycro_main, auxDataName='Bin_FFT'))

################################################################################################
# The additional functionality of PycroDataset is enabled through several functions in hdf_utils. Below, we provide
# several such examples along with comparisons with performing the same operations in a simpler manner using
# the PycroDataset object:

# A function to describe the nature of the contents within a dataset
print(px.hdf_utils.get_data_descriptor(h5_raw))

# this functionality can be accessed in PycroDatasets via:
print(pycro_main.data_descriptor)

################################################################################################
# Using Ancillary Datasets
# ========================
#
# As mentioned earlier, the ancillary datasets contain information about the dimensionality of the original
# N-dimensional dataset.  Here we see how we can extract the size and corresponding names of each of the spectral
# and position dimensions.

# We can use the getAuxData function again to get the ancillary datasets linked with the main dataset:
# The [0] slicing is to take the one and only position indices and spectroscopic indices linked with the dataset
h5_pos_inds = px.hdf_utils.getAuxData(h5_raw, auxDataName='Position_Indices')[0]
h5_spec_inds = px.hdf_utils.getAuxData(h5_raw, auxDataName='Spectroscopic_Indices')[0]

# Need to state that the array needs to be of the spectral shape.
print('Spectroscopic dimensions:')
print(px.hdf_utils.get_formatted_labels(h5_spec_inds))
print('Size of each dimension:')
print(px.hdf_utils.get_dimensionality(h5_spec_inds))
print('Position dimensions:')
print(px.hdf_utils.get_formatted_labels(h5_pos_inds))
print('Size of each dimension:')
print(px.hdf_utils.get_dimensionality(h5_pos_inds[()].T))

################################################################################################
# The same tasks can very easily be accomplished via the PycroDataset object

# an alternate way to get the spectroscopic indices is simply via:
print(pycro_main.h5_spec_inds)

# We can get the spectral / position labels and dimensions easily via:
print('Spectroscopic dimensions:')
print(pycro_main.spec_dim_descriptors)
print('Size of each dimension:')
print(pycro_main.spec_dim_sizes)
print('Position dimensions:')
print(pycro_main.pos_dim_descriptors)
print('Size of each dimension:')
print(pycro_main.pos_dim_sizes)

################################################################################################
# In a few cases, the spectroscopic / position dimensions are not arranged in descending order of rate of change.
# In other words, the dimensions in these ancillary matrices are not arranged from fastest-varying to slowest.
# To account for such discrepancies, hdf_utils has a very handy function that goes through each of the columns or
# rows in the ancillary indices matrices and finds the order in which these dimensions vary.
#
# Below we illustrate an example of sorting the names of the spectroscopic dimensions from fastest to slowest in
# a BEPS data file:

spec_sort_order = px.hdf_utils.get_sort_order(h5_spec_inds)
print('Spectroscopic dimensions arranged as is:')
unsorted_spec_labels = px.hdf_utils.get_formatted_labels(h5_spec_inds)
print(unsorted_spec_labels)
sorted_spec_labels = np.array(unsorted_spec_labels)[np.array(spec_sort_order)]
print('Spectroscopic dimensions arranged from fastest to slowest')
print(sorted_spec_labels)

################################################################################################
# When visualizing the data it is essential to plot the data against appropriate values on the X, Y, Z axes.
# Extracting a simple list or array of values to plot against may be challenging especially for multidimensional
# dataset such as the one under consideration. Fortunately, hdf_utils has a very handy function for this as well:

h5_spec_inds = px.hdf_utils.getAuxData(pycro_main, auxDataName='Spectroscopic_Indices')[0]
h5_spec_vals = px.hdf_utils.getAuxData(pycro_main, auxDataName='Spectroscopic_Values')[0]
dimension_name = 'DC_Offset'
dc_dict = px.hdf_utils.get_unit_values(h5_spec_inds, h5_spec_vals, dim_names=dimension_name)
print(dc_dict)
dc_val = dc_dict[dimension_name]

fig, axis = plt.subplots()
axis.plot(dc_val)
axis.set_title(dimension_name)
axis.set_xlabel('Points in dimension')

################################################################################################
# Yet again, this process is simpler when using the PycroDataset object:

dv_val = pycro_main.get_spec_values(dim_name=dimension_name)

fig, axis = plt.subplots()
axis.plot(dc_val)
axis.set_title(dimension_name)
axis.set_xlabel('Points in dimension')

################################################################################################
# Reshaping Data
# ==============
#
# Pycroscopy stores N dimensional datasets in a flattened 2D form of position x spectral values. It can become
# challenging to retrieve the data in its original N-dimensional form, especially for multidimensional datasets
# such as the one we are working on. Fortunately, all the information regarding the dimensionality of the dataset
# are contained in the spectral and position ancillary datasets. hdf_utils has a very useful function that can
# help retrieve the N-dimensional form of the data using a simple function call:

ndim_form, success, labels = px.hdf_utils.reshape_to_Ndims(h5_raw, get_labels=True)
if success:
    print('Succeeded in reshaping flattened 2D dataset to N dimensions')
    print('Shape of the data in its original 2D form')
    print(h5_raw.shape)
    print('Shape of the N dimensional form of the dataset:')
    print(ndim_form.shape)
    print('And these are the dimensions')
    print(labels)
else:
    print('Failed in reshaping the dataset')

################################################################################################
# The whole process is simplified further when using the PycroDataset object:

ndim_form = pycro_main.get_n_dim_form()
print('Shape of the N dimensional form of the dataset:')
print(ndim_form.shape)
print('And these are the dimensions')
print(pycro_main.n_dim_labels)

################################################################################################
two_dim_form, success = px.hdf_utils.reshape_from_Ndims(ndim_form,
                                                        h5_pos=h5_pos_inds,
                                                        h5_spec=h5_spec_inds)
if success:
    print('Shape of flattened two dimensional form')
    print(two_dim_form.shape)
else:
    print('Failed in flattening the N dimensional dataset')

################################################################################################

# Close and delete the h5_file
h5_f.close()
os.remove(h5_path)
