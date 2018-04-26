########################################################################################################################
# Primer to HDF5 and h5py
# ========================
# Suhas Somnath
# 
# 4/18/2018
# 
# Introduction
# -------------
# We create and consume digital information stored in various file formats on a daily basis such as news presented in
# HTML files, scientific journal articles in PDF files, tabular data in XLSX spreadsheets and so on. Commercially
# available scientific instruments generate data in a variety of, typically proprietary, file formats. The proprietary
# nature of the data impedes scientific research of individual researchers and the collaboration within the scientific
# community at large. Hence, pycroscopy stores all relevant information including the measurement data, metadata etc.
# in the most popular file format for scientific data - Hierarchical Data Format (HDF5) files.
# 
# HDF5 is a remarkably straightforward file format to understand since it mimics the familiar folders and files paradigm
# exposed to users by all operating systems such as Windows, Mac OS, Linux, etc. HDF5 files can contain:
# * Datasets - similar to spreadsheets and text files with tabular data. 
# * Groups - similar to folders in a regular file system
# * Attributes - small metadata that provide additional information about the Group or Dataset they are attached to.
# * other advanced features such as hard links, soft links, object and region references, etc.
# 
# h5py is the official software package for reading and writing to HDF5 files in python. Consequently, Pycroscopy relies
# entirely on h5py for all file related operations. While there are several high-level functions that simplify the
# reading and writing of Pycroscopy stylized data, it is still crucial that the users of Pycroscopy understand the
# basics of HDF5 files and are familiar with the basic functions in h5py. There are several tutorials available
# elsewhere to explain h5py in great detail. This document serves as a quick primer to the basics of interacting with
# HDF5 files via h5py.
# 
# Import all necessary packages
# -----------------------------
# For this primer, we only need some very basic packages, all of which come with the standard Anaconda distribution:
# * os - to manipulate and remove files
# * numpy - for basic numerical work
# * h5py - the package that will be the focus of this primer

from __future__ import print_function, division, unicode_literals
import os
import numpy as np
import h5py

########################################################################################################################
# Creating a HDF5 files using h5py is similar to the process of creating a conventional text file using python. The File
# class of h5py requires the path for the desired file with a .h5, .hdf5, or similar extension.

h5_path = 'hdf5_primer.h5'
h5_file = h5py.File('hdf5_primer.h5')
print(h5_file)

########################################################################################################################
# At this point, a file in the path specified by h5_path has been created and is now open for modification. The returned
# value - h5_file is necessary to perform other operations on the file including creating groups and datasets.
# 
# Groups
# ======
# create_group()
# --------------
# We can use the create_group() function on an existing object such as the open file handle (h5_file) to create a group:

h5_group_1 = h5_file.create_group('Group_1')
print(h5_group_1)

########################################################################################################################
# The output of the above print statement reveals that a group named 'Group_1' was successfully created at location: '/'
# (which stands for the root of the file). Furthermore, this group contains 0 objects or members.
# .name
# -----
# One can find the full / absolute path where this object is located from its 'name' property:

print(h5_group_1.name)

########################################################################################################################
# Groups in Groups
# ----------------
# Much like folders in a computer, these groups can themselves contain more groups and datasets.
# 
# Let us create a few more groups the same way. Except, let us create these groups within the newly created. To do this,
# we would need to call the create_group() function on the h5_group_1 object and not the h5_file object. Doing the
# latter would result in groups created under the file at the same level as Group_1 instead of inside Group_1.

h5_group_1_1 = h5_group_1.create_group('Group_1_1')
h5_group_1_2 = h5_group_1.create_group('Group_1_2')

########################################################################################################################
# Now, when we print h5_group, it will reveal that we have two objects - the two groups we just created:

print(h5_group_1)

########################################################################################################################
# Lets see what a similar print of one of the newly created groups looks like:

print(h5_group_1_1)

########################################################################################################################
# The above print statement shows that this group named 'Group_1_1' exists at a path: "/Group_1/Group_1_1". In other
# words, this is similar to a folder contained inside another folder.
#
# .parent
# -------
# The heirarchical nature of HDF5 allows us to access datasets and datagroups using relationships or paths. For example,
# every HDF5 object has a parent. In the case of 'Group_1' - its parent is the root or h5_file itself. Similarly, the
# parent object of 'Group_1_1' is 'Group_1':

print('Parent of "Group_1" is {}'.format(h5_group_1.parent))
print('Parent of "Group_1_1" is {}'.format(h5_group_1_1.parent))

########################################################################################################################
# In fact the .parent of an object is an HDF5 object (either a HDF5 group or HDF5 File object). So we can check if the
# parent of the h5_group_1_1 variable is indeed the h5_group_1 variable:

print(h5_group_1_1.parent == h5_group_1)

########################################################################################################################
# Accessing H5 objects
# --------------------
# Imagine a file or a folder on a computer that is several folders deep from where one is (e.g. -
# /Users/Joe/Documents/Projects/2018/pycroscopy).One could either reach the desired file or folder by opening one folder
# after another or directly by using a long path string. If you were at root (/), you would need to paste the entire
# path (absolute path) of the desired file -  /Users/Joe/Documents/Projects/2018/pycroscopy. Alternatively, if you were
# in an intermediate directory (e.g. -  /Users/Joe/Documents/), you would need to paste what is called the relative path
# (in this case -  Projects/2018/pycroscopy) to get to the desired file.
# 
# In the same way, we can also access HDF5 objects either through relative paths, or absolute paths. Here are a few ways
# one could get to the group 'Group_1_2':

print(h5_file['/Group_1/Group_1_2'])
print(h5_group_1['Group_1_2'])
print(h5_group_1_1.parent['Group_1_2'])
print(h5_group_1_1.parent.parent['Group_1/Group_1_2'])

########################################################################################################################
# Now let us look at how one can iterate through the datasets and Groups present within a HDF5 group:

for item in h5_group_1:
    print(item)

########################################################################################################################
# .items()
# --------
# Essentially, h5py group objects contain a dictionary of key-value pairs where they key is the name of the object and
# the value is a reference to the object itself.
# 
# What the above for loop does is it iterates only over the keys in this dictionary which are all strings. In order to
# get the actual dataset object itself, we would need to use the aforementioned addressing techniques to get the actual
# Group objects.
# 
# Let us see how we would then try to find the object for the goup named 'Group_1_2':

# In[11]:

for key, value in h5_group_1.items():
    if key == 'Group_1_2':
        print('Found the desired object: {}'.format(value))

########################################################################################################################
# Datasets
# ========
# create_dataset()
# ----------------
# We can create a dataset within 'Group_1' using a function that is similar to create_group(), called create_dataset().
# Unlike create_group() which just takes the path of the desired group as an input, create_dataset() is highly
# customizable and flexible.
# 
# In our experience, there are three modes of creating datasets that are highly relevant for scientific applications:
# * dataset with data at time of creation - where the data is already available at the time of creating the dataset
# * empty dataset - when one knows the size of data but the entire data is not available
# * resizable dataset - when one does not even know how large the data can be. 
# 
# Creating Dataset with available data:
# -------------------------------------
# Let as assume we want to store a simple greyscale (floating point values) image with 256 x 256 pixels. We would create
# and store the data as shown below. As the size of the dataset becomes very large, the precision with which the data is
# stored can signfiicantly affect the size of hte dataset and the file. Therefore, we recommend purposefully specifying
# the dtype during creation.

h5_simple_dataset = h5_group_1.create_dataset('Simple_Dataset', data=np.random.rand(256, 256), dtype=np.float32)
print(h5_simple_dataset)

########################################################################################################################
# Accessing data
# ~~~~~~~~~~~~~~
# We can access data contained in the dataset just like accessing a numpy arary. For example, if we want the value at
# row 29 and column 167, we would read it as:

print(h5_simple_dataset[29, 167])

########################################################################################################################
# Again, just as before, we can address this dataset in many ways:

print(h5_group_1['Simple_Dataset'])
print(h5_file['/Group_1/Simple_Dataset'])

########################################################################################################################
# Creating (potentially large) empty datasets:
# --------------------------------------------
# In certain situations, we know how much space to allocate for the final dataset but we may not have all the data at
# once. Alternatively, the dataset is so large that we cannot fit the entire data in the computer memory before writing
# to the HDF5 file. Another possible circumstance is when we have to read N files, each containing a small portion of
# the data and then write the contents into each slot in the HDF5 dataset.
# 
# For example, assume that we have 128 files each having 1D spectra (amplitude + phase or complex value) of length 1024.
# Here is how one may create the HDF5 dataset to hold the data:

h5_empty_dataset = h5_group_1.create_dataset('Empty_Dataset', shape=(128, 1024), dtype=np.complex64)
print(h5_empty_dataset)

########################################################################################################################
# Note that unlike before, this particular dataset is empty since we only allocated space, so we would be reading zeros
# when attempting to access data:

print(h5_empty_dataset[5, 102])

########################################################################################################################
# populating with data
# ~~~~~~~~~~~~~~~~~~~~
# One could populate each chunk of the dataset just like filling in a numpy array:

h5_empty_dataset[0] = np.random.rand(1024) + 1j * np.random.rand(1024)

########################################################################################################################
# flush()
# ~~~~~~~
# It is a good idea to ensure that this data is indeed commited to the file using regular flush() operations. There are
# chances where the data is still in the memory / buffer and not yet in the file if one does not flush():

h5_file.flush()

########################################################################################################################
# Creating resizeable datasets:
# -----------------------------
# This solution is relevant to those situations where we only know how large each unit of data would be but we don't
# know the number of units. This is especially relevant when acquiring data from an instrument.
# 
# For example, if we were acquiring spectra of length 128 on a 1D grid of 256 locations, we may have created an empty 2D
# dataset of shape (265, 128) using the aforementioned function. The data was being collected ordinarily over the first
# 13 positions but a change in parameters resulted in spectra of length 175 instead. The data from the 14th positon
# cannot be stored in the empty array due to a size mismatch. Therefore, we would need to create another empty 256 x 175
# dataset to hold the data. If changes in parameters cause 157 changes in spectra length, that would result in the
# creation of 157 datasets each with a whole lot of wasted space since datasets cannot be shrunk easily.
# 
# In such cases, it is easier just to create datasets that can expand one pixel at a time. For this specific example,
# one may want to create a 2D dataset of shape (1, 128) that could grow up to a maxshape of (256, 128) as shown below:

h5_expandable_dset = h5_group_1.create_dataset('Expandable_Dataset', shape=(1, 128), maxshape=(256, 128),
                                               dtype=np.float32)
print(h5_expandable_dset)

########################################################################################################################
# Space has been allocated for the first pixel, so the data could be written in as:

h5_expandable_dset[0] = np.random.rand(128)

########################################################################################################################
# For the next pixel, we would need to expand the dataset before filling it in:

h5_expandable_dset.resize(h5_expandable_dset.shape[0] + 1, axis=0)
print(h5_expandable_dset)

########################################################################################################################
# Notice how the dataset has increased in size in the first dimension allowing the second pixel to be stored. The second
# pixel's data would be stored in the same way as in the first pixel and the cycle of expand and populate-with-data
# would continue.
# 
# It is very important to note that there is a non-trivial storage overhead associated with each resize operation. In
# other words, a file containing this resizeable dataset that has been resized 255 times will certainly be larger than
# a similar file where the dataset space was preallocated and never expanded. Therefore this mode of creating datasets
# should used sparingly.
# 
# Attributes
# ==========
# * are metadata that can convey information that cannot be efficently conveyed using Group or Dataset objects. 
# * are almost exactly like python dictionaries in that they have a key-value pairs. 
# * can be stored in either Group or Dataset objects. 
# * are not appropriate for storing large amounts of information. Consider datasets instead
# * are best suited for things like experimental parameter such as beam intensity, scan rate, scan width, etc.
# Writing
# -------
# Storing attributes in objects is identical to appending to python dictionaries. Lets store some simple attributes in
# the group named 'Group_1':

h5_simple_dataset.attrs['single_num'] = 36.23
h5_simple_dataset.attrs.update({'list_of_nums': [1, 6.534, -65],
                               'single_string': 'hello'})

########################################################################################################################
# Reading
# -------
# We would read the attributes just like we would treat a dictionary in python:

for key, val in h5_simple_dataset.attrs.items():
    print('{} : {}'.format(key, val))

########################################################################################################################
# Lets read the attributes one by one and verify that we read what we wrote:

print('single_num: {}'.format(h5_simple_dataset.attrs['single_num'] == 36.23))
print('list_of_nums: {}'.format(np.all(h5_simple_dataset.attrs['list_of_nums'] == [1, 6.534, -65])))
print('single_string: {}'.format(h5_simple_dataset.attrs['single_string'] == 'hello'))

########################################################################################################################
# Caveat
# ------
# While the low-level attribute writing and reading does appear to work and is simple, it does not work for a list of
# strings in python 3. Hence the following line will not work and will cause problems.
# 
# h5_simple_dataset.attrs['list_of_strings'] = ['a', 'bc', 'def']
# 
# Instead, we recommend writing lists of strings by casting them as numpy arrays:

h5_simple_dataset.attrs['list_of_strings'] = np.array(['a', 'bc', 'def'], dtype='S')

########################################################################################################################
# In the same way, reading attributes that are lists of strings is also not straightforward:

print('list_of_strings: {}'.format(h5_simple_dataset.attrs['list_of_strings'] == ['a', 'bc', 'def']))

########################################################################################################################
# A similar decoding step needs to be taken to extract the actual string values. 
# 
# To avoid manual encoding and decoding of attributes (different strategies for different versions of python), we
# recommend:
# * writing attributes using: pycroscopy.hdf_utils.write_simple_attrs()
# * reading attributes using: pycroscopy.hdf_utils.get_attr() or get_attributes()
# 
# Both these functions work reliably and consistently across all python versions and fix this problem in h5py. 
# 
# Besides strings and numbers, we tend to store references to datasets as attributes. Here is how one would link the
# empty dataset to the simple dataset:

h5_simple_dataset.attrs['Dataset_Reference'] = h5_empty_dataset.ref
print(h5_simple_dataset.attrs['Dataset_Reference'])

########################################################################################################################
# Here is how one would get a handle to the actual dataset from the reference:

# Read the attribute how you normally would
h5_ref = h5_simple_dataset.attrs['Dataset_Reference']
# Get the handle to the actual dataset:
h5_dset = h5_file[h5_ref]
# Check if this object is indeed the empty dataset:
print(h5_empty_dataset == h5_dset)

########################################################################################################################
# Once we are done reading or manipulating an HDF5 file, we need to close it to avoid and potential damage:

h5_file.close()
os.remove(h5_path)

########################################################################################################################
# As mentioned in the beginning this is not meant to be a comprehensive overview of HDF5 or h5py, but rather just a
# quick overview of the important functionality we recommend everyone to be familiar with. We encourage you to read more
# about h5py and HDF5 if you are interested.
