"""
===========================================
Writing to hdf5 using the Microdata objects
===========================================



"""

# Code source: Chris Smith -- cq6@ornl.gov
# Liscense: MIT

import os
import numpy as np
import pycroscopy as px

##############################################################################
# Create some MicroDatasets and MicroDataGroups that will be written to the file.
# With h5py, groups and datasets must be created from the top down,
# but the Microdata objects allow us to build them in any order and link them later.

# First create some data
data1 = np.random.rand(5, 7)

##############################################################################
# Now use the array to build the dataset.  This dataset will live
# directly under the root of the file.  The MicroDataset class also implements the
# compression and chunking parameters from h5py.Dataset.
ds_main = px.MicroDataset('Main_Data', data=data1, parent='/')

##############################################################################
# We can also create an empty dataset and write the values in later
# With this method, it is neccessary to specify the dtype and maxshape kwarg parameters.
ds_empty = px.MicroDataset('Empty_Data', data=[], dtype=np.float32, maxshape=[7, 5, 3])

##############################################################################
# We can also create groups and add other MicroData objects as children.
# If the group's parent is not given, it will be set to root.
data_group = px.MicroDataGroup('Data_Group', parent='/')

root_group = px.MicroDataGroup('/')

# After creating the group, we then add an existing object as its child.
data_group.addChildren([ds_empty])
root_group.addChildren([ds_main, data_group])

##############################################################################
# The showTree method allows us to view the data structure before the hdf5 file is
# created.
root_group.showTree()

##############################################################################
# Now that we have created the objects, we can write them to an hdf5 file

# First we specify the path to the file
h5_path = 'microdata_test.h5'

# Then we use the ioHDF5 class to build the file from our objects.
hdf = px.ioHDF5(h5_path)

##############################################################################
# The writeData method builds the hdf5 file using the structure defined by the
# MicroData objects.  It returns a list of references to all h5py objects in the
# new file.
h5_refs = hdf.writeData(root_group, print_log=True)

# We can use these references to get the h5py dataset and group objects
h5_main = px.io.hdf_utils.getH5DsetRefs(['Main_Data'], h5_refs)[0]
h5_empty = px.io.hdf_utils.getH5DsetRefs(['Empty_Data'], h5_refs)[0]

##############################################################################
# Compare the data in our dataset to the original
print(np.allclose(h5_main[()], data1))

##############################################################################
# As mentioned above, we can now write to the Empty_Data object
data2 = np.random.rand(*h5_empty.shape)
h5_empty[:] = data2[:]

##############################################################################
# Now that we are using h5py objects, we must use flush to write the data to file
# after it has been altered.
# We need the file object to do this.  It can be accessed as an attribute of the
# hdf object.
h5_file = hdf.file
h5_file.flush()

##############################################################################
# Now that we are done, we should close the file so that it can be accessed elsewhere.
h5_file.close()
os.remove(h5_path)
