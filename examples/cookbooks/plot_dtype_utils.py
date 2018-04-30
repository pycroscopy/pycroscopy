"""
================================================================================
Utilities for handling data types and transformations
================================================================================

**Suhas Somnath**

4/18/2018
"""
################################################################################
# Introduction
# -------------
# The general nature of pycroscopy facilitates the representation of any kind of measurement data.
# This includes:
#
#  1. Conventional data represented using floating point numbers such as 1.2345
#  2. Integer data (with or without sign) such as 1, 2, 3, 4
#  3. Complex-valued data such as 1.23 + 4.5i
#  4. Multi-valued or compound valued data cells such as ('Frequency': 301.2, 'Amplitude':1.553E-3, 'Phase': 2.14)
#     where a single value or measurement is represented by multiple elements, each with their own names, and data types
#
# While HDF5 datasets are capable of storing all of these kinds of data, many conventional data analysis techniques
# such as decomposition, clustering, etc. are either unable to handle complicated data types such as complex-valued
# datasets and compound valued datasets, or the results from these techniques do not produce physically meaningful
# results. For example, most singular value decomposition algorithms are capable of processing complex-valued datasets.
# However, while the eigenvectors can have complex values, the resultant complex-valued abundance maps are meaningless.
# These algorithms would not even work if the original data was compound valued!
#
# To avoid such problems, we need functions that transform the data to and from the necessary type (integer, real-value
# etc.)
#
# The pycrocsopy.dtype_utils module facilitates comparisons, validations, and most importantly, transformations of one
# data-type to another. We will be going over the many useful functions in this module and explaining how, when and why
# one would use them.
#
# Recommended pre-requisite reading
# -----------------------------------
# * `pycroscopy data format <https://pycroscopy.github.io/pycroscopy/data_format.html>`_
# * `Crash course on HDF5 and h5py <https://pycroscopy.github.io/pycroscopy/auto_examples/cookbooks/plot_h5py.html>`_
#
# Import all necessary packages
# -------------------------------
# Before we begin demonstrating the numerous functions in ``pycroscopy.dtype_utils``, we need to import the necessary
# packages. Here are a list of packages besides pycroscopy that will be used in this example:
#
# * ``h5py`` - to manipulate HDF5 files
# * ``numpy`` - for numerical operations on arrays in memory

from __future__ import print_function, division, unicode_literals
import os
import h5py
import numpy as np
# Finally import pycroscopy.
try:
    import pycroscopy as px
except ImportError:
    # Warning package in case something goes wrong
    from warnings import warn
    warn('pycroscopy not found.  Will install with pip.')
    import pip
    pip.main(['install', 'pycroscopy'])
    import pycroscopy as px

################################################################################
# Utilities for validating data types
# =====================================
# pycroscopy.dtype_utils contains some handy functions that make it easy to write robust and safe code by simplifying
# common data type checking and validation.
#
# contains_integers()
# ---------------------
# The ``contains_integers()`` function checks to make sure that each item in a list is indeed an integer. Additionally, it
# can be configured to ensure that all the values are above a minimum value. This is particularly useful when building
# indices matrices based on the size of dimensions - specified as a list of integers for example.

item = [1, 2, -3, 4]
print('{} : contains integers? : {}'.format(item, px.dtype_utils.contains_integers(item)))
item = [1, 4.5, 2.2, -1]
print('{} : contains integers? : {}'.format(item, px.dtype_utils.contains_integers(item)))

item = [1, 5, 8, 3]
min_val = 2
print('{} : contains integers >= {} ? : {}'.format(item, min_val,
                                                px.dtype_utils.contains_integers(item, min_val=min_val)))

################################################################################
# validate_dtype()
# -----------------
# The ``validate_dtype()`` function ensure that a provided object is indeed a valid h5py or numpy data type. When writing
# a main dataset along with all ancillary datasets, pycroscopy meticulously ensures that all intputs are valid before
# writing data to the file. This comes in very handy when we want to follow the 'measure twice, cut once' ethos.

for item in [np.float16, np.complex64, np.uint8, np.int16]:
    print('Is {} a valid dtype? : {}'.format(item, px.dtype_utils.validate_dtype(item)))


# This function is especially useful on compound or structured data types:

struct_dtype = np.dtype({'names': ['r', 'g', 'b'],
                        'formats': [np.float32, np.uint16, np.float64]})
print('Is {} a valid dtype? : {}'.format(struct_dtype, px.dtype_utils.validate_dtype(struct_dtype)))

################################################################################
# get_compound_sub_dtypes()
# --------------------------
# One common hassle when dealing with compound / structured array dtypes is that it can be a little challenging to
# quickly get the individual datatypes of each field in such a data type. The ``get_compound_sub_dtypes()`` makes this a
# lot easier:

sub_dtypes = px.dtype_utils.get_compound_sub_dtypes(struct_dtype)
for key, val in sub_dtypes.items():
    print('{} : {}'.format(key, val))

################################################################################
# is_complex_dtype()
# -------------------
# Quite often, we need to treat complex datasets different from compound datasets which themselves need to be treated
# different from real valued datasets. ``is_complex_dtype()`` makes it easier to check if a numpy or HDF5 dataset has a
# complex data type:

for dtype in [np.float32, np.float16, np.uint8, np.int16, struct_dtype, bool]:
    print('Is {} a complex dtype?: {}'.format(dtype, (px.dtype_utils.is_complex_dtype(dtype))))

for dtype in [np.complex, np.complex64, np.complex128, np.complex256]:
    print('Is {} a complex dtype?: {}'.format(dtype, (px.dtype_utils.is_complex_dtype(dtype))))

################################################################################
# Data transformation
# ====================
# Perhaps the biggest benefit of ``dtype_utils`` is the ability to flatten complex, compound datasets to real-valued
# datasets and vice versa. As mentioned in the introduction, this is particularly important when attempting to use
# machine learning algorithms on complex or compound-valued datasets. In order to enable such pipelines, we need
# functions to transform:
# * complex / compound valued datasets to real-valued datasets
# * real-valued datasets back to complex / compound valued datasets
#
# flatten_complex_to_real()
# --------------------------
# As the name suggests, this function stacks the imaginary values of a N-dimensional numpy / HDF5 dataset below its
# real-values. Thus, applying this function to a complex valued dataset of size ``(a, b, c)`` would result in a
# real-valued dataset of shape ``(a, b, 2 * c)``:

length = 3
complex_array = np.random.randint(-5, high=5, size=length) + 1j * np.random.randint(-5, high=5, size=length)
stacked_real_array = px.dtype_utils.flatten_complex_to_real(complex_array)
print('Complex value: {} has shape: {}'.format(complex_array, complex_array.shape))
print('Stacked real value: {} has shape: '
      '{}'.format(stacked_real_array, stacked_real_array.shape))

################################################################################
# flatten_compound_to_real()
# ----------------------------
# This function flattens a compound-valued dataset of shape ``(a, b, c)`` into a real-valued dataset of shape
# ``(a, b, k * c)`` where ``k`` is the number of fields within the structured array / compound dtype. Here we will
# demonstrate this on a 1D array of 5 elements each containing 'r', 'g', 'b' fields:

num_elems = 5
structured_array = np.zeros(shape=num_elems, dtype=struct_dtype)
structured_array['r'] = np.random.random(size=num_elems) * 1024
structured_array['g'] = np.random.randint(0, high=1024, size=num_elems)
structured_array['b'] = np.random.random(size=num_elems) * 1024
real_array = px.dtype_utils.flatten_compound_to_real(structured_array)

print('Structured array is of shape {} and have values:'.format(structured_array.shape))
print(structured_array)
print('\nThis array converted to regular scalar matrix has shape: {} and values:'.format(real_array.shape))
print(real_array)

################################################################################
# flatten_to_real()
# -----------------
# This function checks the data type of the provided dataset and then uses either of the above functions to
# (if necessary) flatten the dataset into a real-valued matrix. By checking the data type of the dataset, it obviates
# the need to explicitly call the aforementioned functions (that still do the work). Here is an example of the function
# being applied to the compound valued numpy array again:

real_array = px.dtype_utils.flatten_to_real(structured_array)
print('Structured array is of shape {} and have values:'.format(structured_array.shape))
print(structured_array)
print('\nThis array converted to regular scalar matrix has shape: {} and values:'.format(real_array.shape))
print(real_array)

################################################################################
# The next three functions perform the inverse operation of taking real-valued matrices or datasets and converting them
# to complex or compound-valued datasets.
#
# stack_real_to_complex()
# ------------------------
# As the name suggests, this function collapses a N dimensional real-valued array of size ``(a, b, 2 * c)`` to a
# complex-valued array of shape ``(a, b, c)``. It assumes that the first c values in real-valued dataset are the real
# components and the following c values are the imaginary components of the complex value. This will become clearer
# with an example:


real_val = np.hstack([5 * np.random.rand(6),
                      7 * np.random.rand(6)])
print('Real valued dataset of shape {}:'.format(real_val.shape))
print(real_val)

comp_val = px.dtype_utils.stack_real_to_complex(real_val)

print('\nComplex-valued array of shape: {}'.format(comp_val.shape))
print(comp_val)

################################################################################
# stack_real_to_compound()
# --------------------------
# Similar to the above function, this function shrinks the last axis of a real valued dataset to create the desired
# compound valued dataset. Here we will demonstrate it on the same 3-field ``(r,g,b)`` compound datatype:

num_elems = 5
real_val = np.concatenate((np.random.random(size=num_elems) * 1024,
                           np.random.randint(0, high=1024, size=num_elems),
                           np.random.random(size=num_elems) * 1024))
print('Real valued dataset of shape {}:'.format(real_val.shape))
print(real_val)

comp_val = px.dtype_utils.stack_real_to_compound(real_val, struct_dtype)

print('\nStructured array of shape: {}'.format(comp_val.shape))
print(comp_val)

################################################################################
# stack_real_to_target_dtype()
# -----------------------------
# This function performs the inverse of ``flatten_to_real()`` - stacks the provided real-valued dataset into a complex or
# compound valued dataset using the two above functions. Note that unlike ``flatten_to_real()``, the target data type must
# be supplied to the function for this to work:

print('Real valued dataset of shape {}:'.format(real_val.shape))
print(real_val)

comp_val = px.dtype_utils.stack_real_to_target_dtype(real_val, struct_dtype)

print('\nStructured array of shape: {}'.format(comp_val.shape))
print(comp_val)

################################################################################
# check_dtype()
# --------------
# ``check_dtype()`` is a master function that figures out the data type, necessary function to transform a HDF5 dataset to
# a real-valued array, expected data shape, etc. Before we demonstrate this function, we need to quickly create an
# example HDF5 dataset.

file_path = 'dtype_utils_example.h5'
if os.path.exists(file_path):
    os.remove(file_path)
with h5py.File(file_path) as h5_f:
    num_elems = (5, 7)
    structured_array = np.zeros(shape=num_elems, dtype=struct_dtype)
    structured_array['r'] = 450 * np.random.random(size=num_elems)
    structured_array['g'] = np.random.randint(0, high=1024, size=num_elems)
    structured_array['b'] = 3178 * np.random.random(size=num_elems)
    _ = h5_f.create_dataset('compound', data=structured_array)
    _ = h5_f.create_dataset('real', data=450 * np.random.random(size=num_elems), dtype=np.float16)
    _ = h5_f.create_dataset('complex', data=np.random.random(size=num_elems) + 1j * np.random.random(size=num_elems),
                            dtype=np.complex64)
    h5_f.flush()

################################################################################
# Now, lets test the the function on compound-, complex-, and real-valued HDF5 datasets:


def check_dataset(h5_dset):
    print('\tDataset being tested: {}'.format(h5_dset))
    func, is_complex, is_compound, n_features, type_mult = px.dtype_utils.check_dtype(h5_dset)
    print('\tFunction to transform to real: %s' % func)
    print('\tis_complex? %s' % is_complex)
    print('\tis_compound? %s' % is_compound)
    print('\tShape of dataset in its current form: {}'.format(h5_dset.shape))
    print('\tAfter flattening to real, shape is expected to be: ({}, {})'.format(h5_dset.shape[0], n_features))
    print('\tByte-size of a single element in its current form: {}'.format(type_mult))


with h5py.File(file_path, mode='r') as h5_f:
    print('Checking a compound-valued dataset:')
    check_dataset(h5_f['compound'])
    print('')
    print('Checking a complex-valued dataset:')
    check_dataset(h5_f['complex'])
    print('')
    print('Checking a real-valued dataset:')
    check_dataset(h5_f['real'])
os.remove(file_path)
