Package Organization
====================
Sub-packages
------------
The package structure is simple, with 5 main modules:

1. **io**: Translators to extract data from custom & proprietary microscope formats and write them to HDF5 files.
2. **analysis**: Physics-dependent analysis of information.
3. **processing**: Physics-independent processing of data including  machine learning, image processing, signal
   filtering.
4. **viz**: Plotting functions and interactive jupyter widgets for scientific applications
5. **core**: Science-independent engineering components that form the foundation for the aforementioned four packages.
   In the next release of pycroscopy, this engineering-focused sub-package will become an independent package that will
   support science-focused packages such as pycroscopy similar to how numpy supports scipy.

core subpackage
---------------
The structure of the `.core` subpackage mimics that of pycroscopy and it has three main modules:
1. **io**: utilities that simplify the storage and accessing of data stored in pycroscopy formatted HDF5 files
2. **processing**: utilities and classes that support the piecewise (parallel) processing of arbitrarily large datasets
3. **viz**: plotting utilities and jupyter widgets that simplify common scientific visualization problems

`io`
~~~~
* **hdf_utils** - Utilities for greatly simplifying reading and writing to pycroscopy formatted HDF5 files.
* **write_utils** - Utilities that assist in writing to HDF5 files
* **dtype_utils** - Utilities for data transformation (to and from real-valued, complex-valued, and compound-valued data
  arrays) and validation operations
* **io_utils** - Utilities for simplifying common computational, communication, and string formatting operations
* **PycroDataset** - extends h5py.Dataset. We expect that users will use this class at every opportunity in order to
  simplify common operations on datasets such as interactive visualization in jupyter notebooks, slicing by the
  dataset's N-dimensional form and simplified access to supporting information about the dataset.
* **Translator** - An abstract class that provides the blueprint for other Translator classes to extract data and
  meta-data from other raw-data files and write them into pycroscopy formatted HDF5 files
* **ImageTranslator** - Translates data in common image file formats such as .png and .tiff to a
  Pycroscopy formatted HDF5 file
* **NumpyTranslator** - A generic translator that simplifies writing of a dataset in memory into a pycroscopy formatted
  HDF5 file

`processing`
~~~~~~~~~~~~
* **Process** - Modularizes, formalizes, and simplifies robust data processing
* **parallel_compute** - Highly simplified one-line call to perform parallel computing on a data array

`viz`
~~~~~
* **plot_utils** - utilities to simplify common scientific tasks such as plotting a set of curves within the same or
  separate plots, plotting a set of 2D images in a grid, custom color-bars, etc.
* **jupyter_utils** - utilities to enable interactive visualization on generic 4D datasets within jupyter notebooks

Branches
--------
* ``master`` : Stable code based off which the pip installer works. Recommended for most people.
* ``dev`` : Experimental code with new features that will be made available in ``master`` periodically after thorough
  testing. By its very definition, this package Recommended only for developers.