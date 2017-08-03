.. pycroscopy documentation master file, created by
   sphinx-quickstart on Thu Jul 20 11:56:03 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pycroscopy's documentation!
======================================
0. Description
--------------
A python package for image processing and scientific analysis of imaging modalities such as multi-frequency scanning probe microscopy,
scanning tunneling spectroscopy, x-ray diffraction microscopy, and transmission electron microscopy.
Classes implemented here are ported to a high performance computing platform at Oak Ridge National Laboratory (ORNL).

1. Package Structure
--------------------
The package structure is simple, with 4 main modules:
   1. `io`: Input/Output from custom & proprietary microscope formats to HDF5.
   2. `processing`: Multivariate Statistics, Machine Learning, and Filtering.
   3. `analysis`: Model-dependent analysis of image information.
   4. `viz`: Visualization and interactive slicing of high-dimensional data by lightweight Qt viewers.

Once a user converts their microscope's data format into an HDF5 format, by simply extending some of the classes in `io`, the user gains access to the rest of the utilities present in `pycroscopy.*`.

.. toctree::
   :maxdepth: 4
   :caption: Contents:

   pycroscopy.analysis
   pycroscopy.io
   pycroscopy.processing
   pycroscopy.tests
   pycroscopy.viz
   gallery_examples/index


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
