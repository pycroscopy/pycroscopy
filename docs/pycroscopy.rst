Pycroscopy API Reference
========================

Package Structure
-----------------
The package structure is simple, with 4 main modules:
   1. `io`: Input/Output from custom & proprietary microscope formats to HDF5.
   2. `processing`: Multivariate Statistics, Machine Learning, and Filtering.
   3. `analysis`: Model-dependent analysis of image information.
   4. `viz`: Visualization and interactive slicing of high-dimensional data by lightweight Qt viewers.

Once a user converts their microscope's data format into an HDF5 format, by simply extending some of the classes in `io`, the user gains access to the rest of the utilities present in `pycroscopy.*`.

.. currentmodule:: pycroscopy

.. automodule:: pycroscopy
    :no-members:
    :no-inherited-members:

:py:mod:`pycroscopy`:

.. autosummary::
    :toctree: _autosummary/
    :template: module.rst

    pycroscopy.analysis
    pycroscopy.io
    pycroscopy.processing
    pycroscopy.tests
    pycroscopy.viz
