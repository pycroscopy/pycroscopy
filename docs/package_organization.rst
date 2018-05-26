Package Organization
====================
Sub-packages
------------
The package structure is simple, with 5 main modules:

1. ``io``: Translators to extract data from custom & proprietary microscope formats and write them to HDF5 files.
2. ``analysis``: Physics-dependent analysis of information.
3. ``processing``: Physics-independent processing of data including  machine learning, image processing, signal
   filtering.
4. ``viz``: Plotting functions and interactive jupyter widgets for scientific applications
5. ``core``: Science-independent engineering components that form the foundation for the aforementioned four packages.
   In the next release of pycroscopy, this engineering-focused sub-package will become an independent package that will
   support science-focused packages such as pycroscopy similar to how scipy depends on numpy.


``pycroscopy.io``
~~~~~~~~~~~~~~~~~~~
* ``HDFWriter`` (previously named ``ioHDF5``) - Legacy class that was used for writing HDF5 files. This is now deprecated and will be removed in a future release.
* ``VirtualDataset`` (previously named ``MicroDataset``), ``VirtualGroup`` (previously named ``MicroDataGroup``) - Legacy
  classes that were virtual representations of HDF5 dataset and HDF5 group objects. Both these objects have been deprecated and will be removed in a future release.
* ``write_utils`` - Utilities that assist in writing ancillary datasets using ``VirtualDataset`` objects. The functions
  in this module are deprecated and will be removed in a future release.
* ``translators`` - subpackage containing several ``Translator`` classes that extract data from custom & proprietary microscope formats and write them to HDF5 files.

  * ``df_utils`` - any utilities that assist in the functioning of the ``Translator`` classes in ``translators``

``pycroscopy.processing``
~~~~~~~~~~~~~~~~~~~~~~~~~
* ``Cluster`` - Facilitates data transformation to perform clustering using sklearn and writing of results to file
* ``Decomposition`` - Facilitates data transformation to perform decomposition using sklearn and writing of results to file
* ``SVD`` - Facilitates data transformation to perform SVD using sklearn and writing of results to file. Plus SVD related utilities such as rebuilding data using SVD results
* ``SignalFilter`` - Class that facilitates FFT-based signal filtering
* ``fft`` - module with FFT and FFT filtering related functions
* ``gmode_utils`` - Utilities that support ``SignalFilter``
* ``tree`` - utilities that facilitate the propagation of information (for example endmembers from clustering)
* ``proc_utils`` - utilities used by Cluster, SVD, Decomposition
* ``histogram`` - Utilities for generating histograms on spectral data
* ``image_processing`` - Classes for enabling image-windowing and SVD-based image cleaning
* ``contrib`` - user contributed code

``pycroscopy.analysis``
~~~~~~~~~~~~~~~~~~~~~~~~
* ``Fitter`` - An absrtact class that supports science-agnostic functions that facilitate the fitting of data to analytical functions
* ``optimize`` - A class and set of utilities that facilitate parallel functional fitting using scipy.optimize
* ``guess_methods`` - A set of functions that provide good initial values for functional fitting
* ``fit_methods`` - A set of functions that facilitate functional fitting.
* ``BESHOFitter`` - A class that handles the fitting of piezoresponse spectra to a simple harmonic oscillator model
* ``BELoopFitter`` - A class that handles fitting of piezoresponse hysteresis loops to an analytical function
* ``GIVBayesian`` - A class that performs Bayesian inference on G-mode current-voltage data
* ``contrib`` - A collection of user contribution code

``pycroscopy.core``
~~~~~~~~~~~~~~~~~~~
The structure of the ``pycroscopy.core`` subpackage mimics that of pycroscopy and it has three main modules:
1. ``io``: utilities that simplify the storage and accessing of data stored in pycroscopy formatted HDF5 files
2. ``processing``: utilities and classes that support the piecewise (parallel) processing of arbitrarily large datasets
3. ``viz``: plotting utilities and jupyter widgets that simplify common scientific visualization problems

``io``
^^^^^^^^
* ``hdf_utils`` - Utilities for greatly simplifying reading and writing to pycroscopy formatted HDF5 files.
* ``write_utils`` - Utilities that assist in writing to HDF5 files
* ``dtype_utils`` - Utilities for data transformation (to and from real-valued, complex-valued, and compound-valued data
  arrays) and validation operations
* ``io_utils`` - Utilities for simplifying common computational, communication, and string formatting operations
* ``PycroDataset`` - extends h5py.Dataset. We expect that users will use this class at every opportunity in order to
  simplify common operations on datasets such as interactive visualization in jupyter notebooks, slicing by the
  dataset's N-dimensional form and simplified access to supporting information about the dataset.
* ``Translator`` - An abstract class that provides the blueprint for other Translator classes to extract data and
  meta-data from other raw-data files and write them into pycroscopy formatted HDF5 files
* ``ImageTranslator`` - Translates data in common image file formats such as .png and .tiff to a
  Pycroscopy formatted HDF5 file
* ``NumpyTranslator`` - A generic translator that simplifies writing of a dataset in memory into a pycroscopy formatted
  HDF5 file

``processing``
^^^^^^^^^^^^^^^^
* ``Process`` - Modularizes, formalizes, and simplifies robust data processing
* ``parallel_compute()`` - Highly simplified one-line call to perform parallel computing on a data array

``viz``
^^^^^^^^
* ``plot_utils`` - utilities to simplify common scientific tasks such as plotting a set of curves within the same or
  separate plots, plotting a set of 2D images in a grid, custom color-bars, etc.
* ``jupyter_utils`` - utilities to enable interactive visualization on generic 4D datasets within jupyter notebooks

Branches
--------
* ``master`` : Stable code based off which the pip installer works. Recommended for most people.
* ``dev`` : Experimental code with new features that will be made available in ``master`` periodically after thorough
  testing. By its very definition, this branch is recommended only for developers.