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
5. ``simulation``: Simulations and modelling here

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

``pycroscopy.simulation``
~~~~~~~~~~~~~~~~~~~~~~~~~
* AFM simulation and many more to come.

``pycroscopy.core``
~~~~~~~~~~~~~~~~~~~
This engineering-focused sub-package has been moved to a new package - `pyUSID <https://pycroscopy.github.io/pyUSID/about.html>`_.
pyUSID supports science-focused packages such as pycroscopy similar to how scipy depends on numpy.
The current release of pycroscopy imports pyUSID and makes it available as ``pycroscopy.core`` so that existing imports do not break.
In the next release of pycroscopy, this implicit import will be removed and the modules would have to be imported directly from ``pyUSID``.
See the `what's new <./whats_new.html>`_ under **June 28 2018**.


Branches
--------
* ``master`` : Stable code based off which the pip installer works. Recommended for most people.
* ``dev`` : Experimental code with new features that will be made available in ``master`` periodically after thorough
  testing. By its very definition, this branch is recommended only for developers.