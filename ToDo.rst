.. contents::

Goals
-------

Immediate
~~~~~~~~~
* Solicit brand names:

  * Data model (does NOT include storage in HDF5):

    * Keywords: Open, Spectroscopy, Imaging, MultiDimensional, Tensor, Model / Data Format
    * @rama - Not Another Data Format - NADF
    * @rama - Spectral Data Format - SDF
    * @ssomnath - Open MutiDimensional Imaging and Spectroscopy Data - MISD / OMDISD
    * @ssomnath - Open Spectral Data - OSD
    * @ssomnath - Open HyperSpectral Data - OSD / OHSD
  * File Format - h5<Data Model Acronym>
  * package:

    * Keywords: Open, Python, HDF5, + all from above
    * @rama - pyNADF, pySDF
* Swap out remaining usages of ``VirtualData`` + ``HDFWriter`` to ``hdf_utils`` (especially outside ``io.translators``)
* Test all existing translators and to make sure they still work.
* Move ``requirements`` to requirements.txt
* Gateway translators for SPM:

  * ``Gwyddion``:

    * Fill in details into new skeleton translator. Use jupyter notebook for reference.
    * For native ``.gwy`` format use package - `gwyfile <https://github.com/tuxu/gwyfile>`_ (already added to requirements)
    * For simpler ``.gsf`` format use ``gsf_read()``
  * ``WsXM``:

    * Adopt `Iaroslav Gaponenko's reader <https://github.com/paruch-group/distortcorrect/blob/master/afm/filereader/readWSxM.py>`_
    * Note that the above file only reads scan images and does not appear to work on other kinds of data

* Translators for popular AFM / SPM formats

  * Bruker / Veeco / Digital Instruments - Done but look into `Iaroslav Gaponenko's code <https://github.com/paruch-group/distortcorrect/blob/master/afm/filereader/readNanoscope.py>`_
    to add any missing functionality / normalization of data. etc. Also check against `this project <https://github.com/nikhartman/nanoscope>`_.
  * Nanonis - done but look into `Iaroslav Gaponenko reader <https://github.com/paruch-group/distortcorrect/blob/master/afm/filereader/nanonisFileReader.py>`_
    to make sure nothing is missed out / done incorrectly.

    * Address .SXM, .3DS, and .DAT translation issues
  * Asylum ARDF - Use Liam's data + files from asylum
  * Park Systems - Yunseok is helping here
  * JPK - No data / code available
  * Niche data formats:

    * NT-MDT - Data available. translator pending.
    * PiFM - J. Kong from Ginger Group will write a translator upon her visit to CNMS in summer of 2018.
    * Anasys - import `anasyspythontools <https://github.com/AnasysInstruments/anasys-python-tools>`_ - comes with test data

      * This package does `NOT <https://pypi.org/search/?q=anasyspythontools>`_ have a `pypi installer <https://github.com/AnasysInstruments/anasys-python-tools/issues/2>`_
      * This package does not look like it is finished.

* Move ``.core`` out of pycroscopy

  * continuous integration
  * documentation, website, etc.
  * pypi and conda installers
  * pycroscopy should import core and alias it for now
* Write a comprehensive document on "Contribution guidelines"
* Embed presentation into "About"
* Start writing the journal paper!
* Write plugins to export to pycroscopy HDF5 for ``ImageJ`` and ``FIJI``. There are HDF5 plugins already available.

Short-term - by Jul 1 2018
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* development of community standards for AFM, STM, STEM, etc.
* region reference related functionality

  * Move functions to separate file outside ``hdf_utils``
  * Relax restrictions with regards to expecting region references
* ``PycroDataset`` - Do slicing on ND dataset if available by flattening to 2D and then slicing
* Chris - ``Image Processing`` must be a subclass of ``Process`` and implement resuming of computation and checking for old (both already handled quite well in Process itself) - here only because it is used and requested frequently + should not be difficult to restructure.
* Clear and obvious examples showing what pycroscopy actually does for people

  * Image cleaning - Chris
  * Signal Filtering - Suhas
  * Two more examples
* Upload clean exports of paper notebooks + add notebooks for new papers + add new papers (Sabine + Liam)
* Update change-log with version numbers / releases instead of pull numbers
* unit tests for basic data science (``Cluster``, ``SVD``, ``Decomposition``)
* Add requirements.txt
* Add ability to export data as txt probably via numpy.savetext
* Examples within docs for popular functions
* file dialog for Jupyter not working on Mac OS
* Revisit and address as many pending TODOs as possible

Medium-term - by Aug 1 2018
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Technical poster for pycroscopy
* Explore Azure Notebooks for live tutorials
* ``PycroDataset.slice()`` and ``get_n_dim_form()`` should return ``Xarray`` objects
* Notebooks need to be updated to require pycroscopy version
* Itk for visualization - https://github.com/InsightSoftwareConsortium/itk-jupyter-widgets
* New package for facilitating **scalable ensemble runs**:

  * Compare scalability, simplicity, portability of various solutions:
    
    * MPI4py
    * Dask (Matthew Rocklin)
    * pyspark
    * ipyparallel... 
  * Deploy on CADES SHPC Condo, Eos, Rhea (CPU partition).
  * Use stand-alone GIV or SHO Fitting as an example
  * Develop some generalized class equivalent to / close to ``Process``

Long-term
~~~~~~~~~~
* Rewrite ``Process`` to use ``Dask`` and ignore ``parallel_compute()`` - try on SHO guess
* Think about implementing costly algorithms in a deep learning framework like ``TensorFlow`` / ``PyTorch`` to use GPUs. Test with full Bayesian / simple Bayesian (anything computationally expensive)
* Look into versioneer
* A sister package with the base labview subvis that enable writing pycroscopy compatible hdf5 files. The actual acquisition can be ignored.
* Intelligent method (using timing) to ensure that process and Fitter compute over small chunks and write to file periodically. Alternatively expose number of positions to user and provide intelligent guess by default
* Consider developing a generic curve fitting class a la `hyperspy <http://nbviewer.jupyter.org/github/hyperspy/hyperspy-demos/blob/master/Fitting_tutorial.ipynb>`_
* function for saving sub-tree to new h5 file
* Windows compatible function for deleting sub-tree
* Chris - Demystify analyis / optimize. Use parallel_compute instead of optimize and guess_methods and fit_methods
* Consistency in the naming of and placement of attributes (chan or meas group) in all translators - Some put attributes in the measurement level, some in the channel level! hyperspy appears to create datagroups solely for the purpose of organizing metadata in a tree structure!
* Batch fitting - need to consider notebooks for batch processing of BELINE and other BE datasets. This needs some thought, but a basic visualizer that allows selection of a file from a list and plotting of the essential graphs is needed.
* Profile code to see where things are slow

Back-burner
~~~~~~~~~~~~
* Cloud deployment
  * Container installation
  * Check out HDF5Cloud
  * AWS cloud cluster
* Look into ``Tasmanian`` (mainly modeling) - Miroslav Stoyanov
* Look into ``Adios`` i(William G; Norbert is affiliated with ADWG)
* ``Pydap.client``: wrapper of ``opendap`` – accessing data remotely and remote execution of notebooks - https://github.com/caseyjlaw/jupyter-notebooks/blob/master/vlite_hdfits_opendap_demo.ipynb
* Alternate visualization packages - http://lightning-viz.org

Reogranization
---------------

1.  Reorganize code - This is perhaps the last opportunity for major restructuring and renaming.

  * Subpackages within processing: statistics, image, signal, misc
  * How does one separate tested code from untested code? For example - SHO fitting is currently not tested but may become tested in the future.
  * hdf_utils is becoming very big and all the functions deal with h5 in some form whether it is for reading or writing. Perhaps it should be split into read_utils and write_utils? hdf is implied.
  * Make room (in terms of organization) for deep learning - implementation will NOT be part of 0.60.0:

    * pycroscopy hdf5 to tfrecords / whatever other frameworks use
    * What science specific functions can be generalized and curated?
  * Usage of package (only Clustering + SHO fitting for example) probably provides clues about how the package should / could be reorganized (by analysis / process). Typically, most analysis and Process classes have science-specific plotting. Why not insert Procoess / Analysis specific plotting / jupyter functions along with the Process / Fitter class?
  * Think about whether the rest of the code should be organized by instrument

    * One possible strategy - .core, .process (science independent), .instrument?. For example px.instrument.AFM.BE would contain translators under a .translators, the two analysis modules and accompanying functions under .analysis and visualization utilities under a .viz submodule. The problem with this is that users may find this needlessly complicated. Retaining existing package structure means that all the modalities are mixed in .analysis, .translators and .viz.

External user contributions
----------------------------
* Sabine Neumeyer's cKPFM code
* Incorporate sliding FFT into pycroscopy - Rama
* Create an IR analysis notebook - Suhas should have something written in IF Drive
* Li Xin classification code - Li Xin
* Ondrej Dyck’s atom finding code – written well but needs to work on images with different kinds of atoms
* Nina Wisinger’s processing code (Tselev) – in progress
* Port everything from IFIM Matlab -> Python translation exercises
* Iaroslav Gaponenko's `Distort correct <https://github.com/paruch-group/distortcorrect>`_

Scaling to HPC
--------------
We have two kinds of large computational jobs and one kind of large I/O job:

* I/O - reading and writing large amounts of data:

  * MPI clearly works with very high performance parallel read and write
  * Dask also works but performance is a question. Look at NERSC (Matthew Rocklin et al.)
  * Spark / HDFS requires investigation - Apparently does not work well with HDF5 files

* Computation:

  1. Machine learning and Statistics

    * Use custom algorithms developed for BEAM - NO one is willing to salvage code

      * Advantage - Optimized (and tested) for various HPC environments
      * Disadvantages:

        * Need to integrate non-python code
        * We only have a handful of these. NOT future compatible

    * OR continue using a single FAT node for these jobs

      * Advantages:

        * No optimization required
        * Continue using the same scikit learn packages
      * Disadvantage - Is not optimized for HPC

    * OR use pbdR / write pbdPy (wrappers around pbdR)

      * Advantages:

        * Already optimized / mature project
        * In-house project (good support)
      * Disadvantages:

        * Dependant on pbdR for implementing new algorithms

  2. Embarrasingly parallel analysis / processing. Can be scaled using:

    * Dask - An inplace replacement of multiprocessing will work on laptops and clusters. More elegant and easier to write and maintain compared to MPI at the cost of efficiency

      * simple dask netcdf example: http://matthewrocklin.com/blog/work/2016/02/26/dask-distributed-part-3
    * MPI - Need alternatives to Optimize / Process classes - Best efficiency but a pain to implement
    * Spark?
    * ipyParallel?
