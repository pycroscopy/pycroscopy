What's New
==========

`Jun 28 2018: <https://github.com/pycroscopy/pycroscopy/pull/181>`_
------------------------------------------------------------------------
Moved ``pycroscopy.core`` into separate package - `pyUSID <https://pycroscopy.github.io/pyUSID/about.html>`_
pyUSID will be the engineering package that supports science-focused packages such as pycroscopy similar to how scipy depends on numpy.
All references to pycroscopy.core within the pycroscopy package are now referencing pyusid instead.
The current release of pycroscopy imports pyUSID and makes it available as ``pycroscopy.core`` so that existing imports in user-code do not break.
In the next release of pycroscopy, this implicit import will be removed and the following modules would have to be imported directly from ``pyUSID``:

* ``hdf_utils``
* ``write_utils``
* ``dtype_utils``
* ``io_utils``
* ``PycroDataset`` - renamed to ``USIDataset``
* ``Translator``
* ``ImageTranslator``
* ``NumpyTranslator``
* ``Process``
* ``parallel_compute()``
* ``plot_utils``
* ``jupyter_utils``

Thus, imports and usages of such modules as:

.. code:: python

  import pycroscopy as px
  px.plot_utils.plot_map(...)
  px.hdf_utils.print_tree(h5_file)
  px.PycroDataset(h5_dset)
  # Other non-core classes:
  px.processing.SignalFilter(h5_main, ...)

would need to be changed to:

.. code:: python

  # Now import pyUSID along with pycroscopy
  import pyUSID as usid
  import pycroscopy as px
  # functions and classes still work the same way
  # just use usid instead of px for anything that was in core (see list above).
  usid.plot_utils.plot_map(...)
  usid.hdf_utils.print_tree(h5_file)
  # The only difference is the renaming of the PycroDataset to USIDataset:
  usid.USIDataset(h5_dset)
  # Other classes and functions outside .core are addressed just as before:
  px.processing.SignalFilter(h5_main, ...)

`Jun 19 2018: <https://github.com/pycroscopy/pycroscopy/pull/180>`_
------------------------------------------------------------------------
* Thanks to `@Liambcollins <https://github.com/Liambcollins>`_ for bug-fixes to ``GTuneTranslator``

`Jun 18 2018: <https://github.com/pycroscopy/pycroscopy/pull/177>`_
------------------------------------------------------------------------
* Thanks to `@Liambcollins <https://github.com/Liambcollins>`_ for bug-fixes to ``GLineTranslator``

`Jun 15 2018: <https://github.com/pycroscopy/pycroscopy/pull/175>`_
------------------------------------------------------------------------
* Thanks to `@ramav87 <https://github.com/ramav87>`_ for bug-fixes in BEPS related translators and notebooks

`Jun 14 2018: <https://github.com/pycroscopy/pycroscopy/pull/173>`_
------------------------------------------------------------------------
* Thanks to `@ealopez <https://github.com/ealopez>`_ for adding AFM simulations
* Thanks to `@nmosto <https://github.com/nmosto>`_ for guides for python novices

`Jun 13 2018: <https://github.com/pycroscopy/pycroscopy/pull/172>`_
------------------------------------------------------------------------
* Thanks to `@str-eat <https://github.com/str-eat>`_ for implementing a PycroDataset to csv exporter

`Jun 04 2018: <https://github.com/pycroscopy/pycroscopy/pull/166>`_
------------------------------------------------------------------------
* Thanks to `@donpatrice <https://github.com/donpatrice>`_ for fixing a UTF8 isue with the ``NanonisTranslator``

`Jun 01 2018: <https://github.com/pycroscopy/pycroscopy/pull/162>`_
------------------------------------------------------------------------
* First skeleton ``GwyddionTranslator`` being worked on by `@str-eat <https://github.com/str-eat>`_
* Added guidelines for contributing code

`May 31 2018: <https://github.com/pycroscopy/pycroscopy/pull/162>`_
------------------------------------------------------------------------
* All ``Translators`` now use absolute paths
* Improved examples and documentation

`May 30 2018: <https://github.com/pycroscopy/pycroscopy/pull/162>`_
------------------------------------------------------------------------
* Thanks to `@carlodri <https://github.com/carlodri>`_ for donating his utility to read Gwyddion Simple File (gsf) reader
* Added ``gwyfile`` to the requirements of pycroscopy
* ``NumpyTranslator`` now accepts extra datasets and keyword arguments that will be passed on to ``hdf_utils.write_main_dataset()``

`May 26 2018: <https://github.com/pycroscopy/pycroscopy/pull/158>`_
------------------------------------------------------------------------
* Implemented a general function for reading sections of binary files
* First version of the ``BrukerTranslator`` for translating Bruker Icon and other AFM files

`May 03 2018: <https://github.com/pycroscopy/pycroscopy/pull/153>`_
------------------------------------------------------------------------
* ``plot_utils.plot_map()`` now accepts the extent or the tick values
* Fixed bug in ``hdf_utils.write_reduced_spec_dsets()`` and ``analysis.BESHOFitter``
* General improvements to the ``analysis.Fitter`` class
* Documentation updates

`May 02 2018: <https://github.com/pycroscopy/pycroscopy/pull/151>`_
------------------------------------------------------------------------
* Fixed bug in ``svd_rebuild()``

`May 01 2018: <https://github.com/pycroscopy/pycroscopy/pull/149>`_
------------------------------------------------------------------------
* Minor corrections to documentation formatting
* ``pycroscopy.hdf_utils.get_auxillary_datasets()`` renamed to ``pycroscopy.hdf_utils.get_auxiliary_datasets()``
* Example on parallel computing rewritten to focus on ``pycroscopy.parallel_compute()``
* Added ``setUp()`` and ``tearDown()`` to unit testing classes for ``hdf_utils`` and ``PycroDataset``
* Fixed bug in the sorting capability of ``pycroscopy.hdf_utils.reshape_to_n_dims()``
* Added logo to website

`Apr 29 2018 2: <https://github.com/pycroscopy/pycroscopy/pull/148>`_
------------------------------------------------------------------------
* Centralized verification of slice dictionary in ``pycroscopy.PycroDataset``
* The ``slice_dict`` kwarg in ``pycroscopy.PycroDataset.slice()`` now the first required argument
* Lots of minor formatting changes to examples.
* Removed jupyter notebooks from which the examples were generated.

`Apr 29 2018 1: <https://github.com/pycroscopy/pycroscopy/pull/147>`_
------------------------------------------------------------------------
* Fixed errors in broken examples
* Replaced example BE datasets with ones where the central datasets now have ``quantity`` and ``units`` attributes to make them ``Main datasets``
* Replaced example STS dataset with a zip file which will download a lot faster for examples. Corrected the example on NumpyTranslator

`Apr 28 2018 2: <https://github.com/pycroscopy/pycroscopy/pull/146>`_
-----------------------------------------------------------------------
* Fixed unit tests for python 2. assertWarns() only applied to python 3 now
* Added ``from future`` import statement to all modules in ``pycroscopy.core``

`Apr 28 2018 1: <https://github.com/pycroscopy/pycroscopy/pull/143>`_
-----------------------------------------------------------------------
**(Massive) merge of (skunkworks) branch unity_dev into master``**

* Added unit tests for all (feasible) modules in ``pycroscopy.core``
* Added examples for every single function or class in ``pycroscopy.core`` (**10** cookbooks in total!)
* Added a primer to h5py and HDF5
* Added document with instructions on converting unit tests to examples of functions.
* Added web page with links to external tutorials
* Added web page describing contents of package, organization,
* Added web page with FAQs
* Moved a simplified (non ptychography version of) ImageTranslator to ``pycroscopy.core``
* Package reconfigured to use ``pytests`` instead of ``Nose``
* Converted last few ``assert`` statements into descriptive ``Errors``
* Legacy HDF writing classes and functions **deprecated now** and will be removed in a future release:

    * ``hdf_writer`` and ``virtual_data`` modules moved out of ``pycroscopy.core.io`` and back into ``pycroscopy.io``.
    * Moved functions in ``pycroscopy.write_utils`` using above deprecated classes into ``pycroscopy.io.write_utils``. These functions are also deprecated
    * ``pycroscopy.translators.BEODFTranslator``, ``pycroscopy.analysis.BESHOFitter``, and ``pycroscopy.BELoopFitter``,
      ``pycroscopy.processing.SignalFilter``, ``pycroscopy.translators.GIVTranslator``, ``pycroscopy.analysis.GIVBayesian``, ``pycroscopy.processing.gmode_utils``, etc.
      now do **not** use deprecated classes as proof that even the most complex classes can easily be transitioned to using
      functions in ``pycroscopy.core.io.hdf_utils`` and ``pycroscopy.core.io.write_utils``
    * Unit tests for modules in ``pycroscopy.core.io`` rewritten to not use deprecated functions or classes.
    * Deprecated classes only being used in translators, two analyses modules and two process modules
    * Removed old examples and tutorials, especially on deprecated classes
* Upgrades to ``pycroscopy.Process``:

    * ``pycroscopy.Process`` now has a new function - ``test()`` that allows much easier in-place testing of processes before applying to the entire dataset
    * `pycroscopy.processing.Cluster``, ``pycroscopy.processing.Decomposition``, ``pycroscopy.processing.SVD``, ``pycroscopy.processing.SignalFilter``,
      ``pycroscopy.processing.GIVBayesian`` all implement the new ``test()`` functionality - return results as correct N-dimensional datasets in expected datatypes
    * ``pycroscopy.processing.Cluster``, ``pycroscopy.processing.Decomposition`` now use a user-configured sklearn objects as inputs instead of creating an estimator object
    * ``SVD``, ``Cluster``, ``Decomposition`` now correctly write results as ``Main`` datasets
* More robust ``pycroscopy.gmode_utils`` functions
* Updates to ``pycroscopy.plot_utils``:

    * ``plot_complex_loop_stack`` merged into ``plot_complex_spectra()``
    * new function that provides best row / column configuration for (identical) subplots: ``get_plot_grid_size()``
    * moved clustering related utilities into ``pycroscopy.viz.cluster_utils`` <-- significantly revised many functions in there
    * ``plot_map_stack()`` accepts x, y labels. ``plot_map()`` accepts X and Y vectors instead of sizes for more granular control
    * All compound functions now pass kwargs to underlying functions wherever possible

* Updates to ``pycroscopy.write_utils``:

    * ``pycroscopy.write_utils.AncillaryDescriptor`` and ``pycroscopy.jupyter_utils.VizDimension`` merged and significantly simplified to ``pycroscopy.write_utils.Dimension``
    * Swapped all usages of ``AncillaryDescriptor`` with ``Dimension`` in the entire package
    * More robust handling of attributes with numpy strings as values
    * Added new functions to simplify building of matrices for ancillary datasets - ``build_ind_val_matrices()``

* Updates to ``pycroscopy.hdf_utils``:

    * Functions updated to using the new ``Dimension`` objects
    * Added a few new functions such as ``write_book_keeping_attrs()``, ``create_indexed_group()``, ``create_results_group()``
    * ``write_main_dataset()`` can now write empty datasets, use different prefixes for ancillary dataset names, etc.
    * Centralized the writing of book-keeping attributes to ``write_book_keeping_attrs()``
    * generalized certain functions such as ``copy_attributes``, ``write_simple_attributes()`` so that they can be applied to any HDF5 object
    * ``write_main_dataset()`` and ``create_empty_dataset()`` now validate the ``dtype`` correctly
    * ``print_tree()`` now prints cleaner versions of the tree, only ``Main datasets`` if requested.
    * ``write_book_keeping_attrs()`` now writes the operating system version and pycroscopy version in addition to the timestamp and machine ID
    * Region references functions such as ``copy_region_refs()`` now more robust
* bug fixes to BE translation, visualization, plotting


`Mar 27 2018: <https://github.com/pycroscopy/pycroscopy/pull/138>`_
-------------------------------------------------------------------
* Small changes to make pycroscopy available on ``Conda forge``. Thanks to @carlodri !
* ``pycroscopy.translators.NanonisTranslator`` added to translate Nanonis data files

`Mar 2 2018: <https://github.com/pycroscopy/pycroscopy/pull/133>`_
-------------------------------------------------------------------
* Fixed decode error in ``pycroscopy.translators.IgorTranslator`` relevant for new versions of Asylum Research microscope software versions

`Mar 3 2018: <https://github.com/pycroscopy/pycroscopy/pull/131>`_ (on ``unity_dev`` and not on ``master``)
-------------------------------------------------------------------------------------------------------------
* ``pycroscopy.plot_utils.plot_map`` now accepts X and Y vectors
* Lots of small bug fixes
* More checks for more robust code in ``pycroscopy.core``
* New handy function - ``pycroscopy.hdf_utils.get_region()`` - directly returns the referenced data as a numpy array
* Added two new examples on ``pycroscopy.io_utils`` and ``pycroscopy.dtype_utils``

`Feb 18 2018: <https://github.com/pycroscopy/pycroscopy/pull/131>`_ (on ``unity_dev`` and not on ``master``)
-------------------------------------------------------------------------------------------------------------
**Massive restructuring and overhaul of code:**

* Renamed ``pycroscopy.ioHDF`` to ``pycroscopy.HDFWriter``
* Renamed ``pycroscopy.MicroDataset`` and `pycroscopy.MicroDataGroup`` to ``pycroscopy.VirtualDataset`` and ``pycroscopy.VirtualGroup``
* Data type manipulation functions moved out of ``pycroscopy.io_utils`` into ``pycroscopy.dtype_utils``
* Moved core foundational / science agnostic / engineering elements of pycroscopy into a new subpackage - ``pycroscopy.core``.
  Rule for move - nothing in ``.core`` should import anything out of ``.core``. This may be spun off as its own package at a later stage if deemed appropriate.
  Contents of ``pycroscopy.core``:

    * ``core.io`` - ``HDFWriter``, ``VirtualData``, ``hdf_utils``, ``write_utils``, ``io_utils``, ``dtype_utils``, ``Translator``, ``NumpyTranslator``
    * ``core.processing`` - ``Process``, ``parallel_compute()``
    * ``core.viz`` - ``plot_utils``, ``jupyter_utils``
* Started adding a lot of type and value checks for safer and more robust file reading/writing. Expect a lot of descriptive
  Exceptions that will help in identifying problems easier and sooner.
* Implemented modular and standalone functions in ``pycroscopy.hdf_utils`` that form a (much simpler and more robust) feature-equivalent alternative to
  ``pycroscopy.HDFWriter`` + ``pyroscopy.VirtualData``. ``pycroscopy.HDFWriter`` + ``pyroscopy.VirtualData`` **will be phased out in the near future**.

    * First implementation of what may be one of the most popular and important functions - ``pycroscopy.hdf_utils.write_main_dataset()`` -

        * Thoroughly checks and validates all inputs. Only if these pass,
        * Writes the a dataset containing the central data
        * Creates / reuses ``ancillary datasets``
        * links ``Ancillary datasets`` to create a ``Main dataset``
        * writes ``quantity`` and ``units`` attributes - **now mandatory**
        * Also writes any other attributes

    * Other notable functions include ``write_simple_attrs()``, ``write_region_references``, ``write_ind_val_dsets()``
* ``pycroscopy.NumpyTranslator`` now simply calls ``pycroscopy.hdf_utils.write_main_dataset()``

    * ``pycroscopy.Translator.simple_write()`` removed. Translators can extend ``NumpyTranslator`` instead.

* Added first batch of unit tests for modules in ``pycroscopy.core``.
* More robust ``pycroscopy.parallel_compute()`` via type checking
* Added a new class called ``pycroscopy.AuxillaryDescriptor`` to describe Position and spectroscopic dimensions.
  All major functions like ``write_main_dataset()`` and ``write_ind_val_dsets()`` to use this descriptor

`Jan 16 2018: <https://github.com/pycroscopy/pycroscopy/pull/129>`_ (on ``unity_dev`` and not on ``master``)
-------------------------------------------------------------------------------------------------------------
* ``pycroscopy.processing.Cluster`` and ``pycroscopy.processing.Decomposition`` now extend ``pycroscopy.Process``
* More robust HDF functions for checking the existence of prior results groups.
* Fixed important bugs for better python2 compatibility (HDF I/O, plotting, etc.)
* More FFT signal filtering functions
* Several bug fixes to ``pycroscopy.viz.plot_utils``
* Simplifications to the ``image cleaning`` and ``GIV`` notebooks to use the new capabilities of ``pycroscopy.processing.SVD``, ``pycroscopy.processing.Cluster``

`Dec 7 2017: <https://github.com/pycroscopy/pycroscopy/pull/127>`_
---------------------------------------------------------------------
* New function (``visualize()``) added to ``pycroscopy.PycroDataset`` to facilitate interactive visualization of data in for any dataset (< 4 dimensions)
* Significantly more customizable plotting functions in ``pycroscopy.plot_utils``
* Improved ``pycroscopy.Process`` that provides the framework for:

  * checking for prior instances of a process run on the same dataset with the same parameters
  * resuming an aborted process / computation
* Reorganized ``doSVD()`` into a new Process called ``pycroscopy.processing.SVD``  to take advantage of above advancements.
  
  * The same changes will be rolled out to ``pycroscopy.processing.Cluster`` and ``pycroscopy.processing.Decomposition`` soon

`Nov 17 2017: <https://github.com/pycroscopy/pycroscopy/pull/126>`_
---------------------------------------------------------------------
* Significant improvements and bug fixes to Bayesian Inference for G-mode IV.

`Nov 11 2017: <https://github.com/pycroscopy/pycroscopy/pull/125>`_
---------------------------------------------------------------------
* New robust class for Bayesian Inference on G-mode IV data - ``pycroscopy.processing.GIVBayesian``
* Utilities for reading files from Nanois controllers
* New robust class for FFT Signal Filtering on any data including G-mode - ``pycroscopy.processing.SignalFilter``
* FFT filtering rewritten and simplified to use objects

`Oct 9 2017: <https://github.com/pycroscopy/pycroscopy/pull/124>`_
---------------------------------------------------------------------
* Added ``pycroscopy.PycroDataset`` - a class that simplifies handling, reshaping, and interpretation of ``Main`` datasets.

`Sep 6 2017: <https://github.com/pycroscopy/pycroscopy/pull/123>`_
---------------------------------------------------------------------
* Added ``pycroscopy.Process`` - New class that provides a framework for data processing in Pycroscopy.

`Sep 5 2017: <https://github.com/pycroscopy/pycroscopy/pull/122>`_
---------------------------------------------------------------------
* Improved the example on parallel computing

`Aug 31 2017: <https://github.com/pycroscopy/pycroscopy/pull/118>`_
---------------------------------------------------------------------
* New plot function - ``single_img_cbar_plot()`` (now merged into ``plot_map()``) for nicer 2D image plots with color-bars.

`Aug 29 2017: <https://github.com/pycroscopy/pycroscopy/pull/117>`_
---------------------------------------------------------------------
* Improvements to Bayesian Inference on G-mode IV data including resistance compensation.


