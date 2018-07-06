.. contents::

Goals
-------

Immediate
~~~~~~~~~
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
  * Park Systems - DaeHee Seol and Prof. Yunseok Kim may help here
  * JPK - No data available. `GitHub project <https://github.com/rosscarter3/JPKforceparse>`_ available for translation from Ross Carter
  * Niche data formats:

    * NT-MDT - Data available. translator pending.
    * PiFM - J. Kong from Ginger Group will write a translator upon her visit to CNMS in summer of 2018.
    * Anasys - import `anasyspythontools <https://github.com/AnasysInstruments/anasys-python-tools>`_ - comes with test data

      * This package does `NOT <https://pypi.org/search/?q=anasyspythontools>`_ have a `pypi installer <https://github.com/AnasysInstruments/anasys-python-tools/issues/2>`_
      * This package does not look like it is finished.

* Move ``.core`` out of pycroscopy

  * pycroscopy should import core and alias it for now
* Embed presentation into "About"
* Start writing the journal paper!
* Write plugins to export to pycroscopy HDF5 for ``ImageJ``, and possibly ``Gwyddion``. There are HDF5 plugins already available for ImageJ.

Short-term - by Jul 1 2018
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Look into making notebooks for workshops available through `mybinder <https://mybinder.org>`_
* development of community standards for AFM, STM, STEM, etc.
* Chris - ``Image Processing`` must be a subclass of ``Process`` and implement resuming of computation and checking for old (both already handled quite well in Process itself) - here only because it is used and requested frequently + should not be difficult to restructure.
* Clear and obvious examples showing what pycroscopy actually does for people

  * Image cleaning - Chris
  * Signal Filtering - Suhas
  * Two more examples
* Upload clean exports of paper notebooks + add notebooks for new papers + add new papers (Sabine + Liam)
* Update change-log with version numbers / releases instead of pull numbers
* unit tests for basic data science (``Cluster``, ``SVD``, ``Decomposition``)
* Add requirements.txt
* Examples within docs for popular functions
* Revisit and address as many pending TODOs as possible

Medium-term - by Aug 1 2018
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Technical poster for pycroscopy
* Explore Azure Notebooks for live tutorials
* Notebooks need to be updated to require pycroscopy version
* Itk for visualization - https://github.com/InsightSoftwareConsortium/itk-jupyter-widgets
* Use / integrate hyperspy with pycroscopy if it adds enough functionality

Long-term
~~~~~~~~~~
* A sister package with the base labview subvis that enable writing pycroscopy compatible hdf5 files. The actual acquisition can be ignored.
* Consider developing a generic curve fitting class a la `hyperspy <http://nbviewer.jupyter.org/github/hyperspy/hyperspy-demos/blob/master/Fitting_tutorial.ipynb>`_
* Chris - Demystify analyis / optimize. Use parallel_compute instead of optimize and guess_methods and fit_methods
* Consistency in the naming of and placement of attributes (chan or meas group) in all translators - Some put attributes in the measurement level, some in the channel level! hyperspy appears to create datagroups solely for the purpose of organizing metadata in a tree structure!
* Batch fitting - need to consider notebooks for batch processing of BELINE and other BE datasets. This needs some thought, but a basic visualizer that allows selection of a file from a list and plotting of the essential graphs is needed.
* Profile code to see where things are slow

Back-burner
~~~~~~~~~~~~
* Look into ``Tasmanian`` (mainly modeling) - Miroslav Stoyanov

Reogranization
---------------

1.  Reorganize code - This is perhaps the last opportunity for major restructuring and renaming.

  * Subpackages within processing: statistics, image, signal, misc
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