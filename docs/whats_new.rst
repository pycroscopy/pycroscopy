What's New
==========

`Dec 7 2017: <https://github.com/pycroscopy/pycroscopy/pull/127>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* New function (``visualize()``) added to ``pycroscopy.PycroDataset`` to facilitate interactive visualization of data in for any dataset (< 4 dimensions)
* Significantly more customizable plotting functions in ``pycroscopy.plot_utils``
* Improved ``pycroscopy.Process`` that provides the framework for:

  * checking for prior instances of a process run on the same dataset with the same parameters
  * resuming an aborted process / computation
* Reorganized ``doSVD()`` into a new Process called ``pycroscopy.processing.SVD``  to take advantage of above advancements.
  
  * The same changes will be rolled out to ``pycroscopy.processing.Cluster`` and ``pycroscopy.processing.Decomposition`` soon

`Nov 17 2017: <https://github.com/pycroscopy/pycroscopy/pull/126>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Significant improvements and bug fixes to Bayesian Inference for G-mode IV.

`Nov 11 2017: <https://github.com/pycroscopy/pycroscopy/pull/125>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* New robust class for Bayesian Inference on G-mode IV data - ``pycroscopy.processing.GIVBayesian``
* Utilities for reading files from Nanois controllers
* New robust class for FFT Signal Filtering on any data including G-mode - ``pycroscopy.processing.SignalFilter``
* FFT filtering rewritten and simplified to use objects

`Oct 9 2017: <https://github.com/pycroscopy/pycroscopy/pull/124>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Added ``pycroscopy.PycroDataset`` - a class that simplifies handling, reshaping, and interpretation of ``Main`` datasets.

`Sep 6 2017: <https://github.com/pycroscopy/pycroscopy/pull/123>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Added ``pycroscopy.Process`` - New class that provides a framework for data processing in Pycroscopy.

`Sep 5 2017: <https://github.com/pycroscopy/pycroscopy/pull/122>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Improved the example on parallel computing

`Sep 5 2017: <https://github.com/pycroscopy/pycroscopy/pull/121>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Added a new tutorial on parallel computing

`Aug 31 2017: <https://github.com/pycroscopy/pycroscopy/pull/118>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* New plot function - ``single_img_cbar_plot()`` (now merged into ``plot_map()``) for nicer 2D image plots with color-bars.

`Aug 29 2017: <https://github.com/pycroscopy/pycroscopy/pull/117>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Improvements to Bayesian Inference on G-mode IV data including resistance compensation.


