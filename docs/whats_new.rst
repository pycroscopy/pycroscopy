What's New
==========

v 0.59.2
--------

`Dec 7 2017: <https://github.com/pycroscopy/pycroscopy/pull/127>`_
~~~~~~~~
* `Interactive (Jupyter) visualization <https://github.com/pycroscopy/pycroscopy/blob/master/pycroscopy/viz/jupyter_utils.py>`_ for any dataset (< 4 dimensions)

  * Handy shortcut to this function added to `Pycrodataset.visualize() <https://github.com/pycroscopy/pycroscopy/blob/master/pycroscopy/io/pycro_data.py>`_
* Significantly more customizable plotting functions - `plot_utils <https://github.com/pycroscopy/pycroscopy/blob/master/pycroscopy/viz/plot_utils.py>`_
* Improved `Process class <https://github.com/pycroscopy/pycroscopy/blob/master/pycroscopy/processing/process.py>`_ that provides the framework for:

  * checking for prior instances of a process run on the same dataset with the same parameters
  * resuming an aborted process / computation
* Reorganized doSVD() into a new Process called `SVD <https://github.com/pycroscopy/pycroscopy/blob/master/pycroscopy/processing/svd_utils.py>`_ to take advantage of above advancements. 
  
  * The same changes will be rolled out to `Cluster <https://github.com/pycroscopy/pycroscopy/blob/master/pycroscopy/processing/cluster.py>`_ and `Decomposition <https://github.com/pycroscopy/pycroscopy/blob/master/pycroscopy/processing/decomposition.py>`_ soon

`Nov 17 2017: <https://github.com/pycroscopy/pycroscopy/pull/126>`_
~~~~~~~~
* Significant improvements and bug fixes to `Bayesian Inference <https://github.com/pycroscopy/pycroscopy/blob/master/pycroscopy/processing/giv_bayesian.py>`_ for `G-mode IV <https://github.com/pycroscopy/pycroscopy/blob/master/pycroscopy/processing/giv_utils.py>`_.

`Nov 11 2017: <https://github.com/pycroscopy/pycroscopy/pull/125>`_
~~~~~~~~
* New robust class for `Bayesian Inference <https://github.com/pycroscopy/pycroscopy/blob/master/pycroscopy/processing/giv_bayesian.py>`_ on G-mode IV data
* Significant bug fixes for `Bayesian Inference <https://github.com/pycroscopy/pycroscopy/blob/master/pycroscopy/processing/giv_utils.py>`_.
* Utilies for reading files from `Nanois controllers <https://github.com/pycroscopy/pycroscopy/tree/master/pycroscopy/io/translators/df_utils/nanonispy>`_
* New robust class for `FFT Signal Filtering <https://github.com/pycroscopy/pycroscopy/blob/master/pycroscopy/processing/signal_filter.py>`_ on any data including G-mode
* FFT filtering rewritten as easy to use `objects <https://github.com/pycroscopy/pycroscopy/blob/master/pycroscopy/processing/fft.py>`_.

`Oct 9 2017: <https://github.com/pycroscopy/pycroscopy/pull/124>`_
~~~~~~~~
* New `PycroDataset class <https://github.com/pycroscopy/pycroscopy/blob/master/pycroscopy/io/pycro_data.py>`_ that simplifies handling, reshaping, and interpretation of **Main** datasets.

`Sep 6 2017: <https://github.com/pycroscopy/pycroscopy/pull/123>`_
~~~~~~~~
* New `Process class <https://github.com/pycroscopy/pycroscopy/blob/master/pycroscopy/processing/process.py>`_ that provides a framework for data processing in Pycroscopy.

`Sep 5 2017: <https://github.com/pycroscopy/pycroscopy/pull/122>`_
~~~~~~~~
* Improved the example on `parallel computing <https://pycroscopy.github.io/pycroscopy/auto_examples/dev_tutorials/plot_tutorial_04_parallel_computing.html#sphx-glr-auto-examples-dev-tutorials-plot-tutorial-04-parallel-computing-py>`_

`Sep 5 2017: <https://github.com/pycroscopy/pycroscopy/pull/121>`_
~~~~~~~~
* Added a new tutorial on `parallel computing <https://pycroscopy.github.io/pycroscopy/auto_examples/dev_tutorials/plot_tutorial_04_parallel_computing.html#sphx-glr-auto-examples-dev-tutorials-plot-tutorial-04-parallel-computing-py>`_

`Aug 31 2017: <https://github.com/pycroscopy/pycroscopy/pull/118>`_
~~~~~~~~
* New plot function - single_img_cbar_plot (now merged into plot_map) for nicr `2D image plots <https://github.com/pycroscopy/pycroscopy/blob/master/pycroscopy/viz/plot_utils.py>`_ with colobars.

`Aug 29 2017: <https://github.com/pycroscopy/pycroscopy/pull/117>`_
~~~~~~~~
* Improvements to `Bayesian Inference <https://github.com/pycroscopy/pycroscopy/blob/master/pycroscopy/processing/giv_utils.py>`_ on G-mode IV data including resistance compensation.


