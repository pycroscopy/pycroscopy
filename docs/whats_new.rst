What's New
----------

`0.59.3 <https://github.com/pycroscopy/pycroscopy/pull/127>`_ - 12/07/2017
~~~~~~~~
* `Interactive (Jupyter) visualization <https://github.com/pycroscopy/pycroscopy/blob/master/pycroscopy/viz/jupyter_utils.py>`_ for any dataset (< 4 dimensions)

  * Handy shortcut to this function added to `Pycrodataset.visualize() <https://github.com/pycroscopy/pycroscopy/blob/master/pycroscopy/io/pycro_data.py>`_
* Significantly more customizable plotting functions - `plot_utils <https://github.com/pycroscopy/pycroscopy/blob/master/pycroscopy/viz/plot_utils.py>`_
* Improved `Process class <https://github.com/pycroscopy/pycroscopy/blob/master/pycroscopy/processing/process.py>`_ that provides the framework for:

  * checking for prior instances of a process run on the same dataset with the same parameters
  * resuming an aborted process / computation
* Reorganized doSVD() into a new Process called `SVD <https://github.com/pycroscopy/pycroscopy/blob/master/pycroscopy/processing/svd_utils.py>`_ to take advantage of above advancements. 
  
  * The same changes will be rolled out to `Cluster <https://github.com/pycroscopy/pycroscopy/blob/master/pycroscopy/processing/cluster.py>`_ and `Decomposition <https://github.com/pycroscopy/pycroscopy/blob/master/pycroscopy/processing/decomposition.py>`_ soon

`0.59.2 <https://github.com/pycroscopy/pycroscopy/pull/126>`_ - 11/17/ 2017
~~~~~~~~
* Significant improvements and bug fixes to `Bayesian Inference <https://github.com/pycroscopy/pycroscopy/blob/master/pycroscopy/processing/giv_bayesian.py>`_ for `G-mode IV <https://github.com/pycroscopy/pycroscopy/blob/master/pycroscopy/processing/giv_utils.py>`_.

`0.59.2 <https://github.com/pycroscopy/pycroscopy/pull/125>`_ - 11/11/ 2017
~~~~~~~~
* New robust class for `Bayesian Inference <https://github.com/pycroscopy/pycroscopy/blob/master/pycroscopy/processing/giv_bayesian.py>`_ on G-mode IV data
* Significant bug fixes for `Bayesian Inference <https://github.com/pycroscopy/pycroscopy/blob/master/pycroscopy/processing/giv_utils.py>`_.
* Utilies for reading files from `Nanois controllers <https://github.com/pycroscopy/pycroscopy/tree/master/pycroscopy/io/translators/df_utils/nanonispy>`_
* New robust class for `FFT Signal Filtering <https://github.com/pycroscopy/pycroscopy/blob/master/pycroscopy/processing/signal_filter.py>`_ on any data including G-mode
* FFT filtering rewritten as easy to use `objects <https://github.com/pycroscopy/pycroscopy/blob/master/pycroscopy/processing/fft.py>`_.

`0.59.2 <https://github.com/pycroscopy/pycroscopy/pull/126>`_ - 10/09/ 2017
~~~~~~~~
