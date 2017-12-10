.. contents::

v 1.0 goals
-----------
1. test utils - 2+ weeks
2. DONE - good utilities for interrogating data - pycro data
3. partially done - good documentation for both users and developers

  * Need more on dealing with data + plot_utils tour
  * (for developers) explaining what is where and why + io utils + hdf utils tour etc.
4. mostly done - generic visualizer
5. settle on a structure for process and analysis - moderate ~ 1 day
 
  * Model needs to catch up with Process
6. mostly done - good utils for generating publishable plots - easy ~ 1 day
7. Reorganize package - promote / demote lesser used utilites to processes / analyses. 

Short-term goals
--------------------
* Multi-node compute capability
* More documentation to help users / developers + PAPER
* Cleaned versions of the main modules (Analysis pending) + enough documentation for users and developers

Documentation
-------------
* Upload clean exports of paper notebooks - Stephen and Chris
*	Include examples in documentation
* Links to references for all functions and methods used in our workflows.

Fundamental tutorials on how to use pycroscopy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* A tour of what is where and why
* A tour of the hdf_utils functions used for writing h5 files since these functions need data to show / explain them.
  
  * chunking the main dataset
* A tour of the io_utils functions since these functions need data to show / explain them.
* A tour of plot_utils
* pycroscopy pacakge organization - a short writeup on what is where and differences between the process / analyis submodules
* How to write your own analysis class based on the (to-be simplified) Model class
* Links to tutorials on how to use pycharm, Git, 

Rama's (older and more applied / specific) tutorial goals
~~~~~~~~~~~~~~~~~~~~
1. Open a translated and fitted FORC-PFM file, and plot the SHO Fit from cycle k corresponding to voltage p, along with the raw spectrogram for that location and the SHO guess. Plot both real and imaginary, and do so for both on and off-field.
2. Continuing above, determine the average of the quality factor coming from cycles 1,3,4 for spatial points stored in vector b for the on-field part for a predetermined voltage range given by endpoints [e,f]. Compare the results with the SHO guess and fit for the quality factor.
3. After opening a h5 file containing results from a relaxation experiment, plot the response at a particular point and voltage, run exponential fitting and then store the results of the fit in the same h5 file using iohdf and/or numpy translators.
4. Take a FORC IV ESM dataset and break it up into forward and reverse branches, along with positive and negative branches. Do correlation analysis between PFM and IV for different branches and store the results in the file, and readily access them for plotting again.
5. A guide to using the model fitter for parallel fitting of numpy array-style datasets. This one can be merged with number 

New features
------------
Core development
~~~~~~~~~~~~~~~~
* EVERY process tool should implement two new features:
  
  1. Check if the same process has been performed with the same paramters. When initializing the process, throw an exception. This is better than checking in the notebook stage.
  2. (Gracefully) Abort and resume processing.
  
* consolidate _get_component_slice used in Cluster with duplicate in svd_utils
* Reogranize processing and analysis - promote / demote classes etc.
* Legacy processes **MUST** extend Process:

  * Image Windowing
  * Image Cleaning
  * As time permits, ensure that these can resume processing
  * All these MUST implement the check for previous computations at the very least
  * As time permits, ensure that these can resume processing
  
* Absorb functionality from Process into Model
* Bayesian GIV should actually be an analysis <-- depends on above
* multi-node computing capability in parallel_compute
* Demystify analyis / optimize. Use parallel_compute instead of optimize and guess_methods and fit_methods
* Consistency in the naming of and placement of attributes (chan or meas group) in all translators - Some put attributes in the measurement level, some in the channel level! hyperspy appears to create datagroups solely for the purpose of organizing metadata in a tree structure! 
* Consider developing a generic curve fitting class a la `hyperspy <http://nbviewer.jupyter.org/github/hyperspy/hyperspy-demos/blob/master/Fitting_tutorial.ipynb>`_
* Improve visualization of file contents in print_tree() like hyperspy's `metadata <http://hyperspy.org/hyperspy-doc/current/user_guide/metadata_structure.html>`_

GUI
~~~~~~~~~~~
* Make the generic interactive visualizer for 3 and 4D float numpy arrays ROBUST

  * Allow slicing at the pycrodataset level to handle > 4D datasets - 20 mins
  * Need to handle appropriate reference values for the tick marks in 2D plots - 20 mins
  * Handle situation when only one position and one spectral axis are present. - low priority - 20 mins
* TRULY Generic visualizer in plot.lly / dash? that can use the PycroDataset class
*	Switch to using plot.ly and dash for interactive elements
*	Possibly use MayaVi for 3d plotting

Plot Utils
~~~~~~~~~
* move plot_image_cleaning_results to a application specific module
* move save_fig_filebox_button and export_fig_data to jupyter_utils
* ensure most of these functions result in publication-ready plots (good proportions, font sizes, etc.)
* plot_map 

  1. allow the tick labels to be specified instead of just the x_size and y_size. 

* plot_loops
 
  1. Legend at the bottom
  
* plot_map_stack:

  1. Add ability to manually specify x and y tick labels - see plot_cluster_results_together for inspiration
  2. See all other changes that were made for the image cleaning paper

* plot_cluster_results_together

  1. Use plot_map and its cleaner color bar option
  2. Option to use a color bar for the centroids instead of a legend - especially if number of clusters > 7
  3. See G-mode IV paper to see other changes

* plot_cluster_results_separate
  
  1. Use same guidelines as above

* plot_cluster_dendrogram - this function has not worked recently to my knowledge. Fortunately, it is not one of the more popular functions so it gets low priority for now. Use inspiration from image cleaning paper

* plot_histograms - not used frequently. Can be ignored for this pass

External user contributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Li Xin classification code 
* Ondrej Dyck’s atom finding code – written well but needs to work on images with different kinds of atoms
* Nina Wisinger’s processing code (Tselev) – in progress
* Sabine Neumeyer's cKPFM code
* Iaroslav Gaponenko's Distort correct code from - https://github.com/paruch-group/distortcorrect.
* Port everything from IFIM Matlab -> Python translation exercises
* Other workflows/functions that already exist as scripts or notebooks

Formatting changes
------------------
*	Fix remaining PEP8 problems
*	Ensure code and documentation is standardized
*	Classes and major Functions should check to see if the results already exist

Notebooks
---------
*	Investigate using JupyterLab

Testing
-------
*	Write test code
*	Unit tests for simple functions
*	Longer tests using data (real or generated) for the workflow tests
*  measure coverage using codecov.io and codecov package

Software Engineering
--------------------
* Consider releasing bug fixes (to onsite CNMS users) via git instead of rapid pypi releases 
   * example release steps (incl. git tagging): https://github.com/cesium-ml/cesium/blob/master/RELEASE.txt
* Use https://docs.pytest.org/en/latest/ instead of nose (nose is no longer maintained)
* Add requirements.txt
* Consider facilitating conda installation in addition to pypi

Scaling to clusters
-------------------
We have two kinds of large computational jobs and one kind of large I/O job:

* I/O - reading and writing large amounts of data
   * Dask and MPI are compatible. Spark is probably not
* Computation
   1. Machine learning and Statistics
   
      1.1. Use custom algorithms developed for BEAM
         * Advantage - Optimized (and tested) for various HPC environments
         * Disadvantages:
            * Need to integarate non-python code
            * We only have a handful of these. NOT future compatible            
      1.2. OR continue using a single FAT node for these jobs
         * Advantages:
            * No optimization required
            * Continue using the same scikit learn packages
         * Disadvantage - Is not optimized for HPC
       1.3. OR use pbdR / write pbdPy (wrappers around pbdR)
         * Advantages:
            * Already optimized / mature project
            * In-house project (good support) 
         * Disadvantages:
            * Dependant on pbdR for implementing new algorithms
            
   2. Parallel parametric search - analyze subpackage and some user defined functions in processing. Can be extended using:
   
      * Dask - An inplace replacement of multiprocessing will work on laptops and clusters. More elegant and easier to write and maintain compared to MPI at the cost of efficiency
         * simple dask netcdf example: http://matthewrocklin.com/blog/work/2016/02/26/dask-distributed-part-3
      * MPI - Need alternatives to Optimize / Process classes - Better efficiency but a pain to implement
