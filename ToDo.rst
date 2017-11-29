.. contents::

Short-term goals
--------------------
* More documentation to help users / developers + PAPER
* Cleaned versions of the main modules (Analysis pending) + enough documentation for users and developers
* Multi-node compute capability

Documentation
-------------
*	Include examples in documentation
* Links to references for all functions and methods used in our workflows.

Fundamental tutorials on how to use pycroscopy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
* Generic interactive visualizer for 3 and 4D float numpy arrays.

  * No need to handle h5py datasets, compound datasets, complex datasets etc.
  * Add features time permitting.
* EVERY process tool should implement two new features:
  
  1. Check if the same process has been performed with the same paramters. When initializing the process, throw an exception. This is better than checking in the notebook stage.
  2. (Gracefully) Abort and resume processing.
* Clean up Cluser results plotting
* Consider implementing doSVD as a Process. Maybe the Decomposition and Cluster classes could extend Process?
* Simplify and demystify analyis / optimize. Use parallel_compute instead of optimize and gues_methods and fit_methods
* multi-node computing capability in parallel_compute
* Data Generators
* Consistency in the naming of and placement of attributes (chan or meas group) in all translators

GUI
~~~~~~~~~~~

* Need to be able to run a visualizer even on sliced data. What would this object be? (sliced Main data + vectors for each axis + axis labels ....). Perhaps the slice() function should return this object instead of a numpy array? As it stands, the additional information (for the visualizer) is not returned by the slice function.
* Generic visualizer in plot.lly / dash? that can use the PycroDataset class
   * One suggestion is 2 (or more panes). 
         * Left hand side for positions
               * 1D lines or 2D images
               * Ability to select individual pixels, points within a polygon.
               * What quantity to display for these images? Select one within P fields for compound datasets. Perhaps we need sliders / dropdowns for all spectral dimensions here to for the user to slices?
         * Right hand side for spectral
               * 1D spectra or 2D images. 
               * Users will be asked to slice N-1 or N-2 spectral dimensions
*	Switch to using plot.ly and dash for interactive elements
*	Possibly use MayaVi for 3d plotting

Plot Utils
~~~~~~~~~
* _add_loop_parameters - is BE specific and should be moved out of plot_utils

* rainbow_plot - 

  1. pop cmap from kwargs instead of specifying camp as a separate argument. 
  2. Rename parameters from ax to axis, ao_vec to x_values, ai_vec to y_values. 
  3. Use same methodology from single_img_cbar_plot to add color bar. You will need to expect the figure handle as well for this.

* plot_line_family - 

  1. Rename x_axis parameter to something more sensible like x_values
  2. Remove c map as one of the arguments. It should come from kwargs
  3. Optional color bar (don’t show legend in this case)

* plot_map -combine this with single_img_cbar_plot

* single_img_cbar_plot - It is OK to spend a lot of time on single_img_cbar_plot and plot_map since these will be used HEAVILY for papers.

  1. Combine with plot_map
  2. allow the tick labels to be specified instead of just the x_size and y_size. 
  3. Rename this function to something more sensible
  4. Color bar should be shown by default

* plot_loops

  1. Allow excitation_waveform to also be a list - this will allow different x resolutions for each line family. 
  2. Apply appropriate x, y, label font sizes etc. This should look very polished and ready for publications
  3. Enable use of kwargs - to specify line widths etc.
  4. Ensure that the title is not crammed somewhere behind the subtitles

* Plot_complex_map_stack

  1. allow kwargs. 
  2. Use plot_map 
  3. Respect font sizes for x, y labels, titles - use new kwargs wherever necessary 
  4. Remove map as a kwarg
  5. Show color bars
  6. Possibly allow horizontal / vertical configurations? (Optional)

* plot_complex_loop_stack

  1. Respect font sizes for x, y labels, titles - use new kwargs wherever necessary 
  2. Allow individual plots sizes to be specified
  3. Allow **kwargs and pass two plot functions

* plotScree

  1. rename to plot_scree
  2. Use **kwargs on the plot function

* plot_map_stack:

  1. Do something about the super title getting hidden behind the subtitles
  2. Respect tick, x label, y label, title, etc font sizes
  3. Add ability to manually specify x and y tick labels - see plot_cluster_results_together for inspiration
  4. See all other changes that were made for the image cleaning paper

* plot_cluster_results_together

  1. Use plot_map and its cleaner color bar option
  2. Respect font sizes
  3. Option to use a color bar for the centroids instead of a legend - especially if number of clusters > 7
  4. See mode IV paper to see other changes

* plot_cluster_results_separate
  
  1. Use same guidelines as above

* plot_cluster_dendrogram - this function has not worked recently to my knowledge. Fortunately, it is not one of the more popular functions so it gets low priority for now. Use inspiration from image cleaning paper

* plot_1d_spectrum

  1. Respect font sizes
  2. Do not save figure here. This should be done in the place where this function is called
  3. Use **kwargs and pass to the plot functions
  4. Title should be optional

* plot_2d_spectrogram

  1. Respect font sizes
  2. Use plot_map - show color bar
  3. Don’t allow specification of figure_path here. Save elsewhere

* plot_histograms - not used frequently. Can be ignored for this pass
Examples / Tutorials

External user contributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Li Xin classification code 
* Ondrej Dyck’s atom finding code – written but needs work before fully integrated
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
