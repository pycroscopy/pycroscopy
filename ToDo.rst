.. contents::

Roadmap / Milestones
--------------------
1. Sep 2017 end - Cleaned versions of the main modules (Analysis pending) + enough documentation for users and developers
2. Oct 2017 mid - Multi-node compute capability

New features
------------
Core development
~~~~~~~~~~~~~~~~
* A new class (pycro_data?) for simplifying the many data slicing, referencing operations on **main** datasets.
    * Essentially, the goal is to turn the **main** datasets into powerful python objects that obviate the need for users to dig into ancillary datasets to slice, understand the datasets. Pycroscopy chooses to use a rather generalized representation of data at the cost of simplictiy. This object should bring back the simplicity of accessing the data. 
    * In the process of enabling greater insight into a dataset, this class would read and analyze ancillary datasets once and reuse this knowledge when the user requests another operation (that most likely also requires references to ancillary datasets etc. anyway).
    * Nearly all the functionality has been implemented in hdf_utils and some in io_utils. This class can simply reuse these general functions.
* Generic visualizer in plot.lly / dash? that can use the pycrodata class
   * One suggestion is 2 (or more panes). 
         * Left hand side for positions
               * 1D lines or 2D images
               * Ability to select individual pixels, points within a polygon.
               * What quantity to display for these images? Select one within P fields for compound datasets. Perhaps we need sliders / dropdowns for all spectral dimensions here to for the user to slices?
         * Right hand side for spectral
               * 1D spectra or 2D images. 
               * Users will be asked to slice N-1 or N-2 spectral dimensions
* Simplify and demystify analyis / optimize. Use parallel_compute (joblib instead of multiprocessing)
* multi-node computing capability in parallel_compute
* Data Generators

External user contributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Li Xin classification code 
* Ondrej Dyck’s atom finding code – written but needs work before fully integrated
* Nina Wisinger’s processing code (Tselev) – in progress
* Sabine Neumeyer's cKPFM code
* Iaroslav Gaponenko's Distort correct code from - https://github.com/paruch-group/distortcorrect.
* Port everything from IFIM Matlab -> Python translation exercises
* Other workflows/functions that already exist as scripts or notebooks

Plotting updates
----------------
*	Switch to using plot.ly and dash for interactive elements
*	Possibly use MayaVi for 3d plotting

Examples / Tutorials
--------------------
Short tutorials on how to use pycroscopy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Access h5 files
* Find a specific dataset/group in the file
* Select data within a dataset in various ways
* micro datasets / microdata groups
* chunking the main dataset
* Links to tutorials on how to use pycharm, Git, 

Longer examples (via specific scientific usecases)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Done:

* Data formatting in pycroscopy
* How to write a Translator
* How to write (back) to H5
* Spectral Unmixing with pycroscopy
* Basic introduction to loading data in pycroscopy
* Handling multidimensional (6D) datasets
* Visualizing data (interactively using widgets) (needs some tiny automation in the end)
* How to write your write your own parallel computing function using the process module

Pending:

* How to write your own analysis class
* A tour of the many functions in hdf_utils and io_utils since these functions need data to show / explain them.
* pycroscopy pacakge organization - a short writeup on what is where and differences between the process / analyis submodules

Rama's (older and more applied / specific) tutorial goals
~~~~~~~~~~~~~~~~~~~~
1. Open a translated and fitted FORC-PFM file, and plot the SHO Fit from cycle k corresponding to voltage p, along with the raw spectrogram for that location and the SHO guess. Plot both real and imaginary, and do so for both on and off-field.
2. Continuing above, determine the average of the quality factor coming from cycles 1,3,4 for spatial points stored in vector b for the on-field part for a predetermined voltage range given by endpoints [e,f]. Compare the results with the SHO guess and fit for the quality factor.
3. After opening a h5 file containing results from a relaxation experiment, plot the response at a particular point and voltage, run exponential fitting and then store the results of the fit in the same h5 file using iohdf and/or numpy translators.
4. Take a FORC IV ESM dataset and break it up into forward and reverse branches, along with positive and negative branches. Do correlation analysis between PFM and IV for different branches and store the results in the file, and readily access them for plotting again.
5. A guide to using the model fitter for parallel fitting of numpy array-style datasets. This one can be merged with number 

Documentation
-------------
*	Switch from static examples to dynamic jupyter notebook like examples:
   * http://scikit-image.org/docs/dev/auto_examples/ 
   * http://scikit-learn.org/stable/auto_examples/index.html 
   * more complicated analyses -  http://nipy.org/dipy/examples_index.html
   * Done for existing documentation
   * Work will be needed after examples are done
*	Include examples in documentation

Formatting changes
------------------
*	Fix remaining PEP8 problems
*	Ensure code and documentation is standardized
*	Switch to standard version formatting
*	Classes and major Functions should check to see if the results already exist

Notebooks
---------
*	Direct downloading of notebooks (ipynb an html)
  * nbviewer?
  * Host somewhere other than github?
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
