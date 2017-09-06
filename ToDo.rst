.. contents::

New features
------------
Core development
~~~~~~~~~~~~~~~~
* Simplify and demystify analyis / optimize. Use parallel_compute and joblib
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
* Select data within a dataset in various ways - effectively just examples of hdf_utils functions
* Slicing and dicing multidimensional datasets
* micro datasets / microdata groups
* position and spectroscopic datasets with multidimensional datasets
* chunking the main dataset
* Setting up interactive visualizers
* Links to tutorials on how to use pycharm, Git, 

Longer examples (probably scientific workflows / pipelines)
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
