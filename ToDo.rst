.. contents::

New features
------------
Core development
~~~~~~~~~~~~~~~~
* Data Generators
* Parallel framework for Processing - Should be similar to Optimize from Analysis.
* Either host real data for people to play with
External user contributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Li Xin classification code 
* Ondrej Dyck’s atom finding code – written but needs work before fully integrated
* Nina Wisinger’s processing code (Tselev) – in progress
* Sabine Neumeyer's cKPFM code
* Josh Agar's convex hull code
* Iaroslav Gaponenko's Distort correct code from - https://github.com/paruch-group/distortcorrect.
* Port everything from CNMS Matlab > Python translation exercises
* Other workflows/functions that already exist as scripts or notebooks

Plotting updates
----------------
*	Switch to using plot.ly and dash for interactive elements
*	Possibly use MayaVi for 3d plotting
* Switch plot_map_stack order to (layer, x, y) from (x, y, layer)

Examples / Tutorials
--------------------
Short notebooks / tutorials on how to use pycroscopy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*	Access h5 files
*	Find a specific dataset/group in the file
*	Select data within a dataset in various ways
*	Specific function examples as needed
* Writing data back to h5 - both creating a quick translator AND maing a quick analysis / processing routine on the fly
    * Use the STS dataset as an example (simple and small 3D data)  
    * micro datasets / microdata groups
    * position and spectroscopic datasets
    * chunking the main dataset
    * ioHDF5
    * linking as main etc.
* Setting up interactive visualizers
* Link to tutorials on pycharm, Git, 
Longer examples (probably scientific workflows / pipelines)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*	Ttranslators – probably only one needed
*	Processing
  *	Clustering
  *	Decomposition
*	Analysis
  *	How to use Optimize
*	Vis

Documentation
-------------
*	Switch to using Sphinx-gallery for documentation:
   * https://sphinx-gallery.readthedocs.io/en/latest/
   * http://scikit-image.org/docs/dev/auto_examples/ 
   * http://scikit-learn.org/stable/auto_examples/index.html 
   * more complicated analyses -  http://nipy.org/dipy/examples_index.html
   * Done for existing documentation
   * Work will be needed after examples are done
*	Host docs somewhere other than readthedocs - On github?
*	Include examples in documentation

Formatting changes
------------------
*	Fix remaining PEP8 problems
*	Ensure code and documentation is standardized
*	Switch to standard version formatting
*	Classes and major Functions should check to see if the results already exist

Notebooks
---------
*	Add new workflows
*	Direct downloading of notebooks (ipynb an html)
  * nbviewer?
  * Host somewhere other than github?
*	Investigate using Jupyter hub and Jupyter lab

Testing
-------
*	Write test code
*	Unit tests for simple functions
*	Longer tests using data (real or generated) for the workflow tests
*  measure coverage using codecov.io and codecov package

Software Engineering
--------------------
* Use Travis-CI or Circle for automatic testing, document generation, versioning, uploading, etc.
   * https://ilovesymposia.com/2014/10/15/continuous-integration-in-python-4-set-up-travis-ci/  
   * (good example: scikit-learn: https://github.com/scikit-learn/scikit-learn/blob/master/.travis.yml
   * https://github.com/scikit-learn/scikit-learn/tree/master/build_tools/circle)
* Consider releasing bug fixes (to onsite CNMS users) via git instead of rapid pypi releases 
   * example release steps (incl. git tagging): https://github.com/cesium-ml/cesium/blob/master/RELEASE.txt
* Proper pypi versioning - https://www.python.org/dev/peps/pep-0440/#developmental-releases
* Use https://docs.pytest.org/en/latest/ instead of nose (nose is no longer maintained)
* Add requirements.txt

Scaling to clusters
-------------------
We have two kinds of large computational jobs and one kind of large I/O job:

* I/O - reading and writing large amounts of data
   * Dask and MPI are compatible. Spark is probably not
* Computation
   1. Machine learning and Statistics
   
      1.1. Either use custom algorithms developed for BEAM
         * Advantage - Optimized (and tested) for various HPC environments
         * Disadvantages:
            * Need to integarate non-python code
            * We only have a handful of these. NOT future compatible            
      1.2. Or continue using a single FAT node for these jobs
         * Advantages:
            * No optimization required
            * Continue using the same scikit learn packages
         * Disadvantage - Is not optimized for HPC
   2. Parallel parametric search - analyze subpackage and some user defined functions in processing. Can be extended using:
   
      * Dask - An inplace replacement of multiprocessing will work on laptops and clusters. More elegant and easier to write and maintain compared to MPI at the cost of efficiency
         * simple dask netcdf example: http://matthewrocklin.com/blog/work/2016/02/26/dask-distributed-part-3
      * MPI - Need alternatives to Optimize / Process classes - Better efficiency but a pain to implement
