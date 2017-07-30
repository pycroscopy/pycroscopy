.. contents::

New features
------------
Core development
~~~~~~~~~~~~~~~~
* Data Generators
*	Parallel framework for Processing - Should be similar to Optimize from Analysis.
External user contributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Li Xin classification code 
* Ondrej’s atom finding code – written but needs work before fully integrated
*	Nina’s processing code (Tselev) – in progress
* Sabine's cKPFM code
* Josh Agar's convex hull
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
*	Switch to using Sphinx-gallery for documentation
  *	Done for existing documentation
  *	Work will be needed after examples are done
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
* Use Travis-CI for automatic testing, document generation, versioning, uploading, etc.
