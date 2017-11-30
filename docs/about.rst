==========
Pycroscopy
==========

**Scientific analysis of nanoscale materials imaging data**

What?
--------------------
pycroscopy is a `python <http://www.python.org/>`_ package for image processing and scientific analysis of imaging modalities such as multi-frequency scanning probe microscopy, scanning tunneling spectroscopy, x-ray diffraction microscopy, and transmission electron microscopy. pycroscopy uses a data-centric model wherein the raw data collected from the microscope, results from analysis and processing routines are all written to standardized hierarchical data format (HDF5) files for traceability, reproducibility, and provenance.

With  pycroscopy we aim to:
	1. Serve as a hub for collaboration across scientific domains (microscopists, material scientists, biologists...)
	2. provide a community-developed, open standard for data formatting 
	3. provide a framework for developing data analysis routines 
	4. significantly lower the barrier to advanced data analysis procedures by simplifying I/O, processing, visualization, etc.

To learn more about the motivation, general structure, and philosophy of pycroscopy, please read this `short introduction <https://github.com/pycroscopy/pycroscopy/blob/master/docs/pycroscopy_2017_07_11.pdf>`_.

Who?
-----------
This project begun largely as an effort by scientists and engineers at the **C**\enter for **N**\anophase
**M**\aterials **S**\ciences (`CNMS <https://www.ornl.gov/facility/cnms>`_) to implement a python library
that can support the I/O, processing, and analysis of the gargantuan stream of images that their microscopes
generate (thanks to the large CNMS users community!).

By sharing our methodology and code for analyzing materials imaging we hope that it will benefit the wider
community of materials science/physics. We also hope, quite ardently, that other materials scientists would
follow suit.

**The (core) pycroscopy team:**

* `@ssomnath <https://github.com/ssomnath>`_ (Suhas Somnath), 
* `@CompPhysChris <https://github.com/CompPhysChris>`_ (Chris R. Smith), 
* `@nlaanait <https://github.com/nlaanait>`_ (Numan Laanait), 
* `@stephenjesse <https://github.com/stephenjesse>`_ (Stephen Jesse) 
* and many more...

Why?
---------------
There is that little thing called open science...

As we see it, there are a few  opportunities in microscopy / imaging and materials science:

**1. Growing data sizes**
  * Cannot use desktop computers for analysis
  * *Need: High performance computing, storage resources and compatible, scalable file structures*

**2. Increasing data complexity**
  * Sophisticated imaging and spectroscopy modes resulting in 5,6,7... dimensional data
  * *Need: Robust software and generalized data formatting*

**3. Multiple file formats**
  * Different formats from each instrument. Proprietary in most cases
  * Incompatible for correlation
  * *Need: Open, instrument independent data format*

**4. Disjoint communities**
  * Similar analysis routines written by each community (SPM, STEM, TOF SIMs, XRD...) *independently*!
  * *Need: Centralized repository, instrument agonistic analysis routines that bring communities together*

**5. Expensive analysis software**
  * Software supplied with instruments often insufficient / incapable of custom analysis routines
  * Commercial software (Eg: Matlab, Origin..) are often prohibitively expensive.
  * *Need: Free, powerful, open souce, user-friendly software*

How?
-----------------
* pycroscopy uses an **instrument agnostic data structure** that facilitates the storage of data, regardless
  of dimensionality (conventional 2D images to 9D multispectral SPM datasets) or instrument of origin (AFMs,
  STMs, STEMs, TOF SIMS, and many more). 
* This general defenition of data allows us to write a single and
  generalized version of analysis and processing functions that can be applied to any kind of data.
* The data is stored in `heirarchical
  data format (HDF5) <http://extremecomputingtraining.anl.gov/files/2015/03/HDF5-Intro-aug7-130.pdf>`_
  files which:

  * Allow easy and open acceess to data from any programming language.
  * Accomodate datasets ranging from kilobytes (kB) to petabytes (pB)
  * Are readily compaible with supercomputers and support parallel I/O
  * Allows storage of relevant parameters along with data for improved traceability and reproducability of
    analysis
* Scientific workflows are developed and disseminated through `jupyter notebooks <http://jupyter.org/>`_
  that are interactive and portable web applications containing, text, images, code / scripts, and text-based
  and graphical results
* Once a user converts their microscope's data format into a HDF5 format, by simply extending some of the
  classes in \`io\`, the user gains access to the rest of the utilities present in `pycroscopy.\*`.

Package Structure
-----------------
The package structure is simple, with 4 main modules:
   1. **io**: Reading and writing to HDF5 files + translating data from custom & proprietary microscope formats to HDF5.
   2. **processing**: multivariate statistics, machine Learning, and signal filtering.
   3. **analysis**: model-dependent analysis of information.
   4. **viz**: Plotting functions and interactive jupyter widgets to visualize multidimenional data
   
Acknowledgements
----------------
Besides the packages used in pycroscopy, we would like to thank the developers of the following software
packages:

+ `Python <https://www.python.org>`_
+ `Anaconda Python <https://www.continuum.io/anaconda-overview>`_
+ `jupyter <http://jupyter.org/>`_
+ `PyCharm <https://www.jetbrains.com/pycharm/>`_
+ `GitKraken <https://www.gitkraken.com/>`_
