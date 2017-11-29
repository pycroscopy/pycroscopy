.. pycroscopy documentation master file, created by
   sphinx-quickstart on Thu Aug  3 12:58:27 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

==========
Pycroscopy
==========

**Scientific analysis of nanoscale materials imaging data**


.. contents::

.. _what-label:

What?
--------------------
pycroscopy is a `python <http://www.python.org/>`_ package for image processing and scientific analysis of imaging modalities such as multi-frequency scanning probe microscopy, scanning tunneling spectroscopy, x-ray diffraction microscopy, and transmission electron microscopy. pycroscopy uses a data-centric model wherein the raw data collected from the microscope, results from analysis and processing routines are all written to standardized hierarchical data format (HDF5) files for traceability, reproducibility, and provenance.

With  pycroscopy we aim to:
	1. Serve as a hub for collaboration across scientific domains (microscopists, material scientists, biologists...)
	2. provide a community-developed, open standard for data formatting 
	3. provide a framework for developing data analysis routines 
	4. significantly lower the barrier to advanced data analysis procedures by simplifying I/O, processing, visualization, etc.

To learn more about the motivation, general structure, and philosophy of pycroscopy, please read this `short introduction <https://github.com/pycroscopy/pycroscopy/blob/master/docs/pycroscopy_2017_07_11.pdf>`_.

.. _who-label:

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

.. _why-label:

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

.. _how-label:

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

.. _pkgstr-label:

Package Structure
-----------------
The package structure is simple, with 4 main modules:
   1. **io**: Reading and writing to HDF5 files + translating data from custom & proprietary microscope formats to HDF5.
   2. **processing**: multivariate statistics, machine Learning, and signal filtering.
   3. **analysis**: model-dependent analysis of information.
   4. **viz**: Plotting functions and interactive jupyter widgets to visualize multidimenional data

.. _start-label:

Getting Started
---------------
* Follow the instructions on our `GitHub project page <https://github.com/pycroscopy/pycroscopy>`_ to install pycroscopy
* See how we use pycroscopy for our scientific research in these `jupyter notebooks <http://nbviewer.jupyter.org/github/pycroscopy/pycroscopy/tree/master/jupyter_notebooks/>`_. Many of them are linked to journal publications listed below.
	* Please see the official `jupyter <http://jupyter.org>`_ website for more information about notebooks and consider watching this `youtube video <https://www.youtube.com/watch?v=HW29067qVWk>`_.
* See our `examples <https://pycroscopy.github.io/pycroscopy/auto_examples/index.html>`_ to get started on using and writing your own pycroscopy functions
* Videos and other tutorials are available at the `Institute For Functional Imaging of Materials <http://ifim.ornl.gov/resources.html>`_ 
* For more information about our functions and classes, please see our `API <https://pycroscopy.github.io/pycroscopy/pycroscopy.html>`_
* We have many translators that transform data from popular microscope data formats to pycroscopy compatible .h5 files. We also have `tutorials to get you started on importing your data to pycroscopy <https://pycroscopy.github.io/pycroscopy/auto_examples/tutorial_01_translator.html>`_. 
* Details regarding the defention, implementation, and guidelines for pycroscopy's `data format <https://pycroscopy.github.io/pycroscopy/Data_Format.html>`_ for `HDF5 <https://github.com/pycroscopy/pycroscopy/blob/master/docs/Pycroscopy_Data_Formatting.pdf>`_ are also available. 
* If you are interested in contributing and are looking for topics we are / will work on, please look at our `To Do <https://github.com/pycroscopy/pycroscopy/blob/master/ToDo.rst>`_ page

.. _papers-label:

Journal Papers using pycroscopy
-------------------------------
1. `Big Data Analytics for Scanning Transmission Electron Microscopy Ptychography <https://www.nature.com/articles/srep26348>`_ by S. Jesse et al., Scientific Reports (2015); jupyter notebook `here <http://nbviewer.jupyter.org/github/pycroscopy/pycroscopy/blob/master/jupyter_notebooks/Ptychography.ipynb>`_
 
2. `Rapid mapping of polarization switching through complete information acquisition <http://www.nature.com/articles/ncomms13290>`_ by S. Somnath et al., Nature Communications (2016); jupyter notebook `here <http://nbviewer.jupyter.org/github/pycroscopy/pycroscopy/blob/master/jupyter_notebooks/G_mode_filtering.ipynb>`_
 
3. `Improving superconductivity in BaFe2As2-based crystals by cobalt clustering and electronic uniformity <http://www.nature.com/articles/s41598-017-00984-1>`_ by L. Li et al., Scientific Reports (2017); jupyter notebook `here <http://nbviewer.jupyter.org/github/pycroscopy/pycroscopy/blob/master/jupyter_notebooks/STS_LDOS.ipynb>`_
 
4. `Direct Imaging of the Relaxation of Individual Ferroelectric Interfaces in a Tensile-Strained Film <http://onlinelibrary.wiley.com/doi/10.1002/aelm.201600508/full>`_ by L. Li et al.; Advanced Electronic Materials (2017), jupyter notebook `here <http://nbviewer.jupyter.org/github/pycroscopy/pycroscopy/blob/master/jupyter_notebooks/BE_Processing.ipynb>`_

5. `Decoding apparent ferroelectricity in perovskite nanofibers <http://pubs.acs.org/doi/pdf/10.1021/acsami.7b14257>`_ by R. Ganeshkumar et al., ACS Applied Materials & Interfaces (2017). 

6. Ultrafast Current Imaging via Bayesian Inference by S. Somnath et al., accepted at Nature Communications (2017).

7. Feature extraction via similarity search: application to atom finding and denosing in electon and scanning probe microscopy imaging by S. Somnath et al.; under review at Advanced Structural and Chemical Imaging (2017), jupyter notebook `here 5 <http://nbviewer.jupyter.org/github/pycroscopy/pycroscopy/blob/master/jupyter_notebooks/Image_Cleaning_Atom_Finding.ipynb>`_

8. Many more coming soon....

.. _conferences-label:

International conferences and workshops using pycroscopy
--------------------------------------------------------
* Dec 2017 - Materials Research Society conference
* Oct 31 2017 @ 6:30 PM - American Vacuum Society conference;  Session: SP-TuP1; poster 1641
* Aug 9 2017 @ 8:30 - 10:00 AM - Microscopy and Microanalysis conference; X40 - Tutorial session on `Large Scale Data Acquisition and Analysis for Materials Imaging and Spectroscopy <http://microscopy.org/MandM/2017/program/tutorials.cfm>`_ by S. Jesse and S. V. Kalinin
* Aug 8 2017 @ 10:45 AM - Microscopy and Microanalysis conference - poster session
* Apr 2017 - Lecture on `atom finding <https://physics.appstate.edu/events/aberration-corrected-stem-teaching-machines-and-atomic-forge>`_
* Dec 2016 - Poster + `abstract <https://mrsspring.zerista.com/poster/member/85350>`_ at the 2017 Spring Materials Research Society (MRS) conference

.. _contact-label:

Contact us
----------
* We are interested in collaborating with industry members to integrate pycroscopy into instrumentation or analysis software and can help in exporting data to pycroscopy compatible .h5 files 
* We can work with you to convert your file formats into pycroscopy compatible HDF5 files and help you get started with data analysis.
* Join our slack project at https://pycroscopy.slack.com to discuss about pycroscopy
* Feel free to get in touch with us at pycroscopy (at) gmail [dot] com
* If you find any bugs or if you want a feature added to pycroscopy, raise an `issue <https://github.com/pycroscopy/pycroscopy/issues>`_. You will need a free Github account to do this


Acknowledgements
----------------
Besides the packages used in pycroscopy, we would like to thank the developers of the following software
packages:

+ `Python <https://www.python.org>`_
+ `Anaconda Python <https://www.continuum.io/anaconda-overview>`_
+ `jupyter <http://jupyter.org/>`_
+ `PyCharm <https://www.jetbrains.com/pycharm/>`_
+ `GitKraken <https://www.gitkraken.com/>`_

Documentation Index
-------------------
.. currentmodule:: index

.. autosummary::
   :template: module.rst

.. toctree::
    about
    getting_started
    data_format
    papers_conferences
    api
    contact
    auto_examples/index

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
