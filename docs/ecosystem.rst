Pycroscopy Ecosystem
--------------------
The following diagram provides a graphical representation and description of the several python packages that
make up the pycroscopy ecosystem.

.. image:: ./pycroscopy_ecosystem.png

Here is a brief overview of various technologies and packages that comprise the pycroscopy ecosystem:

* General scientific packages:

  * `SciFiReaders <https://pycroscopy.github.io/SciFiReaders/about.html>`_ – tools to extract data and metadata out of vendor specific data files. Extracted information is stored only in memory
  * `Pycroscopy <https://pycroscopy.github.io/pycroscopy/about.html>`_ – scientific and data analytics tools that can be used across scientific domains
* Application scientific packages:

  * `pyTEMlib <https://pycroscopy.github.io/pyTEMlib/about.html>`_ - Physics model-based TEM data quantification library
  * `stemtool <https://github.com/pycroscopy/stemtool>`_ - Tools to analyze atomic resolution STEM, 4D-STEM and STEM-EELS datasets
  * `atomAI <https://github.com/pycroscopy/atomai>`_ - Deep and machine learning for atomic-scale and mesoscale data
  * `AICrystallographer <https://github.com/pycroscopy/AICrystallographer>`_ - Deep and machine learning models that aid in automated analysis of atomically resolved images
  * `BGlib <https://pycroscopy.github.io/BGlib/index.html>`_ - Utilities to analyze Band Excitation and General Mode Data for `ORNL's Center for Nanophase Materials Science SPM <https://www.ornl.gov/content/advanced-afm>`_ users
  * `FFTA <https://github.com/rajgiriUW/ffta>`_ - Fast Free Transient Analysis of atomic force microscopy data
* Data infrastructure - domain agnostic

  * `sidpy <pycroscopy.github.io/sidpy/>`_ – core engineering tools to support scientific packages and file I/O packages
  * Abstract models to represent data:

    * `USID – Universal Spectroscopy and Imaging and Data model <pycroscopy.github.io/usid/about.html>`_ – General model for representing data with or without N-dimensional forms
    * `NSID – N-Dimensional Spectroscopy and Imaging Data model <https://pycroscopy.github.io/pyNSID/nsid.html>`_ – Model for data with a clear N-dimensional form
  * Interfaces to reading and writing pycroscopy formatted data into `Hierarchical Data Format Files (HDF5) <http://extremecomputingtraining.anl.gov/files/2015/03/HDF5-Intro-aug7-130.pdf>`_:

    * `pyUSID <https://pycroscopy.github.io/pyUSID/about.html>`_ – Python interface USID
    * `pyNSID <https://pycroscopy.github.io/pyNSID/about.html>`_ – Python interface to NSID


With pycroscopy we aim to:

#. significantly lower the barrier to advanced data analysis procedures by simplifying I/O, processing, visualization, etc.
#. serve as a hub for collaboration across scientific domains (microscopists, material scientists, biologists...)

Why?
-----
As we see it, there are a few opportunities in scientific imaging (that surely apply to several other scientific domains):

**1. Growing data sizes**
  * Cannot use desktop computers for analysis
  * *Need: High performance computing, storage resources and compatible, scalable file structures*

**2. Increasing data complexity**
  * Sophisticated imaging and spectroscopy modes resulting in 5,6,7... dimensional data
  * *Need: Robust software and generalized data formatting*

**3. Multiple file formats**
  * Different formats from each instrument. Proprietary in most cases
  * Incompatible for correlation
  * *Need: Open, instrument-independent data format*

**4. Disjoint communities**
  * Similar analysis routines written by each community (SPM, STEM, TOF SIMs, XRD...) *independently*!
  * *Need: Centralized repository, instrument agnostic analysis routines that bring communities together*

**5. Expensive analysis software**
  * Software supplied with instruments often insufficient / incapable of custom analysis routines
  * Commercial software (Eg: Matlab, Origin..) are often prohibitively expensive.
  * *Need: Free, powerful, open source, user-friendly software*

**6. Closed science**
  * Analysis software and data not shared
  * No guarantees of reproducibility or traceability
  * *Need: open source data structures, file formats, centralized code and data repositories*

How?
-----
* pycroscopy uses the `Universal Spectroscopy and Imaging Data model <../../USID/index.html>`_ that facilitates the storage of data, regardless
  of dimensionality (conventional 1D spectra and 2D images to 9D hyperspectral datasets and beyond!) or instrument of origin (AFMs, STEMs, Raman spectroscopy etc.).
* This generalized representation of data allows us to write a single and
  generalized version of analysis and processing functions that can be applied to any kind of data.
* The data are stored in `hierarchical
  data format (HDF5) <http://extremecomputingtraining.anl.gov/files/2015/03/HDF5-Intro-aug7-130.pdf>`_
  files which have numerous benefits including flexibility in storing multiple datasets of arbitrary sizes and dimensionality,
  supercomputer compatibility, storage of important metadata.
* Once the relevant data and metadata are extracted from proprietary raw data files and written into USID HDF5 files
  via a `translation process <https://pycroscopy.github.io/pyUSID/auto_examples/beginner/plot_numpy_translator.html>`_,
  the user gains access to the rest of the utilities present in ``pycroscopy.*``.
* Scientific workflows are developed and disseminated through `jupyter notebooks <http://jupyter.org/>`_
  that are interactive and portable web applications containing text, images, code / scripts, and graphical results.
  Notebooks containing the complete / parts of workflow from raw data to publishable figures often become supplementary
  material for `journal publications <./papers_conferences.html>`_ thereby enabling traceability, reproducibility for open science.