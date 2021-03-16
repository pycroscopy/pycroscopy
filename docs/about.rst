==========
Pycroscopy
==========

**Scientific analysis of nanoscience data**

.. attention::

   The pycroscopy ecosystem of packages have been undergoing a major restructuring process.
   The pycroscopy python package is being completely restructured under the `phoenix <https://github.com/pycroscopy/pycroscopy/tree/phoenix>`_ branch.
   Those wishing to continue to use the legacy version of pycroscopy can do so using the `legacy <https://github.com/pycroscopy/pycroscopy/tree/legacy>`_ branch.

.. note::
   **Weekly Hackathons**

   We run weekly hackathons to develop the pycroscopy ecosystem of python packages.
   Hackathons are held every Friday 3-5PM USA Eastern time.
   The requirements for participation are: knowledge of python, git,
   and the basic structure and philosophy of the pycroscopy ecosystem (available through documentation).
   If you would like to participate, please email us at vasudevanrk *at* ornl.gov

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


Pycroscopy Package
------------------


What?
------
* pycroscopy is a `python <http://www.python.org/>`_ package for analyzing and visualizing nanoscience data.
* pycroscopy uses the **Universal Spectroscopy and Imaging Data (USID)** `model <../../USID/about.html>`_ as its foundation, which:

  * facilitates the representation of any spectroscopic or imaging data regardless of its origin, modality, size, or dimensionality.
  * enables the development of instrument- and modality- agnostic data processing and analysis algorithms.
* pycroscopy uses a data-centric model wherein the raw data collected from the instrument, results from analysis
  and processing routines are all written to USID **hierarchical data format (HDF5)** files for traceability, reproducibility, and provenance.
* pycroscopy uses `pyUSID <https://pycroscopy.github.io/pyUSID/about.html>`_ which provides tools to read, write, visualize, and process **USID** stored in HDF5 files.
  In addition, pycroscopy uses popular packages such as numpy, scipy, scikit-image, scikit-learn, joblib, matplotlib, etc. for most of the computation, analysis and visualization.
* You can choose to perform your analysis outside pycroscopy if you prefer and use pyUSID to standardize the data storage.
* For more information, please consider reading our `Arxiv <https://arxiv.org/abs/1903.09515>`_ **paper**.
* See `scientific research enabled by pycroscopy <https://pycroscopy.github.io/pycroscopy/papers_conferences.html>`_.
* Jump to our `GitHub project <https://github.com/pycroscopy/pycroscopy>`_

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

Who?
-----
* This project begun largely as an effort by scientists and engineers at the **I**\nstitute for **F**\unctional **I**\maging of **M**\aterials (`IFIM <https://ifim.ornl.gov>`_) to implement a python library that can support the I/O, processing, and analysis of the gargantuan stream of images that their microscopes generate (thanks to the large IFIM users community!).
* By sharing our methodology and code for analyzing scientific imaging data we hope that it will benefit the wider scientific community. We also hope, quite ardently, that other scientists would follow suit.
* Please visit our `credits and acknowledgements <./credits.html>`_ page for more information about the people behind pycroscopy.
