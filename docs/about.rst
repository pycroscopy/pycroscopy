==========
Pycroscopy
==========

**Scientific analysis of nanoscale materials imaging data**

.. note::
   **Weekly hackathons**
   We are running weekly hackathons for pycroscopy and pyUSID to encourage community engagement, increase development, issue bug fixes and generally improve the packages. Hackathons are run every Friday 3-5PM USA Eastern time. The requirements for participation are: knowledge of python, git, pyUSID, and we expect that you will be comfortable in understanding the structure of the packages (and have gone through the documentation). If you would like to contribute, please let us know. Email vasudevanrk *at* ornl.gov

What?
------
* pycroscopy is a `python <http://www.python.org/>`_ package for processing, analyzing, and visualizing multidimensional imaging and spectroscopy data.
* pycroscopy uses the **Universal Spectroscopy and Imaging Data (USID)** `model <../../USID/about.html>`_ as its foundation, which:

  * facilitates the representation of any spectroscopic or imaging data regardless of its origin, modality, size, or dimensionality.
  * enables the development of instrument- and modality- agnostic data processing and analysis algorithms.
* pycroscopy uses a data-centric model wherein the raw data collected from the instrument, results from analysis
  and processing routines are all written to USID **hierarchical data format (HDF5)** files for traceability, reproducibility, and provenance.
* pycroscopy uses `pyUSID <https://pycroscopy.github.io/pyUSID/about.html>`_ which provides tools to read, write, visualize, and process **USID** stored in HDF5 files.
  In addition, pycroscopy uses popular packages such as numpy, scipy, scikit-image, scikit-learn, joblib, matplotlib, etc. for most of the computation, analysis and visualization.
* You can choose to perform your analysis outside pycroscopy if you prefer and use pyUSID to standardize the data storage.
* For more information, please consider reading our `Arxiv <https://arxiv.org/abs/1903.09515>`_ **paper**.
* See a high-level overview of pycroscopy in this `presentation <https://github.com/pycroscopy/pycroscopy/blob/master/docs/USID_pyUSID_pycroscopy.pdf>`_
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
