==========
Pycroscopy
==========

**Scientific analysis of nanoscale materials imaging data**

What?
------
pycroscopy is a `python <http://www.python.org/>`_ package for storing, processing, analyzing, and visualizing multidimensional scientific data.
pycroscopy uses a data-centric model wherein the raw data collected from the instrument, results from analysis
and processing routines are all written to standardized **hierarchical data format (HDF5)** files for traceability, reproducibility, and provenance.

With pycroscopy we aim to:

1. significantly lower the barrier to advanced data analysis procedures by simplifying I/O, processing, visualization, etc.
2. serve as a hub for collaboration across scientific domains (microscopists, material scientists, biologists...)
3. provide a community-driven, open standard for data formatting
4. provide a framework for developing origin-agnostic / universal data analysis routines


To learn more about the motivation, general structure, and philosophy of pycroscopy, please read this
`short introduction <https://github.com/pycroscopy/pycroscopy/blob/master/docs/pycroscopy_2017_07_11.pdf>`_.

Jump to our `GitHub project <https://github.com/pycroscopy/pycroscopy>`_

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
  * *Need: Open, instrument independent data format*

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
* pycroscopy uses an `instrument agnostic data structure <https://pycroscopy.github.io/pycroscopy/data_format.html>`_ that facilitates the storage of data, regardless
  of dimensionality (conventional 1D spectra and 2D images to 9D hyperspectral datasets and beyond!) or instrument of origin (AFMs, STEMs, Raman spectroscopy etc.).
* This generalized representation of data allows us to write a single and
  generalized version of analysis and processing functions that can be applied to any kind of data.
* The data is stored in `hierarchical
  data format (HDF5) <http://extremecomputingtraining.anl.gov/files/2015/03/HDF5-Intro-aug7-130.pdf>`_
  files which have numerous benefits including flexibility in storing multiple datasets of arbitrary sizes and dimensionality,
  supercomputer compatibility, storage of important metadata.
* Once the relevant data and metadata are extracted from proprietary raw data files and written into pycroscopy formatted HDF5 files
  via a `translation process <https://pycroscopy.github.io/pycroscopy/auto_examples/cookbooks/plot_numpy_translator.html>`_,
  the user gains access to the rest of the utilities present in ``pycroscopy.\*``.
* Scientific workflows are developed and disseminated through `jupyter notebooks <http://jupyter.org/>`_
  that are interactive and portable web applications containing text, images, code / scripts, and graphical results.
  Notebooks containing the complete / parts of workflow from raw data to publishable figures often become supplementary
  material for `journal publications <https://pycroscopy.github.io/pycroscopy/papers_conferences.html>`_ thereby enabling traceability, reproducibility for open science.

Who?
-----
* This project begun largely as an effort by scientists and engineers at the **I**\nstitute for **F**\unctional **I**\maging of **M**\aterials (`IFIM <https://ifim.ornl.gov>`_) to implement a python library that can support the I/O, processing, and analysis of the gargantuan stream of images that their microscopes generate (thanks to the large IFIM users community!).
* It is now being developed and maintained by `Suhas Somnath <https://github.com/ssomnath>`_ of the **A**\dvanced **D**\ata & **W**\orkflows **G**\roup (ADWG) at the **O**\ak Ridge National Laboratory **L**\eadership **C**\omputing **F**\acility (`OLCF <https://www.olcf.ornl.gov>`_) and `Chris R. Smith <https://github.com/CompPhysChris>`_ of IFIM.
* By sharing our methodology and code for analyzing scientific imaging data we hope that it will benefit the wider scientific community. We also hope, quite ardently, that other scientists would follow suit.
* Please visit our `credits and acknowledgements <https://pycroscopy.github.io/pycroscopy/credits.html>`_ page for more information.