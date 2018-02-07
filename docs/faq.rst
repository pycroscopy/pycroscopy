Frequently asked questions
==========================

.. contents::

What is pycroscopy and how is it different from python?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`Python <https://www.python.org>`_ is an (interpreted) programming langauge similar to R, Java, C, C++, Fortran etc. `Pycroscopy <https://pycroscopy.github.io/pycroscopy/about.html#what>`_ is an addon module to python that provides it the ability to analyze scientific imaging / microscopy data. As an (oversimplified) analogy, think of python as Windows or Mac OS and pycroscopy as Firefox or Chrome or Safari. 

How is pycroscopy different from ImageJ / FIJI / ImageSXM / WSxM?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`ImageJ <https://imagej.nih.gov/ij/>`_ / `FIJI <https://fiji.sc>`_ / `ImageSXM <https://www.liverpool.ac.uk/~sdb/ImageSXM/>`_ / `WSxM <http://www.wsxm.es/download.html>`_ are all excellent software packages for dealing with conventional and popular microscopy data such as 2D images or a handful of spectra. Pycroscopy was built for a completely different purpose and with a unique philosophy. Pycroscopy solves some truly challenging problems that none of the other packages do - It can handle arbitrarily large datasets (Gigabytes / Terabytes / or even larger) with arbitrary dimensionality (any combination of dimensionality in position and spectral space. We have had absolutely no problem dealing with datasets with more than 9 unique dimensions - see `Band Excitation Polarization Switching + First Order Reversal Curve probing <https://pycroscopy.github.io/pycroscopy/auto_examples/dev_tutorials/plot_tutorial_03_multidimensional_data.html#sphx-glr-auto-examples-dev-tutorials-plot-tutorial-03-multidimensional-data-py>`_). In addition, Pycroscopy takes a data centric approach aimed towards open science wherein all the data starting from the raw measurement from the instrument, all the way to the final data that is plotted in the resulting scientific publication, are contained in the same file. The processing steps applied to the data are completely transparent and traceable. Furthermore, pycroscopy was developed from the ground up to run on laptops while aiming towards compatibility with supercomputers.  

Why not use established file formats from other domains?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
It is true that we really don't want yet another format in our lives. We wanted to adopt a file format that is already widely accepted in supercomputing, scientific research, can be accessed from any programming language. We chose HDF5 since it suits our needs perfectly. We found that existing data formats in science such as the `Nexus data format <http://www.nexusformat.org>`_ and `NetCDF <https://www.unidata.ucar.edu/software/netcdf/>`_ were designed for narrow scientific purposes and we did not want to shoehorn our data structure into those formats. Furthermore, despite being some of the more popular scientific data formats, it is not immidiately straightforward to read those files on every computer using any programming language. For example - the Anaconda python distribution does not come with any packages for reading these file formats. Moreover, `Adios <https://www.olcf.ornl.gov/center-projects/adios/>`_, Nexus, NetCDF, and even `Matlab's .mat <https://www.mathworks.com/help/matlab/import_export/mat-file-versions.html>`_ files are actually (now) just custom flavors of HDF5 files thereby unanimously validating our decision to choose HDF5 as our file format. Unlike Nexus, NetCDF, Matlab's .mat files, pycroscopy does not impose any strict restrictions or requirements on the HDF5 file structure. Instead, implementing the pycroscopy data format only increases the functionality of the very same datasets in pycroscopy. 

Can Pycroscopy read data files from instrument X?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Pycroscopy has numerous translators that extract the data and metadata (e.g. - instrument / imaging parameters) from some popular file formats and store the information in HDF5 files. You can find a list of available `translators here <https://github.com/pycroscopy/pycroscopy/tree/master/pycroscopy/io/translators>`_.

I could not find a data translator for my data format. What do I do now?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
You can easily write your own Translator to transfer all relevant information from the raw data files to pycroscopy formatted HDF5 files. We have `examples available here <https://pycroscopy.github.io/pycroscopy/auto_examples/index.html#developer-tutorials>`_.

I am not finding an example on topic X / I find tutorial X confusing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We appreciate your feedback regarding the documentation. Please send us an email at pycroscopy@gmail.com or send us a message on our `slack group <https://pycroscopy.slack.com/>`_ and we will work with you to add / improve our documentation.  
