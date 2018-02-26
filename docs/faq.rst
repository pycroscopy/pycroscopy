Frequently asked questions
==========================

.. contents::

What is pycroscopy and how is it different from python?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`Python <https://www.python.org>`_ is an (interpreted) programming langauge similar to R, Java, C, C++, Fortran etc. `Pycroscopy <https://pycroscopy.github.io/pycroscopy/about.html#what>`_ is an addon module to python that provides it the ability to analyze scientific imaging / microscopy data. As an (oversimplified) analogy, think of python as Windows or Mac OS and pycroscopy as Firefox or Chrome or Safari. 

Who uses pycroscopy?
~~~~~~~~~~~~~~~~~~~~
`The Institute for Functional Imaging of Materials (IFIM) <http://ifim.ornl.gov>`_ at `Oak Ridge National Laboratory <www.ornl.gov>`_ uses pycroscopy exclusively for in-house research as well as supporting the numerous users who visit IFIM to use their state-of-art scanning probe microscopy techniques. There are several research groups in university who are beginning to use this package for their research.

How is pycroscopy different from ImageJ, FIJI, ImageSXM, WSxM?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`ImageJ <https://imagej.nih.gov/ij/>`_, `FIJI <https://fiji.sc>`_, `ImageSXM <https://www.liverpool.ac.uk/~sdb/ImageSXM/>`_, `WSxM <http://www.wsxm.es/download.html>`_, `SpectraFox <https://spectrafox.com>`_, and `OpenFovea <http://www.freesbi.ch/en/openfovea>`_ are all excellent software packages for dealing with conventional and popular microscopy data such as 2D images or a handful of (simple) spectra. We think that pycroscopy is complementary to these  other softwares and packages. Pycroscopy was built for a completely different purpose and with a unique philosophy. Pycroscopy solves some truly challenging problems that none of the other packages do - It can handle arbitrarily large datasets (Gigabytes / Terabytes / or even larger) with arbitrary dimensionality (any combination of dimensionality in position and spectral space. We have had absolutely no problem dealing with datasets with more than 9 unique dimensions - see `Band Excitation Polarization Switching + First Order Reversal Curve probing <https://pycroscopy.github.io/pycroscopy/auto_examples/dev_tutorials/plot_tutorial_03_multidimensional_data.html#sphx-glr-auto-examples-dev-tutorials-plot-tutorial-03-multidimensional-data-py>`_). In addition, Pycroscopy takes a data centric approach aimed towards open science wherein all the data starting from the raw measurement from the instrument, all the way to the final data that is plotted in the resulting scientific publication, are contained in the same file. The processing steps applied to the data are completely transparent and traceable. Furthermore, pycroscopy was developed from the ground up to run on laptops while aiming towards compatibility with supercomputers. `GXSM <http://gxsm.sourceforge.net>`_ is another software package that focuses more on the data acquistion than advanced data analysis

Why not use established file formats from other domains?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
It is true that we really don't want yet another format in our lives. We wanted to adopt a file format that is already widely accepted in supercomputing, scientific research, can be accessed from any programming language. We chose HDF5 since it suits our needs perfectly. We found that existing data formats in science such as the `Nexus data format <http://www.nexusformat.org>`_ and `NetCDF <https://www.unidata.ucar.edu/software/netcdf/>`_ were designed for narrow scientific purposes and we did not want to shoehorn our data structure into those formats. Furthermore, despite being some of the more popular scientific data formats, it is not immidiately straightforward to read those files on every computer using any programming language. For example - the `Anaconda <https://www.anaconda.com/what-is-anaconda/>`_ python distribution does not come with any packages for reading these file formats. Moreover, `Adios <https://www.olcf.ornl.gov/center-projects/adios/>`_, Nexus, NetCDF, and even `Matlab's .mat <https://www.mathworks.com/help/matlab/import_export/mat-file-versions.html>`_ files are actually (now) just custom flavors of HDF5 files thereby unanimously validating our decision to choose HDF5 as our file format. Unlike Nexus, NetCDF, Matlab's .mat files, pycroscopy does not impose any strict restrictions or requirements on the HDF5 file structure. Instead, implementing the pycroscopy data format only increases the functionality of the very same datasets in pycroscopy. 

Can Pycroscopy read data files from instrument X?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Pycroscopy has numerous translators that extract the data and metadata (e.g. - instrument / imaging parameters) from some popular file formats and store the information in HDF5 files. You can find a list of available `translators here <https://github.com/pycroscopy/pycroscopy/tree/master/pycroscopy/io/translators>`_.

I could not find a data translator for my data format. What do I do now?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Chances are that there is a (or multiple) python package out there already that reads the data from your file into python. All you will need to do is to write the data and metadata to HDF5 files by writing your own Translator. We have `examples available here <https://pycroscopy.github.io/pycroscopy/auto_examples/index.html#developer-tutorials>`_.

Why doesn't pycroscopy use graphical user interfaces?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Traditional graphical interfaces are rather time consuming to develop. Instead, we provide jupyter widgets to interact with data wherever possible. Here are some great examples that use jupyter widgets to simplify interaction with the data:

* `Band Excitation jupyter notebook <http://nbviewer.jupyter.org/github/pycroscopy/pycroscopy/blob/master/jupyter_notebooks/BE_Processing.ipynb>`_ developed by The Institute for Functional Imaging of Materials for supporting its users
* `Image cleaning and atom finding notebook <http://nbviewer.jupyter.org/github/pycroscopy/pycroscopy/blob/master/jupyter_notebooks/Image_Cleaning_Atom_Finding.ipynb>`_

I am not able to find an example on topic X / I find tutorial Y confusing / I need help!
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We appreciate your feedback regarding the documentation. Please send us an email at pycroscopy@gmail.com or send us a message on our `slack group <https://pycroscopy.slack.com/>`_ and we will work with you to add / improve our documentation. 

I don't know programming. Does this preclude me from using pycroscopy?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Not at all. One of the tenets of pycroscopy is lowering the barrier for scientists and researchers. To this end, we have developed `several notebooks <http://nbviewer.jupyter.org/github/pycroscopy/pycroscopy/blob/master/jupyter_notebooks/>`_ that make data visualization, discovery, and analysis interactive. You should have absolutely no trouble in using these notebooks even if you do not know programming. That being said, you would be able to make the fullest use of pycroscopy if you knew basic programming in python. 

I don't know python / I don't think I write great python code. Does this preclude me from contributing to pycroscopy?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Not really. Python is far easier to learn than many languages. If you know Matlab, Julia, C++, Fortran or any other programming language. You should not have a hard time reading our code or contributing to the codebase. 

You can still contribute your code. 

I would like to help but I don't know programming
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Your contributions are very valuable to the microscopy, imaging, and scientific community at large. You can help even if you DON'T know how to program! 

* You can spread the word - tell anyone who you think may benefit from using pycroscopy. 
* Tell us what you think of our documentation or share your own. 
* Let us know what you would like to see in pycroscopy. 

I would like to help and I am OK at programming
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Chances are that you are far better at python than you might think! Interesting tidbit - The (first version of the) first module of pycroscopy was written less than a week after we learnt how to write code in python. We weren't great programmers when we began but we would like to think that we have gotten a lot better since then. 

You can contribute in numerous ways including but not limited to:

* Writing translators to convert data from proprietary formats to the pycroscopy format - We are missing some for Park Systems, Bruker, Anasys AFMs and certain electron microscopy formats. 
* Writing image processing, signal processing code, functional fitting, etc.

Our current efforts are focussed on `making pycroscopy substantially more robust and user-friendly <https://github.com/pycroscopy/pycroscopy/blob/master/ToDo.rst#v-1-0-goals>`_. We could certainly use your help there too. Send us an email at pycroscopy@gmail.com or a message on our `slack group <https://pycroscopy.slack.com/>`_. 

Can you add my code to pycroscopy?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We would like to thank you and several others who have offered their code. We are more than happy to add your code to this project. Just as we strive to ensure that you get the best possible software from us, we ask that you do the same for others. We do NOT ask that your code be as efficient as possible. Instead, we have some simpler and easier requests:

* Encapsulate indpendent sections of your code into functions that can be used individually if required.
* Ensure that your code (functions) is well documented (numpy format) - expected inputs and outputs, purpose of functions
* Ensure that your code works in python 2.7 and python 3.5 (ideally using packages that are easy to install on Windows, Mac, and Linux)
* Provide a few examples on how one might use your code

You can look at our code in our `GitHub project <https://github.com/pycroscopy/pycroscopy>`_ to get an idea of how we organize, document, and submit our code.
