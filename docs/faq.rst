Frequently asked questions
==========================

.. contents::

My question was still not answered here
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Please see a `complementary list of FAQ <../../pyUSID/faq.html>`_ on our sister package - pyUSID.

Pycroscopy philosophy
---------------------

What is pycroscopy and how is it different from python?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
`Python <https://www.python.org>`_ is an (interpreted) programming langauge similar to R, Java, C, C++, Fortran etc. `Pycroscopy <https://pycroscopy.github.io/pycroscopy/about.html#what>`_ is an addon module to python that provides it the ability to analyze scientific data (especially imaging data). As an (oversimplified) analogy, think of python as Windows or Mac OS and pycroscopy as Firefox or Chrome or Safari.

Is pycroscopy only for the scientific imaging / microscopy communities? 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
**Not at all**. We have ensured that the basic data and file formatting paradigm is general enough that it can be extended to any other scientific domain so long as each experiment involves ``N`` identical measurements of ``S`` values.

Note that one of the major strengths of pycroscopy is that it can be **science- and instrument-agnostic**. For example, some of our scientific analysis algorithms such as curve-fitting for spectra, image denoising are written in a general enough manner that they can easily find applications in scientific domains beyond imaging and microscopy.

Our data and file format as well as programming framework can easily be extended to or adopted by other scientific domains such as neutron science, nuclear sciences, etc.

We are eager to hear about the many research domains that find our data format and package useful. Please send us an email at pycroscopy@gmail.com

Who uses pycroscopy?
~~~~~~~~~~~~~~~~~~~~
* `The Institute for Functional Imaging of Materials (IFIM) <http://ifim.ornl.gov>`_ at `Oak Ridge National Laboratory <www.ornl.gov>`_ uses pycroscopy exclusively for in-house research as well as supporting the numerous users who visit IFIM to use their state-of-art scanning probe microscopy techniques.
* Synchrotron Radiation Research at Lund University
* Nuclear Engineering and Health Physics, Idaho State University
* Prof. David Ginger's group at Department of Chemistry, University of Washington
* Idaho National Laboratory
* Central Michigan University
* Iowa State University
* George Western University
* Brown University
* University of Mons
* and many more groups in universities and national labs.
* Please get in touch with us if you would like your group / university to be added here.

How is pycroscopy different from ImageJ, ImageSXM, WSxM, Gwyddion, or xarray?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* **Data sizes and dimensionality**: `ImageJ <https://imagej.nih.gov/ij/>`_, `FIJI <https://fiji.sc>`_, `ImageSXM <https://www.liverpool.ac.uk/~sdb/ImageSXM/>`_,
  `WSxM <http://www.wsxm.es/download.html>`_, `SpectraFox <https://spectrafox.com>`_, and `OpenFovea <http://www.freesbi.ch/en/openfovea>`_ are all excellent
  software packages for dealing with conventional and popular microscopy data such as 2D images or a handful of (simple) spectra that are at best 1- 100 MB in size. We think that pycroscopy is
  complementary to these  other software and packages. Pycroscopy was built from scratch to handle **arbitrarily large datasets** (gigabytes / terabytes) of
  datasets which regularly have a large number of position or spectroscopic dimensions (we have had absolutely no problem dealing with datasets with more than
  9 unique dimensions - see `Band Excitation Polarization Switching + First Order Reversal Curve probing <https://pycroscopy.github.io/pycroscopy/auto_examples/dev_tutorials/plot_tutorial_03_multidimensional_data.html#sphx-glr-auto-examples-dev-tutorials-plot-tutorial-03-multidimensional-data-py>`_).
* **Data centric**: In addition, Pycroscopy takes a data centric approach aimed towards open science wherein all the data starting from the raw measurement from
  the instrument, all the way to the final data that is plotted in the resulting scientific publication, are contained in the same file. The processing steps applied
  to the data are completely transparent and traceable.
* **Data Processing framework**: Most of the aforementioned packages are a collection of several popularly used algorithms applied to data. Again, most of these are applied to small datasets in memory.
  Perhaps more importantly, the algorithms are tied to a specific scientific domain / application. In contrast, the universal data format used by pycroscopy allows the development of a
  single version of an algorithm that can be applied to any data.
* **Scalable**: Furthermore, pycroscopy was developed from the ground up to run on laptops while aiming towards compatibility with supercomputers. Nearly all the aforementioned
  software are applicable to laptops only. Supercomputer / cloud computing scaling in pycroscopy will arrive in the later part of 2018.
* **Flexibility / Customizable**: Like ImageJ / FIJI, it is far easier to add features to pycroscopy when compared to  WSxM or Gwyddion
* **User Interface**: Pycroscopy relies on Jupyter notebooks + interactive widgets instead of graphical interface used in most other alternatives.
* **Other complimentary software**:

  * `GXSM <http://gxsm.sourceforge.net>`_ is another software package that focuses more on the data acquisition from instruments rather than advanced data analysis.
  * `xarray <https://github.com/pydata/xarray>`_ has many similar and more advanced features for handling scientific multidimensional data compared to pycroscopy. However, while pycroscopy is a file-based package, xarray enables the features for data in memory only. We see xarray as a package that is complementary to pycroscopy.
* For simple data operations such as flattening, finding maximum, etc. on 2D images or spectra, ImageJ / Gwyddion / WSxM may be better alternatives to pycroscopy.

What's the difference between pyUSID and pycroscopy?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pyUSID is the **science-agnostic** package that mainly contains the file handling tools upon which pycroscopy is built.
Pycroscopy contains scientific and data analysis functions, instrument-specific data translators, etc.
Pycroscopy depends on pyUSID just as the scipy family of packages depend on numpy.

When should I use pyUSID and pycroscopy?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
See the entry under `June 28th 2018 <./whats_new.html>`_

Using pycroscopy
----------------

Why doesn't pycroscopy use graphical user interfaces?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Traditional graphical interfaces are rather time consuming to develop. Instead, we provide jupyter widgets to interact with data wherever possible. Here are some great examples that use jupyter widgets to simplify interaction with the data:

* `Band Excitation jupyter notebook <http://nbviewer.jupyter.org/github/pycroscopy/pycroscopy/blob/master/jupyter_notebooks/BE_Processing.ipynb>`_ developed by The Institute for Functional Imaging of Materials for supporting its users
* `Image cleaning and atom finding notebook <http://nbviewer.jupyter.org/github/pycroscopy/pycroscopy/blob/master/jupyter_notebooks/Image_Cleaning_Atom_Finding.ipynb>`_

What do I do when something is broken?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Often, others may have encountered the same problem and may have brought up a similar issue. Try searching on google and trying out some suggested solutions.
If this does not work, get in touch with us on our `Google group <https://groups.google.com/forum/#!forum/pycroscopy>`_.
If something is indeed broken, please raise an ``issue`` `here <https://github.com/pycroscopy/pycroscopy/issues>`_ and one of us will work with you to resolve the problem.

Do I still need to use standard software for plotting figures for papers?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Not at all. Python has an excellent set of libraries for generating even complicated figures for journal papers. Pycroscopy has `several functions <https://pycroscopy.github.io/pycroscopy/auto_examples/user_tutorials/plot_utils.html#sphx-glr-auto-examples-user-tutorials-plot-utils-py>`_ that make it easier to quickly generate publication-ready figures. There are `several publications <https://pycroscopy.github.io/pycroscopy/papers_conferences.html#journal-papers-using-pycroscopy>`_ that have only used pycroscopy and matplotlib to generate figures for papers. If you are still not convinced, you can always export your data to text / csv files and use conventional softwares like `Origin Pro <https://www.originlab.com>`_.

How can I reference pycroscopy?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Please reference our `Arxiv <https://arxiv.org/abs/1903.09515>`_ paper for now.
This manuscript was submitted to Advanced Structural and Chemical Imaging recently and is currently being peer-reviewed.

Can Pycroscopy read data files from instrument X?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Pycroscopy has numerous translators that extract the data and metadata (e.g. - instrument / imaging parameters) from some
popular file formats and store the information in HDF5 files.
You can find a list of available `translators here <./translators.html>`_.

Becoming a part of the effort
-----------------------------
I would like to help but I don't know programming
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Your contributions are very valuable to the imaging and scientific community at large. You can help even if you DON'T know how to program!

* You can spread the word - tell anyone who you think may benefit from using pycroscopy. 
* Tell us what you think of our documentation or share your own. 
* Let us know what you would like to see in pycroscopy.
* Put us in touch with others working on similar efforts so that we can join forces.
* Guide us in `developing data translators <./translators.html>`_

I would like to help and I am OK at programming
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Chances are that you are far better at python than you might think! Interesting tidbit - The (first version of the) first module of pycroscopy was written less than a week after we learnt how to write code in python. We weren't great programmers when we began but we would like to think that we have gotten a lot better since then. 

You can contribute in numerous ways including but not limited to:

* Writing ``translators`` to convert data from proprietary formats to the pycroscopy format
* Writing image processing, signal processing code, functional fitting, etc.

Send us an email at pycroscopy@gmail.com or a message on our `slack group <https://pycroscopy.slack.com/>`_.

Can you add my code to pycroscopy?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Please see our `guidelines for contributing code <./contribution_guidelines.html>`_
