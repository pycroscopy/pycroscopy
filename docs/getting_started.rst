Getting Started
---------------
* Follow `these instructions <./install.html>`_ to install pycroscopy
* See how we use pycroscopy for our scientific research in these `jupyter notebooks <http://nbviewer.jupyter.org/github/pycroscopy/pycroscopy/tree/master/jupyter_notebooks/>`_.
* See `journal publications <./papers_conferences.html>`_ that used pycroscopy.
* We have also compiled a list of `handy tutorials <./external_guides.html>`_ on basic / prerequisite topics such as programming in python, data analytics, machine learning, etc.
* See our `examples <./auto_examples/index.html>`_ to get started on using and writing your own pycroscopy functions
* We already have `many translators <./translators.html>`_ that transform data from popular microscope data formats to pycroscopy compatible HDF5 files.
    * We also have `tutorials  <./auto_examples/cookbooks/plot_numpy_translator.html>`_ to get you started on importing your other data to pycroscopy.
* Details regarding the definition, implementation, and guidelines for pycroscopy's `data format <./data_format.html>`_ for HDF5 are also available.
* Please see our document on the `organization of pycroscopy <./package_organization.html>`_ to find out more on what is where and why.
* If you are interested in contributing your code to pycroscopy, please look at our `guidelines <./contribution_guidelines.html>`_
* If you need detailed documentation on all our classes, functions, etc., please visit our `API <./api.html>`_
* Have questions? See our `FAQ <./faq.html>`_ to see if we have already answered them.
* Need help or need to get in touch with us? See our `contact <./contact.html>`_ information.

Guide for python novices
~~~~~~~~~~~~~~~~~~~~~~~~
For the python novices by a python novice - **Nick Mostovych, Brown University**

#. Learn about the `philosophy, purpose, etc. of pycroscopy <./about.html>`_.
#. Get an idea of the different resources available by reading the `getting started <./getting_started.html>`_ section
#. Watch the video on `installing Anaconda <https://www.youtube.com/watch?v=YJC6ldI3hWk>`_ from the `Tutorials on Basics <./external_guides.html>`_ page
#. Follow instructions on the `installation <./install.html>`_ page to install Anaconda.
#. Watch the `video tutorial <https://www.youtube.com/watch?v=HW29067qVWk>`_ from the ``Jupyter Notebooks`` section in `the Tutorials on Basics <./external_guides.html>`_ page
#. Read the whole `Tutorial on Basics page <./external_guides.html>`_. Do NOT proceed unless you are familiar with basic python programming and usage.
#. Read `Pycroscopy Data and File Format <./data_format.html>`_ This is very important and highlights the advantages of using pycroscopy. New users should not jump to the examples until they have a good understanding of the data format.
#. Depending on your needs, go through the `recommended sequence of tutorials and examples <https://pycroscopy.github.io/pycroscopy/auto_examples/index.html>`_

Tips and pitfalls
~~~~~~~~~~~~~~~~~
For the python novices by a python novice - **Nick Mostovych, Brown University**

* Documentation and examples on this website are for the latest version of pycroscopy. If something does not work as shown on this website,
  chances are that you may be using an older version of pycroscopy. Follow the instructions to `update pycroscopy to the latest version <./install.html#updating-pycroscopy>`_
* Pycroscopy has excellent documentation (+ examples too) for all functions. If you are ever confused with the usage of a
  function or class and you can get help in numerous ways:

  * If you are using jupyter notebooks, just hit the ``Shift+Tab`` keys after typing the name of your function.
    See `this quick video <https://www.youtube.com/watch?v=TgqMK1SG7XI>`_ for a demo.
    E.g. - type ``px.PycroDataset(``. Hit ``Shift+Tab`` twice or four times. You should be able to see the documentation for the
    class / function to learn how to supply inputs / extract outputs
  * Use the search function and reference the source code in the `API section <./api.html>`_ for detailed comments.
    Most detailed questions are answered there.
* Use the `PycroDataset <https://pycroscopy.github.io/pycroscopy/auto_examples/cookbooks/plot_pycro_dataset.html>`_ everywhere possible to simplify your tasks.
* Many functions in pycroscopy have a ``verbose`` keyword argument that can be set to ``True`` to get detailed print logs of intermediate steps in the function.
  This is **very** handy for debugging code

If there are tips or pitfalls you would like to add to this list, please `write to us <./contact.html>`_
