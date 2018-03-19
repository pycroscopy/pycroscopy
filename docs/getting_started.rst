Getting Started
---------------
* Follow `these instructions <https://pycroscopy.github.io/pycroscopy/install.html>`_ to install pycroscopy
* See how we use pycroscopy for our scientific research in these `jupyter notebooks <http://nbviewer.jupyter.org/github/pycroscopy/pycroscopy/tree/master/jupyter_notebooks/>`_. Many of them are linked to `journal publications <https://pycroscopy.github.io/pycroscopy/papers_conferences.html>`_.
	* Please see the official `jupyter <http://jupyter.org>`_ website for more information about notebooks or consider watching this `youtube video <https://www.youtube.com/watch?v=HW29067qVWk>`_.
* See our `examples <https://pycroscopy.github.io/pycroscopy/auto_examples/index.html>`_ to get started on using and writing your own pycroscopy functions
* Videos and other tutorials are available at the `Institute For Functional Imaging of Materials <http://ifim.ornl.gov/resources.html>`_ 
* For more information about our functions and classes, please see our `API <https://pycroscopy.github.io/pycroscopy/api.html>`_
* We already have many translators that transform data from popular microscope data formats to pycroscopy compatible .h5 files. We also have `tutorials  <https://pycroscopy.github.io/pycroscopy/auto_examples/dev_tutorials/plot_tutorial_01_translator.html>`_ to get you started on importing your other data to pycroscopy. 
* Details regarding the definition, implementation, and guidelines for pycroscopy's `data format <https://pycroscopy.github.io/pycroscopy/data_format.html>`_ for `HDF5 <https://github.com/pycroscopy/pycroscopy/blob/master/docs/Pycroscopy_Data_Formatting.pdf>`_ are also available. 
* We can guide you to convert your file formats into pycroscopy compatible HDF5 files and help you get started with data analysis.
* We are interested in collaborating with industry members to integrate pycroscopy into instrumentation or analysis software.
* If you are interested in contributing and are looking for topics we are / will work on, please look at our `To Do <https://github.com/pycroscopy/pycroscopy/blob/master/ToDo.rst>`_ page

Branches
~~~~~~~~
* ``master`` : Stable code based off which the pip installer works. Recommended for most people.
* ``dev`` : Experimental code with new features that will be made available in ``master`` periodically after thorough testing. Note that certain features may be broken on this branch. Also note that we have currently frozen the addition of new features and are focusing on releasing a version 1.0 via the ``unity_dev`` branch. 
* ``unity_dev`` : Substantially restructured version of ``master`` aimed towards `version 1.0 <https://github.com/pycroscopy/pycroscopy/blob/master/ToDo.rst#v-1-0-goals>`_ for pycroscopy. Besides pycroscopy.core.io, no guarantees are made for the rest of the package. This branch will eventually become ``master`` by around mid 2018. Developers encouraged to add features on this branch.  
* Other branches belong to individual users / developers.
