Tutorials on Basics
====================
Here are a list of excellent tutorials from other websites and sources that describe some of the many important topics
on reading, using / running and writing code:

.. contents::

Python and  packages
--------------------
There are several concepts such as machine learning, statistics, image processing, parallel computing, file operations
that are heavily used and applied in pycroscopy. Most of these concepts are realized using add-ons or packages in
python. Here is a compilation of useful tutorials:

Python
~~~~~~
The following tutorials go over the basics of python programming:

* Python `Tutorial <https://docs.python.org/3/tutorial/>`_
* Introduction to programming in `Python 3 <https://pythonprogramming.net/beginner-python-programming-tutorials/>`_

Data Science
~~~~~~~~~~~~
The following tutorials provide an overview of different concepts ranging from data wraningling to statistical and
machine learning:

* Coursera `Machine learning course by Andrew Ng <https://www.coursera.org/learn/machine-learning>`_
* Python for Data Science book `with Notebooks <https://github.com/jakevdp/PythonDataScienceHandbook>`_
* Python `DataCamp <https://www.datacamp.com/courses/tech:python?tap_a=5644-dce66f&tap_s=75426-9cf8ad>`_
* Coursera `Deep Learning specialization by Andrew Ng <https://www.coursera.org/specializations/deep-learning>`_

HDF5 and h5py
~~~~~~~~~~~~~
pycroscopy uses the h5py python package to store data in hierarchical data format (HDF5) files. Given that pycroscopy is
designed to be file-centric, we highly recommend learning more about HDF5 and h5py:

* `Basics of HDF5 <https://portal.hdfgroup.org/display/HDF5/Learning+HDF5>`_ (especially the last three tutorials)
* `Quick start <http://docs.h5py.org/en/latest/quick.html>`_ to h5py
* Another `tutorial on HDF5 and h5py <https://www.nersc.gov/assets/Uploads/H5py-2017-Feb23.pdf>`_
* The `O-Reilly book <http://shop.oreilly.com/product/0636920030249.do>`_ where we learnt h5py

Installing software
-------------------
python
~~~~~~~
Tutorial for `installing Anaconda <https://www.youtube.com/watch?v=YJC6ldI3hWk>`_ (Python + all necessary packages)

python packages
~~~~~~~~~~~~~~~~
Two popular methods for installing packages in python are:

* `pip <https://packaging.python.org/tutorials/installing-packages/>`_:
    * included with basic python and standard on Linux and Mac OS
    * Works great for installing pure python and other simple packages
* `conda <https://conda.io/docs/user-guide/tasks/manage-pkgs.html>`_
    * included with Anaconda installation
    * Ideally suited for installing packages that have complex dependencies
* Here's a nice tutorial on `installing packages using both pip and conda <https://www.youtube.com/watch?v=Z_Kxg-EYvxM>`_

Writing code
------------
Text Editors
~~~~~~~~~~~~
These software often do not have any advanced features found in IDEs such as syntax highlighting,
real-time code-checking etc. but are simple, and most importantly, open files quickly.  Here are some excellent
text editors for each class of operating system:

* Mac OS - `Atom <https://atom.io/>`_
* Linux - `gEdit <https://wiki.gnome.org/Apps/Gedit>`_
* Windows - `Notepad++ <https://notepad-plus-plus.org/>`_

Integrated Development Environments (IDE)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
These applications often come with a built-in text editor, code management
capabilities, a python console, a terminal, integration with software repositories, etc. that make them ideal for
executing and developing code. We only recommend two IDEs at this point: Spyder for users, PyCharm for developers.
Both of these work in Linux, Mac OS, and Windows.

* `Spyder <https://en.wikipedia.org/wiki/Spyder_(software)>`_ is a great IDE that is simple and will be immediately
  familiar for users of Matlab.

    * `Basics of Spyder <https://www.youtube.com/watch?v=a1P_9fGrfnU>`_
    * `Python  with Spyder <http://datasciencesource.com/python-with-spyder-tutorial/>`_ - this was written with
      Python 2.7 in mind, but most concepts will still apply

* `Pycharm <https://www.jetbrains.com/pycharm/>`_

    * Official `PyCharm Tutorial <https://confluence.jetbrains.com/display/PYH/PyCharm+Tutorials>`_ from Jetbrains

Jupyter Notebooks
~~~~~~~~~~~~~~~~~
These are `interactive documents <http://jupyter.org/>`_ containing live cells with code, equations,
visualizations, and narrative text. The interactive nature of the document makes Jupyter notebooks an ideal medium for
conveying information and a narrative. These documents are neither text editors nor IDEs and are a separate category.

* Notebook `basics <http://nbviewer.jupyter.org/github/jupyter/notebook/blob/master/docs/source/examples/Notebook/Notebook%20Basics.ipynb>`_
* `Video <https://www.datacamp.com/community/tutorials/tutorial-jupyter-notebook>`_ tutorial
* Another `video overview <https://www.youtube.com/watch?v=HW29067qVWk>`_.

Software development basics
---------------------------
This section is mainly focused on the other tools that are mainly necessary for those interested in developing their own
code and possibly contributing back to pycroscopy.

Environments
~~~~~~~~~~~~
Environments allow users to set up and segregate software sandboxes. For example, one could set up separate environments
in python 2 and 3 to ensure that a certain desired code works in both python 2 and 3. For python users, there are two
main and popular modes of creating and managing environments - **virtual environments** and **conda environments**.

* `Virtual environment <https://docs.python.org/3/tutorial/venv.html>`_
    * Basic python ships with virtual enviroments. Anaconda is not required for this
    * How to `use venv <http://www.pythonforbeginners.com/basics/how-to-use-python-virtualenv>`_

* Conda environments
    * `Basics  <https://conda.io/docs/user-guide/getting-started.html>`_ of Conda
    * How to `manage environments in conda <https://conda.io/docs/user-guide/tasks/manage-environments.html>`_
    * `Managing Python Environments <https://www.youtube.com/watch?v=EGaw6VXV3GI>`_ with Conda

Version control
~~~~~~~~~~~~~~~
`Version control <https://vimeo.com/41027679>`_ is a tool used for managing changes in code over time. It lifts the
burden of having to check for changes line-by-line when multiple people are working on the same project. For example,
pycroscopy uses `Git <https://git-scm.com/>`_, the most popular version control software (VCS) for tracking changes etc. By default, git
typically only comes with a command-line interface. However, there are several software packages that provide a
graphical user interface on top of git. One other major benefit of using an IDE over jupyter or a text editor is that
(some) IDEs come with excellent integration with VCS like Git. Here are a collection of useful resources to get you
started on git:

* Tutorial on the `basics of git <https://www.atlassian.com/git/tutorials>`_
* Our favorite git client - `GitKraken <https://support.gitkraken.com/>`_
* Our favorite IDE with `excellent integration with Git: PyCharm <https://www.youtube.com/watch?v=vIReqoQYud8>`_
* Our own guide to `setting up and using git with PyCharm <https://github.com/pycroscopy/pycroscopy/blob/master/docs/Using%20PyCharm%20to%20manage%20repository.pdf>`_