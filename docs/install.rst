Installation
============

.. attention::

  Note that only the legacy version of pycroscopy can be installed at this time.
  We are currently working on the thoroughly reimagined version of pycroscopy.

Preparing for pycroscopy
~~~~~~~~~~~~~~~~~~~~~~~~
`Pycroscopy <https://github.com/pycroscopy/pycroscopy>`_ requires many commonly used scientific and numeric python packages such as numpy, h5py etc.
To simplify the installation process, we recommend the installation of Anaconda which contains most of the prerequisite packages,
`conda <https://conda.io/docs/>`_ - a package / environment manager,
as well as an `interactive development environment <https://en.wikipedia.org/wiki/Integrated_development_environment>`_ - `Spyder <https://www.coursera.org/learn/python-programming-introduction/lecture/ywcuv/introduction-to-the-spyder-ide>`_.

Do you already have Anaconda installed?

- No? `Download and install Anaconda <https://www.anaconda.com/download/>`_ for Python 3.6
- Yes? Is your Anaconda based on python 3.6+?

  - No? Uninstall existing Python / Anaconda distribution(s). Restart computer afterwards.
  - Yes? Proceed to install pycroscopy

Compatibility
-------------
* Pycroscopy is expected to only work on Python 3.6+.
* We currently do not support 32 bit architectures

Installing pycroscopy
~~~~~~~~~~~~~~~~~~~~~
Once the appropriate Anaconda distribution has been successfully installed, pycroscopy can be installed easily either via the python package index (pypi / pip) or conda-forge (conda).

pip installation
----------------
Open a terminal (mac / linux) or command prompt (windows - be sure to install in a location where you have write access.  Don't install as administrator unless you are required to do so.) and type:
   	
.. code:: bash

  pip install pycroscopy
  
conda installation
------------------

First, open a terminal / command window. Add `conda-forge` to your channels with:

.. code:: bash

  conda config --append channels conda-forge

Once the `conda-forge` channel has been enabled, `pycroscopy` can be installed with:

.. code:: bash

  conda install pycroscopy

Installing the legacy version
-----------------------------
The upcoming version of pycroscopy will not share any code or capabilities with the legacy version.
In order to install the last legacy version of pycroscopy before this major revision, please use the following commands in a terminal:

.. code:: bash

  pip install pycroscopy==0.60.7

  
Installing from a specific branch (advanced users **ONLY**)
-----------------------------------------------------------

Here, we are installing pycroscopy from the latest development branch. Note that we do not recommend installing pycroscopy this way. 

Before you can install pycroscopy, you need to install git.

.. code:: bash

  conda install git

Once git has installed, you can install a specific branch of pycroscopy (``dev`` in this case):

.. code:: bash

  pip install -U git+https://github.com/pycroscopy/pycroscopy@dev

  
Updating pycroscopy
~~~~~~~~~~~~~~~~~~~

We recommend periodically updating your anaconda distribution.  To fully update run the following commands in a terminal / command window.

.. code:: bash

    conda upgrade anaconda
    conda update --all

If you installed pycroscopy via conda, the last command should update pycroscopy as well. 

Updating via pip
----------------

If you already have pycroscopy installed and want to update to the latest version, use the following command in a terminal / command window:

.. code:: bash

  pip install -U --no-deps pycroscopy
  
If it does not work try reinstalling the package:

.. code:: bash

  pip uninstall pycroscopy
  pip install pycroscopy

Updating via conda
------------------
If you installed pycroscopy via `conda`, open a terminal / command window and type:

.. code:: bash

  conda update pycroscopy
