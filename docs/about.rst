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


Who?
-----
* This project begun largely as an effort by scientists and engineers at the **I**\nstitute for **F**\unctional **I**\maging of **M**\aterials (`IFIM <https://ifim.ornl.gov>`_) to implement a python library that can support the I/O, processing, and analysis of the gargantuan stream of images that their microscopes generate (thanks to the large IFIM users community!).
* By sharing our methodology and code for analyzing scientific imaging data we hope that it will benefit the wider scientific community. We also hope, quite ardently, that other scientists would follow suit.
* Please visit our `credits and acknowledgements <./credits.html>`_ page for more information about the people behind pycroscopy.
