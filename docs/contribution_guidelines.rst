Guidelines for Contribution
============================

We would like to thank you and several others who have offered / are willing to contribute their code.
We are more than happy to add your code to this project.
Just as we strive to ensure that you get the best possible software from us, we ask that you do the same for others.
We do NOT ask that your code be as efficient as possible. Instead, we have some simpler and easier requests.
We have compiled a list of best practices below with links to additional information.
If you are confused or need more help, please feel free to `contact us <./contact.html>`_.

General guidelines
------------------
Our sister project, pyUSID, has `comprehensive guidelines <https://pycroscopy.github.io/pyUSID/contribution_guidelines.html>`_ about contributing code.
Please refer to that page before following pycroscopy specific guidelines below.

Pycroscopy guidelines
---------------------
At a minimum, we request that everyone follow these guidelines:

* Engineering / science-agnostic tools fit better into pyUSID while scientific functionality go into pycroscopy.
* Please ensure that your code files fit into our `package structure <./package_structure.html>`_ (``io``, ``processing``, ``viz``, ``analysis``, or ``simulation``)
* Provide a few examples on how one might use your code - preferably via a jupyter notebook.

While not mandatory, we encourage everyone to follow the additional guidelines below to ensure that your code is fully pycroscopy-compatible and is used by everyone:

* All functionality in pyUSID and pycroscopy revolves around the `Universal Spectroscopic and Imaging Data (USID) <../USID/index.html>`_ model
  and its implementation into HDF5 files. Data is read from HDF5 files, processed, and written back to it.
  Therefore, it will be much easier to understand the rationale for certain practices in pyUSID and pycroscopy once USID is understood.
* Please consider familiarizing yourself with the `examples <https://pycroscopy.github.io/pyUSID/auto_examples/index.html>`_
  on functionality available in pyUSID so that you can use the available functionality to simplify your code
  in addition to avoiding the development of duplicate code.
  If you have not yet begun developing your code, please note that it will be far easier to understand USID, pyUSID and
  use tools in pyUSID while developing the code rather than adapting pre-written code (that has not used pyUSID) to work within pycroscopy.
* If you are contributing any data processing / analysis algorithms or tools to operate on USID HDF5 files, please consider
  using the `Process class <https://pycroscopy.github.io/pyUSID/auto_examples/intermediate/plot_process.html>`_ to formalize and standardize your computation.
* If you are interested in contributing translators (code that extracts data and metadata from proprietary file formats and writes all information in to USID HDF5 files),
  please see our `tutorials on writing translators <https://pycroscopy.github.io/pyUSID/auto_examples/beginner/plot_numpy_translator.html>`_

