Data Translators
=================
* Pycroscopy uses ``Translators`` to extract data and metadata from files (often measurement data stored in instrument-generated proprietary file formats) and write them into `Universal Spectroscopy and Imaging Data (USID) HDF5 files <../../pyUSID/data_format.html>`_.
* You can write your own ``Translator`` easily by following `this example <https://pycroscopy.github.io/pyUSID/auto_examples/beginner/plot_numpy_translator.html>`_ on our sister project's documentation.
* Below is a list of ``Translators`` already available in pycroscopy to translate data.
* These translators can be accessed via ``pycroscopy.io.translators`` or ``pycroscopy.translators``
* We tend to add new ``Translators`` to this list frequently.

  * We understand that this list does not (yet) comprehensively cover all modalities and instrument manufacturers, but we are working towards a comprehensive set of translators.
  * We only have access to a small subset of all available instruments and data which limits our ability to add more translators.
  * Given that this is a **community-driven effort**, you can help by providing:

    * Example datasets (scans, force curves, spectra, force-maps, spectra acquired on grids of locations, etc.)
    * Links to existing packages that have figured out how to extract this data. (More often than not, there are researchers who have put up their code)
    * Your own code for extracting data. We invite you to come onboard and add your tools to the package.
    * Guidance in correctly extracting the metadata (parameters) and data
    * Your time. We are interested in collaborating with you to develop translators.
* We are also interested in collaborating with instrument manufacturers to integrate pycroscopy into instrumentation or analysis software.
* We are working on writing translators to popular open-source software / formats such as ``WSxM``, ``Gwyddion``, and ``ImageJ``.

Generic File Formats
--------------------
* PNG, TIFF images - ``ImageTranslator``
* Movie (frames of) represented by a stack of images - ``MovieTranslator``
* numpy data in memory - ``NumpyTranslator``
* Gwyddion - ``GwyddionTranslator`` - work in progress

Scanning Transmission Electron Microscopy (STEM)
------------------------------------------------
* Nion Company - NData - ``NDataTranslator``
* One View - ``OneViewTranslator``
* Digital Micrograph DM3 and DM4 files - ``PtychographyTranslator``
* TIFF image stack for 4D STEM - ``PtychographyTranslator``

Scanning Tunnelling Microscopy (STM)
------------------------------------
* Omicron STMs - Scanning Tunnelling Spectroscopy - ``AscTranslator``
* Nanonis Controllers - ``NanonisTranslator``

Atomic Force Microscopy (AFM)
-----------------------------
Common formats
~~~~~~~~~~~~~~~
* Asylum Research - Igor IBWs for images and force curves - ``IgorIBWTranslator``
* Bruker / Veeco / Digital Instruments - images, force curves, force maps - ``BrukerAFMTranslator``
* Nanonis Controllers - ``NanonisTranslator``

IFIM specific
~~~~~~~~~~~~~~
* Band Excitation (BE):

  * BE-Line and BEPS (Pre 2013) - ``BEodfTranslator``
  * BEPS (Post 2013) - ``BEPSndfTranslator``
  * BE Relaxation - ``BEodfRelaxationTranslator``
  * Time Resolved Kelvin Probe Force Microscopy (trKPFM) - ``TRKPFMTranslator``
  * Synthetic BEPS data generator - ``FakeBEPSGenerator``
  * Post 2016 Band Excitation data patcher - ``LabViewH5Patcher``

* General Mode (G-mode):

  * G-Mode Line - ``GLineTranslator``
  * G-Mode Current-Voltage (G-IV) - ``GIVTranslator``
  * G-Mode Frequency Tune - ``GTuneTranslator``
  * General Dynamic Mode (GDM) - ``GDMTranslator``
  * Speedy First Order Reversal Curve (SPORC) - ``SporcTranslator``
