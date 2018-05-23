Data Translators
=================
* Pycroscopy uses ``Translators`` to extract data and metadata from files (often measurement data stored in instrument-generated proprietary file formats) and write them into `pycroscopy formatted HDF5 files <https://pycroscopy.github.io/pycroscopy/data_format.html>`_. 
* You can write your own ``Translator`` easily by following `this example <https://pycroscopy.github.io/pycroscopy/auto_examples/cookbooks/plot_numpy_translator.html>`_
* Below is a list of ``Translators`` already available in pycroscopy to translate data. We tend to add new ``Translators`` to this list frequently. 
* We are interested in collaborating with industry members to integrate pycroscopy into instrumentation or analysis software.
* These translators can be accessed via ``pycroscopy.io.translators`` or ``pycroscopy.translators``

Generic File Formats
--------------------
* PNG, TIFF images - ``ImageTranslator``
* Movie (frames of) represented by a stack of images - ``MovieTranslator``
* numpy data in memory - ``NumpyTranslator``

Scanning Transmission Electron Microscopy (STEM)
------------------------------------------------
* Nion Company - NData - ``NDataTranslator``
* One View - ``OneViewTranslator``
* Digital Micrograph DM3 and DM4 files - ``PtychographyTranslator``
* TIFF image stack for 4D STEM - ``PtychographyTranslator``

Atomic Force Microscopy
-----------------------
Common formats
~~~~~~~~~~~~~~~
* Asylum Research - Igor IBWs for images and force curves - ``IgorIBWTranslator``
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
