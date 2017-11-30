===============================
Pycroscopy Data and File Format
===============================

**Suhas Somnath**

8/8/2017

In this document we aim to provide a comprehensive overview, guidelines,
and specifications for storing imaging data using the community-driven
pycroscopy format. The credit for guidelines on structuring the data
goes to **Dr. Stephen Jesse** and the credit for implementation goes to
**Dr. Suhas Somnath** and **Chris R. Smith**

Why should you care?
--------------------

The quest for understanding more about samples has necessitated the
development of a multitude of microscopes, each capable of numerous
measurement modalities.

Typically, each commercial microscope generates data files formatted in
proprietary data formats by the instrument manufacturer. The proprietary
natures of these data formats impede scientific progress in the
following ways: 1. By making it challenging for researchers to extract
data from these files 2. Impeding the correlation of data acquired from
different instruments. 3. Inability to store results back into the same
file 4. Inflexibility to accomodate few kilobytes to several gigabytes
of data 5. Requiring different versions of analysis routines for each
format 6. In some cases, requiring proprietary software provided with
the microscope to access the data

Future concerns: 1. Several fields are moving towards the open science
paradigm which will require journals and researchers to support journal
papers with data and analysis software 2. US Federal agencies that
support scientific research require curation of datasets in a clear and
organized manner

To solve the above and many more problems, we have developed an
**instrument agnostic data format** that can be used to represent data
from any instrument, size, dimensionality, or complexity. We store data
in **heirarchical data format (HDF5)** files because we find them to be
best suited for the pycroscopy data format.

Pycroscopy data format
----------------------

Data in pycroscopy files are stored in three main kinds of datasets:

#. ``Main`` datasets that contain the raw measurements recorded from
   the instrument as well as results from processing or analysis routines
   applied to the data
#. Mandatory ``Ancillary`` datasets that are necessary to explain the
   ``main`` data
#. ``Extra`` datasets store any other data that may be of value

``Main`` Datasets
~~~~~~~~~~~~~~~~~

Regardless of origin, modality or complexity, imaging data have one
thing in common:

**The same measurement is performed at multiple spatial locations**

The data format in pycroscopy is based on this one simple ground truth.
The data always has some ``spatial dimensions`` (X, Y, Z) and some
``spectroscopic dimensions`` (time, frequency, intensity, wavelength,
temperature, cycle, voltage, etc.). **In pycroscopy, the spatial
dimensions are collapsed onto a single dimension and the spectroscopic
dimensions are flattened to the other dimensions.** Thus, all data are
stored as two dimensional grids. While the data could indeed be stored
in the original N-dimensional form, there are a few key advantages to
the 2D format:

* Researchers want to acquire ever larger datasets that
  take much longer to acquire. This has necessitated approaches such as
  sparse sampling or `compressed sensing
  <https://en.wikipedia.org/wiki/Compressed_sensing>`__ wherein
  measurements are acqurired from a few randomly sampled positions and the
  data for the rest of the positions are inferred using complex
  algorithms. Storing such sparse sampled data in the N dimensional format
  would baloon the size of the stored data even though the majority of the
  data is actually empty. Two dimensional datasets would allow the random
  measurements to be written without any empty sections.
* When acquiring measurement data, users often adjust experimental parameters
  during the experiment that may affect the size of the data, especially the
  spectral sizes. Thus, changes in experimental parameters would mean that the
  existing N dimensional set would have to be left partially (in most cases
  largely) empty and a new N dimensional dataset would have to be allocated
  with the first few positions left empty. In the case of flattened datasets,
  the current dataset can be truncated at the point of the parameter change
  and a new dataset can be created to start from the current measurement.
  Thus, no space would be wasted.

Here are some examples of how some familar data can be represented using
this paradigm:

-  **Grayscale photographs**: A single value (intensity) in is recorded
   at each pixel in a two dimensional grid. Thus, there are are two
   spatial dimensions - X, Y and one spectroscopic dimension -
   "Intensity". The data can be represented as a N x 1 matrix where N is
   the product of the number of rows and columns of pixels. The second
   axis has size of 1 since we only record one value (intensity) at each
   location. *The positions will be arranged as row0-col0, row0-col1....
   row0-colN, row1-col0....* Color images or photographs will be
   discussed below due to some very important subtleties about the
   measurement.
-  A **single Raman spectra**: In this case, the measurement is recorded
   at a single location. At this position, data is recorded as a
   function of a single (spectroscopic) variable such as wavelength.
   Thus this data is represented as a 1 x P matrix, where P is the
   number of points in the spectra
-  **Scanning Tunelling Spectroscopy or IV spectroscopy**: The current
   (A 1D array of size P) is recorded as a function of voltage at each
   position in a two dimensional grid of points (two spatial
   dimensions). Thus the data would be represented as a N x P matrix,
   where N is the product of the number of rows and columns in the grid
   and P is the number of spectroscopic points recorded.

Using prefixes ``i`` for position and ``j`` for spectroscopic, the main
dataset would be structured as:

+------------+------------+------------+--------+--------------+--------------+
| i0, j0     | i0, j1     | i0, j2     | ....   | i0, jP-2     | i0, jP-1     |
+------------+------------+------------+--------+--------------+--------------+
| i1, j0     | i1, j1     | i1, j2     | ....   | i1, jP-2     | i1, jP-1     |
+------------+------------+------------+--------+--------------+--------------+
| ........   | ........   | ........   | ....   | ..........   | ..........   |
+------------+------------+------------+--------+--------------+--------------+
| iN-2, j0   | iN-2, j1   | iN-2, j2   | ....   | iN-2, jP-2   | iN-2, jP-1   |
+------------+------------+------------+--------+--------------+--------------+
| iN-1, j0   | iN-1, j1   | iN-1, j2   | ....   | iN-1, jP-1   | iN-1, jP-1   |
+------------+------------+------------+--------+--------------+--------------+

* If the same voltage sweep were performed twice at each location, the data would be represented as N x 2 P.
  The data is still saved as a long (2*P) 1D array at each location. The number of spectroscopic dimensions
  would change from just ['Voltage'] to ['Voltage', 'Cycle'] where the second spectroscopic dimension would
  account for repetitions of this bias sweep.

  * **The spectroscopic data would be stored as it would be recorded as volt_0-cycle_0, volt_1-cycle_0.....
    volt_P-1-cycle_0, volt_0-cycle_1.....volt_P-1-cycle-1. Just like the positions**

* Now, if the bias was swept thrice from -1 to +1V and then thrice again from -2 to 2V, the data bacomes
  N x 2 * 3 P. The data now has two position dimensions (X, Y) and three spectrosocpic dimensions ['Voltage',
  'Cycle', 'Step']. The data is still saved as a (P * 2 * 3) 1D array at each location.

-  A collection of ``k`` chosen spectra would also be considered
   ``main`` datasets since the data is still structured as
   ``[instance, features]``. Some examples include:
-  the cluster centers obtained from a clustering algorithm like
   ``k-Means clustering``.
-  The abundance maps obtained from decomposition algorithms like
   ``Singular Value Decomposition (SVD)`` or
   ``Non-negetive matrix factorization (NMF)``

Compound Datasets:
^^^^^^^^^^^^^^^^^^

There are instances where multiple values are associate with a
single position and spectroscopic value in a dataset.  In these cases,
we use the Compound Dataset functionality in HDF5 to store all of the
values at each point.  This also allows us to access any combination of
the values without needing to read all of them.  Pycroscopy actually uses
compound datasets a lot more frequently than one would think. The need
and utility of compound datasets are best described with examples:

* **Color images**: Each position in these datasets contain three (red,
  blue, green) or four (cyan, black, magenta, yellow) values. One would
  naturally be tempted to simply treat these datasets as N x 3 datasets
  and it certainly is not wrong to represent data this way. However,
  storing the data in this format would mean that the red intensity was
  collected first, followed by the green, and finally by the blue. In
  other words, a notion of chronology is attached to both the position
  and spectroscopic axis if one strictly follows the pycroscopy defenition.
  While the intensities for each color may be acquired sequentially in
  detectors, we will assume that they are acquired simultaneously for
  this argument. In these cases, we store data using ``compound datasets``
  that allow the storage of multiple pieces of data within the same cell.
  While this may seem confusing or implausible, remember that computers
  store complex numbers in the same way. The complex numbers have a *real*
  and an *imaginary* component just like color images have *red*, *blue*,
  and *green* components that describe a single pixel. Therefore, color
  images in pycroscopy would be represented by a N x 1 matrix with
  compound values instead of a N x 3 matrix with real or integer values.
  One would refer to the red component at a particular position as
  ``dataset[position_index, spectroscopic_index]['red']``.
* **Functional fits**: Let's take the example of a N x P dataset whose
  spectra at each location are fitted to a complicated equation. Now the P
  points in the spectra will be represented by S coefficients that don't
  necessarily follow any order. Consequently, the result of the functional
  fit should actually be a N x 1 dataset where each element is a compound
  value made up of the S coefficients. Note that while some form of sequence
  can be forced onto the coefficients if the spectra were fit to polynomial
  equations, the drawbacks outweight the benefits:

  * Storing data in compund datasets circumvents (slicing) problems associated
    with getting a specific / the kth coeffient if the data were stored in a
    real-valued matrix instead.
  * Visualization also becomes a lot simpler since compound datasets cannot
    be plotted without specifying the component / coefficient of interest. This
    avoids plots with alternating coefficients that are several orders of
    magnitude larger / smaller than each other.

For more information on compound datasets see the `tutorial
<https://support.hdfgroup.org/HDF5/Tutor/compound.html>` from the HDF Group
and the `h5py Datasets documentation
<http://docs.h5py.org/en/latest/high/dataset.html#reading-writing-data>`

``Ancillary`` Datasets
~~~~~~~~~~~~~~~~~~~~~~

Each ``main`` dataset is always accompanied by four ancillary datasets to
help make sense of the flattened ``main`` dataset. These are the:

* The ``Position Values`` and ``Position Indices`` describe the index and
  value of any given row or spatial position in the dataset.
* The ``Spectroscopic Values`` and ``Spectroscopic Indices`` describe the
  spectroscopic information at the specific time.

In addition to serving as a legend or the key for the , these ancillary
datasets are necessary for explaining:

* the original dimensionality of the dataset
* how to reshape the data back to its N dimensional form

Much like ``main`` datasets, the ``ancillary`` datasets are also two
dimensional matricies regardless of the number of position or
spectroscopic dimensions. Given a main dataset with ``N`` positions in
``U`` dimensions and ``P`` spectral values in ``V`` dimensions:

* The ``Position Indices`` and ``Position Values`` datasets would both of the
  same size of ``N x U``, where ``U`` is the number of position
  dimensions. The columns would be arranged in ascending order of rate of
  change. In other words, the first column would be the fastest changing
  position dimension and the last column would be the slowest.

  * A simple grayscale photograph with N pixels would have ancillary position
    datasets of size N x 2. The first column would be the columns (faster)
    and the second would be the rows assuming that the data was collected
    column-by-column and then row-by-row.

* The ``Spectroscopic Values`` and ``Spectroscopic Indices`` dataset would
  both be ``V x S`` in shape, where ``V`` is the number of spectroscopic
  dimensions. Similarly to the position dimensions, the first row would be
  the fastest changing spectroscopic dimension while the last row would be
  the slowest.

The ancillary datasets are better illustrated using an example. Let's
take the **IV Spectorscopy** example from above, which has two position
dimensions - X and Y, and three spectroscopic dimensions - Voltage,
Cycle, Step.

-  If the dataset had 2 rows and 3 columns, the corresponding
   ``Position Indices`` dataset would be:

+-------+-----+
|   X   | Y   |
+=======+=====+
| 0     | 0   |
+-------+-----+
| 1     | 0   |
+-------+-----+
| 2     | 0   |
+-------+-----+
| 0     | 1   |
+-------+-----+
| 1     | 1   |
+-------+-----+
| 2     | 1   |
+-------+-----+

-  Note that indices start from 0 instead of 1 and end at N-1 instead of
   N in lines with common programming languages such as C or python.
-  Correpondingly, if the measurements were performed at X locations:
   0.0, 1.5, and 3.0 microns and Y locations: -7.0 and 2.3 nanometers,
   the ``Position Values`` dataset may look like the table below:

+----------+----------+
| X [um]   | Y [nm]   |
+==========+==========+
| 0.0      | -7.0     |
+----------+----------+
| 1.5      | -7.0     |
+----------+----------+
| 3.0      | -7.0     |
+----------+----------+
| 0.0      | 2.3      |
+----------+----------+
| 1.5      | 2.3      |
+----------+----------+
| 3.0      | 2.3      |
+----------+----------+

-  Note that X and Y have different units - microns and nanometers.
   Pycroscopy has been designed to handle variations in the units for
   each of these dimensions. Details regarding how and where to store
   the information regarding the ``labels`` ('X', 'Y') and ``units`` for
   these dimensions ('um', 'nm') will be discussed below.
-  If the dataset had 3 bias values in each cycle, each cycle repeated 2
   times, and there were 5 such bias waveforms or steps; the
   ``Spectroscopic Indices`` would be:

+---------+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
| Bias    | 0   | 1   | 2   | 0   | 1   | 2   | 0   | 1   | 2   | .   | .   | .   | 0   | 1   | 2   | 0   | 1   | 2   | 0   | 1   | 2   |
+=========+=====+=====+=====+=====+=====+=====+=====+=====+=====+=====+=====+=====+=====+=====+=====+=====+=====+=====+=====+=====+=====+
| Cycle   | 0   | 0   | 0   | 1   | 1   | 1   | 0   | 0   | 0   | .   | .   | .   | 1   | 1   | 1   | 0   | 0   | 0   | 1   | 1   | 1   |
+---------+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
| Step    | 0   | 0   | 0   | 0   | 0   | 0   | 1   | 1   | 1   | .   | .   | .   | 3   | 3   | 3   | 4   | 4   | 4   | 4   | 4   | 4   |
+---------+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+

-  Similarly, the ``Spectroscopic Values`` table would be structured as:

+------------+--------+-------+-------+--------+-------+-------+--------+-------+-------+-----+-----+-----+-------+--------+-------+-------+--------+-------+-------+
| Bias [V]   | -6.5   | 0.0   | 6.5   | -6.5   | 0.0   | 6.5   | -6.5   | 0.0   | 6.5   | .   | .   | .   | 6.5   | -6.5   | 0.0   | 6.5   | -6.5   | 0.0   | 6.5   |
+============+========+=======+=======+========+=======+=======+========+=======+=======+=====+=====+=====+=======+========+=======+=======+========+=======+=======+
| Cycle []   | 0      | 0     | 0     | 1      | 1     | 1     | 0      | 0     | 0     | .   | .   | .   | 1     | 0      | 0     | 0     | 1      | 1     | 1     |
+------------+--------+-------+-------+--------+-------+-------+--------+-------+-------+-----+-----+-----+-------+--------+-------+-------+--------+-------+-------+
| Step []    | 0      | 0     | 0     | 0      | 0     | 0     | 1      | 1     | 1     | .   | .   | .   | 3     | 4      | 4     | 4     | 4      | 4     | 4     |
+------------+--------+-------+-------+--------+-------+-------+--------+-------+-------+-----+-----+-----+-------+--------+-------+-------+--------+-------+-------+

-  Thus, the data at the fourth row and seventh column of the main
   dataset can be explained using these ancillary datasets as data from:

   -  X index of 0, with value of 0.0 microns
   -  Y index of 1 and value of 2.3 nm
      where a bias of index 0 and value of -6.5 V was being applied
      on the first cycle
      of the second bias waveform step.
-  A simple glance at the shape of these datsets would be enough to
   reveal that the data has two position dimensions (from the second
   axis of the ``Position Indices`` dataset) and three spectroscopic
   dimensions (from the first axis of the ``Spectroscopic Indices``
   dataset)

Channels
^^^^^^^^

The pycroscopy data format also allows multiple channels of information
to be recorded as separate datasets in the same file. For example, one
channel could be a spectra (1D array) collected at each location on a 2D
grid while another could be the temperature (single value) recorded by
another sensor at the same spatial positions. In this case, the two
datasets could indeed share the same ancilalry position datasets but
different spectroscopic datasets. Alternativeley, there could be other
cases where the average measurement over multiple spatial points is
recorded separately (possibly by another detector). In this case, the
two measurement datasets would not share the ancillary position datasets
as well. Other specifics regarding the implementation of different
channels will be discussed in a later section.

File Format (HDF5)
------------------

While it is indeed possible to store data in the pycroscopy format in
multiple kinds of file formats such at .mat files, plain binary files,
etc., we chose the `HDF5 file
format <https://support.hdfgroup.org/HDF5/doc/H5.intro.html>`__ since it
comfortably accomodates the pycroscopy format and offers several
advantageous features.

Information can be stored in HDF5 files in several ways:

* ``Datasets`` allow the storageo of data matricies and these are the vessels used for storing the ``main``,
  ``ancillary``, and any extra data matricies
* ``Datagroups`` are similar to folders in conventional file systems and can be used to store any number of datasets or
  datagroups themselves
* ``Attributes`` are small pieces of information, such as experimental or analytical parameters, that are stored in
  key-value pairs in the same way as dictionaries in python.  Both datagroups and datasets can store attributes.
* While they are not means to store data, ``Links`` or ``references`` can be used to provide shortcuts and aliases to
  datasets and datagroups. This feature is especially useful for avoiding duplication of datasets when two ``main``
  datasets use the same ancillary datasets.

Among the `various benefits <http://extremecomputingtraining.anl.gov/files/2015/03/HDF5-Intro-aug7-130.pdf>`__
that they offer, HDF5 files:

* are readily compatible with high-performance computing facilities
* scale very efficiently from few kilobytes to several terabytes
* can be read and modified using any language including Python, Matlab,
  C/C++, Java, Fortran, Igor Pro, etc.
* store data in a intuitive and familiar heirarchical / tree-like
  structure that is similar to files and folders in personal computers.
* faciliates storage of any number of experimental or analysis parameters
  in addition to regular data.

Implementation
--------------

Here we discuss guidelines and specifications for implementing the
pycroscopy format in HDF5 files.

``Main`` data:
~~~~~~~~~~~~~~

**Dataset** structured as (positions x time or spectroscopic values)

* ``dtype`` : uint8, float32, complex64, compound if necessary, etc.
* *Required* attributes:

  * ``quantity`` - Single string that explains the data. The physical
    quantity contained in each cell of the dataset – eg –
    'Current' or 'Deflection'
  * ``units`` – Single string for units. The units for the physical
    quantity like 'nA', 'V', 'pF', etc.
  * ``Position_Indices`` - Reference to the position indices dataset
  * ``Position_Values`` - Reference to the position values dataset
  * ``Spectroscopic_Indices`` - Reference to the spectroscopic indices
    dataset
  * ``Spectroscopic_Values`` - Reference to the spectroscopic values
    dataset

* `chunking <https://support.hdfgroup.org/HDF5/doc1.8/Advanced/Chunking/index.html>`__
  : HDF group recommends that chunks be between 100 kB to 1 MB. We
  recommend chunking by whole number of positions since data is more
  likely to be read by position rather than by specific spectral indices.

Note that we are only storing references to the ancillary datasets. This
allows multiple ``main`` datasets to share the same ancillary datasets
without having to duplicate them.

``Ancillary`` data:
~~~~~~~~~~~~~~~~~~~

**Position\_Indices** structured as (positions x spatial dimensions)

* dimensions are arranged in ascending order of rate of change. In other
  words, the fastest changing dimension is in the first column and the
  slowest is in the last or rightmost column.
* ``dtype`` : uint32
* Required attributes:

  * ``labels`` - list of strings for the column names like ['X', 'Y']
  * ``units`` – list of strings for units like ['um', 'nm']

* Optional attributes:
  * Region references based on column names

**Position\_Values** structured as (positions x spatial dimensions)

* dimensions are arranged in ascending order of rate of change. In other
  words, the fastest changing dimension is in the first column and the
  slowest is in the last or rightmost column.
* ``dtype`` : float32
* Required attributes:

  * ``labels`` - list of strings for the column names like ['X', 'Y']
  * ``units`` – list of strings for units like ['um', 'nm']

* Optional attributes:
  * Region references based on column names

**Spectroscopic\_Indices** structured as (spectroscopic dimensions x
time)

* dimensions are arranged in ascending order of rate of change.
  In other words, the fastest changing dimension is in the first row and
  the slowest is in the last or lowermost row.
* ``dtype`` : uint32
* Required attributes:

  * ``labels`` - list of strings for the column names like ['Bias', 'Cycle']
  * ``units`` – list of strings for units like ['V', ''].
    Empty string for dimensionless quantities

* Optional attributes:
  * Region references based on row names

**Spectroscopic\_Values** structured as (spectroscopic dimensions x
time)

* dimensions are arranged in ascending order of rate of change.
  In other words, the fastest changing dimension is in the first row and
  the slowest is in the last or lowermost row.
* ``dtype`` : float32
* Required attributes:

  * ``labels`` - list of strings for the column names like ['Bias', 'Cycle']
  * ``units`` – list of strings for units like ['V', ''].
    Empty string for dimensionless quantities

* Optional attributes:

  * Region references based on row names

Attributes
~~~~~~~~~~

-  All datagroups and datasets must be created with the following two
   **mandatory** attributes for better traceability:
-  ``time_stamp`` : '2017\_08\_15-22\_15\_45' (date and time of creation
   of the datagroup or dataset formatted as 'YYYY\_MM\_DD-HH\_mm\_ss' as
   a string)
-  ``machine_id`` : 'mac1234.ornl.gov' (a fully qualified domain name as
   a string)

Datagroups
~~~~~~~~~~

Datagroups in pycroscopy are used to organize datasets in an intuitive
manner.

Measurement data
^^^^^^^^^^^^^^^^

-  As mentioned earlier, microscope users may change experimental
   parameters during measurements. Even if these changes are minor, they
   can lead to misinterpretation of data if the changes are not handled
   robustly. To solve this problem, we recommend storing data under
   datagroups named as **``Measurement_00x``**. Each time the parameters
   are changed, the dataset is truncated to the point until which data
   was collected and a new datagroup is created to store the upcoming
   new measurement data.
-  Each **channel** of information acquired during the measurement gets
   its own datagroup.
-  The ``main`` datasets would reside within these channel datagroups.
-  Similar to the measurement datagroups, the channel datagroups are
   named as ``Channel_00x``. The index for the datagroup is incremented
   according to the index of the information channel.
-  Depending on the circumstances, the ancillary datasets can be shared
   among channels.

   -  Instead of the main dataset in Channel\_001 having references to
      the ancillary datasets in Channel\_000, we recommend placing the
      ancillary datasets outside the Channel datagroups in a area common
      to both channel datagroups. Typically, this is the
      Measurement\_00x datagroup.

-  This is what the tree structure in the file looks like when
   experimental parameters were changed twice and there are two channels
   of information being acquired during the measurements.
-  ``/`` (Root - also considered a datagroup)
-  Datasets common to all measurement groups (perhaps some calibration
   data that is acquired only once before all measurements)
-  ``Measurement_000`` (datagroup)

   -  ``Channel_000`` (datagroup)

      -  Datasets here

   -  ``Channel_001`` (datagroup)

      -  Datasets here

   -  Datasets common to Channel\_000 and Channel\_001

-  ``Measurement_001`` (datagroup)

   -  ``Channel_000`` (datagroup)

      -  Datasets here

   -  ``Channel_001`` (datagroup)

      -  Datasets here

   -  Datasets common to Channel\_000 and Channel\_001

-  ...

Tool (analysis / processing)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  Each time an analysis or processing routine, refered generally as
   ``tool``, is performed on a dataset of interest, the results are
   stored in new datasets within a datagroup.
-  A completely new dataset(s) and datagroup are created even if a minor
   operation is being performed on the dataset.
-  Almost always, the tool is applied to a ``main`` dataset (refered to
   as the ``parent`` dataset) and at least one of the results is
   typically also a ``main`` dataset. These new ``main`` datasets will
   either need to be linked to the ancillary matricies of the ``parent``
   or to new ancillary datasets that will need to be created.
-  The resultant dataset(s) are always stored in a datagroup whose name
   is derived from the names of the tool and the dataset. This makes the
   data **traceable**, meaning that the names of the datasets and
   datagroups are sufficient to understand what processing or analysis
   steps were applied to the data to bring it to a particular point.
-  The datagroup is named as ``Parent_Dataset-Tool_Name_00x``, where a
   ``tool`` named ``Tool_Name`` is applied to a ``main`` dataset named
   ``Parent_Dataset``.

   -  Since there is a possibility that the same tool could be applied
      to the very same dataset multiple times, we store the results of
      each run of the tool in a separate datagroup. These datagroups are
      differentiated by the index that is appened to the base-name of
      the datagroup.
   -  Note that a ``-`` separates the dataset name from the tool name
      and anything after the last ``_`` will be assumed to be the index
      of the datagroup

-  In general, the results from tools applied to datasets should be
   stored as:

    -  ``Parent_Dataset``
    -  ``Parent_Dataset-Tool_Name_000`` (datagroup comtaining results from
       first run of the ``tool`` on ``Parent_Dataset``)

       -  Attributes:

          -  ``time_stamp``
          -  ``machine_id``
          -  ``algorithm``
          -  Other tool-relevant attributes

       -  ``Dataset_Result0``
       -  ``Dataset_Result1`` ...

    -  ``Parent_Dataset-Tool_Name_001`` (datagroup comtaining results from
       second run of the ``tool`` on ``Parent_Dataset``)

-  This methodolody is illustrated with an example of applying
   ``K-Means Clustering`` on the ``Raw_Data`` acquired from a
   mesurement:

    -  ``Raw_Data`` (``main`` dataset)
    -  ``Raw_Data-Cluster_000`` (datagroup)
    -  Attributes:

       -  ``time_stamp`` : '2017\_08\_15-22\_15\_45'
       -  ``machine_id`` : 'mac1234.ornl.gov'      \* ``algorithm`` :
          'K-Means'

    -  ``Label_Indices`` (ancillary spectroscopic dataset)
    -  ``Label_Values`` (ancillary spectroscopic dataset)
    -  ``Labels`` (main dataset)

       -  Attributes:

          -  ``quantity`` : 'Cluster labels'
          -  ``units`` : ''
          -  ``Position_Indicies`` : Reference to ``Position_Indices`` from
             attribute of ``Raw_Data``
          -  ``Position_Values`` : Reference to ``Position_Values`` from
             attribute of ``Raw_Data``
          -  ``Spectrocopic_Indices`` : Reference to ``Label_Indices``
          -  ``Spectrocopic_Values`` : Reference to ``Label_Values``

    -  ``Cluster_Indices`` (ancillary positions dataset)
    -  ``Cluster_Values`` (ancillary positions dataset)
    -  ``Mean_Response`` (main dataset) <- This dataset stores the endmember
       or mean response for each cluster

       -  Attributes:

          -  ``quantity`` : copy from the ``quantity`` attribute in
             ``Raw_Data``
          -  ``units`` : copy from the ``units`` attribute in ``Raw_Data``
          -  ``Position_Indicies`` : Reference to ``Cluster_Indices``
          -  ``Position_Values`` : Reference to ``Cluster_Values``
          -  ``Spectrocopic_Indices`` : Reference to
             ``Spectrocopic_Indices`` from attribute of ``Raw_Data``
          -  ``Spectrocopic_Values`` : Reference to ``Spectrocopic_Values``
             from attribute of ``Raw_Data``

-  Note that the spectroscopic datasets that the ``Labels`` dataset link
   to are not called ``Spectroscopic_Indices`` or
   ``Spectroscopic_Values`` themselves. They only need to follow the
   specifications outlined above. The same is true for the position
   datasets for ``Mean_Response``.

Advanced topics:
----------------

-  ``Region references`` - These are references to sections of a
   ``main`` or ``ancillary`` dataset that make it easy to access data
   specfic to a specific portion of the measurement, or each column or
   row in the ancillary datasets just by their alias (intuitive strings
   for names).
