

.. _sphx_glr_auto_examples_plot_load_dataset_example.py:


============
Load Dataset
============

Conventionally, the h5py package is used to create, read, write, and modify h5 files.

Pycroscopy uses h5py to read hdf5 files and the ioHDF5 subpackage (a light wrapper around h5py) within pycroscopy
to create / write back to the file. Please see the example on writing hdf5 files for more information on creating and
writing to h5 files using pycroscopy.

In the event that modification / addition of data to the existing file is of interest,
it is recommended that the file be opened using ioHDF5. The same h5py handles can be obtained easily from ioHDF5.
Note that ioHDF5 always reads the files in the 'r+' mode that allows modification of the file.

In this example, we will be loading the Raw_Data dataset from the hdf5 file.






.. rst-class:: sphx-glr-script-out

 Out::

    /
    Measurement_000
    Measurement_000/Channel_000
    Measurement_000/Channel_000/Bin_FFT
    Measurement_000/Channel_000/Bin_Frequencies
    Measurement_000/Channel_000/Bin_Indices
    Measurement_000/Channel_000/Bin_Step
    Measurement_000/Channel_000/Bin_Wfm_Type
    Measurement_000/Channel_000/Excitation_Waveform
    Measurement_000/Channel_000/Noise_Floor
    Measurement_000/Channel_000/Position_Indices
    Measurement_000/Channel_000/Position_Values
    Measurement_000/Channel_000/Raw_Data
    Measurement_000/Channel_000/Spatially_Averaged_Plot_Group_000
    Measurement_000/Channel_000/Spatially_Averaged_Plot_Group_000/Bin_Frequencies
    Measurement_000/Channel_000/Spatially_Averaged_Plot_Group_000/Mean_Spectrogram
    Measurement_000/Channel_000/Spatially_Averaged_Plot_Group_000/Spectroscopic_Parameter
    Measurement_000/Channel_000/Spatially_Averaged_Plot_Group_000/Step_Averaged_Response
    Measurement_000/Channel_000/Spectroscopic_Indices
    Measurement_000/Channel_000/Spectroscopic_Values
    Measurement_000/Channel_000/UDVS
    Measurement_000/Channel_000/UDVS_Indices
    h5_raw1:  <HDF5 dataset "Raw_Data": shape (16384, 119), type "<c8">
    h5_meas_grp: <HDF5 group "/Measurement_000" (1 members)>
    h5_raw1_alias_1: <HDF5 dataset "Raw_Data": shape (16384, 119), type "<c8">
    h5_dsets: [<HDF5 dataset "Raw_Data": shape (16384, 119), type "<c8">
    located at: 
    /Measurement_000/Channel_000/Raw_Data 
    Data contains: 
    Unknown quantity (unknown units) 
    Data dimensions and original shape: 
    Position Dimensions: 
    X - size: 128 
    Y - size: 128 
    Spectroscopic Dimensions: 
    Frequency - size: 119]
    h5_raw1_alias_2: <HDF5 dataset "Raw_Data": shape (16384, 119), type "<c8">
    located at: 
    /Measurement_000/Channel_000/Raw_Data 
    Data contains: 
    Unknown quantity (unknown units) 
    Data dimensions and original shape: 
    Position Dimensions: 
    X - size: 128 
    Y - size: 128 
    Spectroscopic Dimensions: 
    Frequency - size: 119
    All aliases of the same dataset? False
    h5_raw2: <HDF5 dataset "Raw_Data": shape (16384, 119), type "<c8">




|


.. code-block:: python


    # Code source: pycroscopy
    # Liscense: MIT

    from __future__ import division, print_function, absolute_import, unicode_literals
    import h5py
    try:
        # This package is not part of anaconda and may need to be installed.
        import wget
    except ImportError:
        import pip
        pip.main(['install', 'wget'])
        import wget

    from os import remove
    import pycroscopy as px

    # Downloading the file from the pycroscopy Github project
    url = 'https://raw.githubusercontent.com/pycroscopy/pycroscopy/master/data/BELine_0004.h5'
    h5_path = 'temp.h5'
    _ = wget.download(url, h5_path)

    # h5_path = px.io_utils.uiGetFile(caption='Select .h5 file', filter='HDF5 file (*.h5)')

    # Read the file using using h5py:
    h5_file1 = h5py.File(h5_path, 'r')

    # Look at the contents of the file:
    px.hdf_utils.print_tree(h5_file1)

    # Access the "Raw_Data" dataset from its absolute path
    h5_raw1 = h5_file1['Measurement_000/Channel_000/Raw_Data']
    print('h5_raw1: ', h5_raw1)

    # We can get to the same dataset through relative paths:

    # Access the Measurement_000 group first
    h5_meas_grp = h5_file1['Measurement_000']
    print('h5_meas_grp:', h5_meas_grp)

    # Now we can access the "Channel_000" group via the h5_meas_grp object
    h5_chan_grp = h5_meas_grp['Channel_000']

    # And finally, the same raw dataset can be accessed as:
    h5_raw1_alias_1 = h5_chan_grp['Raw_Data']
    print('h5_raw1_alias_1:', h5_raw1_alias_1)

    # Another way to get this dataset is via functions written in pycroscopy:
    h5_dsets = px.hdf_utils.getDataSet(h5_file1, 'Raw_Data')
    print('h5_dsets:', h5_dsets)

    # In this case, there is only a single Raw_Data, so we an access it simply as:
    h5_raw1_alias_2 = h5_dsets[0]
    print('h5_raw1_alias_2:', h5_raw1_alias_2)

    # Let's just check to see if these are indeed aliases of the same dataset:
    print('All aliases of the same dataset?', h5_raw1 == h5_raw1_alias_1 and h5_raw1 == h5_raw1_alias_2)

    # Let's close this file
    h5_file1.close()

    # Load the dataset with pycroscopy
    hdf = px.ioHDF5(h5_path)

    # Getting the same h5py handle to the file:
    h5_file2 = hdf.file

    h5_raw2 = h5_file2['Measurement_000/Channel_000/Raw_Data']
    print('h5_raw2:', h5_raw2)

    h5_file2.close()

    # Delete the temporarily downloaded h5 file:
    remove(h5_path)

**Total running time of the script:** ( 0 minutes  42.592 seconds)



.. only :: html

 .. container:: sphx-glr-footer


  .. container:: sphx-glr-download

     :download:`Download Python source code: plot_load_dataset_example.py <plot_load_dataset_example.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: plot_load_dataset_example.ipynb <plot_load_dataset_example.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_
