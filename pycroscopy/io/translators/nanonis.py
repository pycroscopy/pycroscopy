# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import, unicode_literals

import os
import numpy as np
import h5py
from pyUSID.io.hdf_utils import create_indexed_group, write_main_dataset, write_simple_attrs, Dimension, \
    write_ind_val_dsets
from pyUSID.io.translator import Translator
from pyUSID.io.write_utils import get_aux_dset_slicing
from .df_utils.nanonis_utils import read_nanonis_file
# TODO: Adopt any missing features from https://github.com/paruch-group/distortcorrect/blob/master/afm/filereader/nanonisFileReader.py


class NanonisTranslator(Translator):
    """
    Translate Nanonis data files (3ds, sxm, and dat) into Pycroscopy compatible HDF5 files
    """
    def __init__(self, filepath, *args, **kwargs):
        """

        Parameters
        ----------
        filepath : str
           Path to the input data file.
        """
        super(Translator, self).__init__(*args, **kwargs)

        filepath = os.path.abspath(filepath)
        folder, basename = self._parse_file_path(filepath)

        self.data_path = filepath
        self.folder = folder
        self.basename = basename
        self.parm_dict = None
        self.data_dict = None
        self.h5_path = os.path.join(folder, basename + '.h5')

    def get_channels(self):
        """
        Read the file and print the list of channels.

        Returns
        -------
        None
        """
        self._read_data(self.data_path)

        print("The following channels were found in the file:")
        for channel in self.parm_dict['channels']:
            print(channel)

        print('You may specify which channels to use when calling translate.')

        return

    def translate(self, data_channels=None, verbose=False):
        """
        Translates the data in the Nanonis file into a Pycroscopy compatible HDF5 file.

        Parameters
        ----------
        data_channels : (optional) list of str
            Names of channels that will be read and stored in the file.
            If not given, all channels in the file will be used.
        verbose : (optional) Boolean
            Whether or not to print statements

        Returns
        -------
        h5_path : str
            Filepath to the output HDF5 file.

        """
        if self.parm_dict is None or self.data_dict is None:
            self._read_data(self.data_path)

        if data_channels is None:
            print('No channels specified.  All channels in file will be used.')
            data_channels = self.parm_dict['channels']

        if verbose:
            print('Using the following channels')
            for channel in data_channels:
                print(channel)

        if os.path.exists(self.h5_path):
            os.remove(self.h5_path)

        h5_file = h5py.File(self.h5_path, 'w')

        meas_grp = create_indexed_group(h5_file, 'Measurement')

        dc_offset = self.data_dict['sweep_signal']

        spec_label, spec_units = self.parm_dict['sweep_signal'].split()
        spec_units = spec_units.strip('()')
        spec_dim = Dimension(spec_label, spec_units, dc_offset)
        pos_dims = self.data_dict['Position Dimensions']

        h5_pos_inds, h5_pos_vals = write_ind_val_dsets(meas_grp, pos_dims, is_spectral=False)
        h5_spec_inds, h5_spec_vals = write_ind_val_dsets(meas_grp, spec_dim, is_spectral=True)

        num_points = h5_pos_inds.shape[0]

        for data_channel in data_channels:
            raw_data = self.data_dict[data_channel].reshape([num_points, -1]) * 1E9  # Convert to nA

            chan_grp = create_indexed_group(meas_grp, 'Channel')

            data_label, data_unit = data_channel.rsplit(maxsplit=1)
            data_unit = data_unit.strip('()')

            write_main_dataset(chan_grp, raw_data, 'Raw_Data',
                               data_label, data_unit,
                               None, None,
                               h5_pos_inds=h5_pos_inds, h5_pos_vals=h5_pos_vals,
                               h5_spec_inds=h5_spec_inds, h5_spec_vals=h5_spec_vals)

            h5_file.flush()

        h5_file.close()
        print('Nanonis translation complete.')

        return self.h5_path

    def _read_data(self, grid_file_path):
        """
        Handles reading the data from the file and extracting the needed parameters for translating.

        Parameters
        ----------
        grid_file_path : str
            File path to the source data file

        Returns
        -------
        None

        """
        header_dict, signal_dict = read_nanonis_file(grid_file_path)

        parm_dict = dict()
        for key, parm_grid in zip(header_dict['fixed_parameters'] + header_dict['experimental_parameters'],
                                  signal_dict['params'].T):
            parm_dict[key] = parm_grid

        parm_dict['channels'] = header_dict['channels']
        parm_dict['sweep_signal'] = header_dict['sweep_signal']
        nx, ny = header_dict['dim_px']
        parm_dict['num_cols'] = nx
        parm_dict['num_rows'] = ny

        num_points = nx * ny
        pos_vals = np.hstack([parm_dict['X (m)'].reshape(-1, 1), parm_dict['Y (m)'].reshape(-1, 1)])
        z_data = signal_dict['Z (m)'][:, :, 0].reshape([num_points, -1])
        pos_vals = np.hstack([pos_vals, z_data])
        pos_vals *= 1E9

        pos_dims = (Dimension(label, 'nm', values) for label, values in zip(['X', 'Y', 'Z'],
                                                                            pos_vals.T))

        self.parm_dict = parm_dict
        self.data_dict = signal_dict
        self.data_dict['Position Dimensions'] = pos_dims

        return

    def _parse_file_path(self, file_path):
        """
        Get the folder and base filename for the input data file

        Parameters
        ----------
        file_path : str
            Path to the input data file

        Returns
        -------
        folder_path : str
            Path to the directory containing the input data file
        basename : str
            The base of the input file after stripping away the extension and folder
            from the path

        """
        # Get the folder and basename from the file path
        (folder_path, basename) = os.path.split(file_path)
        (basename, _) = os.path.splitext(basename)

        return folder_path, basename
