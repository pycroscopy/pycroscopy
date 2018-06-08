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
            raw_data = self.data_dict[data_channel].reshape([num_points, -1])

            chan_grp = create_indexed_group(meas_grp, 'Channel')
            data_label, data_unit = data_channel.rsplit(maxsplit=1)
            data_unit = data_unit.strip('()')
            write_simple_attrs(chan_grp, self.parm_dict['channel_parms'][data_label])

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
        data, file_ext = read_nanonis_file(grid_file_path)

        header_dict = data.header
        signal_dict = data.signals

        if file_ext == '.3ds':
            parm_dict = self._parse_3ds_parms(header_dict, signal_dict)
        elif file_ext == '.sxm':
            parm_dict = self._parse_sxm_parms(header_dict, signal_dict)
        else:
            parm_dict = self._parse_dat_parms(header_dict, signal_dict)

        nx = parm_dict['num_cols']
        ny = parm_dict['num_rows']
        num_points = nx * ny

        if 'X (m)' in parm_dict:
            row_vals = parm_dict.pop('X (m)')
        else:
            row_vals = np.arange(nx, dtype=np.float32)

        if 'Y (m)' in parm_dict:
            col_vals = parm_dict.pop('Y (m)')
        else:
            col_vals = np.arange(ny, dtype=np.float32)

        pos_vals = np.hstack([row_vals.reshape(-1, 1), col_vals.reshape(-1, 1)])
        pos_names = ['X', 'Y']
        if file_ext == '.3ds':
            z_data = signal_dict['Z (m)'][:, :, 0].reshape([num_points, -1])
            pos_vals = np.hstack([pos_vals, z_data])
            pos_names.append('Z')
        pos_vals *= 1E9

        pos_dims = [Dimension(label, 'nm', values) for label, values in zip(pos_names,
                                                                            pos_vals.T)]

        self.parm_dict = parm_dict
        self.data_dict = signal_dict
        self.data_dict['Position Dimensions'] = pos_dims

        return

    @staticmethod
    def _parse_sxm_parms(header_dict, signal_dict):
        """

        Parameters
        ----------
        header_dict
        signal_dict

        Returns
        -------
        parm_dict

        """
        parm_dict = dict()
        parm_dict['channels'] = header_dict.pop('scan>channels').split(';')
        parm_dict['sweep_signal'] = 'Single Point'
        signal_dict['sweep_signal'] = np.arange(1, dtype=np.float32)
        nx, ny = header_dict['scan_pixels']
        parm_dict['num_cols'] = nx
        parm_dict['num_rows'] = ny
        # Reorganize the channel parameters
        info_dict = header_dict.pop('data_info', dict())
        chan_names = info_dict.pop('Name')
        parm_dict['channel_parms'] = {name: dict() for name in chan_names}
        for field_name, field_val in info_dict.items():
            for name, val in zip(chan_names, field_val):
                parm_dict['channel_parms'][name][field_name] = val
        return parm_dict

    @staticmethod
    def _parse_3ds_parms(header_dict, signal_dict):
        """

        Parameters
        ----------
        header_dict
        signal_dict

        Returns
        -------
        parm_dict

        """
        parm_dict = dict()
        for key, parm_grid in zip(header_dict['fixed_parameters'] + header_dict['experimental_parameters'],
                                  signal_dict['params'].T):
            # Collapse the parm_grid along one axis if it's constant along said axis
            if parm_grid.ndim > 1:
                dim_slice = list()
                # Find dimensions that are constant
                for idim in range(parm_grid.ndim):
                    tmp_grid = np.moveaxis(parm_grid.copy(), idim, 0)
                    if np.all(np.equal(tmp_grid[0], tmp_grid[1])):
                        dim_slice.append(0)
                    else:
                        dim_slice.append(slice(None))
                # print(key, dim_slice)
                # print(parm_grid[tuple(dim_slice)])
                parm_grid = parm_grid[tuple(dim_slice)]
            parm_dict[key] = parm_grid

        parm_dict['channels'] = header_dict['channels']
        parm_dict['sweep_signal'] = header_dict['sweep_signal']
        nx, ny = header_dict['dim_px']
        parm_dict['num_cols'] = nx
        parm_dict['num_rows'] = ny
        return parm_dict

    @staticmethod
    def _parse_dat_parms(header_dict, signal_dict):
        """

        Parameters
        ----------
        header_dict
        signal_dict

        Returns
        -------

        """
        pass

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
