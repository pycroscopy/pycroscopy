# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import os
from warnings import warn
import numpy as np
import h5py

from sidpy.sid import Translator
from sidpy.hdf.hdf_utils import write_simple_attrs

from pyUSID.io.hdf_utils import create_indexed_group, write_main_dataset,\
    write_ind_val_dsets
from pyUSID import Dimension

from .df_utils.nanonis_utils import read_nanonis_file
# TODO: Adopt any missing features from https://github.com/paruch-group/distortcorrect/blob/master/afm/filereader/nanonisFileReader.py


class NanonisTranslatorCorrect(Translator):
    """
    Translator for Nanonis data files.

    This translator provides method to translate Nanonis data files
    (3ds, sxm, and dat) into Pycroscopy compatible HDF5 files.

    """
    def __init__(self, *args, **kwargs):
        super(Translator, self).__init__(*args, **kwargs)

        self.data_path = None
        self.folder = None
        self.basename = None
        self.parm_dict = None
        self.data_dict = None
        self.h5_path = None

    def get_channels(self):
        """
        Read the file and print the list of channels.

        Returns
        -------
        None
        """
        self._read_data(self.data_path)

        print("The following channels were found in the file:")
        for channel in self.parm_dict['channel_parms'].keys():
            print(channel)

        print('You may specify which channels to use when calling translate.')

        return

    def translate(self, filepath, data_channels=None, verbose=False):
        """
        Translate the data into a Pycroscopy compatible HDF5 file.

        Parameters
        ----------
        filepath : str
            Path to the input data file.
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
        filepath = os.path.abspath(filepath)
        folder, basename = self._parse_file_path(filepath)

        self.data_path = filepath
        self.folder = folder
        self.basename = basename
        self.h5_path = os.path.join(folder, basename + '.h5')

        if self.parm_dict is None or self.data_dict is None:
            self._read_data(self.data_path)

        if data_channels is None:
            print('No channels specified. All channels in file will be used.')
            data_channels = self.parm_dict['channel_parms'].keys()

        if verbose:
            print('Using the following channels')
            for channel in data_channels:
                print(channel)

        if os.path.exists(self.h5_path):
            os.remove(self.h5_path)

        h5_file = h5py.File(self.h5_path, 'w')

        # Create measurement group and assign attributes
        meas_grp = create_indexed_group(h5_file, 'Measurement')
        write_simple_attrs(
            meas_grp, self.parm_dict['meas_parms']
        )

        # Create datasets for positional and spectroscopic indices and values
        spec_dim = self.data_dict['Spectroscopic Dimensions']
        pos_dims = self.data_dict['Position Dimensions']
        h5_pos_inds, h5_pos_vals = write_ind_val_dsets(meas_grp, pos_dims,
                                                       is_spectral=False)
        h5_spec_inds, h5_spec_vals = write_ind_val_dsets(meas_grp, spec_dim,
                                                         is_spectral=True)

        # Create the datasets for all the channels
        num_points = h5_pos_inds.shape[0]
        for data_channel in data_channels:
            raw_data = self.data_dict[data_channel].reshape([num_points, -1])

            chan_grp = create_indexed_group(meas_grp, 'Channel')
            data_label = data_channel
            data_unit = self.parm_dict['channel_parms'][data_channel]['Unit']
            write_simple_attrs(
                chan_grp, self.parm_dict['channel_parms'][data_channel]
            )
            write_main_dataset(chan_grp, raw_data, 'Raw_Data',
                               data_label, data_unit,
                               None, None,
                               h5_pos_inds=h5_pos_inds,
                               h5_pos_vals=h5_pos_vals,
                               h5_spec_inds=h5_spec_inds,
                               h5_spec_vals=h5_spec_vals)
            h5_file.flush()

        h5_file.close()
        print('Nanonis translation complete.')

        return self.h5_path

    def _read_data(self, file_path):
        """
        Extracting data and parameters from Nanonis files.

        Parameters
        ----------
        file_path : str
            File path to the source data file

        Returns
        -------
        None

        """
        data, file_ext = read_nanonis_file(file_path)

        header_dict = data.header
        signal_dict = data.signals

        if file_ext == '.3ds':
            parm_dict, data_dict = self._parse_3ds_parms(header_dict,
                                                         signal_dict)
        elif file_ext == '.sxm':
            parm_dict, data_dict = self._parse_sxm_parms(header_dict,
                                                         signal_dict)
        else:
            parm_dict, data_dict = self._parse_dat_parms(header_dict,
                                                         signal_dict)

        self.parm_dict = parm_dict
        self.data_dict = data_dict

        return

    @staticmethod
    def _parse_sxm_parms(header_dict, signal_dict):
        """
        Parse sxm files.

        Parameters
        ----------
        header_dict : dict
        signal_dict : dict

        Returns
        -------
        parm_dict : dict

        """
        parm_dict = dict()
        data_dict = dict()

        # Create dictionary with measurement parameters
        meas_parms = {key: value for key, value in header_dict.items()
                      if value is not None}
        info_dict = meas_parms.pop('data_info')
        parm_dict['meas_parms'] = meas_parms

        # Create dictionary with channel parameters
        channel_parms = dict()
        channel_names = info_dict['Name']
        single_channel_parms = {name: dict() for name in channel_names}
        for field_name, field_value, in info_dict.items():
            for channel_name, value in zip(channel_names, field_value):
                single_channel_parms[channel_name][field_name] = value
        for value in single_channel_parms.values():
            if value['Direction'] == 'both':
                value['Direction'] = ['forward', 'backward']
            else:
                direction = [value['Direction']]
        scan_dir = meas_parms['scan_dir']
        for name, parms in single_channel_parms.items():
            for direction in parms['Direction']:
                key = ' '.join((name, direction))
                channel_parms[key] = dict(parms)
                channel_parms[key]['Direction'] = direction
                data = signal_dict[name][direction]
                if scan_dir == 'up':
                    data = np.flip(data, axis=0)
                if direction == 'backward':
                    data = np.flip(data, axis=1)
                data_dict[key] = data
        parm_dict['channel_parms'] = channel_parms

        # Position dimensions
        num_cols, num_rows = header_dict['scan_pixels']
        width, height = header_dict['scan_range']
        pos_names = ['X', 'Y']
        pos_units = ['nm', 'nm']
        pos_vals = np.vstack([
            np.linspace(0, width, num_cols),
            np.linspace(0, height, num_rows),
        ])
        pos_vals *= 1e9
        pos_dims = [Dimension(name, unit, values) for name, unit, values
                    in zip(pos_names, pos_units, pos_vals)]
        data_dict['Position Dimensions'] = pos_dims

        # Spectroscopic dimensions
        spec_dims = Dimension('arb.', 'a. u.', np.arange(1, dtype=np.float32))
        data_dict['Spectroscopic Dimensions'] = spec_dims

        return parm_dict, data_dict

    @staticmethod
    def _parse_3ds_parms(header_dict, signal_dict):
        """
        Parse 3ds files.

        Parameters
        ----------
        header_dict : dict
        signal_dict : dict

        Returns
        -------
        parm_dict : dict

        """
        parm_dict = dict()
        data_dict = dict()

        # Create dictionary with measurement parameters
        meas_parms = {key: value for key, value in header_dict.items()
                      if value is not None}
        channels = meas_parms.pop('channels')
        for key, parm_grid in zip(meas_parms.pop('fixed_parameters')
                                  + meas_parms.pop('experimental_parameters'),
                                  signal_dict['params'].T):
            # Collapse the parm_grid along one axis if it's constant
            # along said axis
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
            meas_parms[key] = parm_grid
        parm_dict['meas_parms'] = meas_parms

        # Create dictionary with channel parameters and
        # save channel data before renaming keys
        data_channel_parms = dict()
        for chan_name in channels:
            splitted_chan_name = chan_name.split(maxsplit=2)
            if len(splitted_chan_name) == 2:
                direction = 'forward'
            elif len(splitted_chan_name) == 3:
                direction = 'backward'
                splitted_chan_name.pop(1)
            name, unit = splitted_chan_name
            key = ' '.join((name, direction))
            data_channel_parms[key] = {'Name': name,
                                       'Direction': direction,
                                       'Unit': unit.strip('()'),
                                       }
            data_dict[key] = signal_dict.pop(chan_name)
        parm_dict['channel_parms'] = data_channel_parms

        # Add remaining signal_dict elements to data_dict
        data_dict.update(signal_dict)

        # Position dimensions
        nx, ny = header_dict['dim_px']
        if 'X (m)' in parm_dict:
            row_vals = parm_dict.pop('X (m)')
        else:
            row_vals = np.arange(nx, dtype=np.float32)

        if 'Y (m)' in parm_dict:
            col_vals = parm_dict.pop('Y (m)')
        else:
            col_vals = np.arange(ny, dtype=np.float32)
        pos_vals = np.hstack([row_vals.reshape(-1, 1),
                              col_vals.reshape(-1, 1)])
        pos_names = ['X', 'Y']
        pos_dims = [Dimension(label, 'nm', values)
                    for label, values in zip(pos_names, pos_vals.T)]
        data_dict['Position Dimensions'] = pos_dims

        # Spectroscopic dimensions
        sweep_signal = header_dict['sweep_signal']
        spec_label, spec_unit = sweep_signal.split(maxsplit=1)
        spec_unit = spec_unit.strip('()')
        # parm_dict['sweep_signal'] = (sweep_name, sweep_unit)
        dc_offset = data_dict['sweep_signal']
        spec_dim = Dimension(spec_label, spec_unit, dc_offset)
        data_dict['Spectroscopic Dimensions'] = spec_dim

        return parm_dict, data_dict

    @staticmethod
    def _parse_dat_parms(header_dict, signal_dict):
        """
        Parse dat files.

        Parameters
        ----------
        header_dict : dict
        signal_dict : dict

        Returns
        -------
        parm_dict : dict

        """
        pass

    def _parse_file_path(self, file_path):
        """
        Get the folder and base filename for the input data file.

        Parameters
        ----------
        file_path : str
            Path to the input data file

        Returns
        -------
        folder_path : str
            Path to the directory containing the input data file
        basename : str
            The base of the input file after stripping away the
            extension and folder from the path

        """
        # Get the folder and basename from the file path
        (folder_path, basename) = os.path.split(file_path)
        (basename, _) = os.path.splitext(basename)

        return folder_path, basename


class NanonisTranslator(NanonisTranslatorCorrect):

    def __init__(self, filepath, *args, **kwargs):
        """
        Instantiates the translator class

        Parameters
        ----------
        filepath : str
            Path to the input data file.
        args
        kwargs
        """
        super(NanonisTranslator, self).__init__(*args, **kwargs)
        warn(
            'In the future, you will need to pass the file path to the "translate()" function instead of here',
            FutureWarning)
        self.data_path = filepath

    def translate(self, data_channels=None, verbose=False):
        """
        Translate the data into a Pycroscopy compatible HDF5 file.

        Parameters
        ----------
        filepath : str
            Path to the input data file.
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
        return super(NanonisTranslator, self).translate(self.data_path,
                                                        data_channels=data_channels,
                                                        verbose=verbose)