import os
import numpy as np

from ...core.io.hdf_utils import get_h5_obj_refs, link_as_main
from ...core.io.translator import Translator
from ...core.io.write_utils import get_aux_dset_slicing
from .df_utils.nanonis_utils import read_nanonis_file
from ..virtual_data import VirtualDataset, VirtualGroup
from ..hdf_writer import HDFwriter
from ..write_utils import build_ind_val_dsets


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

        num_points = self.data_dict['Position Indices'].shape[0]
        dc_offset = self.data_dict['sweep_signal']

        spec_label, spec_units = self.parm_dict['sweep_signal'].split()
        spec_units = spec_units.strip('()')
        ds_spec_inds, ds_spec_vals = build_ind_val_dsets([dc_offset.size], labels=[spec_label], units=[spec_units])
        ds_spec_vals.data[:] = dc_offset

        ds_pos_inds = VirtualDataset('Position_Indices', self.data_dict['Position Indices'])
        ds_pos_vals = VirtualDataset('Position_Values', self.data_dict['Position Values'])

        ds_pos_inds.attrs['labels'] = self.data_dict['Position labels']
        ds_pos_inds.attrs['units'] = self.data_dict['Position units']
        ds_pos_vals.attrs['labels'] = self.data_dict['Position labels']
        ds_pos_vals.attrs['units'] = self.data_dict['Position units']

        ds_meas_grp = VirtualGroup('Measurement_')
        ds_meas_grp.addChildren([ds_spec_vals, ds_spec_inds, ds_pos_inds, ds_pos_vals])

        if os.path.exists(self.h5_path):
            os.remove(self.h5_path)
        hdf = HDFwriter(self.h5_path)
        h5_refs = hdf.writeData(ds_meas_grp, print_log=True)

        aux_ds_names = ['Position_Indices', 'Position_Values',
                        'Spectroscopic_Indices', 'Spectroscopic_Values']

        h5_pos_inds, h5_pos_vals, h5_spec_inds, h5_spec_vals = get_h5_obj_refs(aux_ds_names, h5_refs)

        for data_channel in data_channels:
            raw_data = self.data_dict[data_channel].reshape([num_points, -1]) * 1E9  # Convert to nA

            ds_raw = VirtualDataset('Raw_Data', raw_data)
            data_label, data_unit = data_channel.rsplit(maxsplit=1)
            data_unit = data_unit.strip('()')
            ds_raw.attrs['units'] = data_unit
            ds_raw.attrs['quantity'] = data_label

            ds_chan_grp = VirtualGroup('Channel_', parent=ds_meas_grp.name)

            ds_chan_grp.addChildren([ds_raw])
            ds_meas_grp.addChildren([ds_chan_grp])

            h5_refs = hdf.writeData(ds_chan_grp, print_log=verbose)
            h5_main = get_h5_obj_refs(['Raw_Data'], h5_refs)[0]
            link_as_main(h5_main, h5_pos_inds, h5_pos_vals, h5_spec_inds, h5_spec_vals)

            hdf.file.flush()

        hdf.close()
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
        xinds, yinds = np.ogrid[0:nx, 0:ny]
        xinds = np.repeat(xinds.flatten(), ny)
        yinds = np.tile(yinds.flatten(), nx)
        pos_dims = np.hstack([parm_dict['X (m)'].reshape(-1, 1), parm_dict['Y (m)'].reshape(-1, 1)])
        pos_inds = np.hstack([xinds.reshape(-1, 1), yinds.reshape(-1, 1)])
        z_data = signal_dict['Z (m)'][:, :, 0].reshape([num_points, -1])
        pos_dims = np.hstack([pos_dims, z_data])
        pos_dims *= 1E9
        pos_inds = np.hstack([pos_inds, np.arange(z_data.size).reshape(z_data.shape)])
        pos_labs = get_aux_dset_slicing(['X', 'Y', 'Z'], is_spectroscopic=False)
        pos_units = ['nm', 'nm', 'nm']

        self.parm_dict = parm_dict
        self.data_dict = signal_dict
        self.data_dict['Position Indices'] = pos_inds
        self.data_dict['Position Values'] = pos_dims
        self.data_dict['Position labels'] = pos_labs
        self.data_dict['Position units'] = pos_units

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
