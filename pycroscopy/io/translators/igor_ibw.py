# -*- coding: utf-8 -*-
"""
Created on Wed Dec 07 16:04:34 2016

@author: Suhas Somnath, Chris R. Smith, Raj Giri
"""

from __future__ import division, print_function, absolute_import, unicode_literals
from os import path, remove  # File Path formatting
import numpy as np  # For array operations
import h5py
from igor import binarywave as bw

from pyUSID.io.translator import Translator, \
    generate_dummy_main_parms  # Because this class extends the abstract Translator class
from pyUSID.io.write_utils import VALUES_DTYPE, Dimension
from pyUSID.io.hdf_utils import create_indexed_group, write_main_dataset, write_simple_attrs, write_ind_val_dsets


class IgorIBWTranslator(Translator):
    """
    Translates Igor Binary Wave (.ibw) files containing images or force curves to .h5
    """

    def translate(self, file_path, verbose=False, parm_encoding='utf-8'):
        """
        Translates the provided file to .h5

        Parameters
        ----------
        file_path : String / unicode
            Absolute path of the .ibw file
        verbose : Boolean (Optional)
            Whether or not to show  print statements for debugging
        parm_encoding : str, optional
            Codec to be used to decode the bytestrings into Python strings if needed.
            Default 'utf-8'

        Returns
        -------
        h5_path : String / unicode
            Absolute path of the .h5 file
        """
        file_path = path.abspath(file_path)
        # Prepare the .h5 file:
        folder_path, base_name = path.split(file_path)
        base_name = base_name[:-4]
        h5_path = path.join(folder_path, base_name + '.h5')
        if path.exists(h5_path):
            remove(h5_path)

        h5_file = h5py.File(h5_path, 'w')

        # Load the ibw file first
        ibw_obj = bw.load(file_path)
        ibw_wave = ibw_obj.get('wave')
        parm_dict = self._read_parms(ibw_wave, parm_encoding)
        chan_labels, chan_units = self._get_chan_labels(ibw_wave, parm_encoding)

        if verbose:
            print('Channels and units found:')
            print(chan_labels)
            print(chan_units)

        # Get the data to figure out if this is an image or a force curve
        images = ibw_wave.get('wData')

        if images.shape[2] != len(chan_labels):
            chan_labels = chan_labels[1:]  # for layer 0 null set errors in older AR software

        if images.ndim == 3:  # Image stack
            if verbose:
                print('Found image stack of size {}'.format(images.shape))
            type_suffix = 'Image'

            num_rows = parm_dict['ScanLines']
            num_cols = parm_dict['ScanPoints']

            images = images.transpose(2, 1, 0)  # now ordered as [chan, Y, X] image
            images = np.reshape(images, (images.shape[0], -1, 1))  # 3D [chan, Y*X points,1]

            pos_desc = [Dimension('X', 'm', np.linspace(0, parm_dict['FastScanSize'], num_cols)),
                        Dimension('Y', 'm', np.linspace(0, parm_dict['SlowScanSize'], num_rows))]

            spec_desc = Dimension('arb', 'a.u.', [1])

        else:  # single force curve
            if verbose:
                print('Found force curve of size {}'.format(images.shape))

            type_suffix = 'ForceCurve'
            images = np.atleast_3d(images)  # now [Z, chan, 1]
            images = images.transpose((1, 2, 0))  # [chan ,1, Z] force curve

            # The data generated above varies linearly. Override.
            # For now, we'll shove the Z sensor data into the spectroscopic values.

            # Find the channel that corresponds to either Z sensor or Raw:
            try:
                chan_ind = chan_labels.index('ZSnsr')
                spec_data = np.atleast_2d(VALUES_DTYPE(images[chan_ind]))
            except ValueError:
                try:
                    chan_ind = chan_labels.index('Raw')
                    spec_data = np.atleast_2d(VALUES_DTYPE(images[chan_ind]))
                except ValueError:
                    # We don't expect to come here. If we do, spectroscopic values remains as is
                    spec_data = np.arange(images.shape[2])

            pos_desc = Dimension('X', 'm', [1])
            spec_desc = Dimension('Z', 'm', spec_data)

        # Create measurement group
        meas_grp = create_indexed_group(h5_file, 'Measurement')

        # Write file and measurement level parameters
        global_parms = generate_dummy_main_parms()
        global_parms['data_type'] = 'IgorIBW_' + type_suffix
        global_parms['translator'] = 'IgorIBW'
        write_simple_attrs(h5_file, global_parms)

        write_simple_attrs(meas_grp, parm_dict)

        # Create Position and spectroscopic datasets
        h5_pos_inds, h5_pos_vals = write_ind_val_dsets(meas_grp, pos_desc, is_spectral=False)
        h5_spec_inds, h5_spec_vals = write_ind_val_dsets(meas_grp, spec_desc, is_spectral=True)

        # Prepare the list of raw_data datasets
        for chan_data, chan_name, chan_unit in zip(images, chan_labels, chan_units):
            chan_grp = create_indexed_group(meas_grp, 'Channel')

            write_main_dataset(chan_grp, np.atleast_2d(chan_data), 'Raw_Data',
                               chan_name, chan_unit,
                               None, None,
                               h5_pos_inds=h5_pos_inds, h5_pos_vals=h5_pos_vals,
                               h5_spec_inds=h5_spec_inds, h5_spec_vals=h5_spec_vals,
                               dtype=np.float32)

        if verbose:
            print('Finished preparing raw datasets')

        h5_file.close()
        return h5_path

    @staticmethod
    def _read_parms(ibw_wave, codec='utf-8'):
        """
        Parses the parameters in the provided dictionary

        Parameters
        ----------
        ibw_wave : dictionary
            Wave entry in the dictionary obtained from loading the ibw file
        codec : str, optional
            Codec to be used to decode the bytestrings into Python strings if needed.
            Default 'utf-8'

        Returns
        -------
        parm_dict : dictionary
            Dictionary containing parameters
        """
        parm_string = ibw_wave.get('note')
        if type(parm_string) == bytes:
            try:
                parm_string = parm_string.decode(codec)
            except:
                parm_string = parm_string.decode('ISO-8859-1')  # for older AR software
        parm_string = parm_string.rstrip('\r')
        parm_list = parm_string.split('\r')
        parm_dict = dict()
        for pair_string in parm_list:
            temp = pair_string.split(':')
            if len(temp) == 2:
                temp = [item.strip() for item in temp]
                try:
                    num = float(temp[1])
                    parm_dict[temp[0]] = num
                    try:
                        if num == int(num):
                            parm_dict[temp[0]] = int(num)
                    except OverflowError:
                        pass
                except ValueError:
                    parm_dict[temp[0]] = temp[1]

        # Grab the creation and modification times:
        other_parms = ibw_wave.get('wave_header')
        for key in ['creationDate', 'modDate', 'bname']:
            try:
                parm_dict[key] = other_parms[key]
            except KeyError:
                pass
        return parm_dict

    @staticmethod
    def _get_chan_labels(ibw_wave, codec='utf-8'):
        """
        Retrieves the names of the data channels and default units

        Parameters
        ----------
        ibw_wave : dictionary
            Wave entry in the dictionary obtained from loading the ibw file
        codec : str, optional
            Codec to be used to decode the bytestrings into Python strings if needed.
            Default 'utf-8'

        Returns
        -------
        labels : list of strings
            List of the names of the data channels
        default_units : list of strings
            List of units for the measurement in each channel
        """
        temp = ibw_wave.get('labels')
        labels = []
        for item in temp:
            if len(item) > 0:
                labels += item
        for item in labels:
            if item == '':
                labels.remove(item)

        default_units = list()
        for chan_ind, chan in enumerate(labels):
            # clean up channel names
            if type(chan) == bytes:
                chan = chan.decode(codec)
            if chan.lower().rfind('trace') > 0:
                labels[chan_ind] = chan[:chan.lower().rfind('trace') + 5]
            # Figure out (default) units
            if chan.startswith('Phase'):
                default_units.append('deg')
            elif chan.startswith('Current'):
                default_units.append('A')
            else:
                default_units.append('m')

        return labels, default_units

    def _parse_file_path(self, input_path):
        pass

    def _read_data(self):
        pass
