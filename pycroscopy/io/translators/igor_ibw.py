# -*- coding: utf-8 -*-
"""
Created on Wed Dec 07 16:04:34 2016

@author: Suhas Somnath
"""

from __future__ import print_function, division  # int/int = float
from os import path, remove  # File Path formatting
import numpy as np  # For array operations

from igor import binarywave as bw

from .translator import Translator  # Because this class extends the abstract Translator class
from .utils import generateDummyMainParms
from ..hdf_utils import getH5DsetRefs, linkRefs
from ..io_hdf5 import ioHDF5  # Now the translator is responsible for writing the data.
from ..microdata import MicroDataGroup, \
    MicroDataset  # The building blocks for defining hierarchical storage in the H5 file


class IgorIBWTranslator(Translator):
    """
    Translates Igor Binary Wave (.ibw) files containing images or force curves to .h5
    """

    def translate(self, file_path):
        """
        Translates the provided file to .h5

        Parameters
        ----------
        file_path : String / unicode
            Absolute path of the .ibw file

        Returns
        -------
        h5_path : String / unicode
            Absolute path of the .h5 file
        """

        # Load the ibw file first
        ibw_obj = bw.load(file_path)
        ibw_wave = ibw_obj.get('wave')
        parm_dict = self._read_parms(ibw_wave)
        chan_labels, chan_units = self._get_chan_labels(ibw_wave)
        print(chan_labels)
        print(chan_units)

        # Get the data to figure out if this is an image or a force curve
        images = ibw_wave.get('wData')
        if images.ndim == 3:  # Image stack
            print('Found image stack of size {}'.format(images.shape))
            type_suffix = 'Image'
            images = images.transpose(2, 0, 1)  # [chan, Y, X] image
            images = np.reshape(images, (images.shape[0], -1))  # 2D [chan, Y*X points]

            num_rows = parm_dict['ScanLines']
            num_cols = parm_dict['ScanPoints']
            ds_pos_ind, ds_pos_val = self._buildpositiondatasets([num_rows, num_cols],
                                                                 steps=[1.0 * parm_dict['SlowScanSize'] / num_rows,
                                                                        1.0 * parm_dict['FastScanSize'] / num_cols],
                                                                 labels=['Y', 'X'],
                                                                 units=['m', 'm'])

            ds_spec_inds, ds_spec_vals = self._buildspectroscopicdatasets([1], steps=1, labels=['arb'], units=['a.u.'])

        else:  # single force curve
            print('Found force curve of size {}'.format(images.shape))
            type_suffix = 'ForceCurve'
            images = images.transpose()  # [chan x Z] force curve

            ds_pos_ind, ds_pos_val = self._buildpositiondatasets([1], steps=[25E-9], labels=['X'], units=['m'])

            ds_spec_inds, ds_spec_vals = self._buildspectroscopicdatasets([images.shape[1]], labels=['Z'], units=['m'])
            # The data generated above varies linearly. Override.
            # For now, we'll shove the Z sensor data into the spectroscopic values.

            # Find the channel that corresponds to either Z sensor or Raw:
            try:
                chan_ind = chan_labels.index('ZSnsr')
                ds_spec_vals.data = images[chan_ind, :]
            except ValueError:
                try:
                    chan_ind = chan_labels.index('Raw')
                    ds_spec_vals.data = images[chan_ind, :]
                except ValueError:
                    # We don't expect to come here. If we do, spectroscopic values remains as is
                    pass

        # Prepare the list of raw_data datasets
        chan_raw_dsets = list()
        for chan_data, chan_name, chan_unit in zip(images, chan_labels, chan_units):
            ds_raw_data = MicroDataset('Raw_Data', data=chan_data, dtype=np.float32, compression='gzip')
            ds_raw_data.attrs['labels'] = [chan_name]
            ds_raw_data.attrs['units'] = [chan_unit]
            chan_raw_dsets.append(ds_raw_data)
        print('Finished preparing raw datasets')

        # Prepare the tree structure
        # technically should change the date, etc.
        spm_data = MicroDataGroup('')
        global_parms = generateDummyMainParms()
        global_parms['data_type'] = 'IgorIBW_' + type_suffix
        global_parms['translator'] = 'IgorIBW'
        spm_data.attrs = global_parms
        meas_grp = MicroDataGroup('Measurement_000')
        meas_grp.attrs = parm_dict
        spm_data.addChildren([meas_grp])
        print('Finished preparing tree trunk')

        # Prepare the .h5 file:
        folder_path, base_name = path.split(file_path)
        base_name = base_name[:-4]
        h5_path = path.join(folder_path, base_name + '.h5')
        if path.exists(h5_path):
            remove(h5_path)

        # Write head of tree to file:
        hdf = ioHDF5(h5_path)
        # spm_data.showTree()
        hdf.writeData(spm_data, print_log=False)
        print('Finished writing tree trunk')

        # Standard list of auxiliary datasets that get linked with the raw dataset:
        aux_ds_names = ['Position_Indices', 'Position_Values', 'Spectroscopic_Indices', 'Spectroscopic_Values']

        # Create Channels, populate and then link:
        for chan_index, raw_dset in enumerate(chan_raw_dsets):
            chan_grp = MicroDataGroup('{:s}{:03d}'.format('Channel_', chan_index), '/Measurement_000/')
            chan_grp.addChildren([ds_pos_ind, ds_pos_val, ds_spec_inds, ds_spec_vals, raw_dset])
            h5_refs = hdf.writeData(chan_grp, print_log=False)
            h5_raw = getH5DsetRefs(['Raw_Data'], h5_refs)[0]
            linkRefs(h5_raw, getH5DsetRefs(aux_ds_names, h5_refs))

        print('Finished writing all channels')
        hdf.close()
        return h5_path

    @staticmethod
    def _read_parms(ibw_wave):
        """
        Parses the parameters in the provided dictionary

        Parameters
        ----------
        ibw_wave : dictionary
            Wave entry in the dictionary obtained from loading the ibw file

        Returns
        -------
        parm_dict : dictionary
            Dictionary containing parameters
        """
        parm_string = ibw_wave.get('note')
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
    def _get_chan_labels(ibw_wave):
        """
        Retrieves the names of the data channels and default units

        Parameters
        ----------
        ibw_wave : dictionary
            Wave entry in the dictionary obtained from loading the ibw file

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
        for chan in labels:
            if chan.startswith('Phase'):
                default_units.append('deg')
            elif chan.startswith('Current'):
                default_units.append('A')
            else:
                default_units.append('m')

        return labels, default_units

    def _parsefilepath(self, input_path):
        pass

    def _read_data(self):
        pass
