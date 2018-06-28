# -*- coding: utf-8 -*-
"""
Created on Fri May 25 16:04:34 2016

@author: Suhas Somnath
"""

from __future__ import division, print_function, absolute_import, unicode_literals
from os import path, remove
from collections import OrderedDict
import numpy as np
import h5py

from pyUSID.io.translator import Translator, generate_dummy_main_parms
from pyUSID.io.write_utils import Dimension
from pyUSID.io.hdf_utils import create_indexed_group, write_main_dataset, write_simple_attrs, write_ind_val_dsets
from .df_utils.base_utils import read_binary_data
# TODO: Adopt missing aspects / features from https://github.com/paruch-group/distortcorrect/blob/master/afm/filereader/readNanoscope.py


class BrukerAFMTranslator(Translator):

    def translate(self, file_path, *args, **kwargs):
        """
        Translates a given Bruker / Veeco / Nanoscope AFM derived file to HDF5. Currently handles scans, force curves,
        and force-distance maps

        Note that this translator was written with a single example file for each modality and may be buggy.

        Parameters
        ----------
        file_path : str / unicode
            path to data file

        Returns
        -------
        h5_path : str / unicode
            path to translated HDF5 file
        """
        self.file_path = path.abspath(file_path)
        self.meta_data, other_parms = self._extract_metadata()

        # These files are weirdly named with extensions such as .001
        h5_path = file_path.replace('.', '_') + '.h5'

        if path.exists(h5_path):
            remove(h5_path)

        h5_file = h5py.File(h5_path, 'w')

        type_suffixes = ['Image', 'Force_Curve', 'Force_Map']
        # 0 - stack of scan images
        # 1 - single force curve
        # 2 - force map
        force_count = 0
        image_count = 0
        for class_name in self.meta_data.keys():
            if 'Ciao force image list' in class_name:
                force_count += 1
            elif 'Ciao image list' in class_name:
                image_count += 1
        data_type = 0
        if force_count > 0:
            if image_count > 0:
                data_type = 2
            else:
                data_type = 1

        global_parms = generate_dummy_main_parms()
        global_parms['data_type'] = 'Bruker_AFM_' + type_suffixes[data_type]
        global_parms['translator'] = 'Bruker_AFM'
        write_simple_attrs(h5_file, global_parms)

        # too many parameters. Making a dummy group just for the parameters.
        h5_parms_grp = h5_file.create_group('Parameters')
        # We currently have a dictionary of dictionaries. This needs to be flattened
        flat_dict = dict()
        for class_name, sub_dict in other_parms.items():
            for key, val in sub_dict.items():
                flat_dict[class_name + '_' + key] = val
        write_simple_attrs(h5_parms_grp, flat_dict)

        # Create measurement group
        h5_meas_grp = create_indexed_group(h5_file, 'Measurement')

        # Call the data specific translation function
        trans_funcs = [self._translate_image_stack, self._translate_force_curve, self._translate_force_map]
        trans_funcs[data_type](h5_meas_grp)

        # wrap up and return path
        h5_file.close()
        return h5_path

    def _translate_force_curve(self, h5_meas_grp):
        """
        Reads the force curves from the proprietary file and writes them to HDF5 datasets

        Parameters
        ----------
        h5_meas_grp : h5py.Group object
            Reference to the measurement group
        """
        # since multiple channels will share the same position and spectroscopic dimensions, why not share them?
        h5_pos_inds, h5_pos_vals = write_ind_val_dsets(h5_meas_grp, Dimension('single', 'a. u.', 1), is_spectral=False)

        # Find out the size of the force curves from the metadata:
        layer_info = None
        for class_name in self.meta_data.keys():
            if 'Ciao force image list' in class_name:
                layer_info = self.meta_data[class_name]
                break
        tr_rt = [int(item) for item in layer_info['Samps/line'].split(' ')]

        h5_spec_inds, h5_spec_vals = write_ind_val_dsets(h5_meas_grp, Dimension('Z', 'nm', int(np.sum(tr_rt))),
                                                         is_spectral=True)

        for class_name in self.meta_data.keys():
            if 'Ciao force image list' in class_name:
                layer_info = self.meta_data[class_name]
                quantity = layer_info.pop('Image Data_4')
                data = self._read_data_vector(layer_info)
                h5_chan_grp = create_indexed_group(h5_meas_grp, 'Channel')
                write_main_dataset(h5_chan_grp, np.expand_dims(data, axis=0), 'Raw_Data',
                                   # Quantity and Units needs to be fixed by someone who understands these files better
                                   quantity, 'a. u.',
                                   None, None, dtype=np.float32, compression='gzip',
                                   h5_pos_inds=h5_pos_inds, h5_pos_vals=h5_pos_vals,
                                   h5_spec_inds=h5_spec_inds, h5_spec_vals=h5_spec_vals)
                # Think about standardizing attributes
                write_simple_attrs(h5_chan_grp, layer_info)

    def _translate_image_stack(self, h5_meas_grp):
        """
        Reads the scan images from the proprietary file and writes them to HDF5 datasets

        Parameters
        ----------
        h5_meas_grp : h5py.Group object
            Reference to the measurement group
        """
        # since multiple channels will share the same position and spectroscopic dimensions, why not share them?
        h5_spec_inds, h5_spec_vals = write_ind_val_dsets(h5_meas_grp, Dimension('single', 'a. u.', 1), is_spectral=True)

        # Find out the size of the force curves from the metadata:
        layer_info = None
        for class_name in self.meta_data.keys():
            if 'Ciao image list' in class_name:
                layer_info = self.meta_data[class_name]
                break

        h5_pos_inds, h5_pos_vals = write_ind_val_dsets(h5_meas_grp, [Dimension('X', 'nm', layer_info['Samps/line']),
                                                                     Dimension('Y', 'nm',
                                                                               layer_info['Number of lines'])],
                                                       is_spectral=False)

        for class_name in self.meta_data.keys():
            if 'Ciao image list' in class_name:
                layer_info = self.meta_data[class_name]
                quantity = layer_info.pop('Image Data_2')
                data = self._read_image_layer(layer_info)
                h5_chan_grp = create_indexed_group(h5_meas_grp, 'Channel')
                write_main_dataset(h5_chan_grp, np.reshape(data, (-1, 1)), 'Raw_Data',
                                   # Quantity and Units needs to be fixed by someone who understands these files better
                                   quantity, 'a. u.',
                                   None, None, dtype=np.float32, compression='gzip',
                                   h5_pos_inds=h5_pos_inds, h5_pos_vals=h5_pos_vals,
                                   h5_spec_inds=h5_spec_inds, h5_spec_vals=h5_spec_vals)
                # Think about standardizing attributes for rows and columns
                write_simple_attrs(h5_chan_grp, layer_info)

    def _translate_force_map(self, h5_meas_grp):
        """
        Reads the scan image + force map from the proprietary file and writes it to HDF5 datasets

        Parameters
        ----------
        h5_meas_grp : h5py.Group object
            Reference to the measurement group
        """
        # First lets write the image into the measurement group that has already been created:
        image_parms = self.meta_data['Ciao image list']
        quantity = image_parms.pop('Image Data_2')
        image_mat = self._read_image_layer(image_parms)
        h5_chan_grp = create_indexed_group(h5_meas_grp, 'Channel')
        write_main_dataset(h5_chan_grp, np.reshape(image_mat, (-1, 1)), 'Raw_Data',
                           # Quantity and Units needs to be fixed by someone who understands these files better
                           quantity, 'a. u.',
                           [Dimension('X', 'nm', image_parms['Samps/line']),
                            Dimension('Y', 'nm', image_parms['Number of lines'])],
                           Dimension('single', 'a. u.', 1), dtype=np.float32, compression='gzip')
        # Think about standardizing attributes for rows and columns
        write_simple_attrs(h5_chan_grp, image_parms)

        # Now work on the force map:
        force_map_parms = self.meta_data['Ciao force image list']
        quantity = force_map_parms.pop('Image Data_4')
        force_map_vec = self._read_data_vector(force_map_parms)
        tr_rt = [int(item) for item in force_map_parms['Samps/line'].split(' ')]
        force_map_2d = force_map_vec.reshape(image_mat.size, np.sum(tr_rt))
        h5_chan_grp = create_indexed_group(h5_meas_grp, 'Channel')
        write_main_dataset(h5_chan_grp, force_map_2d, 'Raw_Data',
                           # Quantity and Units needs to be fixed by someone who understands these files better
                           quantity, 'a. u.',
                           [Dimension('X', 'nm', image_parms['Samps/line']),
                            Dimension('Y', 'nm', image_parms['Number of lines'])],
                           Dimension('Z', 'nm', int(np.sum(tr_rt))), dtype=np.float32, compression='gzip')
        # Think about standardizing attributes
        write_simple_attrs(h5_chan_grp, force_map_parms)

    def _extract_metadata(self):
        """
        Reads the metadata in the header

        Returns
        -------
        meas_parms : OrderedDict
            Ordered dictionary of Ordered dictionaries (one per image / force channel, etc.)
        other_parms : OrderedDict
            Ordered Dictionary of Ordered dictionaries containing all other metadata
        """
        other_parms = OrderedDict()
        meas_parms = OrderedDict()
        curr_category = ''
        temp_dict = OrderedDict()
        with open(self.file_path, "rb") as file_handle:
            for ind, line in enumerate(file_handle):
                line = line.decode("utf-8", 'ignore')
                trimmed = line.strip().replace("\\", "").replace('@', '')
                split_data = trimmed.split(':')

                # First account for wierdly formatted metadata that
                if len(split_data) == 3:
                    split_data = [split_data[1] + '_' + split_data[0], split_data[-1]]
                elif len(split_data) > 3:
                    # Date:
                    split_ind = trimmed.index(':')
                    split_data = [trimmed[:split_ind], trimmed[split_ind + 1:]]

                # At this point, split_data should only contain either 1 (class header) or 2 elements
                if len(split_data) == 1:
                    if len(temp_dict) > 0:
                        if 'Ciao image list' in curr_category or 'Ciao force image list' in curr_category:
                            # In certain cases the same class name occurs multiple times.
                            # Append suffix to existing name and to this name
                            count = 0
                            for class_name in meas_parms.keys():
                                if curr_category in class_name:
                                    count += 1
                            if count == 0:
                                meas_parms[curr_category] = temp_dict.copy()
                            else:
                                if count == 1:
                                    for class_name in meas_parms.keys():
                                        if curr_category == class_name:
                                            # Remove and add back again with suffix
                                            # This should only ever happen once.
                                            # The next time we come across the same class, all elements already have
                                            # suffixes
                                            meas_parms[curr_category + '_0'] = meas_parms.pop(curr_category)
                                            break
                                meas_parms[curr_category + '_' + str(count)] = temp_dict.copy()
                        else:
                            curr_category = curr_category.replace('Ciao ', '')
                            other_parms[curr_category] = temp_dict.copy()

                    if "*File list end" in trimmed:
                        break
                    curr_category = split_data[0].replace('*', '')
                    temp_dict = OrderedDict()
                elif len(split_data) == 2:
                    split_data = [item.strip() for item in split_data]
                    try:
                        num_val = float(split_data[1])
                        if int(num_val) == num_val:
                            num_val = int(num_val)
                        temp_dict[split_data[0]] = num_val
                    except ValueError:
                        temp_dict[split_data[0]] = split_data[1]
                else:
                    print(split_data)

        return meas_parms, other_parms

    def _read_data_vector(self, layer_info):
        """
        Reads data relevant to a single image, force curve, or force map

        Parameters
        ----------
        layer_info : OrderedDictionary
            Parameters describing the data offset, length and precision in the binary file

        Returns
        -------
        data_vec : np.ndarray
            1D array containing data represented by binary data
        """
        data_vec = read_binary_data(self.file_path, layer_info['Data offset'], layer_info['Data length'],
                                    layer_info['Bytes/pixel'])

        # Remove translation specific values from dictionary:
        for key in ['Data offset', 'Data length', 'Bytes/pixel']:
            _ = layer_info.pop(key)
        return data_vec

    def _read_image_layer(self, layer_info):
        """
        Reads a single scan image layer / channel

        Parameters
        ----------
        layer_info : OrderedDictionary
            Parameters describing the data offset, length and precision in the binary file

        Returns
        -------
        data_mat : numpy.ndarray
            2D array representing the requested channel of information
        """
        data_vec = self._read_data_vector(layer_info)
        data_mat = data_vec.reshape(layer_info['Number of lines'], layer_info['Samps/line'])
        return data_mat
