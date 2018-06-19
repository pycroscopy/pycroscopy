# -*- coding: utf-8 -*-
"""
Created on Fri Jun 01 15:21:46 2018

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

# packages specific to this kind of file
from .df_utils.gsf_read import gsf_read
import gwyfile


class GwyddionTranslator(Translator):

    def translate(self, file_path, *args, **kwargs):
        # Two kinds of files:
        # 1. Simple GSF files -> use metadata, data = gsf_read(file_path)
        # 2. Native .gwy files -> use the gwyfile package
        # I have a notebook that shows how such data can be read.
        # Create the .h5 file from the input file
        file_path = path.abspath(file_path)
        folder_path, base_name = path.split(file_path)
        base_name = base_name[:-4]
        h5_path = path.join(folder_path, base_name + '.h5')
        
        if path.exists(h5_path):
            remove(h5_path)

        h5_file = h5py.File(h5_path, 'w')

        """
        Setup the global parameters
        ---------------------------
        translator: Gywddion
        data_type: depends on file type
                    GwyddionGSF_<gsf_meta['title']>
                    or
                    GwyddionGWY_<gwy_meta['title']>
        """
        global_parms = generate_dummy_main_parms()
        for key in global_parms.keys():
            global_parms[key] = ''
        global_parms['translator'] = 'Gwyddion'
        
        if file_path.endswith('.gsf'):
            """
            For more information on the .gsf file format visit the link below - 
            http://gwyddion.net/documentation/user-guide-en/gsf.html
            """
            # Read the data in from the specified file
            gsf_meta, gsf_values = gsf_read(file_path)
            
            # Write parameters where available specifically for sample_name
            # data_type, comments and experiment_date to file-level parms
            global_parms['sample_name'] = gsf_meta['Title']
            global_parms['data_type'] = 'GwyddionGSF_' + gsf_meta['Title']

            if 'comment' in gsf_meta.keys():
                global_parms['comments'] = gsf_meta['comment']
            else:
                global_parms['comments'] = ''
            
            if 'date' in gsf_meta.keys():
                global_parms['experiment_date'] = gsf_meta['date']
            else:
                global_parms['experiment_date'] = ''
            write_simple_attrs(h5_file, global_parms)

            # Create the measurement group and write measurement level
            # parameters - same parms as file-level parms
            meas_grp = create_indexed_group(h5_file, 'Measurement')
            write_simple_attrs(meas_grp, global_parms)

            # Build the ancillary position datasets
            x_step = gsf_meta['XReal']/gsf_meta['XRes']
            y_step = gsf_meta['YReal']/gsf_meta['YRes']

            if 'XOffset' in gsf_meta.keys():
                x_offset = gsf_meta['XOffset']
                x_vals = list(x_pos * x_step + x_offset for x_pos in range(0, gsf_meta['XRes']))
            else:
                x_vals = list(x_pos * x_step for x_pos in range(0, gsf_meta['XRes']))

            if 'YOffset' in gsf_meta.keys():
                y_offset = gsf_meta['YOffset']
                y_vals = list(y_pos * y_step + y_offset for y_pos in range(0, gsf_meta['YRes']))
            else:
                y_vals = list(y_pos * y_step for y_pos in range(0, gsf_meta['YRes']))

            pos_desc = [Dimension('X', gsf_meta['XYUnits'], x_vals),
                        Dimension('Y', gsf_meta['XYUnits'], y_vals)]

            h5_pos_inds, h5_pos_vals = write_ind_val_dsets(meas_grp, pos_desc, is_spectral=False)            
            
            # Build the ancillary spectral datasets
            spec_desc = Dimension('Raster', gsf_meta['ZUnits'], [1])            
            h5_spec_inds, h5_spec_vals = write_ind_val_dsets(meas_grp, spec_desc, is_spectral=True)

            # Create the channel-level group
            chan_grp = create_indexed_group(meas_grp, 'Channel')
            
            # Build the main dataset
            raster_height = gsf_values
            write_main_dataset(chan_grp,
                    np.atleast_2d(np.reshape(raster_height, len(pos_desc[0].values) * len(pos_desc[1].values))).transpose(),
                        'Raw_Data', 'Raster_Height', gsf_meta['ZUnits'], pos_desc, spec_desc,
                            global_parms, h5_pos_inds, h5_pos_vals, h5_spec_inds, h5_spec_vals,
                                'Raster_', 'Position_')
            
            return h5_path

        if file_path.endswith('gwy'):
            """
            For more information on the .gwy file format visit the link below - 
            http://gwyddion.net/documentation/user-guide-en/gwyfile-format.html
            """
            # Read the data in from the specified file
            gwy_data = gwyfile.load(file_path)
            
            # Write parameters where available specifically for sample_name
            # data_type, comments and experiment_date to file-level parms         
            
            # Create Position and spectroscopic datasets
        
            # Write file and measurement level parameters

            # Prepare the list of raw_data datasets
            return file_path
        

    def _translate_image_stack(self):
        """
        Use this function to write data corrsponding to a stack of scan images (most common)0
        Returns
        -------

        """
        pass

    def _translate_3d_spectroscopy(self):
        """
        Use this to translate force-maps, I-V spectroscopy etc.
        Returns
        -------

        """
        pass

    def _translate_spectra(self):
        """
        Use this to translate simple 1D data like force curves
        Returns
        -------

        """
        pass
