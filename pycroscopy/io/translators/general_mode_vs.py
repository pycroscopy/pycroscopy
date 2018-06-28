# -*- coding: utf-8 -*-
"""
Created on Fri June 22 11:28:45 2018

@author: Rama Vasudevan
"""

from __future__ import division, print_function, absolute_import, unicode_literals

from os import path, remove  # File Path formatting

import numpy as np  # For array operations
from scipy.io.matlab import loadmat  # To load parameters stored in Matlab .mat file
import h5py

from .df_utils.gmode_utils import readGmodeParms
from pyUSID.io.translator import Translator, \
    generate_dummy_main_parms  # Because this class extends the abstract Translator class
from pyUSID.io.write_utils import VALUES_DTYPE, Dimension
from pyUSID.io.hdf_utils import link_h5_objects_as_attrs, create_indexed_group, \
    write_simple_attrs, write_main_dataset


class GVSTranslator(Translator):
    """
    Translates G-mode voltage spectroscopy datasets from .mat files to .h5
    """

    def _read_data(self):
        pass

    def _parse_file_path(self, input_path):
        pass

    def translate(self, parm_path):
        """
        Basic method that translates .mat data files to a single .h5 file
        
        Parameters
        ------------
        parm_path : string / unicode
            Absolute file path of the parameters .mat file. 
            
        Returns
        ----------
        h5_path : string / unicode
            Absolute path of the translated h5 file
        """
        self.parm_path = path.abspath(parm_path)
        (folder_path, file_name) = path.split(parm_path)
        (file_name, base_name) = path.split(folder_path)
        h5_path = path.join(folder_path, base_name + '.h5')

        # Read parameters
        parm_dict = readGmodeParms(parm_path)

        # Add the w^2 specific parameters to this list
        parm_data = loadmat(parm_path, squeeze_me=True, struct_as_record=True)
        #freq_sweep_parms = parm_data['freqSweepParms']
        #parm_dict['freq_sweep_delay'] = np.float(freq_sweep_parms['delay'].item())
        gen_sig = parm_data['genSig']
        #parm_dict['wfm_fix_d_fast'] = np.int32(gen_sig['restrictT'].item())
        #freq_array = np.float32(parm_data['freqArray'])

        # prepare and write spectroscopic values
        samp_rate = parm_dict['IO_down_samp_rate_[Hz]']
        num_bins = int(parm_dict['wfm_n_cycles'] * parm_dict['wfm_p_slow'] * samp_rate)

        w_vec = np.arange(-0.5 * samp_rate, 0.5 * samp_rate, np.float32(samp_rate / num_bins))

        # There is most likely a more elegant solution to this but I don't have the time... Maybe np.meshgrid
        spec_val_mat = np.zeros((len(freq_array) * num_bins, 2), dtype=VALUES_DTYPE)
        spec_val_mat[:, 0] = np.tile(w_vec, len(freq_array))
        spec_val_mat[:, 1] = np.repeat(freq_array, num_bins)

        spec_ind_mat = np.zeros((2, len(freq_array) * num_bins), dtype=np.int32)
        spec_ind_mat[0, :] = np.tile(np.arange(num_bins), len(freq_array))
        spec_ind_mat[1, :] = np.repeat(np.arange(len(freq_array)), num_bins)

        num_rows = parm_dict['grid_num_rows']
        num_cols = parm_dict['grid_num_cols']
        parm_dict['data_type'] = 'GVS'

        num_pix = num_rows * num_cols

        global_parms = generate_dummy_main_parms()
        global_parms['grid_size_x'] = parm_dict['grid_num_cols']
        global_parms['grid_size_y'] = parm_dict['grid_num_rows']
        # assuming that the experiment was completed:
        global_parms['current_position_x'] = parm_dict['grid_num_cols'] - 1
        global_parms['current_position_y'] = parm_dict['grid_num_rows'] - 1
        global_parms['data_type'] = parm_dict['data_type']  # self.__class__.__name__
        global_parms['translator'] = 'GVS'

        # Now start creating datasets and populating:
        if path.exists(h5_path):
            remove(h5_path)

        h5_f = h5py.File(h5_path, 'w')
        write_simple_attrs(h5_f, global_parms)

        meas_grp = create_indexed_group(h5_f, 'Measurement')
        chan_grp = create_indexed_group(meas_grp, 'Channel')
        write_simple_attrs(chan_grp, parm_dict)


        pos_dims = [Dimension('X', 'nm', num_rows),
                    Dimension('Y', 'nm', num_cols)]
        spec_dims = [Dimension('Response Bin', 'a.u.', num_bins),
                     Dimension('Excitation Frequency ', 'Hz', len(freq_array))]

        # Minimize file size to the extent possible.
        # DAQs are rated at 16 bit so float16 should be most appropriate.
        # For some reason, compression is more effective on time series data

        h5_main = write_main_dataset(chan_grp, (num_pix, num_bins), 'Raw_Data',
                                     'Deflection', 'V',
                                     pos_dims, spec_dims,
                                     chunks=(1, num_bins), dtype=np.float32)

        h5_ex_freqs = chan_grp.create_dataset('Excitation_Frequencies', freq_array)
        h5_bin_freq = chan_grp.create_dataset('Bin_Frequencies', w_vec)

        # Now doing link_h5_objects_as_attrs:
        link_h5_objects_as_attrs(h5_main, [h5_ex_freqs, h5_bin_freq])

        # Now read the raw data files:
        pos_ind = 0
        for row_ind in range(1, num_rows + 1):
            for col_ind in range(1, num_cols + 1):
                file_path = path.join(folder_path, 'fSweep_r' + str(row_ind) + '_c' + str(col_ind) + '.mat')
                print('Working on row {} col {}'.format(row_ind, col_ind))
                if path.exists(file_path):
                    # Load data file
                    pix_data = loadmat(file_path, squeeze_me=True)
                    pix_mat = pix_data['AI_mat']
                    # Take the inverse FFT on 2nd dimension
                    pix_mat = np.fft.ifft(np.fft.ifftshift(pix_mat, axes=1), axis=1)
                    # Verified with Matlab - no conjugate required here.
                    pix_vec = pix_mat.transpose().reshape(pix_mat.size)
                    h5_main[pos_ind, :] = np.float32(pix_vec)
                    h5_f.flush()  # flush from memory!
                else:
                    print('File not found for: row {} col {}'.format(row_ind, col_ind))
                pos_ind += 1
                if (100.0 * pos_ind / num_pix) % 10 == 0:
                    print('completed translating {} %'.format(int(100 * pos_ind / num_pix)))

        h5_f.close()

        return h5_path
