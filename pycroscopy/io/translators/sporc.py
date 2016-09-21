# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 16:04:34 2016

@author: Suhas Somnath
"""

from __future__ import print_function, division; # int/int = float

from os import path, remove # File Path formatting

import h5py
import numpy as np; # For array operations
from scipy.io.matlab import loadmat # To load parameters stored in Matlab .mat file

from .translator import Translator # Because this class extends the abstract Translator class
from .utils import makePositionMat, getPositionSlicing, generateDummyMainParms
from ..hdf_utils import getH5DsetRefs, linkRefs
from ..io_hdf5 import ioHDF5 # Now the translator is responsible for writing the data.
from ..microdata import MicroDataGroup, MicroDataset # The building blocks for defining heirarchical storage in the H5 file


class SporcTranslator(Translator):
    """
    Translates G-mode SPORC datasets from .mat files to .h5
    """
    
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
		
        (folder_path, file_name) = path.split(parm_path)
        (file_name, base_name) = path.split(folder_path)
        h5_path = path.join(folder_path,base_name+'.h5')
        
        # Read parameters
        print('reading parameter files')
        parm_dict, excit_wfm, spec_ind_mat = self.__readparms(parm_path)
        parm_dict['data_type'] = 'SPORC'
        
        num_rows = parm_dict['grid_num_rows']
        num_cols = parm_dict['grid_num_cols'] 
        num_pix = num_rows*num_cols        
        
        pos_mat = makePositionMat([num_cols, num_rows])        
        pos_slices = getPositionSlicing(['X','Y'], num_pix)
        
        # new data format
        spec_ind_mat = np.transpose(np.float32(spec_ind_mat))
        
        # Now start creating datasets and populating:
        ds_pos_ind = MicroDataset('Position_Indices', np.uint32(pos_mat))
        ds_pos_ind.attrs['labels'] = pos_slices
        ds_pos_val = MicroDataset('Position_Values', np.float32(pos_mat))
        ds_pos_val.attrs['labels'] = pos_slices
        ds_pos_val.attrs['units'] = ['um','um']
                
        spec_ind_labels = ['x index', 'y index', 'loop index', 'repetition index', 'slope index']
        spec_ind_dict = dict()
        for col_ind, col_name in enumerate(spec_ind_labels):
            spec_ind_dict[col_name] = (slice(col_ind,col_ind+1), slice(None))
        ds_spec_inds = MicroDataset('Spectroscopic_Indices', np.uint32(spec_ind_mat))
        ds_spec_inds.attrs['labels'] = spec_ind_dict
        ds_spec_vals = MicroDataset('Spectroscopic_Values', spec_ind_mat)
        ds_spec_vals.attrs['labels'] = spec_ind_dict
        ds_spec_vals.attrs['units'] = ['V','V','','','']

        ds_excit_wfm = MicroDataset('Excitation_Waveform', np.float32(excit_wfm))
        
        ds_raw_data = MicroDataset('Raw_Data', data=[], 
                                  maxshape=(num_pix,len(excit_wfm)), 
                                dtype=np.float16, chunking=(1,len(excit_wfm)), 
                                compression='gzip')
        
        # technically should change the date, etc.
         
        chan_grp = MicroDataGroup('Channel_000')
        chan_grp.attrs = parm_dict        
        chan_grp.addChildren([ds_pos_ind, ds_pos_val, ds_spec_inds, ds_spec_vals,
                              ds_excit_wfm, ds_raw_data])
                
        global_parms = generateDummyMainParms()
        global_parms['grid_size_x'] = parm_dict['grid_num_cols'];
        global_parms['grid_size_y'] = parm_dict['grid_num_rows'];
        # assuming that the experiment was completed:        
        global_parms['current_position_x'] = parm_dict['grid_num_cols']-1;
        global_parms['current_position_y'] = parm_dict['grid_num_rows']-1;
        global_parms['data_type'] = parm_dict['data_type'] 
        global_parms['translator'] = 'SPORC'
        
        meas_grp = MicroDataGroup('Measurement_000')
        meas_grp.addChildren([chan_grp])
        spm_data = MicroDataGroup('')
        spm_data.attrs = global_parms
        spm_data.addChildren([meas_grp])
        
        if path.exists(h5_path):
            remove(h5_path)
        
        # Write everything except for the main data.
        hdf = ioHDF5(h5_path)
        
        h5_refs = hdf.writeData(spm_data)
                    
        h5_main = getH5DsetRefs(['Raw_Data'], h5_refs)[0]
            
        #Now doing linkrefs:
        aux_ds_names = ['Excitation_Waveform', 'Position_Indices','Position_Values',
                     'Spectroscopic_Indices','Spectroscopic_Values']
        linkRefs(h5_main, getH5DsetRefs(aux_ds_names, h5_refs))
        
        print('reading raw data now...')
        
        # Now read the raw data files:
        pos_ind = 0
        for row_ind in xrange(1,num_rows+1):
            for col_ind in xrange(1,num_cols+1):
                file_path = path.join(folder_path,'result_r'+str(row_ind)+'_c'+str(col_ind)+'.mat')
                #print('Working on row {} col {}'.format(row_ind,col_ind))
                if path.exists(file_path):
                    # Load data file
                    pix_data = loadmat(file_path, squeeze_me=True)
                    # Take the inverse FFT on 1st dimension
                    pix_vec = np.fft.ifft(np.fft.ifftshift(pix_data['data']))
                    # Verified with Matlab - no conjugate required here.
                    h5_main[pos_ind,:] = np.float16(np.real(pix_vec))
                    hdf.flush() # flush from memory!
                else:
                    print('File for row {} col {} not found'.format(row_ind, col_ind))
                pos_ind +=1
                if (100.0*(pos_ind)/num_pix)%10 == 0:
                    print('Finished reading {} % of data'.format(int(100*pos_ind/num_pix)))

        hdf.close()
        
        return h5_path
        
        
    @staticmethod
    def __readparms(parm_path):
        """
        Copies experimental parameters from the .mat file to a dictionary

        Parameters
        ------------
        parm_path : string / unicode
            Absolute path of the parameters file

        Returns
        --------
        (parm_dict, excit_wfm, spec_ind_mat) : tuple

        parm_dict : dictionary
            Dictionary containing all relevant parameters
        excit_wfm : 1d numpy float array
            Excitation waveform
        spec_ind_mat : 2D numpy flaot array
            Spectroscopic indicies matrix
        """

        parm_data = loadmat(parm_path, squeeze_me=True, struct_as_record=True)
        parm_dict = dict()

        IO_parms = parm_data['IOparms']
        parm_dict['IO_samp_rate_[Hz]'] = np.int32(IO_parms['sampRate'].item())
        parm_dict['IO_down_samp_rate_[Hz]'] = np.int32(IO_parms['downSampRate'].item())
        parm_dict['IO_AO0_amp'] = np.int32(IO_parms['AO0_amp'].item())
        parm_dict['IO_AI_chans'] = np.int32(parm_data['aiChans'])

        parm_dict['grid_num_rows'] = parm_data['numrows']
        parm_dict['grid_num_cols'] = parm_data['numcols']

        sporc_parms = parm_data['sporcParms']
        parm_dict['SPORC_V_max_[V]'] = np.float32(sporc_parms['V_max'].item())
        parm_dict['SPORC_N_steps'] = np.int32(sporc_parms['N_steps'].item())
        parm_dict['SPORC_N_reps'] = np.int32(sporc_parms['N_reps'].item())
        parm_dict['SPORC_t_max_[sec]'] = np.float32(sporc_parms['t_max'])
        parm_dict['SPORC_f_cutoff_[Hz]'] = np.int32(sporc_parms['f_cutoff'])
        parm_dict['SPORC_f_rolloff_[Hz]'] = np.int32(sporc_parms['f_rolloff'])

        if 'FORC_vec' in parm_data.keys() and 'ind_vecs' in parm_data.keys():
            excit_wfm = np.squeeze(np.float32(parm_data['FORC_vec']))
            spec_ind_mat = np.transpose(np.float32(parm_data['ind_vecs']))
        else:
            # Look for a second parms file that contains these vectors:
            fold, basename = path.split(parm_path)
            second_path = path.join(fold,'SPORC_wave.mat')
            h5_sporc_parms = h5py.File(second_path,'r') # Use this for v7.3 and beyond.
            excit_wfm = np.squeeze(h5_sporc_parms['FORC_vec'].value)
            spec_ind_mat = np.float32(h5_sporc_parms['ind_vecs'].value)
            h5_sporc_parms.close()

        return parm_dict, excit_wfm, spec_ind_mat