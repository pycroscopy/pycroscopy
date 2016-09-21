# -*- coding: utf-8 -*-
"""
Created on Sun May 29 17:58:35 2016

@author: Suhas Somnath
"""

from __future__ import division # int/int = float

from os import path, remove # File Path formatting
from warnings import warn

import h5py
import numpy as np # For array operations

from .translator import Translator
from .utils import makePositionMat, getPositionSlicing, generateDummyMainParms
from ..hdf_utils import getH5DsetRefs, linkRefs
from ..io_hdf5 import ioHDF5 # Now the translator is responsible for writing the data.
from ..microdata import MicroDataGroup, MicroDataset # The building blocks for defining heirarchical storage in the H5 file


class FastIVTranslator(Translator):
    """
    Translates G-mode Fast IV datasets from .mat files to .h5
    """
    
    def translate(self, parm_path):
        """      
        The main function that translates the provided file into a .h5 file
        
        Parameters
        ------------
        parm_path : string / unicode
            Absolute file path of the parameters .mat file. 
            
        Returns
        ----------
        h5_path : string / unicode
            Absolute path of the translated h5 file
        """
        parm_dict, excit_wfm = self.__getparms(parm_path)
        folder_path, base_name = path.split(parm_path)
        waste, base_name = path.split(folder_path)
        
        h5_path = path.join(folder_path,base_name+'.h5')
        if path.exists(h5_path):
            remove(h5_path)
           
        # prepare and write spectroscopic values
        spec_ind_mat = np.atleast_2d(np.arange(len(excit_wfm)))
        # spec_val_mat = 1.0*spec_ind_mat/parm_dict['IO_samp_rate_[Hz]']
           
        pos_mat = makePositionMat([parm_dict['grid_num_rows']])
        
        pos_slices = getPositionSlicing(['Y'], parm_dict['grid_num_rows'])
        
        # Now start creating datasets and populating:
        ds_pos_ind = MicroDataset('Position_Indices', pos_mat)
        ds_pos_ind.attrs['labels'] = pos_slices
        ds_pos_val = MicroDataset('Position_Values', np.float32(pos_mat) * parm_dict['grid_scan_height_[m]'])
        ds_pos_val.attrs['labels'] = pos_slices
        ds_pos_val.attrs['units'] = ['m']
        
        ds_spec_inds = MicroDataset('Spectroscopic_Indices', spec_ind_mat)
        ds_spec_inds.attrs['labels'] = {'Time Index':(slice(0,1), slice(None))}
        
        ds_spec_vals = MicroDataset('Spectroscopic_Values', np.atleast_2d(excit_wfm))
        ds_spec_vals.attrs['labels'] = {'Bias':(slice(0,1), slice(None))}
        ds_spec_vals.attrs['units'] = ['sec']
                
        ds_ex_efm = MicroDataset('Excitation_Waveform', excit_wfm)
        
        # Minimize file size to the extent possible.
        # DAQs are rated at 16 bit so float16 should be most appropriate.
        # For some reason, compression is more effective on time series data
        ds_raw_data = MicroDataset('Raw_Data', data=[], 
                                    maxshape=(parm_dict['grid_num_rows'],len(excit_wfm)),
                                    dtype=np.float16, chunking=(1,len(excit_wfm)), compression='gzip')
        
        aux_ds_names = ['Excitation_Waveform', 'Position_Indices','Position_Values',
                     'Spectroscopic_Indices','Spectroscopic_Values']
        
        # Until a better method is provided....        
        h5_f = h5py.File(path.join(folder_path,'line_1.mat'),'r')
        num_ai_chans = h5_f['data'].shape[1]
        h5_f.close()        
                
        # technically should change the date, etc.              
        spm_data = MicroDataGroup('')
        global_parms = generateDummyMainParms()
        global_parms['data_type'] = 'fastIV'
        global_parms['translator'] = 'fastIV'
        spm_data.attrs = global_parms
        meas_grp = MicroDataGroup('Measurement_000') 
        spm_data.addChildren([meas_grp])
        
        hdf = ioHDF5(h5_path)
        #spm_data.showTree()
        hdf.writeData(spm_data, print_log=False)

        raw_datasets = list()        
        
        for chan_index in xrange(num_ai_chans):
            
            chan_grp = MicroDataGroup('{:s}{:03d}'.format('Channel_',chan_index),'/Measurement_000/')
            chan_grp.attrs = parm_dict        
            chan_grp.addChildren([ds_pos_ind, ds_pos_val, ds_spec_inds, ds_spec_vals, 
                                  ds_ex_efm, ds_raw_data])
            h5_refs = hdf.writeData(chan_grp, print_log=False)
            h5_raw = getH5DsetRefs(['Raw_Data'], h5_refs)[0]
            linkRefs(h5_raw, getH5DsetRefs(aux_ds_names, h5_refs))
            raw_datasets.append(h5_raw)
            
        # Now that the N channels have been made, populate them with the actual data....
        
        for line_ind in xrange(parm_dict['grid_num_rows']):
            if line_ind % np.round(parm_dict['grid_num_rows']/10) == 0:
                print('Reading data in line {} of {}'.format(line_ind+1,parm_dict['grid_num_rows']))
            file_path = path.join(folder_path,'line_'+str(line_ind+1)+'.mat')
            if path.exists(file_path):
                h5_f = h5py.File(file_path,'r')
                h5_data = h5_f['data']
                if h5_data.shape[0] >= len(excit_wfm) and h5_data.shape[1] == len(raw_datasets):
                    for chan, h5_chan in enumerate(raw_datasets):
                        h5_chan[line_ind,:] = np.float16(h5_data[:-1*parm_dict['excitation_extra_pts'],chan])
                    hdf.flush()
                else:
                    warn('No data found for Line '+str(line_ind))
            else:
                warn('File not found for: line '+str(line_ind))
            
        hdf.close()        
        return h5_path
                    
    
    @staticmethod
    def __getparms(parm_path):
        """
        Copies experimental parameters from the .mat file to a dictionary
        
        Parameters
        ------------
        parm_path : string / unicode
            Absolute path of the parameters file
        
        Returns 
        --------
        (parm_dict, excit_wfm) : tuple
        
        parm_dict : dictionary
            Dictionary containing all relevant parameters
        excit_wfm : 1d numpy float array
            Excitation waveform
        """
        h5_f = h5py.File(parm_path,'r')
        parm_dict = dict()
        
        parm_dict['IO_samp_rate_[Hz]'] = np.uint32(h5_f['samp_rate'][0][0])
        parm_dict['excitation_frequency_[Hz]'] = np.float32(h5_f['frequency'][0][0])
        excit_wfm = np.float32(np.squeeze(h5_f['excit_wfm'].value))
        
        # Make sure to truncate the data to the point when the 
        pts_per_cycle = int(np.round(1.0*parm_dict['IO_samp_rate_[Hz]']/parm_dict['excitation_frequency_[Hz]']))
        extra_pts = len(excit_wfm) % pts_per_cycle
        parm_dict['excitation_extra_pts'] = extra_pts
        line_time = np.float32(h5_f['line_time'][0][0])
        excess_time = line_time - extra_pts/parm_dict['IO_samp_rate_[Hz]']
        parm_dict['excitation_duration_[sec]']=line_time - excess_time
        excit_wfm = excit_wfm[:-extra_pts]
        
        parm_dict['grid_num_rows'] = np.int32(h5_f['num_lines'][0][0])
        parm_dict['grid_num_cols'] = np.int32(h5_f['num_pixels'][0][0])
        
        parm_dict['grid_scan_height_[m]'] = np.float32(h5_f['scan_height'][0][0])
        parm_dict['grid_scan_width_[m]'] = np.float32(h5_f['scan_width'][0][0])
        parm_dict['grid_scan_speed_[ms-1]'] = np.float32(h5_f['scan_speed'][0][0])
        
        h5_f.close()
        return parm_dict, excit_wfm