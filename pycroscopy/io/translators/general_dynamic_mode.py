# -*- coding: utf-8 -*-
"""
Created on Fri Mar 04 11:12:45 2016

@author: Suhas Somnath
"""

from __future__ import print_function, division # int/int = float

from os import path, remove# File Path formatting

import numpy as np # For array operations
from scipy.io.matlab import loadmat # To load parameters stored in Matlab .mat file

from .gmode_utils import readGmodeParms
from .translator import Translator # Because this class extends the abstract Translator class
from .utils import makePositionMat, getPositionSlicing, generateDummyMainParms
from ..hdf_utils import getH5DsetRefs, linkRefs
from ..io_hdf5 import ioHDF5 # Now the translator is responsible for writing the data.
from ..microdata import MicroDataGroup, MicroDataset # The building blocks for defining heirarchical storage in the H5 file


class GDMTranslator(Translator):
    """
    Translates G-mode w^2 datasets from .mat files to .h5
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
        parm_dict = readGmodeParms(parm_path)
        
        # Add the w^2 specific parameters to this list
        parm_data = loadmat(parm_path, squeeze_me=True, struct_as_record=True)
        freq_sweep_parms = parm_data['freqSweepParms']
        parm_dict['freq_sweep_delay'] = np.float(freq_sweep_parms['delay'].item())
        gen_sig = parm_data['genSig']
        parm_dict['wfm_fix_d_fast'] = np.int32(gen_sig['restrictT'].item())
        freq_array = np.float32(parm_data['freqArray'])
        
        # prepare and write spectroscopic values
        samp_rate = parm_dict['IO_down_samp_rate_[Hz]']
        num_bins = int(parm_dict['wfm_n_cycles']*parm_dict['wfm_p_slow']*samp_rate)
        
        w_vec = np.arange(-0.5*samp_rate, 0.5*samp_rate,np.float32(samp_rate/num_bins))
        
        # There is most likely a more elegant solution to this but I don't have the time... Maybe np.meshgrid
        spec_val_mat = np.zeros((len(freq_array)*num_bins,2), dtype=np.float32)
        spec_val_mat[:,0] = np.tile(w_vec,len(freq_array)) 
        spec_val_mat[:,1] = np.repeat(freq_array,num_bins) 
        
        spec_ind_mat = np.zeros((2,len(freq_array)*num_bins), dtype=np.int32)
        spec_ind_mat[0,:] = np.tile(np.arange(num_bins),len(freq_array)) 
        spec_ind_mat[1,:] = np.repeat(np.arange(len(freq_array)),num_bins) 
        
        num_rows = parm_dict['grid_num_rows']
        num_cols = parm_dict['grid_num_cols']
        parm_dict['data_type'] = 'GmodeW2'
            
        num_pix = num_rows*num_cols
        
        pos_mat = makePositionMat([num_cols, num_rows])
        pos_slices = getPositionSlicing(['X','Y'], num_pix)
        
        # Now start creating datasets and populating:
        ds_pos_ind = MicroDataset('Position_Indices', np.uint32(pos_mat))
        ds_pos_ind.attrs['labels'] = pos_slices
        ds_pos_val = MicroDataset('Position_Values', np.float32(pos_mat))
        ds_pos_val.attrs['labels'] = pos_slices
              
        ds_spec_inds = MicroDataset('Spectroscopic_Indices', np.uint32(spec_ind_mat))
        ds_spec_inds.attrs['labels'] = {'Response Bin Index':(slice(0,1), slice(None)), 'Excitation Frequency Index':(slice(1,2),slice(None))}
        
        ds_spec_vals = MicroDataset('Spectroscopic_Values', spec_val_mat)
        ds_spec_vals.attrs['labels'] = {'Response Bin':(slice(0,1), slice(None)), 'Excitation Frequency':(slice(1,2), slice(None))}
                
        ds_ex_freqs = MicroDataset('Excitation_Frequencies', freq_array)
        ds_bin_freq = MicroDataset('Bin_Frequencies', w_vec)
        
        # Minimize file size to the extent possible.
        # DAQs are rated at 16 bit so float16 should be most appropriate.
        # For some reason, compression is more effective on time series data
        ds_main_data = MicroDataset('Raw_Data', data=[], maxshape=(num_pix, len(freq_array)*num_bins), dtype=np.float32, chunking=(1,num_bins), compression='gzip')
        
        chan_grp = MicroDataGroup('Channel_000')		
        chan_grp.attrs = parm_dict        
        chan_grp.addChildren([ds_pos_ind, ds_pos_val, ds_spec_inds, ds_spec_vals, 
								ds_ex_freqs, ds_bin_freq, ds_main_data])
        meas_grp = MicroDataGroup('Measurement_000')
        meas_grp.addChildren([chan_grp])
        
        spm_data = MicroDataGroup('')
        global_parms = generateDummyMainParms()
        global_parms['grid_size_x'] = parm_dict['grid_num_cols'];
        global_parms['grid_size_y'] = parm_dict['grid_num_rows'];
        # assuming that the experiment was completed:        
        global_parms['current_position_x'] = parm_dict['grid_num_cols']-1;
        global_parms['current_position_y'] = parm_dict['grid_num_rows']-1;
        global_parms['data_type'] = parm_dict['data_type'] #self.__class__.__name__
        global_parms['translator'] = 'W2'
        spm_data.attrs = global_parms
        spm_data.addChildren([meas_grp])
        
        if path.exists(h5_path):
            remove(h5_path)
        
        # Write everything except for the main data.
        hdf = ioHDF5(h5_path)
        
        h5_refs = hdf.writeData(spm_data)
                    
        h5_main = getH5DsetRefs(['Raw_Data'], h5_refs)[0]
            
        #Now doing linkrefs:
        aux_ds_names = ['Position_Indices','Position_Values',
                     'Spectroscopic_Indices','Spectroscopic_Values',
                     'Excitation_Frequencies', 'Bin_Frequencies']
        linkRefs(h5_main, getH5DsetRefs(aux_ds_names, h5_refs))

        # Now read the raw data files:
        pos_ind = 0
        for row_ind in xrange(1,num_rows+1):
            for col_ind in xrange(1,num_cols+1):
                file_path = path.join(folder_path,'fSweep_r'+str(row_ind)+'_c'+str(col_ind)+'.mat')
                print('Working on row {} col {}'.format(row_ind,col_ind))
                if path.exists(file_path):
                    # Load data file
                    pix_data = loadmat(file_path, squeeze_me=True)
                    pix_mat = pix_data['AI_mat']
                    # Take the inverse FFT on 2nd dimension
                    pix_mat = np.fft.ifft(np.fft.ifftshift(pix_mat,axes=1), axis=1)
                    # Verified with Matlab - no conjugate required here.
                    pix_vec = pix_mat.transpose().reshape(pix_mat.size)
                    h5_main[pos_ind,:] = np.float32(pix_vec)
                    hdf.flush() # flush from memory!
                else:
                    print('File not found for: row {} col {}'.format(row_ind,col_ind))
                pos_ind +=1
                if (100.0*(pos_ind)/num_pix)%10 == 0:
                    print('completed translating {} %'.format(int(100*pos_ind/num_pix)))
                    
        hdf.close()
        
        return h5_path