# -*- coding: utf-8 -*-
"""
Created on Sat Nov 07 15:21:46 2015

@author: Suhas Somnath
"""
from __future__ import division, print_function # int/int = float

from os import path, listdir, remove # File Path formatting
from warnings import warn

import numpy as np # For array operations
from scipy.io.matlab import loadmat # To load parameters stored in Matlab .mat file

from .be_utils import parmsToDict
from .translator import Translator
from .utils import interpretFreq, makePositionMat, getPositionSlicing, generateDummyMainParms
from ..hdf_utils import getH5DsetRefs, linkRefs
from ..io_hdf5 import ioHDF5 # Now the translator is responsible for writing the data.
from ..microdata import MicroDataGroup, MicroDataset # The building blocks for defining heirarchical storage in the H5 file


class GLineTranslator(Translator):
    """
    Translated G-mode line (bigtimedata.dat) files from actual BE line experiments to HDF5
    """
    
    def translate(self, file_path):
        """
        The main function that translates the provided file into a .h5 file
        
        Inputs:
            data_filepath: Absolute path of the data file (.dat) in
        """
        # Figure out the basename of the data:
        (basename,parm_paths, data_paths) = self.__ParseFilePath(file_path)
        
        (folder_path, unused) = path.split(file_path)
        h5_path = path.join(folder_path,basename+'.h5')
        
        if path.exists(h5_path):
            remove(h5_path)
        
        # Load parameters from .mat file - 'BE_wave', 'total_cols', 'total_rows', 'FFT_BE_wave'
        matread = loadmat(parm_paths['parm_mat'],variable_names=['BE_wave','FFT_BE_wave', 'total_cols', 'total_rows'])
        BE_wave = np.float32(np.squeeze(matread['BE_wave']))         
        # Need to take the complex conjugate if reading from a .mat file
        # FFT_BE_wave = np.conjugate(np.complex64(np.squeeze(matread['FFT_BE_wave'])))
        
        self.num_cols = int(matread['total_cols'][0][0])
        expected_rows = int(matread['total_rows'][0][0])
        self.num_points = len(BE_wave)
        
        # Load parameters from .txt file - 'BE_center_frequency_[Hz]', 'IO rate'
        (isBEPS,parm_dict) = parmsToDict(parm_paths['parm_txt'])
        
        # IO rate is the same for the entire board / any channel
        IO_rate = interpretFreq(parm_dict['IO rate'])
        
        # Get file byte size:
        # For now, assume that bigtime_00 always exists and is the main file
        file_size = path.getsize(data_paths[0])
        
        # Calculate actual number of lines since the first few lines may not be saved
        self.num_rows = 1.0*file_size/(4*self.num_points*self.num_cols)
        if self.num_rows % 1:
            warn('Error - File has incomplete rows')
            return None
        else:
            self.num_rows = int(self.num_rows)
            
        num_pix = self.num_rows * self.num_cols
        
        # Some very basic information that can help the processing crew
        parm_dict['num_bins'] = self.num_points
        parm_dict['num_pix'] = num_pix
        parm_dict['grid_num_rows'] = self.num_rows
        parm_dict['data_type'] = 'GLine' #self.__class__.__name__
            
        if self.num_rows != expected_rows:
            print('Note: {} of {} lines found in data file'.format(self.num_rows,expected_rows))
        
        # Calculate number of points to read per line:
        self.__bytes_per_row__ = int(file_size/self.num_rows)
        
        pos_mat = makePositionMat([self.num_cols, self.num_rows])
        pos_slices = getPositionSlicing(['X','Y'], num_pix)        
        
        # First finish writing all global parameters, create the file too:
        spm_data = MicroDataGroup('')
        global_parms = generateDummyMainParms()
        global_parms['data_type'] = 'GLine' #self.__class__.__name__
        global_parms['translator'] = 'GLine'
        spm_data.attrs = global_parms
        spm_data.addChildren([MicroDataGroup('Measurement_000')])
        
        hdf = ioHDF5(h5_path)
        # hdf.clear()
        hdf.writeData(spm_data)
        
        # Now that the file has been created, go over each raw data file:
        # 1. write all ancillary data. Link data. 2. Write main data sequentially
                
        """ We only allocate the space for the main data here.
        This does NOT change with each file. The data written to it does.
        The auxillary datasets will not change with each raw data file since 
        only one excitation waveform is used"""
        ds_main_data = MicroDataset('Raw_Data', data=[], 
                                  maxshape=(num_pix,self.num_points), 
                                    chunking=(1,self.num_points), dtype=np.float16)
        ds_ex_wfm = MicroDataset('Excitation_Waveform', np.float32(BE_wave))     
        ds_pos_ind = MicroDataset('Position_Indices', np.uint32(pos_mat))
        ds_pos_ind.attrs['labels'] = pos_slices
        ds_pos_val = MicroDataset('Position_Values', np.float32(pos_mat))
        ds_pos_val.attrs['labels'] = pos_slices          
        ds_spec_inds = MicroDataset('Spectroscopic_Indices', np.atleast_2d(np.arange(self.num_points, dtype=np.int32), dtype=np.uint32))
        ds_spec_inds.attrs['labels'] = {'Time': (slice(self.num_points))}          
        ds_spec_vals = MicroDataset('Spectroscopic_Values', np.atleast_2d(np.arange(self.num_points, dtype=np.float32)/IO_rate, dtype=np.float32))            
        ds_spec_vals.attrs['labels'] = {'Time': (slice(self.num_points))} 
        
        aux_ds_names = ['Excitation_Waveform', 'Position_Indices','Position_Values',
                     'Spectroscopic_Indices','Spectroscopic_Values']
        
        for f_index in data_paths.keys():
            
            meas_grp = MicroDataGroup('{:s}{:03d}'.format('Channel_',f_index),'/Measurement_000/')
            meas_grp.attrs = parm_dict
            meas_grp.addChildren([ds_main_data, ds_ex_wfm, ds_pos_ind, 
                                  ds_pos_val, ds_spec_inds, ds_spec_vals])
            
            # print('Writing following treee to file:')
            # meas_grp.showTree()
            h5_refs = hdf.writeData(meas_grp)
            
            h5_main = getH5DsetRefs(['Raw_Data'], h5_refs)[0] # We know there is exactly one main data
            
            # Reference linking can certainly take place even before the datasets have reached their final size         
            linkRefs(h5_main, getH5DsetRefs(aux_ds_names, h5_refs))
            
            # Now transfer scan data in the dat file to the h5 file:
            self.__readdatafile(data_paths[f_index],h5_main)
            
        hdf.close()
            
            
    @staticmethod
    def __ParseFilePath(data_filepath):
        """
        Goes through the file directory and figures out the basename and the 
        parameter (text and .mat), data file paths (for each analog input channel)
        
        Parameters
        -----------------
        data_filepath : string / unicode
            absolute path of the bigtime .dat files
        
        Returns
        ----------------
        basename : string / unicode
            base name of the experiment\n
        parm_paths : dictionary
            paths for the text and .mat parameter files\n
            parm_text : absolute filepath of the parameter text file\n
            parm_mat : absolute filepath of the parmeter .mat file
        data_paths : dictionariy of the paths for the bigtime data files.
            key : index of the analog input that generated the data file\n
            value : absolute filepath of the data file
        """
        # Return (basename, parameter text path)
        (folder_path, basename) = path.split(data_filepath)
        (upper_folder, basename) = path.split(folder_path)
        
        # There may be one or two bigdata files. May need both paths
        parm_paths = dict()
        data_paths = dict()
        targ_str = 'bigtime_0'
        for filenames in listdir(folder_path):
            ind = filenames.find(targ_str)
            if ind > 0:
                data_paths[int(filenames[ind+len(targ_str)])] = path.join(folder_path,filenames)
        
            if filenames.endswith('.txt') and filenames.find('parm') > 0:
                parm_paths['parm_txt'] = path.join(folder_path,filenames)
                
            if filenames.endswith('_all.mat'):
                parm_paths['parm_mat'] = path.join(folder_path,filenames)
                
        return (basename,parm_paths, data_paths)



    def __readdatafile(self,filepath,h5_dset):
        """
        Reads the .dat file and populates the .h5 dataset

        Parameters
        ---------
        filepath : String / unicode
            absolute path of the data file for a particular analog input channel
        h5_dset : HDF5 dataset reference
            Reference to the target Raw_Data dataset

        Returns
        ---------
        None
        """
        # Create data matrix - Only need 16 bit floats (time) 
                
        # Read line by line and write to h5                 
        with open(filepath, 'rb') as file_handl:
            for row_indx in xrange(self.num_rows):
                
                if row_indx % 10 == 0:
                    print('Reading line {} of {}'.format(row_indx,self.num_rows))
                
                file_handl.seek(row_indx*self.__bytes_per_row__,0)
                data_vec = np.fromstring(file_handl.read(self.__bytes_per_row__), dtype='f')
                data_mat = data_vec.reshape(self.num_cols, self.num_points)               
                h5_dset[row_indx*self.num_cols:(row_indx+1)*self.num_cols,:] = np.float16(data_mat)
                h5_dset.file.flush()
                del data_vec, data_mat
        
        print('Finished reading file: %s!' %(filepath))        