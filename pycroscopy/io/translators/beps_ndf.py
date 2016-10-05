# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 15:07:16 2015

@author: Suhas Somnath

"""

from __future__ import division # int/int = float
#import abc # Abstract base class https://pymotw.com/2/abc/ <---- This needs to be implemented in a cleaner way
import numpy as np # For array operations
from os import path, listdir, remove # File Path formatting
from warnings import warn
import xlrd as xlreader # To read the UDVS spreadsheet
from scipy.io.matlab import loadmat; # To load parameters stored in Matlab .mat file
from .utils import makePositionMat, generateDummyMainParms
from .be_utils import trimUDVS, getSpectroscopicParmLabel, parmsToDict, generatePlotGroups, normalizeBEresponse, createSpecVals
from ..microdata import MicroDataGroup, MicroDataset # The building blocks for defining heirarchical storage in the H5 file
from ..io_hdf5 import ioHDF5 # Now the translator is responsible for writing the data.
from .translator import Translator
from ..be_hdf_utils import maxReadPixels
from ..hdf_utils import getH5DsetRefs, linkRefs

class BEPSndfTranslator(Translator):
    """
    Translates Band Excitation Polarization Switching (BEPS) datasets from .dat
    files to .h5
    """       
        
    def translate(self, data_filepath, show_plots=True, save_plots=True, do_histogram=False, debug=False):
        """
        The main function that translates the provided file into a .h5 file
        
        Parameters
        ----------------
        data_filepath : String / unicode
            Absolute path of the data file (.dat)
        show_plots : Boolean (Optional. Default is True)
            Whether or not to show plots
        save_plots : Boolean (Optional. Default is True)
            Whether or not to save the generated plots
        do_histogram : Boolean (Optional. Default is False)
            Whether or not to generate and save 2D histograms of the raw data
        debug : Boolean (Optional. default is false)
            Whether or not to print log statements
            
        Returns
        --------------
        h5_path : String / unicode
            Absolute path of the generated .h5 file
        """
        ## Read the parameter files
        if debug: print('BEndfTranslator: Getting file paths')
        (parm_filepath,udvs_filepath, parms_mat_path) = self.__getParmFilePaths(data_filepath)
        if debug: print('BEndfTranslator: Reading Parms text file')
        (isBEPS,self.parm_dict) = parmsToDict(parm_filepath)
        self.parm_dict['data_type'] = 'BEPSData'
        if not isBEPS:
            warn('This is NOT a BEPS new-data-format dataset!')
            return None
        
        """ Find out if this is a custom experiment and whether in and out of field were acquired
        For a standard experiment where only in / out field is acquired, zeros are stored
        even for those UDVS steps without band excitation"""
        self.field_mode = self.parm_dict['VS_measure_in_field_loops']
        expt_type = self.parm_dict['VS_mode']
        self.spec_label = getSpectroscopicParmLabel(expt_type)
        std_expt = expt_type in ['DC modulation mode','current mode']
        self.halve_udvs_steps = False
        ignored_plt_grps = []
        if std_expt and self.field_mode != 'in and out-of-field':
            self.halve_udvs_steps = True
            if self.field_mode == 'out-of-field':
                ignored_plt_grps = ['in-field']
            else:
                ignored_plt_grps = ['out-of-field']
            
        
        h5_path = path.join(self.folder_path,self.basename+'.h5')
        if path.exists(h5_path):
            remove(h5_path)
        
        if debug: print('BEndfTranslator: Preparing to read parms.mat file')
        self.BE_wave = self.__getExWfm(parms_mat_path)
        
        if debug: print('BEndfTranslator: About to read UDVS file')
        
        (self.udvs_labs, self.udvs_units, self.udvs_mat) = self.__readUDVStable(udvs_filepath)
        # Remove the unused plot group columns before proceeding:
        (self.udvs_mat, self.udvs_labs, self.udvs_units) = trimUDVS(self.udvs_mat, 
                                                                    self.udvs_labs, 
                                                                    self.udvs_units, 
                                                                    ignored_plt_grps)
        if debug: print('BEndfTranslator: Read UDVS file')
            
        self.num_udvs_steps = self.udvs_mat.shape[0];
        self.excit_type_vec = (self.udvs_mat[:,4]).astype(int); # This is absolutely crucial for reconstructing the data chronologically       
        
        # First figure out how many waveforms are present in the data from the UDVS
        unique_waves = self.__getUniqueWaveTypes(self.excit_type_vec) 
        self.__unique_waves__ = unique_waves
        self.__num_wave_types__ = len(unique_waves)
        # print self.__num_wave_types__, 'different excitation waveforms in this experiment'
        
        if debug: print('BEndfTranslator: Preparing to set up parsers')

        # Preparing objects to parse the file(s)
        parsers = self.__assembleParsers()        
        
        # Gathering some basic details before parsing the files:
        self.max_pixels = parsers[0].getNumPixels()
        s_pixels = np.array(parsers[0].getSpatialPixels())
        self.pos_labels = ['Laser Spot','Z','X','Y']
        self.pos_labels = [self.pos_labels[i] for i in np.where(s_pixels>1)[0]]
        self.pos_mat = makePositionMat(s_pixels)
        self.pos_units = ['' for _ in xrange(len(self.pos_labels))]     
#         self.pos_mat = np.int32(self.pos_mat)
        
        # Helping Eric out a bit. Remove this section at a later time:
        main_parms = generateDummyMainParms()
        main_parms['grid_size_x'] = self.parm_dict['grid_num_cols'];
        main_parms['grid_size_y'] = self.parm_dict['grid_num_rows'];
        main_parms['experiment_date'] = self.parm_dict['File_date_and_time']        
        # assuming that the experiment was completed:        
        main_parms['current_position_x'] = self.parm_dict['grid_num_cols']-1;
        main_parms['current_position_y'] = self.parm_dict['grid_num_rows']-1;
        main_parms['data_type'] = 'BEPSData'
        main_parms['translator'] = 'NDF'
        
        # Writing only the root now:
        spm_data = MicroDataGroup('')
        spm_data.attrs = main_parms
        self.hdf = ioHDF5(h5_path)
#         self.hdf.clear()
        
        #cacheSettings = self.hdf.file.id.get_access_plist().get_cache()
        #print 'H5 cache settings: Metadata Objects {}    Data Chunks {}    Raw Data Size(kB) {}'.format(cacheSettings[0],cacheSettings[1],cacheSettings[2]/1024)
        
        self.hdf.writeData(spm_data)
        
        ########################################################
        # Reading and parsing the .dat file(s) 
        
        print('Reading data file(s)')

        self.dset_index = 0
        self.ds_pixel_start_indx = 0            
        
        for pixel_ind in xrange(self.max_pixels):
            
            if (100.0*(pixel_ind+1)/self.max_pixels)%10 == 0:
                print('{} % complete'.format(int(100*(pixel_ind+1)/self.max_pixels)))
            
            # First read the next pixel from all parsers:
            current_pixels = {}
            for prsr in parsers:
                current_pixels[prsr.getWaveType()] = prsr.readPixel()
        
            if pixel_ind == 0:               
                h5_refs = self.__initializeDataSet(self.max_pixels, current_pixels)
                prev_pixels = current_pixels # This is here only to avoid annoying warnings. 
            else: 
                if current_pixels[unique_waves[0]].isDifferentFrom(prev_pixels[unique_waves[0]]):
                    # Some parameter has changed. Write current group and make new group
                    self.__closeDataset(h5_refs, show_plots, save_plots, do_histogram)
                    self.ds_pixel_start_indx = pixel_ind
                    h5_refs = self.__initializeDataSet(self.max_pixels - pixel_ind, current_pixels)
            
            # print('reading Pixel {} of {}'.format(pixel_ind,self.max_pixels))
            self.__appendPixelData(current_pixels)
            
            prev_pixels = current_pixels
        
        self.__closeDataset(h5_refs, show_plots, save_plots, do_histogram)
        
        self.hdf.close()
            
        return h5_path
        
    ###################################################################################################
        
    def __closeDataset(self, h5_refs, show_plots, save_plots, do_histogram):
        """
        Performs following operations : 
            * Updates the number of pixels attribute in the measurement group
            * Writes Noise floor axis labels as region references
            * Writes position values and indices along with region references
            * Links all ancilliary datasets to the main data set
            * Writes the spatiall averaged plot data
        
        Parameters
        ----------
        h5_refs : list of HDF references
            References to the written datasets
        show_plots : Boolean 
            Whether or not to show plots
        save_plots : Boolean
            Whether or not to save the generated plots
        do_histogram : Boolean
            Whether or not to generate and save 2D histograms of the raw data
            
        Returns
        -------
        None
        """
        # Update the number of pixels in the attributes
        meas_grp = self.ds_main.parent
        meas_grp.attrs['num_pix'] = self.ds_pixel_index
        
        # Add region references to the now-completed Noise floor dataset
        noise_slices = dict();
        noise_slices['super_band'] = (slice(None), slice(0,1), slice(None));
        noise_slices['inter_bin_band'] = (slice(None), slice(1,2), slice(None));
        noise_slices['sub_band'] = (slice(None), slice(2,3), slice(None));
        self.hdf.regionRefs(self.ds_noise, noise_slices)
        
        # Write position specific datasets now that the dataset is complete
        pos_slice_dict = dict();
        for spat_ind, spat_dim in enumerate(self.pos_labels):
            pos_slice_dict[spat_dim] = (slice(None),slice(spat_ind,spat_ind+1));
        ds_pos_ind = MicroDataset('Position_Indices', self.pos_mat[self.ds_pixel_start_indx : self.ds_pixel_start_indx + self.ds_pixel_index,:], dtype=np.uint)
        ds_pos_ind.attrs['labels'] = pos_slice_dict
        ds_pos_ind.attrs['units'] = self.pos_units
        ds_pos_val = MicroDataset('Position_Values', np.float32(self.pos_mat[self.ds_pixel_start_indx : self.ds_pixel_start_indx + self.ds_pixel_index,:]))
        ds_pos_val.attrs['labels'] = pos_slice_dict
        ds_pos_val.attrs['units'] = self.pos_units
#         ds_pos_labs = MicroDataset('Position_Labels',np.array(self.pos_labels))          
        meas_grp = MicroDataGroup(meas_grp.name,'/')
        meas_grp.addChildren([ds_pos_ind, ds_pos_val])
        
        h5_refs += self.hdf.writeData(meas_grp)
        
        # Do all the reference linking:
        aux_ds_names = ['Excitation_Waveform', 'Position_Indices','Position_Values',
                     'Spectroscopic_Indices','Bin_Step', 'Bin_Indices', 'Bin_Wfm_Type',
                     'Bin_Frequencies','Bin_FFT', 'UDVS','UDVS_Labels', 'Noise_Floor','Spectroscopic_Values',
                     'UDVS_Indices']
        linkRefs(self.ds_main, getH5DsetRefs(aux_ds_names, h5_refs))

        # While we have all the references and mean data, write the plot groups as well:
        generatePlotGroups(self.ds_main, self.hdf, self.mean_resp, 
                           self.folder_path, self.basename,
                           self.max_resp, self.min_resp, 
                           max_mem_mb=self.max_ram,
                           spec_label = self.spec_label, 
                           show_plots=show_plots, save_plots=save_plots,
                           do_histogram=do_histogram)
        
        # Now that everything about this dataset is complete:
        self.dset_index += 1
        
    ###################################################################################################
            
    def __initializeDataSet(self, num_pix, current_pixels):
        """
        Creates and initializes the primary (and auxillary) datasets and datagroups
        to hold the raw data for the current set of experimental parameters.
        
        Parameters
        ----------
        num_pix : unsigned int
            Number of pixels this datagroup is expected to hold
        current_pixels : dictionary of BEPSndfPixel objects
            Extracted data for the first pixel in this group
            
        Returns
        ---------
        h5_refs : list of HDF5group and HDF5Dataset references 
            references of the written H5 datasets
        """

        tot_bins = 0
        tot_pts = 0
        # Each wavetype can have different number of bins
        for pixl in current_pixels.values():
            tot_bins += pixl.num_bins
            tot_pts += pixl.num_bins*pixl.num_steps
            
        # Need to halve the number of steps when only in / out field is acquired:
        if self.halve_udvs_steps:
            tot_pts = int(tot_pts/2)
        
        # Populate information from the columns within the pixels such as the FFT, bin freq, indices, etc. 
        bin_freqs = np.zeros(shape=(tot_bins), dtype=np.float32)
        bin_inds = np.zeros(shape=(tot_bins), dtype=np.uint32)
        bin_FFT = np.zeros(shape=(tot_bins), dtype=np.complex64)
        exec_bin_vec = np.zeros(shape=(tot_bins), dtype=np.int32)
        pixel_bins = {} # Might be useful later
        stind = 0
        for wave_type in self.__unique_waves__:
            pixl = current_pixels[wave_type]
            exec_bin_vec[stind:stind+pixl.num_bins] = wave_type*np.ones(pixl.num_bins)
            bin_inds[stind:stind+pixl.num_bins] = pixl.BE_bin_ind
            bin_freqs[stind:stind+pixl.num_bins] = pixl.BE_bin_w
            bin_FFT[stind:stind+pixl.num_bins] = pixl.FFT_BE_wave
            pixel_bins[wave_type] = [stind,pixl.num_bins]
            stind+=pixl.num_bins
        del pixl, stind 
                
        # Make the index matrix that has the UDVS step number and bin indices
        spec_inds = np.zeros(shape=(2,tot_pts), dtype=np.uint32)
        stind = 0
        # Need to go through the UDVS file and reconstruct chronologically
        for step_index, wave_type in enumerate(self.excit_type_vec):
            if self.halve_udvs_steps and self.udvs_mat[step_index,2] < 1E-3: # invalid AC amplitude
                    continue # skip
            vals = pixel_bins[wave_type]
            spec_inds[1,stind:stind+vals[1]] = step_index * np.ones(vals[1]) # UDVS step
            spec_inds[0,stind:stind+vals[1]] = np.arange(vals[0],vals[0]+vals[1]) # Bin step
            stind += vals[1]
        del stind,wave_type,step_index
        
        self.spec_inds = spec_inds # will need this for plot group generation
                        
        ds_ex_wfm = MicroDataset('Excitation_Waveform', self.BE_wave)
        ds_bin_freq = MicroDataset('Bin_Frequencies', bin_freqs)
        ds_bin_inds = MicroDataset('Bin_Indices', bin_inds - 1, dtype=np.uint32) # From Matlab (base 1) to Python (base 0)
        ds_bin_FFT = MicroDataset('Bin_FFT', bin_FFT)
        ds_wfm_typ = MicroDataset('Bin_Wfm_Type', exec_bin_vec)
        ds_bin_steps = MicroDataset('Bin_Step', np.arange(tot_bins, dtype=np.uint32)) 
        
        curr_parm_dict = self.parm_dict
        # Some very basic information that can help the processing crew
        curr_parm_dict['num_bins'] = tot_bins
        curr_parm_dict['num_pix'] = num_pix
                
        # technically should change the date, etc.
        self.current_group = '{:s}'.format('Measurement_')
        meas_grp = MicroDataGroup(self.current_group,'/')
        meas_grp.attrs = curr_parm_dict

        chan_grp = MicroDataGroup('Channel_')
        chan_grp.attrs['Channel_Input'] = curr_parm_dict['IO_Analog_Input_1']
        meas_grp.addChildren([chan_grp])
        
        udvs_slices = dict();
        for col_ind, col_name in enumerate(self.udvs_labs):
            udvs_slices[col_name] = (slice(None),slice(col_ind,col_ind+1));
            #print('UDVS column index {} = {}'.format(col_ind,col_name))
        ds_UDVS = MicroDataset('UDVS', self.udvs_mat)
        ds_UDVS.attrs['labels'] = udvs_slices
        ds_UDVS.attrs['units'] = self.udvs_units
        
        actual_udvs_steps = self.num_udvs_steps
        if self.halve_udvs_steps:
            actual_udvs_steps = actual_udvs_steps/2
        
        curr_parm_dict['num_udvs_steps'] = actual_udvs_steps        
        
        ds_UDVS_inds = MicroDataset('UDVS_Indices', self.spec_inds[1])
#         ds_UDVS_inds.attrs['labels'] = {'UDVS_step':(slice(None),)}
        
        
        """
        Create the Spectroscopic Values tables
        """
        spec_vals, spec_inds, spec_vals_labs, spec_vals_units, spec_vals_labs_names =  createSpecVals(self.udvs_mat, spec_inds, bin_freqs, exec_bin_vec, 
                                                                                                      curr_parm_dict, np.array(self.udvs_labs), self.udvs_units)
        
        spec_vals_slices = dict()
        for row_ind, row_name in enumerate(spec_vals_labs):
            spec_vals_slices[row_name]=(slice(row_ind,row_ind+1),slice(None))            
        ds_spec_vals_mat = MicroDataset('Spectroscopic_Values',np.array(spec_vals,dtype=np.float32))
        ds_spec_vals_mat.attrs['labels'] = spec_vals_slices
        ds_spec_vals_mat.attrs['units'] = spec_vals_units
        ds_spec_mat = MicroDataset('Spectroscopic_Indices', spec_inds, dtype=np.uint32)
        ds_spec_mat.attrs['labels'] = spec_vals_slices
        ds_spec_mat.attrs['units'] = spec_vals_units  
        for entry in spec_vals_labs_names:
            label=entry[0]+'_parameters'
            names = entry[1]
            ds_spec_mat.attrs[label]= names
            ds_spec_vals_mat.attrs[label]= names

#         ds_spec_vals_labs = MicroDataset('Spectroscopic_Values_Labels',np.array(spec_vals_labs))
#         ds_main_data = MicroDataset('Raw_Data', np.zeros(shape=(tot_pts,1), dtype=np.complex64), chunking=(tot_bins,1), compression='gzip', dtype=np.complex64, resizable=True)
        """
        New Method for chunking the Main_Data dataset.  Chunking is now done in N-by-N squares of UDVS steps by pixels.
        N is determined dinamically based on the dimensions of the dataset.  Currently it is set such that individual
        chunks are less than 10kB in size.
        
        Chris Smith -- csmith55@utk.edu
        """
        max_bins_per_pixel = np.max(pixel_bins.values())
        pixel_chunking = maxReadPixels(10240, num_pix, max_bins_per_pixel, np.dtype('complex64').itemsize)
        chunking = np.floor(np.sqrt(pixel_chunking))
        chunking = max(1, chunking)
        chunking = min(actual_udvs_steps, num_pix, chunking)
        ds_main_data = MicroDataset('Raw_Data', np.zeros(shape=(1,tot_pts), dtype=np.complex64), chunking=(chunking,chunking*max_bins_per_pixel), resizable=True,compression='gzip')
        
        ds_noise = MicroDataset('Noise_Floor', np.zeros(shape=(1,3,actual_udvs_steps), dtype=np.float32), chunking=(1,3,actual_udvs_steps), compression='gzip', dtype=np.float32, resizable=True)
        noise_labs = ['super_band','inter_bin_band','sub_band']
        noise_slices = dict()
        for col_ind, col_name in enumerate(noise_labs):
            noise_slices[col_name] = (slice(None),slice(col_ind,col_ind+1), slice(None))
        ds_noise.attrs['labels'] = noise_slices
        ds_noise.attrs['units'] = ['','','']
#         ds_noise_labs = MicroDataset('Noise_Labels',np.array(noise_labs))
        # Allocate space for the first pixel for now and write along with the complete tree...
        # Positions CANNOT be written at this time since we don't know if the parameter changed
        
        chan_grp.addChildren([ds_main_data, ds_noise, ds_ex_wfm, ds_spec_mat, ds_wfm_typ,
                              ds_bin_steps, ds_bin_inds, ds_bin_freq, ds_bin_FFT, ds_UDVS, 
                              ds_spec_vals_mat, ds_UDVS_inds])
                              
        #meas_grp.showTree()
        h5_refs = self.hdf.writeData(meas_grp)
        
        self.ds_noise = getH5DsetRefs(['Noise_Floor'], h5_refs)[0] 
        self.ds_main = getH5DsetRefs(['Raw_Data'], h5_refs)[0]
                
        #self.dset_index += 1 # raise dset index after closing only
        self.ds_pixel_index = 0
        
        # Use this for plot groups:
        self.mean_resp = np.zeros(shape=(tot_pts), dtype=np.complex64)
        
        # Used for Histograms
        self.max_resp = np.zeros(shape=(num_pix), dtype = np.float32)
        self.min_resp = np.zeros(shape=(num_pix), dtype = np.float32)
        
        return h5_refs
        
    ###################################################################################################        
        
    def __appendPixelData(self, pixel_data):
        """
        Goes through the list of pixel objects and populates the raw dataset 
        and noise dataset for this spatial pixel.
        
        Parameters
        ----------
        pixel_data : List of BEPSndfPixel objects 
            List containing parsed data for this particular spatial pixel
        
        Returns
        ---------
        None
        """
        
        if self.__num_wave_types__ == 1 and not self.halve_udvs_steps:
            """Technically, this will be taken care of in the later (general) part but 
            since this condition is more common it is worth writing for specifically"""
            
            data_vec = pixel_data[self.__unique_waves__[0]].spectrogram_vec
            noise_mat = np.float32(pixel_data[self.__unique_waves__[0]].noise_floor_mat)
            
        else:

            data_vec = np.zeros(shape=(self.ds_main.shape[1]), dtype=np.complex64)
            noise_mat = np.zeros(shape=(3,self.ds_noise.shape[2]), dtype=np.float32)
            
            internal_step_index = {}
            for wave_type in self.__unique_waves__:
                internal_step_index[wave_type] = 0
                
            stind = 0
            step_counter = 0    
            # Go through each line in the UDVS file and reconstruct chronologically
            for step_index, wave_type in enumerate(self.excit_type_vec):
                # get the noise and data from correct pixel -> address by wave_number.
                
                if self.halve_udvs_steps and self.udvs_mat[step_index,2] < 1E-3: # invalid AC amplitude
                    # print('Step index {} was skipped'.format(step_index))
                    # Not sure why each wavetype has its own counter but there must have been a good reason
                    internal_step_index[wave_type] += 1 
                    continue # skip
                
                data_pix = pixel_data[wave_type].spectrogram_mat
                noise_pix = pixel_data[wave_type].noise_floor_mat
                enind = stind + pixel_data[wave_type].num_bins
                                
                data_vec[stind:enind] = data_pix[:,internal_step_index[wave_type]]
                noise_mat[:,step_counter] = np.float32(noise_pix[:,internal_step_index[wave_type]])
                
                stind = enind;
                internal_step_index[wave_type] += 1
                step_counter += 1
            
            del internal_step_index, stind, enind, step_index, wave_type, step_counter
        
        if self.ds_pixel_index > 0:
            # in the case of the first pixel, we already reserved zeros- no extension
            # for all other lines - we extend the dataset before writing
            self.ds_main.resize(self.ds_main.shape[0]+1, axis = 0)
            self.ds_noise.resize(self.ds_noise.shape[0]+1, axis = 0)

        self.ds_main[-1,:] = data_vec
        self.ds_noise[-1,:,:] = noise_mat
        self.hdf.file.flush()
        
        # Take mean response here:
        self.mean_resp = (1/(self.ds_pixel_index + 1))*(data_vec + self.ds_pixel_index*self.mean_resp)
        
        self.max_resp[self.ds_pixel_index] = np.amax(np.abs(data_vec))
        self.min_resp[self.ds_pixel_index] = np.amin(np.abs(data_vec))
        
        self.ds_pixel_index += 1
               
    ###################################################################################################
    
    def __getParmFilePaths(self,data_filepath):
        """
        Returns the filepaths to the parms text file and UDVS spreadsheet.\n
        Note: This function also initializes the basename and the folder_path for this instance
        
        Parameters
        ---------
        data_filepath : String / unicode
            Absolute path of the data file (.dat) in the newdatafolder
        
        Returns
        ---------
        parm_filepath : String / unicode
            absolute filepath of the parms text file
        udvs_filepath : String / unicode
            absolute file path of the UDVS spreadsheet
        parms_mat_path : String / unicode
            absolute filepath of the .mat parms file
        """
        (self.folder_path, tail) = path.split(data_filepath) # folderpath should end in newdataformat, tail must be the .dat file name
        (main_folder_path,tail) = path.split(self.folder_path) # main_folder_path is the folder above newdataformat or the relative root
        
        parms_mat_path = None
        for filenames in listdir(main_folder_path):
            if filenames.endswith('.txt') and filenames.find('parm') > 0:
                parm_filepath = path.join(main_folder_path,filenames)
            elif filenames.endswith('more_parms.mat'):
                parms_mat_path = path.join(main_folder_path,filenames)
        for filenames in listdir(self.folder_path):
            if (filenames.endswith('.xlsx') or filenames.endswith('.xls')) and filenames.find('UD_VS') > 0:
                udvs_filepath = path.join(self.folder_path,filenames)
                break
                
        (tail,self.basename) = path.split(main_folder_path)     
        
        return (parm_filepath, udvs_filepath, parms_mat_path)
        
    ###################################################################################################
        
    def __assembleParsers(self):
        """
        Returns a list of BEPSndfParser objects per excitation wavetype.
        
        Parameters
        ----------
        None
        
        Returns
        ---------
        parsers: list of BEPSndfParser objects 
            list of the same length as the input numpy array
        """
        parsers = [] # Maybe this needs to be a dictionary instead for easier access?
        for wave_type in self.__unique_waves__:
            filename = self.basename + '_1_'
            if wave_type > 0:
                filename = self.basename + '_1_' + str(wave_type) + '.dat'
            else:
                filename = self.basename + '_1_r' + str(abs(wave_type)) + '.dat'
            datapath = path.join(self.folder_path,filename)
            if path.isfile(datapath) == False:
                warn('Error!!: {}expected but not found!'.format(filename))
                #return
            parsers.append(BEPSndfParser(datapath,wave_type))
        return parsers
        
    ###################################################################################################
        
    def __getExWfm(self, filepath):
        """
        Returns the excitation BE waveform present in the more parms.mat file
        
        Parameters
        ------------
        filepath : String / unicode
            Absolute filepath of the .mat parameter file
        
        Returns
        -----------
        ex_wfm : 1D numpy float array
            Band Excitation waveform
        """
        if not path.exists(filepath):
            warn('BEPSndfTranslator - NO more_parms.mat file found')
            return np.zeros(1000, dtype=np.float32)
            
        matread = loadmat(filepath,variable_names=['FFT_BE_wave']);
        FFT_full = np.complex64(np.squeeze(matread['FFT_BE_wave']));
        return np.float32(np.real(np.fft.ifft(np.fft.ifftshift(FFT_full))))
        
    ###################################################################################################
        
    def __readUDVStable(self,udvs_filepath):
        """
        Reads the UDVS spreadsheet in either .xls or .xlsx format!
        
        Parameters
        ----------
        udvs_filepath : String / unicode
            absolute path to the spreadsheet
            
        Returns
        ----------
        udvs_labs : list of strings
            names of columns in the UDVS table
        udvs_units : list of strings
            units for columns in the UDVS table
        UDVS_mat : 2D numpy float array
            Contents of the UDVS table
        """
        workbook = xlreader.open_workbook(udvs_filepath)
        worksheet = workbook.sheet_by_index(0)
        udvs_labs = list()
        for col in xrange(worksheet.ncols):
            udvs_labs.append(str(worksheet.cell(0,col).value))
        # sometimes, the first few columns are named incorrectly. FORCE them to be named correclty:
        udvs_units = list(['' for _ in xrange(len(udvs_labs))])
        udvs_labs[0:5] = ['step_num','dc_offset','ac_amp','wave_type','wave_mod']
        udvs_units[0:5] = ['', 'V', 'A', '', '']
        
        udvs_mat = np.zeros(shape=(worksheet.nrows-1, worksheet.ncols),dtype=np.float32)
        for row in xrange(1,worksheet.nrows):
            for col in xrange(worksheet.ncols):
                try:
                    udvs_mat[row-1,col] = worksheet.cell(row,col).value
                except ValueError:
                    udvs_mat[row-1,col] = float('NaN')
                except:
                    raise
                    
        # Decrease counter of number of steps by 1 (Python base 0)
        udvs_mat[:,0] -= 1
        
        return (udvs_labs, udvs_units, udvs_mat)
        

        
    def __getUniqueWaveTypes(self,vec):
        """
        Returns a numpy array containing the different waveforms in a 
        BEPS experiment in the format: [1,-1,2,-2.....]
        
        Parameters
        ---------------
        vec : 1D list or 1D numpy array 
            waveform types in the UDVS table
            
        Returns
        ----------
        uniq : 1D numpy array
            Unique waveform types in format listed above
        """
        sorted_all = np.unique(vec)
        pos_vals = sorted_all[sorted_all >= 0]
        neg_vals = sorted_all[sorted_all < 0]
        
        if len(pos_vals) == 0:
            return neg_vals
            
        if len(neg_vals) == 0:
            return pos_vals
            
        uniq = []
        posind = 0 
        negind = len(neg_vals)-1
        while posind < len(pos_vals) or negind >= 0:
            
            if posind == len(pos_vals):
                uniq = uniq + list(neg_vals)
                break
            if negind == len(neg_vals):
                uniq = uniq + list(pos_vals)
                break
            # Otherwise compare
            if pos_vals[posind] < abs(neg_vals[negind]):
                uniq.append(pos_vals[posind])
                posind += 1
            elif pos_vals[posind] == abs(neg_vals[negind]):
                uniq.append(pos_vals[posind])
                uniq.append(neg_vals[negind])
                posind += 1
                negind -= 1
            else:
                uniq.append(neg_vals[negind])
                negind -= 1
                        
        return np.array(uniq)
                
#%% Parser class that parses each data file

class BEPSndfParser(object):
    """
    An object of this class is given the responsibility to step through a 
    BEPS new data format file and return parsed BEPSndfPixel objects.\n
    This class is NOT responsible for actually parsing the byte contents within
    each pixel.\n
    Each wave type is given its own Parser object since it has a file of its own
    """
    
    def __init__(self,file_path,wave_type=1,scout=True):
        """
        Initializes the BEPSndfParser object with following inputs:
        
        Parameters
        -----------
        file_path : string or unicode
            Absolute path of the .dat file
        wave_type : int (optional. Default = 1)
            Integer value signifying type of the excitation waveform\n
        scout : Boolean (optional. Default = true) 
            whether or not the parser should figure out basic details such as 
            the number of pixels, and the spatial dimensionality
        """
        self.__file_handle__ = open(file_path, "rb")
        self.__EOF__ = False
        self.__curr_Pixel__ = 0
        self.__start_point__ = 0
        self.__wave_type__ = wave_type
        self.__filesize__ = path.getsize(file_path)
        self.__pixel_indices__ = list()
        if scout:
            self.__scout()
        
    def getWaveType(self):
        """
        Returns the excitation wave type as an integer
        
        Parameters
        ----------
        None
        
        Returns
        -------
        wave_type : int
            Wave type. Positive number means chirp up, negative number is chirp down.
        """
        return self.__wave_type__
        
    def getNumPixels(self):
        """
        Returns the total number of spatial pixels. This includes X, Y, Z, Laser positions
        
        Parameters
        ----------
        None 
        
        Returns
        -------
        num_pix : unsigned int
            Number of pixels in this file
        """
        return self.__num_pixels__
        
    def getSpatialPixels(self):
        """
        Returns the number of steps in each spatial dimension 
        organized from fastest to slowest varying dimension
        
        Parameters
        ----------
        
        Returns
        -------
        Laser steps, Z steps, Y steps, X steps : unsigned ints
        """
        return (self.__num_laser_steps__,self.__num_z_steps__,self.__num_x_steps__,self.__num_y_steps__)
    
    # Don't use this to figure out if something changes. You need pixel to previous pixel comparison    
    def __scout(self):
        """
        Steps through the file quickly without parsing it. 
        The idea is to calculate the number of pixels ahead of time so that 
        it is easier to parse the dataset. 
        For phase checking, it is recommended that this function be modified to 
        also keep track of the byte positions of the pixels so that pixels can be 
        directly accessed if need be.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        count = 0
        self.__num_pixels__ = 0
        while True:
            self.__pixel_indices__.append(self.__start_point__*4)
            self.__file_handle__.seek(self.__start_point__*4,0)
            spectrogram_length = int(np.fromstring(self.__file_handle__.read(4), dtype='f')[0]) #length of spectrogram
            
            if count == 0:
                self.__file_handle__.seek(self.__start_point__*4,0)
                data_vec = np.fromstring(self.__file_handle__.read(spectrogram_length*4), dtype='f')
                pix = BEPSndfPixel(data_vec,self.__wave_type__)
                self.__num_x_steps__ = pix.num_x_steps
                self.__num_y_steps__ = pix.num_y_steps
                self.__num_z_steps__ = pix.num_z_steps
                self.__num_bins__ = pix.num_bins
                
            count += 1
            self.__start_point__ += spectrogram_length
            
            if self.__filesize__ == self.__start_point__*4:
                self.__num_pixels__ = count
                
                # Laser position spectroscopy is NOT accounted for anywhere. 
                # It is impossible to find out from the parms.txt, UD_VS, or the binary .dat file
                num_laser_steps = 1.0*count/(self.__num_z_steps__*self.__num_y_steps__*self.__num_x_steps__)                
                if num_laser_steps%1.0 != 0:
                    print('Some parameter changed inbetween. BEPS NDF Translator does not handle this usecase at the moment')
                else:
                    self.__num_laser_steps__ = int(num_laser_steps)
                
                break
            
            if self.__filesize__ > self.__start_point__*4:
                self.__num_pixels__ = -1
                
        self.__start_point__ = 0
        
        spat_dim = 0
        if self.__num_z_steps__ > 1:
            #print('Z is varying')
            spat_dim += 1
        if self.__num_y_steps__ > 1:
            #print('Y is varying')
            spat_dim += 1
        if self.__num_x_steps__ > 1:
            #print('X is varying')
            spat_dim += 1
        if self.__num_laser_steps__ > 1:
            # Laser spot position vector is junk in the .dat file
            #print('Laser position / unknown parameter varying')
            spat_dim += 1
        # print('Total of {} spatial dimensions'.format(spat_dim))
        self.__spat_dim__ = spat_dim
            
        
    def readPixel(self):
        """
        Returns a BEpixel object containing the parsed information within a pixel.
        Moves pixel index up by one.
        This is where one could conceivably read the file in one pass instead of making 100,000 file I/Os.

        Returns
        -------
        pixel : BEPSndfPixel
            Object that describes the data contained within the pixel
        """
        
        if self.__filesize__ == self.__start_point__*4:
            print('BEPS NDF Parser - No more pixels left!')
            return -1
        
        self.__file_handle__.seek(self.__start_point__*4,0)
        spectrogram_length = int(np.fromstring(self.__file_handle__.read(4), dtype='f')[0]) #length of spectrogram
        self.__file_handle__.seek(self.__start_point__*4,0)
        data_vec = np.fromstring(self.__file_handle__.read(spectrogram_length*4), dtype='f')
       
        self.__start_point__ += spectrogram_length
        self.__curr_Pixel__ += 1
                
        if self.__filesize__ == self.__start_point__*4:
            print('BEPS NDF Parser reached End of File')
            self.__EOF__ = True
            self.__file_handle__.close()
                
        return BEPSndfPixel(data_vec,abs(self.__wave_type__))    
        

#%% This class parses a data vector 

class BEPSndfPixel(object):
    """
    Stands for BEPS (new data format) Pixel. 
    This class parses (and keeps) the stream of data contained in a single cell of a BEPS data set of the new data format.
    Access desired parameter directly without get methods.
    """
    
    def __init__(self, data_vec,harm=1):
        """
        Initializes the pixel instance by parsing the provided data. 
        
        Parameters
        ----------
        data_vec : 1D float numpy array
            Data contained within each pixel
        harm: unsigned int
            Harmonic of the BE waveform. absolute value of the wave type used to normalize the response waveform.        
        """
        
        harm = abs(harm)
        if harm > 3 or harm < 1:
            harm = 1
            warn('Error in BEPSndfPixel: invalid wave type / harmonic provided.')
        
        # Begin parsing data:
        self.spatial_index = int(data_vec[1])-1
        
        self.spectrogram_length = int(data_vec[0]) 
        
        # calculate indices for parsing
        s1 = int(data_vec[2]) # total rows in pixel
        s2 = int(data_vec[3]) # total cols in pixel
        data_vec1 = data_vec[2:self.spectrogram_length]    
        data_mat1 = data_vec1.reshape(s1,s2)
        spect_size1 = int(data_mat1[1,0]) # total rows in spectrogram set
        self.num_bins = int(spect_size1/2)  # or, len(BE_bin_w)
        self.num_steps = int(data_mat1[1,1]) # total cols in spectrogram set 
        s3 = int(s1-spect_size1) #row index of beginning of spectrogram set
        s4 = int(s2-self.num_steps) #col index of beginning of spectrogram set
            
        self.wave_label =  data_mat1[2,0] # This is useless
        self.wave_modulation_type = data_mat1[2,1] # this is the one with useful information
        #print 'Pixel #',self.spatial_index,' Wave label: ', self.wave_label, ', Wave Type: ', self.wave_modulation_type
        
        # First get the information from the columns:   
        FFT_BE_wave_real = data_mat1[s3:s3-0+self.num_bins,1] #real part of excitation waveform  
        FFT_BE_wave_imag = data_mat1[s3+self.num_bins:s3-0+spect_size1,1]  #imaginary part of excitation waveform  
        
        # Though typecasting the combination of the real and imaginary data looks fine in HDFviewer and Spyder, Labview sees such data as an array of clusters having "r" and "i" elements 
        #self.FFT_BE_wave = np.complex64(FFT_BE_wave_real + 1j*FFT_BE_wave_imag) 
        
        #complex excitation waveform !!! due to a problem in the acquisition software, this may not be normalzed properly
        self.FFT_BE_wave = np.zeros(self.num_bins, dtype=np.complex64);
        self.FFT_BE_wave.real = FFT_BE_wave_real;
        self.FFT_BE_wave.imag = FFT_BE_wave_imag;
        
        del FFT_BE_wave_real, FFT_BE_wave_imag            
        
        self.BE_bin_w = data_mat1[s3:s3-0+self.num_bins,2] # vector of band frequencies
        self.BE_bin_ind = data_mat1[s3+self.num_bins:s3-0+spect_size1,2] # vector of band indices (out of all accesible frequencies below Nyquist frequency)
     
        # Now look at the top few rows to get more information: 
        self.DAQ_channel = data_mat1[2,2]
        self.num_x_steps = int(data_mat1[3,0])
        self.num_y_steps = int(data_mat1[4,0])
        self.num_z_steps = int(data_mat1[5,0])
        self.z_index = int(data_mat1[5,1] - 1)        
        self.y_index = int(data_mat1[4,1] - 1)
        self.x_index = int(data_mat1[3,1] - 1)
     
        self.step_ind_vec = data_mat1[0,s4:] # vector of step indices   
        self.DC_off_vec = data_mat1[1,s4:] # vector of dc offsets  voltages
        self.AC_amp_vec = data_mat1[2,s4:] # vector of ac amplitude voltages 
        self.noise_floor_mat = data_mat1[3:6,s4:] # matrix of noise floor data. Use this information to exclude bins during fitting         
        #plot_group_list_mat = data_mat1[6:s3-2,s4:] # matrix of plot groups
        
        # Here come the optional parameter rows:
        self.deflVolt_vec = data_mat1[s3-2,s4:] # vector of dc cantilever deflection
        # I think this is how the defl setpoint vec should be fixed:
        self.deflVolt_vec[np.isnan(self.deflVolt_vec)] = 0
        # so far, this vector seemed to match the DC offset vector....
        
        self.laser_spot_pos_vec = data_mat1[s3-1,s4:] # NEVER used
        
        # Actual data for this pixel:
        spectrogram_real_mat = data_mat1[s3:s3+self.num_bins,s4:] #real part of response spectrogram
        spectrogram_imag_mat = data_mat1[s3+self.num_bins:s3+spect_size1,s4:] #imaginary part of response spectrogram
        # Be consistent and ensure that the data is also stored as 64 bit complex as in the array creation
        self.spectrogram_mat = np.complex64(spectrogram_real_mat + 1j*spectrogram_imag_mat) #complex part of response spectrogram
        del spectrogram_real_mat, spectrogram_imag_mat 
                
        self.spectrogram_mat = normalizeBEresponse(self.spectrogram_mat, self.FFT_BE_wave, harm)
            
        #  Reshape as one column (its free in Python anyway):
        temp_mat = self.spectrogram_mat.transpose() 
        self.spectrogram_vec = temp_mat.reshape(self.spectrogram_mat.size)
    
            
    def isDifferentFrom(self,prevPixel):
        """
        Compares parameters in this object with those another BEPSndfPixel object 
        to tell if any parameter has changed between these pixels
        
        Parameters
        ----------
        prevPixel : BEPSndfPixel object
            The other pixel object to compare this pixel to
            
        Returns
        -------
        is_different : Boolean
            Whether or not these pixel objects share the same parameters

        Notes
        -----
        *Typical things that change during BEPS*
        1. BE parameters:
            a. Center Frequency, Band Width - changes in the BE_bin_w
            b. Amplitude, Phase Variation, Band Edge Smoothing, Band Edge Trim - Harder to find out what happened exactly - FFT should show changes
            c. BE repeats, desired duration - changes in the spectrogram length?
        2. VS Parameters:
            a. Amplitude, Phase shift - Changes in the AC_amp_vec / DC offset
            b. Offset, Read voltage - Shows up in the DC offset
            c. Steps per full Cycle - Changes in DC offset / AC amplitude ....
            d. Number of cycles, Cycle fraction, In and out of field - changes in the length of DC offset etc.
            e. FORC - should all show up in DC / AC amplitude
        3. IO parameters : don't change really
        4. grid parameters : cannot do anything about this. 
        """
        disp_on = True        
        
        
        if self.spectrogram_length != prevPixel.spectrogram_length:
            if disp_on:
                print('Spectrogram Length changed on pixel {}'.format(self.spatial_index))
            return True
                        
        if self.num_bins != prevPixel.num_bins:
            if disp_on:
                print('Number of bins changed on on pixel {}'.format(self.spatial_index))
            return True
            
        if not np.array_equal(self.BE_bin_w,prevPixel.BE_bin_w):
            if disp_on:
                print('Bin Frequencies changed on pixel {}'.format(self.spatial_index))
            return True
            
        if not np.array_equal(self.FFT_BE_wave, prevPixel.FFT_BE_wave):
            if disp_on:
                print('BE FFT changed on pixel {}'.format(self.spatial_index))
            return True
            
        if not np.array_equal(self.AC_amp_vec, prevPixel.AC_amp_vec):
            if disp_on:
                print('AC amplitude (UDVS) changed on pixel {}'.format(self.spatial_index))
            return True
        
        if not np.array_equal(self.DC_off_vec, prevPixel.DC_off_vec):
            if disp_on:
                print('DC offset (UDVS) changed on pixel {}'.format(self.spatial_index))
            return True
        
        # I was told that this section was just garbage in the file.
        #if not np.array_equal(self.laser_spot_pos_vec, prevPixel.laser_spot_pos_vec):
            #print 'Laser spot position vec was different....'
            #print self.laser_spot_pos_vec
            #return True
            
        if not np.array_equal(self.deflVolt_vec, prevPixel.deflVolt_vec):
            print('deflVolt_vec vec was different....')
            return True
        
        return False