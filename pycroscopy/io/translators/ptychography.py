"""
Created on Feb 9, 2016

@author: Chris Smith
"""

import numpy as np
import os
from scipy.misc import imread
from ..iohdf5 import ioHDF5
from ..microdata import MicroDataGroup, MicroDataset
from ..hdfutils import getH5DsetRefs
from .translator import Translator
from .utils import generateDummyMainParms, makePositionMat, getPositionSlicing

class PtychographyTranslator(Translator):
    """
    Translate Pytchography data from a set of images to an HDF5 file
    """

    def translate(self, h5_path, image_path):
        """
        Basic method that adds Ptychography data to existing hdf5 thisfile
        You must have already done the basic translation with BEodfTranslator
        
        Parameters
        ----------------
        h5_path : Absolute thisfile path for the data thisfile.
            Must be in hdf5 format
        image_path : Absolute path to folder holding the image files 
            
        Returns
        ----------
        None
        """
                
        # Get the list of all files with the .tif extension and the number of files in the list

        file_list, num_files = self.__getfilelist(image_path,'.tif')
          
        # Open the hdf5 thisfile and delete any contents
        try:
            hdf = ioHDF5(h5_path)
            hdf.clear()
        except:
            raise
                
        # Set up the basic parameters associated with this set of images
        
        (usize,vsize),data_type = self.__getimagesize(os.path.join(image_path,file_list[0]))
        num_pixels = usize*vsize
        
        scan_size = np.int(np.sqrt(num_files))
        num_files = scan_size**2
        
        mean_ronch = np.zeros(num_pixels,dtype=np.float32)  
        
        h5_main, h5_mean_spec, h5_ronch = self.__setupH5(num_files, hdf, usize, 
                                                         vsize, np.float32, 
                                                         num_pixels, 
                                                         scan_size)

        for ifile, thisfile in enumerate(file_list[:num_files]):

            selected = (ifile+1) % round(num_files/16) == 0
            if selected:
                print('Processing file...{}% - reading: {}'.format(round(100*ifile/num_files),thisfile))
            
            image = imread(os.path.join(image_path,thisfile))
            image = image.reshape(num_pixels)
            h5_main[ifile,:] = image
            
            h5_mean_spec[ifile] = np.mean(image)
            
            mean_ronch += image
            
            hdf.flush()
        
        #print('{}, {}'.format(np.max(mean_ronch), np.min(mean_ronch)))
        
        h5_ronch[:] = mean_ronch/num_files
        
        hdf.flush()
        hdf.close()
        
        
    def __getfilelist(self, path, ftype='all'):
        """
        Returns a list of all files in the directory given by path
        
        Parameters
        ---------------
        path : string / unicode
            absolute path to directory containing files
        ftype : this file types to return in file_list. (optional. Default is all) 
        
        Returns
        ----------
        file_list : list of strings
            names of all files in directory located at path
        numfiles : unsigned int
            number of files in file_list
        """    
        

        # Get all files in directory
        file_list = os.listdir(path)
        
        # If no file type specified, return full list
        if ftype=='all':
            numfiles = len(file_list)
            return (file_list, numfiles)

        # Remove files of type other than the request ftype from the list
        new_file_list = []
        for this_thing in file_list:
            split = os.path.splitext(this_thing)
            if len(split) <2:
                continue
            ext = split[1]
            if ext == ftype:
                new_file_list.append(this_thing)

        numfiles = len(new_file_list)
        
        return (new_file_list,numfiles)

    def __getimagesize(self,image):
        """
        Returns the x and y size of the image in pixels
        
        Parameters
        ------------
        image : string / unicode
            absolute path to the image file
        
        Returns
        -----------
        (size, tmp.dtype) : Tuple 
        
        size : unsigned integer
            x and y dimenstions of image
        dtype : data type
            Datatype of the image
        """
        tmp = imread(image)
        size = tmp.shape
        
        return (size, tmp.dtype)
    
    def __buildSpectroscopic(self, usize, vsize, num_pixels):
        spec_mat = makePositionMat([usize, vsize])
        spec_slices = {'U': (slice(0,1,None),slice(0,num_pixels,None)),
                       'V': (slice(1,2,None),slice(0,num_pixels,None))}
        ds_spec_ind = MicroDataset('Spectroscopic_Indices', np.transpose(spec_mat), dtype=np.uint32)
        ds_spec_ind.attrs['labels'] = spec_slices
        ds_spec_vals = MicroDataset('Spectroscopic_Values', np.transpose(spec_mat), dtype=np.float32)
        ds_spec_vals.attrs['labels'] = spec_slices
        ds_spec_vals.attrs['units'] = ['', '']
    
        return ds_spec_ind, ds_spec_vals
    
    def __buildPosition(self, num_files, scan_size):
        pos_mat = makePositionMat([scan_size, scan_size])
        pos_slices = getPositionSlicing(['X', 'Y'], num_files)
        ds_pos_ind = MicroDataset('Position_Indices', pos_mat, dtype=np.uint32)
        ds_pos_ind.attrs['labels'] = pos_slices
        ds_pos_val = MicroDataset('Position_Values', pos_mat)
        ds_pos_val.attrs['labels'] = pos_slices
        ds_pos_val.attrs['units'] = ['pixel', 'pixel']
        
        return ds_pos_ind, ds_pos_val

    def __setupH5(self, num_files, hdf, usize, vsize, data_type, num_pixels, scan_size):
        main_parms = generateDummyMainParms()
        main_parms['num_images'] = num_files
        main_parms['datatype'] = 'ptychography'
        main_parms['image_size_u'] = usize
        main_parms['image_size_v'] = vsize
        main_parms['num_pixels'] = num_pixels
        main_parms['translator'] = 'Ptychography'
        main_parms['scan_size_x'] = scan_size
        main_parms['scan_size_y'] = scan_size
    # Create the hdf5 data Group
        meas_grp = MicroDataGroup('Measurement_000',parent='/')
        meas_grp.attrs = main_parms
        chan_grp = MicroDataGroup('Channel_000')
    # Get the Position and Spectroscopic Datasets
        ds_spec_ind, ds_spec_vals = self.__buildSpectroscopic(usize, vsize, num_pixels)
        ds_pos_ind, ds_pos_val = self.__buildPosition(num_files, scan_size)
        
    # Allocate space for Main_Data and Pixel averaged Data
        ds_main_data = MicroDataset('Raw_Data', data=[], 
            maxshape=(num_files, num_pixels), 
            chunking=(1, num_pixels), 
            dtype=data_type, 
            compression='gzip')
        ds_mean_ronch_data = MicroDataset('Mean_Ronchigram', 
            data=np.zeros(num_pixels, 
                dtype=np.float32), 
            dtype=np.float32)
        ds_mean_spec_data = MicroDataset('Spectroscopic_Mean', 
            data=np.zeros(num_files, 
                dtype=np.float32), 
            dtype=np.float32)
    # Add datasets as children of Measurement_000 data group
        chan_grp.addChildren([ds_main_data, ds_spec_ind, ds_spec_vals, ds_pos_ind, ds_pos_val, ds_mean_ronch_data, ds_mean_spec_data])
        meas_grp.addChildren([chan_grp])
        #print('Writing following tree to this file:')
        #meas_grp.showTree()
        h5_refs = hdf.writeData(meas_grp)
        h5_main = getH5DsetRefs(['Raw_Data'], h5_refs)[0]
        h5_ronch = getH5DsetRefs(['Mean_Ronchigram'], h5_refs)[0]
        h5_mean_spec = getH5DsetRefs(['Spectroscopic_Mean'], h5_refs)[0]
        aux_ds_names = ['Spectroscopic_Indices', 
                        'Spectroscopic_Values', 
                        'Position_Indices', 
                        'Position_Values']
        hdf.linkRefs(h5_main, getH5DsetRefs(aux_ds_names, h5_refs))
        
        hdf.flush()
        
        return h5_main, h5_mean_spec, h5_ronch


