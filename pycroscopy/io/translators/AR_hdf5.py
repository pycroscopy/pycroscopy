# -*- coding: utf-8 -*-
"""
Created on Wed Dec 07 16:04:34 2016

@author: Tommaso Costanzo, Suhas Somnath, Chris R. Smith
"""
import numpy as np
import h5py
import os
import sys
import re #used to get note values

from pyUSID.io.translator import Translator
from pyUSID.io.write_utils import make_indices_matrix

def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])
    # Package for downloading online files:
    # Finally import pyUSID.
try:
    import pyUSID as usid
except ImportError:
    warn('pyUSID not found.  Will install with pip.')
    import pip
    install('pyUSID')
    import pyUSID as usid

class ARhdf5(Translator):
    '''
    Translate Asylum Research HDF5 file into pyUSID.

    The ARhdf5 file should be generated with the converter provided
    by Asylum Research called ARDFtoHDF5. Contact David Aue <David.Aue@oxinst.com>
    or Tommaso Costanzo <tommaso.costanzo01@gmail.com> to get a
    copy of the converter. NOTE: the AR converter works only under
    windows.
    '''

    def __init__(self):
        '''
        Initialize the ARhdf5 class

        Parameters
        ----------------
        data_filepath : String / unicode
            Absolute path of the data file
        '''

        self.debug = False
        self.translated = False # if translate has been run successfully

        self.notes = None
        self.segments = None
        self.segments_name = []
        self.map_size = {'X':0, 'Y':0}
        self.channels_name = []
        self.points_per_sec = None
        
    def translate(self, data_filepath, out_filename,
                  verbose=False, debug=False):
        '''
        The main function that translates the provided file into a .h5 file

        Parameters
        ----------------
        data_filepath : String / unicode
            Absolute path of the data file
        out_filename : String / unicode
            Name for the new generated hdf5 file. The new file will be
            saved in the same folder of the input file with
            file name "out_filename".
            NOTE: the .h5 extension is automatically added to "out_filename"
        debug : Boolean (Optional. default is false)
            Whether or not to print log statements

        Returns
        ----------------
        h5_path : String / unicode
            Absolute path of the generated .h5 file
        '''

        self.debug = debug
    
        #Open the datafile
        try:
            data_filepath = os.path.abspath(data_filepath)
            ARh5_file = h5py.File(data_filepath, 'r')
        except:
            print('Unable to open the file', data_filepath)
            raise

        #Get info from the origin file like Notes and Segments
        self.notes = ARh5_file.attrs['Note']
        self.segments = ARh5_file['ForceMap']['Segments'] #shape: (X, Y, 4)
        self.segments_name = list(ARh5_file['ForceMap'].attrs['Segments'])
        self.map_size['X'] = ARh5_file['ForceMap']['Segments'].shape[0]
        self.map_size['Y'] = ARh5_file['ForceMap']['Segments'].shape[1]
        self.channels_name = list(ARh5_file['ForceMap'].attrs['Channels'])
        try:
            self.points_per_sec = np.float(self.note_value('ARDoIVPointsPerSec'))
        except NameError:
            self.points_per_sec = np.float(self.note_value('NumPtsPerSec'))
        if self.debug:
            print('Map size [X, Y]: ', self.map_size)
            print('Channels names: ', self.channels_name)

        # Only the extension 'Ext' segment can change size
        # so we get the shortest one and we trim all the others
        extension_idx = self.segments_name.index('Ext')
        short_ext = np.amin(np.array(self.segments[:, :, extension_idx]))
        longest_ext = np.amax(np.array(self.segments[:, :, extension_idx]))
        difference = longest_ext - short_ext #this is a difference between integers
        tot_length = (np.amax(self.segments) - difference) + 1 # +1 otherwise \
          # array(tot_length) will be of 1 position shorter
        points_trimmed = np.array(self.segments[:, :, extension_idx]) - short_ext
        if self.debug:
            print('Data were trimmed in the extension segment of {} points'.format(difference))

        # Open the output hdf5 file
        folder_path = os.path.dirname(data_filepath)
        h5_path = os.path.join(folder_path, out_filename + '.h5')
        h5_file = h5py.File(h5_path, 'w')

        # Create the measurement group
        h5_meas_group = usid.hdf_utils.create_indexed_group(h5_file, 'Measurement')

        # Create all channels and main datasets
        # at this point the main dataset are just function of time
        x_dim = np.linspace(0, np.float(self.note_value('FastScanSize')),
                             self.map_size['X'])
        y_dim = np.linspace(0, np.float(self.note_value('FastScanSize')),
                             self.map_size['Y'])
        z_dim = np.arange(tot_length) / np.float(self.points_per_sec)
        pos_dims = [usid.write_utils.Dimension('Cols', 'm', x_dim),
                    usid.write_utils.Dimension('Rows', 'm', y_dim)]
        spec_dims = [usid.write_utils.Dimension('Time', 's', z_dim)]

        # This is quite time consuming, but on magnetic drive
        # is limited from the disk, and therefore is not useful
        # to parallelize these loops
        for index, channel in enumerate(self.channels_name):
            cur_chan = usid.hdf_utils.create_indexed_group(h5_meas_group, 'Channel')
            main_dset = np.empty((self.map_size['X'], self.map_size['Y'], tot_length))
            for column in np.arange(self.map_size['X']):
                for row in np.arange(self.map_size['Y']):
                    AR_pos_string = str(column) + ':' + str(row)
                    seg_start = self.segments[column, row, extension_idx] - short_ext
                    main_dset[column, row, :] = ARh5_file['ForceMap'][AR_pos_string][index, seg_start:]

            # Reshape with Fortran order to have the correct position indices
            main_dset = np.reshape(main_dset, (-1, tot_length), order='F')
            if index == 0:
                first_main_dset = cur_chan
                quant_unit = self.get_def_unit(channel)
                h5_raw = usid.hdf_utils.write_main_dataset(cur_chan, # parent HDF5 group
                                                           main_dset, # 2D array of raw data
                                                           'Raw_'+channel, # Name of main dset
                                                           channel, # Physical quantity
                                                           self.get_def_unit(channel), # Unit
                                                           pos_dims, # position dimensions
                                                           spec_dims, #spectroscopy dimensions
                                                           )
            else:
                h5_raw = usid.hdf_utils.write_main_dataset(cur_chan, # parent HDF5 group
                                                           main_dset, # 2D array of raw data
                                                           'Raw_'+channel, # Name of main dset
                                                           channel, # Physical quantity
                                                           self.get_def_unit(channel), # Unit
                                                           pos_dims, # position dimensions
                                                           spec_dims, #spectroscopy dimensions
                                                           # Link Ancilliary dset to the first
                                                           h5_pos_inds=first_main_dset['Position_Indices'],
                                                           h5_pos_vals=first_main_dset['Position_Values'],
                                                           h5_spec_inds=first_main_dset['Spectroscopic_Indices'],
                                                           h5_spec_vals=first_main_dset['Spectroscopic_Values'],
                                                           )

        # Make Channels with IMAGES.
        # Position indices/values are the same of all other channels
        # Spectroscopic indices/valus are they are just one single dimension
        img_spec_dims = [usid.write_utils.Dimension('arb', 'a.u.', [1])]
        for index, image in enumerate(ARh5_file['Image'].keys()):
            main_dset = np.reshape(np.array(ARh5_file['Image'][image]), (-1,1), order='F')
            cur_chan = usid.hdf_utils.create_indexed_group(h5_meas_group, 'Channel')
            if index == 0:
                first_image_dset = cur_chan
                h5_raw = usid.hdf_utils.write_main_dataset(cur_chan, # parent HDF5 group
                                                           main_dset, # 2D array of image (shape: P*Q x 1)
                                                           'Img_'+image, # Name of main dset
                                                           image, # Physical quantity
                                                           self.get_def_unit(image), # Unit
                                                           pos_dims, # position dimensions
                                                           img_spec_dims, #spectroscopy dimensions
                                                           # Link Ancilliary dset to the first
                                                           h5_pos_inds=first_main_dset['Position_Indices'],
                                                           h5_pos_vals=first_main_dset['Position_Values'],
                                                           )
            else:
                h5_raw = usid.hdf_utils.write_main_dataset(cur_chan, # parent HDF5 group
                                                           main_dset, # 2D array of image (shape: P*Q x 1)
                                                           'Img_'+image, # Name of main dset
                                                           image, # Physical quantity
                                                           self.get_def_unit(image), # Unit
                                                           pos_dims, # position dimensions
                                                           img_spec_dims, #spectroscopy dimensions
                                                           # Link Ancilliary dset to the first
                                                           h5_pos_inds=first_main_dset['Position_Indices'],
                                                           h5_pos_vals=first_main_dset['Position_Values'],
                                                           h5_spec_inds=first_image_dset['Spectroscopic_Indices'],
                                                           h5_spec_vals=first_image_dset['Spectroscopic_Values'],
                                                           )

        # Create the new segments that will be stored as attribute
        new_segments = {}
        for seg, name in enumerate(self.segments_name):
            new_segments.update({name:self.segments[0,0,seg] - short_ext})
        usid.hdf_utils.write_simple_attrs(h5_meas_group, {'Segments':new_segments,
                                                          'Points_trimmed':points_trimmed,
                                                          'Notes':self.notes})
        usid.hdf_utils.write_simple_attrs(h5_file,
                                          {'translator':'ARhdf5',
                                           'instrument':'Asylum Research '+self.note_value('MicroscopeModel'),
                                           'AR sftware version':self.note_value('Version')})

        if self.debug:
            print(usid.hdf_utils.print_tree(h5_file))
            print('\n')
            for key, val in usid.hdf_utils.get_attributes(h5_meas_group).items():
                if key != 'Notes':
                    print('{} : {}'.format(key, val))
                else:
                    print('{} : {}'.format(key, 'notes string too long to be written here.'))

        #Clean up
        ARh5_file.close()        
        h5_file.close()
        self.translated = True
        return h5_path

    def note_value(self, name):
        '''
        Get the value of a single note entry with name "name"
        
        Parameters
        ----------------
        name : String / unicode
            Name of the parameter to get teh value

        Returns
        ----------------
        value : String / unicode
            Value of the Note entry requested.
        '''
        try:
            match = re.search(r"^" + name + ":\s+(.+$)", self.notes, re.M)
            if not match:
                raise
        except:
            match = re.search(r"^" + name + ":+(.+$)", self.notes, re.M)
        if (match):
            matched = match.groups()
            if len(matched) == 1:
                return match.groups()[0]
            else:
                # We do not expect to enter here
                print('WARNING! Multiple value matched! \n Only the first is returned')
                return match.groups()[0]
        else:
            raise NameError('Note entry with name "{}" not found'.format(name))

    def get_def_unit(self, chan_name):
        """
        Retrive the default unit from the channel name

        Parameters
        ----------
        chan_name : string
            Name of the channel to get the unit

        Returns
        -------
        default_unit : string
            Default unit of that channel
        """

        # Check if chan_name is string
        if not isinstance(chan_name, basestring):
            raise TypeError('The channel name must be of type string')

        # Find the default unit        
        if chan_name.startswith('Phas'):
            default_unit = 'deg'
        elif chan_name.startswith('Curr'):
            default_unit = 'A'
        elif chan_name.startswith('Freq'):
            default_unit = 'Hz'
        elif chan_name.startswith('Bias'):
            default_unit = 'V'
        elif (chan_name.startswith('Amp') or
              chan_name.startswith('Raw') or
              chan_name.startswith('ZSnsr') or
              chan_name.startswith('Defl') or
              chan_name.startswith('MapHeight')):
            default_unit = 'm'
        elif (chan_name.startswith('Seconds') or
              chan_name == 'TriggerTime'):
            default_unit = 's'
        elif chan_name.startswith('HeaterTemperature'):
            default_unit = 'Celsius'
        elif chan_name == 'MapAdhesion':
            default_unit = 'N/m^2'
        elif chan_name == 'HeaterHumidity':
            default_unit = 'g/m^3'
        elif chan_name.endswith('LVDT'):
            # This should be the laser virtual deflection
            default_unit = 'm'
        else:
            if self.debug:
                print('Unknown unit for channel: {}'.format(chan_name))
                print('Unit set to "unknown"')
            default_unit = 'unknown'

        return default_unit
