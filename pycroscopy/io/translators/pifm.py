import os
import numpy as np
import h5py

from sidpy.sid import Translator
from sidpy.hdf.hdf_utils import write_simple_attrs

from pyUSID.io.write_utils import Dimension, build_ind_val_matrices
from pyUSID.io.hdf_utils import write_main_dataset, create_indexed_group, get_all_main
from pyUSID import USIDataset


class PiFMTranslator(Translator):
    """
    Class that writes images, spectrograms, point spectra and associated ancillary data sets to h5 file in pyUSID data
    structure.
    """

    def translate(self, path, append_path='', grp_name='Measurement', overwrite=False):
        """
        Parameters
        ----------
        file_path : String / unicode
            Absolute path of the .ibw file
        verbose : Boolean (Optional)
            Whether or not to show  print statements for debugging
        append_path : string (Optional)
            h5_file to add these data to, must be a path to the h5_file on disk
        overwrite : bool (optional, default=False)
            If True, will overwrite an existing .h5 file of the same name
        parm_encoding : str, optional
            Codec to be used to decode the bytestrings into Python strings if needed.
            Default 'utf-8'

        Returns
        -------
        h5_path : String / unicode
            Absolute path of the .h5 file
        """
        self.get_path(path)
        self.read_anfatec_params()
        self.read_file_desc()
        self.read_spectrograms()
        self.read_imgs()
        self.read_spectra()
        self.make_pos_vals_inds_dims()
        self.create_hdf5_file(append_path, grp_name, overwrite)
        self.write_spectrograms()
        self.write_images()
        self.write_spectra()
        self.write_ps_spectra()
        
        return self.h5_f


    def get_path(self, path):
        """writes full path, directory, and file name as attributes to class"""
        # get paths/get params dictionary, img/spectrogram/spectrum descriptions
        
        self.path = path
        full_path = os.path.realpath(self.path)
        directory = os.path.dirname(full_path)
        # file name
        basename = os.path.basename(self.path)
        self.full_path = full_path
        self.directory = directory
        self.basename = basename

    #these dictionary parameters will be written to hdf5 file under measurement attributes
    def read_anfatec_params(self):
        """reads the scan parameters and writes them to a dictionary"""
        params_dictionary = {}
        params = True
        with open(self.path, 'r', encoding="ISO-8859-1") as f:
            for line in f:
                if params:
                    sline = [val.strip() for val in line.split(':')]
                    if len(sline) == 2 and sline[0][0] != ';':
                        params_dictionary[sline[0]] = sline[1]
                    #in ANFATEC parameter files, all attributes are written before file references.
                    if sline[0].startswith('FileDesc'):
                        params = False
            f.close()
        self.params_dictionary = params_dictionary
        self.x_len, self.y_len = int(params_dictionary['xPixel']), int(params_dictionary['yPixel'])

    def read_file_desc(self):
        """reads spectrogram, image, and spectra file descriptions and stores all to dictionary where
        the key:value pairs are filename:[all descriptors]"""
        spectrogram_desc = {}
        img_desc = {}
        spectrum_desc = {}
        pspectrum_desc = {}
        with open(self.path,'r', encoding="ISO-8859-1") as f:
            ## can be made more concise...by incorporating conditons with loop control
            lines = f.readlines()
            for index, line in enumerate(lines):
                sline = [val.strip() for val in line.split(':')]
                #if true, then file describes image.
                if sline[0].startswith('FileDescBegin'):
                    no_descriptors = 5
                    file_desc = []
                    for i in range(no_descriptors):
                        line_desc = [val.strip() for val in lines[index+i+1].split(':')]
                        file_desc.append(line_desc[1])
                    #img_desc['filename'] = caption, scale, physical unit, offset
                    img_desc[file_desc[0]] = file_desc[1:]
                #if true, file describes spectrogram (ie hyperspectral image)
                if sline[0].startswith('FileDesc2Begin'):
                    no_descriptors = 10
                    file_desc = []
                    for i  in range(no_descriptors):
                        line_desc = [val.strip() for val in lines[index+i+1].split(':')]
                        file_desc.append(line_desc[1])
                    #caption, bytes perpixel, scale, physical unit, offset, offset, datatype, bytes per reading
                    #filename wavelengths, phys units wavelengths.
                    spectrogram_desc[file_desc[0]] = file_desc[1:]
                if sline[0].startswith('AFMSpectrumDescBegin'):

                    file_desc = []
                    line_desc = [val.strip() for val in lines[index+1].split(':')][1]
                    if 'powerspectrum' in line_desc:
                        no_descriptors = 2
                        for i in range(no_descriptors):
                            line_desc = [val.strip() for val in lines[index+i+1].split(':')]
                            file_desc.append(line_desc[1])
                        #file name, position x, position y
                        pspectrum_desc[file_desc[0]] = file_desc[1:]
                    else:
                        no_descriptors = 7
                        for i in range(no_descriptors):
                            line_desc = [val.strip() for val in lines[index+i+1].split(':')]
                            file_desc.append(line_desc[1])
                        #file name, position x, position y
                        spectrum_desc[file_desc[0]] = file_desc[1:]
            f.close()
        self.img_desc = img_desc
        self.spectrogram_desc = spectrogram_desc
        self.spectrum_desc = spectrum_desc
        self.pspectrum_desc = pspectrum_desc

    def read_spectrograms(self):
        """reads spectrograms, associated spectral values, and saves them in two dictionaries"""
        spectrograms = {}
        spectrogram_spec_vals = {}
        for file_name, descriptors in self.spectrogram_desc.items():
            #load and save spectroscopic values
            spec_vals_i = np.loadtxt(os.path.join(self.directory, file_name.strip('.int') + 'Wavelengths.txt'))
            #if true, data is acquired with polarizer, with an attenuation data column
            if np.array(spec_vals_i).ndim == 2:
                spectrogram_spec_vals[file_name] = spec_vals_i[:, 0]
                attenuation = {}
                attenuation[file_name] = spec_vals_i[:, 1]
                self.attenuation = attenuation
            else:
                spectrogram_spec_vals[file_name] = spec_vals_i
            #load and save spectrograms
            spectrogram_i = np.fromfile(os.path.join(self.directory, file_name), dtype='i4')
            spectrograms[file_name] = np.zeros((self.x_len, self.y_len, len(spec_vals_i)))
            for y, line in enumerate(np.split(spectrogram_i, self.y_len)):
                for x, pt_spectrum in enumerate(np.split(line, self.x_len)):
                    spectrograms[file_name][x, y, :] = pt_spectrum * float(descriptors[2])
        self.spectrograms = spectrograms
        self.spectrogram_spec_vals = spectrogram_spec_vals

    def read_imgs(self):
        """reads images and saves to dictionary"""
        imgs = {}
        for file_name, descriptors in self.img_desc.items():
            img_i = np.fromfile(os.path.join(self.directory, file_name), dtype='i4')
            imgs[file_name] = np.zeros((self.x_len, self.y_len))
            for y, line in enumerate(np.split(img_i, self.y_len)):
                for x, pixel in enumerate(np.split(line, self.x_len)):
                    imgs[file_name][x, y] = pixel * float(descriptors[1])
        self.imgs = imgs

    def read_spectra(self):
        """reads all point spectra and saves to dictionary"""
        spectra = {}
        spectra_spec_vals = {}
        spectra_x_y_dim_name = {}
        for file_name, descriptors in self.spectrum_desc.items():
            spectrum_f = np.loadtxt(os.path.join(self.directory, file_name), skiprows=1)
            spectra_spec_vals[file_name] = spectrum_f[:, 0]
            spectra[file_name] = spectrum_f[:,1]
            with open(os.path.join(self.directory, file_name)) as f:
                spectra_x_y_dim_name[file_name]  = f.readline().strip('\n').split('\t')
        for file_name, descriptors in self.pspectrum_desc.items():
            spectrum_f = np.loadtxt(os.path.join(self.directory, file_name), skiprows=1)
            spectra_spec_vals[file_name] = spectrum_f[:, 0]
            spectra[file_name] = spectrum_f[:,1]
            with open(os.path.join(self.directory, file_name)) as f:
                spectra_x_y_dim_name[file_name]  = f.readline().strip('\n').split('\t')
        self.spectra = spectra
        self.spectra_spec_vals = spectra_spec_vals
        self.spectra_x_y_dim_name = spectra_x_y_dim_name

    def make_pos_vals_inds_dims(self):
        x_range = float(self.params_dictionary['XScanRange'])
        y_range = float(self.params_dictionary['YScanRange'])
        x_center = float(self.params_dictionary['xCenter'])
        y_center = float(self.params_dictionary['yCenter'])

        x_start = x_center-(x_range/2); x_end = x_center+(x_range/2)
        y_start = y_center-(y_range/2); y_end = y_center+(y_range/2)

        dx = x_range/self.x_len
        dy = y_range/self.y_len
        #assumes y scan direction:down; scan angle: 0 deg
        y_linspace = -np.arange(y_start, y_end, step=dy)
        x_linspace = np.arange(x_start, x_end, step=dx)
        pos_ind, pos_val = build_ind_val_matrices(unit_values=(x_linspace, y_linspace), is_spectral=False)
        #Dimension uses ascii encoding, which can not encode
        # micron symbol, so we replace it, if present, with the letter u.
        pos_dims = [Dimension('X', self.params_dictionary['XPhysUnit'].replace('\xb5', 'u'), self.x_len),
                    Dimension('Y', self.params_dictionary['YPhysUnit'].replace('\xb5', 'u'), self.y_len)]
        self.pos_ind, self.pos_val, self.pos_dims = pos_ind, pos_val, pos_dims

    def create_hdf5_file(self, append_path='', grp_name='Measurement', overwrite=False):
        if not append_path:
            h5_path = os.path.join(self.directory, self.basename.replace('.txt', '.h5'))
            if os.path.exists(h5_path):
                if not overwrite:
                    raise FileExistsError('This file already exists). Set attribute overwrite to True')
                else:
                    print('Overwriting file', h5_path)
                    #os.remove(h5_path)
                    
            self.h5_f = h5py.File(h5_path, mode='w')

        else:
            if not os.path.exists(append_path):
                raise Exception('File does not exist. Check pathname.')
            self.h5_f = h5py.File(append_path, mode='r+')

        self.h5_meas_grp = create_indexed_group(self.h5_f, grp_name)
        
        write_simple_attrs(self.h5_meas_grp, self.params_dictionary)

        return
    
    def write_spectrograms(self):
        if bool(self.spectrogram_desc):
            for spectrogram_f, descriptors in self.spectrogram_desc.items():
                channel_i = create_indexed_group(self.h5_meas_grp, 'Channel_')
                spec_vals_i = self.spectrogram_spec_vals[spectrogram_f]
                spectrogram_spec_dims = Dimension('Wavelength', descriptors[8], spec_vals_i)
                h5_raw = write_main_dataset(channel_i,  # parent HDF5 group
                                                           (self.x_len *
                                                            self.y_len, len(spec_vals_i)),  # shape of Main dataset
                                                           'Raw_Data',  # Name of main dataset
                                                           'Spectrogram',  # Physical quantity contained in Main dataset
                                                           descriptors[3],  # Units for the physical quantity
                                                           self.pos_dims,  # Position dimensions
                                                           spectrogram_spec_dims,  # Spectroscopic dimensions
                                                           dtype=np.float32,  # data type / precision
                                                           main_dset_attrs={'Caption': descriptors[0],
                                                                            'Bytes_Per_Pixel': descriptors[1],
                                                                            'Scale': descriptors[2],
                                                                            'Physical_Units': descriptors[3],
                                                                            'Offset': descriptors[4],
                                                                            'Datatype': descriptors[5],
                                                                            'Bytes_Per_Reading': descriptors[6],
                                                                            'Wavelength_File': descriptors[7],
                                                                            'Wavelength_Units': descriptors[8]})
                h5_raw.h5_pos_vals[:, :] = self.pos_val
                h5_raw[:, :] = self.spectrograms[spectrogram_f].reshape(h5_raw.shape)

    def write_images(self):
        if bool(self.img_desc):
            for img_f, descriptors in self.img_desc.items():
                #check for existing spectrogram or image and link position/spec inds/vals
                #at most two channels worth of need to be checked (Fwd and Bwd)
                try:
                    str_main = str(get_all_main(self.h5_f['Measurement_000/Channel_000']))
                    i_beg = str_main.find('located at: \n\t') + 14
                    i_end = str_main.find('\nData contains') - 1
                    data_loc = str_main[i_beg:i_end]
                    channel_data = USIDataset(self.h5_f[data_loc])
                    h5_pos_inds = channel_data.h5_pos_inds
                    h5_pos_vals = channel_data.h5_pos_vals
                    pos_dims = None
                    write_pos_vals = False
                    if channel_data.spec_dim_sizes[0] == 1:
                        h5_spec_inds = channel_data.h5_spec_inds
                        h5_spec_vals = channel_data.h5_spec_vals
                        spec_dims = None
                    #if channel 000 is spectrogram, check next dataset
                    elif channel_data.spec_dim_sizes[0] !=1:
                        str_main = str(get_all_main(self.h5_f['Measurement_000/Channel_001']))
                        i_beg = str_main.find('located at: \n\t') + 14
                        i_end = str_main.find('\nData contains') - 1
                        data_loc = str_main[i_beg:i_end]
                        channel_data = USIDataset(self.h5_f[data_loc])
                        #channel data is an image, & we link their spec inds/vals
                        if channel_data.spec_dim_sizes[0] == 1:
                            h5_spec_inds = channel_data.h5_spec_inds
                            h5_spec_vals = channel_data.h5_spec_vals
                            spec_dims = None
                        else: # If a forward/bwd spectrogram exist
                            h5_spec_inds = None
                            h5_spec_vals = None
                            spec_dims = Dimension('arb', 'a.u', 1)

                #in case where channel does not exist, we make new spec/pos inds/vals
                except KeyError:
                    #pos dims
                    h5_pos_inds = None
                    h5_pos_vals = None
                    pos_dims = self.pos_dims
                    write_pos_vals = True
                    #spec dims
                    h5_spec_inds = None
                    h5_spec_vals = None
                    spec_dims = Dimension('arb', 'a.u', 1)

                channel_i = create_indexed_group(self.h5_meas_grp,'Channel_')
                h5_raw = write_main_dataset(channel_i, #parent HDF5 group
                                                               (self.x_len * self.y_len, 1),  # shape of Main dataset
                                                               'Raw_' + descriptors[0].replace('-', '_'),
                                                               # Name of main dataset
                                                               descriptors[0],
                                                               # Physical quantity contained in Main dataset
                                                               descriptors[2],  # Units for the physical quantity
                                                               h5_pos_inds=h5_pos_inds,
                                                               h5_pos_vals=h5_pos_vals,
                                                               # Position dimensions
                                                               pos_dims=pos_dims,
                                                               # Spectroscopic dimensions
                                                               h5_spec_inds=h5_spec_inds,
                                                               h5_spec_vals=h5_spec_vals,
                                                               spec_dims=spec_dims,
                                                               dtype=np.float32,  # data type / precision
                                                               main_dset_attrs={'Caption': descriptors[0],
                                                                                'Scale': descriptors[1],
                                                                                'Physical_Units': descriptors[2],
                                                                                'Offset': descriptors[3]})
                h5_raw[:, :] = self.imgs[img_f].reshape(h5_raw.shape)
                if write_pos_vals:
                    h5_raw.h5_pos_vals[:, :] = self.pos_val

    def write_spectra(self):
        if bool(self.spectrum_desc):
            for spec_f, descriptors in self.spectrum_desc.items():
                #create new measurement group for ea spectrum
                self.h5_meas_grp = create_indexed_group(self.h5_f, 'Measurement_')
                x_name = self.spectra_x_y_dim_name[spec_f][0].split(' ')[0]
                x_unit = self.spectra_x_y_dim_name[spec_f][0].split(' ')[1]
                y_name = self.spectra_x_y_dim_name[spec_f][1].split(' ')[0]
                y_unit = self.spectra_x_y_dim_name[spec_f][1].split(' ')[1]
                spec_i_spec_dims = Dimension(x_name, x_unit, self.spectra_spec_vals[spec_f])
                spec_i_pos_dims = [Dimension('X',
                                                              self.params_dictionary['XPhysUnit'].replace('\xb5','u'),
                                                              np.array([float(descriptors[1])])),
                                   Dimension('Y',
                                                              self.params_dictionary['YPhysUnit'].replace('\xb5','u'),
                                                              np.array([float(descriptors[1])]))]
                #write data to a channel in the measurement group
                spec_i_ch = create_indexed_group(self.h5_meas_grp, 'Spectrum_')
                h5_raw = write_main_dataset(spec_i_ch,  # parent HDF5 group
                                                           (1, len(self.spectra_spec_vals[spec_f])),  # shape of Main dataset
                                                           'Raw_Spectrum',
                                                           # Name of main dataset
                                                           y_name,
                                                           # Physical quantity contained in Main dataset
                                                           y_unit,  # Units for the physical quantity
                                                           # Position dimensions
                                                           pos_dims=spec_i_pos_dims, spec_dims=spec_i_spec_dims,
                                                           # Spectroscopic dimensions
                                                           dtype=np.float32,  # data type / precision
                                                           main_dset_attrs={'XLoc': descriptors[0],
                                                                            'YLoc': descriptors[1]})
                h5_raw[:, :] = self.spectra[spec_f].reshape(h5_raw.shape)

    def write_ps_spectra(self):
        if bool(self.pspectrum_desc):
            for spec_f, descriptors in self.pspectrum_desc.items():

                # create new measurement group for ea spectrum
                self.h5_meas_grp = create_indexed_group(self.h5_f, 'Measurement_')
                x_name = self.spectra_x_y_dim_name[spec_f][0].split(' ')[0]
                x_unit = self.spectra_x_y_dim_name[spec_f][0].split(' ')[1]
                y_name = self.spectra_x_y_dim_name[spec_f][1].split(' ')[0]
                y_unit = self.spectra_x_y_dim_name[spec_f][1].split(' ')[1]
                spec_i_spec_dims = Dimension(x_name, x_unit, self.spectra_spec_vals[spec_f])
                spec_i_pos_dims = [Dimension('X',
                                                              self.params_dictionary['XPhysUnit'].replace(
                                                                  '\xb5', 'u'),
                                                              np.array([0])),
                                   Dimension('Y',
                                                              self.params_dictionary['YPhysUnit'].replace(
                                                                  '\xb5', 'u'),
                                                              np.array([0]))]
                # write data to a channel in the measurement group
                spec_i_ch = create_indexed_group(self.h5_meas_grp, 'PowerSpectrum_')
                h5_raw = write_main_dataset(spec_i_ch,  # parent HDF5 group
                                                           (1, len(self.spectra_spec_vals[spec_f])),
                                                           # shape of Main dataset
                                                           'Raw_Spectrum',
                                                           # Name of main dataset
                                                           y_name,
                                                           # Physical quantity contained in Main dataset
                                                           y_unit,  # Units for the physical quantity
                                                           # Position dimensions
                                                           pos_dims=spec_i_pos_dims, spec_dims=spec_i_spec_dims,
                                                           # Spectroscopic dimensions
                                                           dtype=np.float32,  # data type / precision
                                                           main_dset_attrs={'XLoc': 0,
                                                                            'YLoc': 0})
                h5_raw[:, :] = self.spectra[spec_f].reshape(h5_raw.shape)

