# Image windowing class

import numpy as np
import sidpy
from sidpy.base.num_utils import build_ind_val_matrices
from scipy.signal.windows import hamming, blackman
from skimage.transform import rescale
import dask

class ImageWindowing:
    """
    This class will generate windows from sidpy dataset objects. At present only 2D windowing is allowed.
    """
    def __init__(self, parms_dict, verbose = False):
        '''Sliding Window Class.

        Parameters
        ----------
        - parms_dict : dictionary
            Dictionary with parameters of the windowing process, see below.

            Keys:
            - 'window_size_x' (integer) (required): size of the window across the x-axis
            - 'window_size_y' (integer) (required): size of the window across the y-axis
            - 'window_step_x' (integer) (required): step size of the window across the x-axis. Must divide into the image size in the x axis
            - 'window_step_y' (integer) (required): step size of the window across the y-axis. Must divide into the image size in the y axis
            - 'mode' (string) (Optional, default is 'image'): One of 'image' or 'fft' which defines the processing to be performed for each window.
                The choice of 'fft' will perform 2D fast Fourier transforms on each image whereas 'image' will not perform any operation on the window
            - 'fft_mode' (string) (Optional, default is 'abs'): If mode is 'fft', choose whether to look at amplitude or phase. Options are 'abs', 'phase' and 'complex'.
            - 'interpol_factor' (float) (Optional, default is 1.0): Interpolation factor for windows to increase or decrease size of the windows.
            - 'zoom_factor' (integer or list of ints) (Optional, default is 1): Zoom the window by this factor, typically done for 'fft' mode to observe higher frequencies clearly
                            If passing a list of ints, this will determine the degree of cropping per axis
            - 'filter' (string) (Optional, default is None): Filtering to use for the image window. Options are 'blackman', 'hamming'.
            The filter is applied to each window before 'mode'.
        - verbose : (Optional) Boolean
            Verbose flag. Default is False.

        Returns
        --------
        Instance of ImageWindowing object setup with parameters defined by the parms_dict above.
       '''

        self.window_step_x = parms_dict['window_step_x']
        self.window_step_y = parms_dict['window_step_y']
        self.window_size_x = parms_dict['window_size_x']
        self.window_size_y = parms_dict['window_size_y']
        self.fft_mode = 'abs'
        self.verbose = verbose

        if 'mode' in parms_dict.keys():
            if parms_dict['mode'] not in ['image', 'fft']:
                raise ValueError("Parameters dictionary field 'mode' must be one of 'image' or 'fft'."
                                 "Try again with one of these two options.")
            else:
                self.mode = parms_dict['mode']
        else:
            self.mode = 'image'
            parms_dict['mode'] = 'image'

        if 'interpol_factor' in parms_dict.keys(): self.interpol_factor = parms_dict['interpol_factor']
        else:
            self.interpol_factor = 1
            parms_dict['interpol_factor'] = 1

        if 'zoom_factor' in parms_dict.keys(): self.zoom_factor = parms_dict['zoom_factor']
        else:
            self.zoom_factor = 1
            parms_dict['zoom_factor'] = 1

        # Based on the zoom and interpolation factors we need to figure out the final size of the window
        self.window_size_final_x, self.window_size_final_y = self._get_window_size()
        #Setup the filter for the window
        if 'filter' in parms_dict.keys():
            if parms_dict['filter'] not in ['blackman', 'hamming']:
                raise ValueError("Parameter 'filter' must be one of 'hamming', 'blackman'")
            else:
                self.filter = parms_dict['filter']
                if self.filter == 'hamming':
                    filter_x = hamming(self.window_size_final_x)
                    filter_y = hamming(self.window_size_final_y)
                    self.filter_mat = np.sqrt(np.outer(filter_x, filter_y))
                elif self.filter == 'blackman':
                    filter_x = blackman(self.window_size_final_x)
                    filter_y = blackman(self.window_size_final_y)
                    self.filter_mat = np.sqrt(np.outer(filter_x, filter_y))
        else:
            self.filter = 'None'

        if self.mode=='fft':
            #load FFT options
            if 'fft_mode' in parms_dict.keys():
                if parms_dict['fft_mode'] not in ['abs', 'phase', 'complex']:
                    raise ValueError("Parameter 'fft_mode' must be \
                    one of 'abs', 'phase' or 'complex' ")
                else:
                    self.fft_mode = parms_dict['fft_mode']
            else:
                self.fft_mode = 'abs' #default to absolute value in case fft mode is not provided
                parms_dict['fft_mode'] = 'abs'
        if self.verbose:
            print('ImageWindowing Object created with parameters {}'.format(parms_dict))

        self.window_parms = parms_dict
        self.window_dataset = None
        return

    def _get_window_size(self):
        '''
        Computes window size based on zoom and interpolation factors
        '''

        image_test = np.random.uniform(size=(self.window_size_x, self.window_size_y))
        image_zoomed = self.zoom(image_test, self.zoom_factor)

        #interpolate it
        zoomed_interpolated = rescale(image_zoomed, self.interpol_factor)
        return zoomed_interpolated.shape[0],zoomed_interpolated.shape[1]
    
    def do_PCA_window_cleaning(self, num_comps = None):
        """
        This function performs PCA cleaning
        Inputs: 
            - num_comps: (int) (Default = None). Number of components to keep in reconstruction. 
            By default, num_comps is 0.5*(window_size_x * window_size_y)
        """
        #Here we assume that the windowing has been done

        if self.window_dataset is None:
            raise ValueError("Windowing has not been done. Please perform windowing first before calling this function")

        assert self.window_size_x == self.window_size_final_x, "Cannot use zoom and interpolation for PCA image cleaning, rerun without these"
        assert self.window_size_y == self.window_size_final_y, "Cannot use zoom and interpolation for PCA image cleaning, rerun without these"
        
        windows_2d = self.window_dataset.fold(method='spaspec')
        u, s, vh = np.linalg.svd(np.array(windows_2d), full_matrices=False )
        
        if num_comps is None:
            num_comps = len(s)//2 #choose half the components as default
        
        s[num_comps:] = 0

        recon = np.dot(u * s, vh)
        recon_4d = recon.reshape(self.window_dataset.shape)

        recon_image = np.zeros(self.image_shape)
        window_size = [self.window_size_final_x, self.window_size_final_y]
        m=0
        for xind in range(recon_4d.shape[0]):
            for yind in range(recon_4d.shape[1]):
                cur_slice = recon_4d[xind,yind,:,:]
                pos =self.pos_vec[m]
                start_stop = [slice(x, x + y, 1) for x, y in zip(pos, window_size)]
                recon_image[tuple(start_stop)] = cur_slice
                m+=1

        self.recon_image = recon_image
        
        #TODO: Need to return as a sidpy dataset

        return recon_image

    def MakeWindows(self, dataset, dim_slice=None):
        '''
            Image should be a sidpy dataset object
            We will take the image to be the first two spatial dimensions,
            unless dimensions are specified

            Inputs:
                - dataset (sidpy.Dataset object of the image to be windowed)
                - dim_slice (List) (Optional). list of integers of the slices over which the
                image windowing should take place. This should be of length number of dimensions of
                the dataset minus two.

            Returns:
                - windowed_dataset (sidpy.Dataset) object with windows created as per
                the parameters passed to the ImageWindowing class.

            '''

        # This is the windowing function. Will generate the windows (but not the FFT)
        num_dimensions = dataset.ndim
        self.dataset = dataset
        if dim_slice is None:
            if num_dimensions > 2:
                raise ValueError('You have specified windowing on a sidpy dataset '
                                 'with more than 2 dimensions without specifying slices')
            else:
                image_source = dataset[:]
                image_dims = [0,1]
        elif dim_slice is not None:
            """Get all spatial dimensions"""
            image_dims = []
            for dim, axis in dataset._axes.items():
                if axis.dimension_type == sidpy.DimensionType.SPATIAL:
                    image_dims.append(dim)
            all_dims = np.arange(0, num_dimensions)
            slice_list = []
            for k in range(num_dimensions):
                if k in image_dims:
                    slice_list.append(slice(None, dataset.shape[k], 1))
                else:
                    slice_list.append(dim_slice)
            image_source = dataset[tuple(slice_list)]

        self.image_shape = image_source.shape

        if self.verbose:
            print('Full image shape is {}'.format(self.image_shape))

        window_step = [self.window_step_x, self.window_step_y]
        window_size = [self.window_size_x, self.window_size_y]
        window_size_final = [self.window_size_final_x, self.window_size_final_y]
        #division_factor_x = self.window_size_x - self.window_step_x
        #division_factor_y = self.window_size_y - self.window_step_y
        if self.window_size_x == self.window_step_x: division_factor_x = self.window_size_x
        if self.window_size_y == self.window_step_y: division_factor_y = self.window_size_y
        
        #assert np.mod(self.image_shape[0] - self.window_size_x, self.window_step_x) ==0, "Image shape along y is {} but window size is {}, window step is ({}) are not divisible \
        #without remainder, change your window size or window step".format(self.image_shape[0], self.window_size_x, self.window_step_x)
        #assert np.mod(self.image_shape[1] - self.window_size_y, self.window_step_y) ==0, "Image shape along x is {} but window size is {}, and window step is ({}) are not divisible \
        #without remainder, change your window size or window step".format(self.image_shape[1], self.window_size_y, self.window_step_y)

        dim_vec = []
        for i in range(2):
            dim_vec.append(np.arange(0, self.image_shape[i] - window_size[i], window_step[i]))
            dim_vec[i] = np.append(dim_vec[i], self.image_shape[i] - window_size[i])

        if self.verbose:
            print("dim vec is {}".format(dim_vec))

        _, pos_vec = build_ind_val_matrices(dim_vec)
        if self.verbose:
            print("Pos vec is {}".format(pos_vec))

        pca_mat = np.zeros(shape=(pos_vec.shape[0], np.prod(window_size_final)), dtype=np.complex64)
        pos_vec = np.int32(pos_vec)
        self.pos_vec = pos_vec

        def make_windows_parallel(ind, pos):
            start_stop = [slice(x, x + y, 1) for x, y in zip(pos, window_size)]
            full_slice = image_source[tuple(start_stop)]
            full_slice = self._return_win_image_processed(full_slice)
            full_slice_flat = full_slice.flatten()
            return full_slice_flat

        window_results = []
        for ind, pos in enumerate(pos_vec):
            lazy_result = dask.delayed(make_windows_parallel)(ind, pos)
            window_results.append(lazy_result)

        pca_mat = dask.compute(*window_results)
        pca_mat = np.array(pca_mat) #it comes out as a tuple, make it array
        
        self.pos_vec = pos_vec

        # Get the positions and make them dimensions
        new_y_vals = np.linspace(dataset._axes[image_dims[0]].values.min(),
                                 dataset._axes[image_dims[0]].values.max(), len(np.unique(pos_vec[:, 0])))

        new_x_vals = np.linspace(dataset._axes[image_dims[1]].values.min(),
                                 dataset._axes[image_dims[1]].values.max(), len(np.unique(pos_vec[:, 1])))
        if self.verbose:
            print("position values x {} and y {}".format(new_y_vals, new_x_vals))
        
        self.pca_mat = pca_mat
        windows_reshaped = pca_mat.reshape(len(new_x_vals), len(new_y_vals),
                                           self.window_size_final_x, self.window_size_final_y)
        if self.verbose:
            print('Reshaped windows size is {}'.format(windows_reshaped.shape))

        # Make a sidpy dataset
        #if the data is complex, then convert it to absolute
        #this needs to be changed..depending on user preferences.
        if np.iscomplexobj(windows_reshaped):
            if self.fft_mode == 'abs':
                windows_reshaped = np.array(np.abs(windows_reshaped), dtype = np.float64)
            elif self.fft_mode == 'phase':
                windows_reshaped = np.array(np.angle(windows_reshaped), dtype=np.float64)

        data_set = sidpy.Dataset.from_array(windows_reshaped,
                                            name='Image_Windowed')

        # Set the data type
        data_set.data_type = 'Image_4d'

        # Add quantity and units
        data_set.units = dataset.units
        data_set.quantity = dataset.quantity

        # Add dimension info

        window_size_fraction_x = window_size[0]/self.image_shape[0]
        window_size_fraction_y = window_size[1] / self.image_shape[1]

        window_extent_x = (dataset._axes[image_dims[0]].values.max() -
                           dataset._axes[image_dims[0]].values.min())*window_size_fraction_x

        window_extent_y = (dataset._axes[image_dims[1]].values.max() -
                           dataset._axes[image_dims[1]].values.min()) * window_size_fraction_y
        window_units = dataset._axes[image_dims[0]].units
        if self.mode =='fft':
            #to check if this is correct
            z_dimx = np.linspace(0, 1.0/(window_extent_x / self.zoom_factor), data_set.shape[2])
            z_dimy = np.linspace(0, 1.0/(window_extent_y / self.zoom_factor), data_set.shape[3])
            window_units = dataset._axes[image_dims[0]].units + '^-1'
        else:
            z_dimx = np.linspace(0, window_extent_x/self.zoom_factor, data_set.shape[2])
            z_dimy = np.linspace(0, window_extent_y/self.zoom_factor, data_set.shape[3])

        data_set.set_dimension(0, sidpy.Dimension(new_x_vals,
                                                  name=dataset._axes[image_dims[0]].name,
                                                  units=dataset._axes[image_dims[0]].units,
                                                  quantity=dataset._axes[image_dims[0]].quantity,
                                                  dimension_type='spatial'))

        data_set.set_dimension(1, sidpy.Dimension(new_y_vals,
                                                  name=dataset._axes[image_dims[1]].name,
                                                  units=dataset._axes[image_dims[1]].units,
                                                  quantity=dataset._axes[image_dims[1]].quantity,
                                                  dimension_type='spatial'))

        data_set.set_dimension(2, sidpy.Dimension(z_dimx,
                                                  name='WindowX',
                                                  units=window_units, quantity='kx',
                                                  dimension_type='spectral'))

        data_set.set_dimension(3, sidpy.Dimension(z_dimy,
                                                  name='WindowY',
                                                  units=window_units, quantity='ky',
                                                  dimension_type='spectral'))

        # append metadata
        data_set.metadata = self._merge_dictionaries(dataset.metadata, self.window_parms)
        self.window_dataset = data_set
        return data_set

    def _return_win_image_processed(self, img_window):
        #Real image slice, returns it back with image processed


        if self.mode == 'fft': # Apply FFT if needed
            img_window = np.array(img_window)
            img_window = np.fft.fftshift(np.fft.fft2(img_window))
            if self.fft_mode == 'amp':
                img_window = np.abs(img_window,)
            elif self.fft_mode == 'phase':
                img_window = np.angle(img_window)
            elif self.fft_mode == 'complex':
                img_window = np.array(img_window, dtype = np.complex64)

        #Zoom and interpolate if needed
        if self.zoom_factor == 1 and self.interpol_factor == 1:
            return img_window
        else:
            img_window = self.zoom(img_window, self.zoom_factor)  # Zoom it
            img_window = self.rescale_win(img_window, self.interpol_factor)  # Rescale

        if self.filter != 'None':
            img_window *= self.filter_mat  # Apply filter

        return img_window

    def _merge_dictionaries(self, dict1, dict2):
        #given two dictionaries, merge them into one
        merged_dict = {**dict1, **dict2}
        return merged_dict

    def zoom(self, img_window, zoom_factor):
        #Zooms by the zoom factor
        if zoom_factor==1:
            return img_window
        else:
            if type(zoom_factor) is int:
                zoom_factor = [zoom_factor, zoom_factor]

        #Find the midpoint
        img_x_mid = img_window.shape[0]//2
        img_y_mid = img_window.shape[1]//2
        zoom_x_size = (img_window.shape[0] / zoom_factor[0])/2
        zoom_y_size = (img_window.shape[1] / zoom_factor[1])/2

        img_window = img_window[int(img_x_mid - zoom_x_size) : int(img_x_mid + zoom_x_size),
                     int(img_y_mid - zoom_y_size ): int(img_y_mid + zoom_y_size)]

        return img_window

    def rescale_win(self, img_window, interpol_factor):
        if self.fft_mode !='complex':
            img_window = np.array(img_window, dtype = np.float32)
            complex_rescaled_image = rescale(img_window, interpol_factor)
        else:
            real_img = np.real(img_window)
            imag_img = np.imag(img_window)
            real_img_scaled = rescale(real_img, interpol_factor)
            imag_img_scaled = rescale(imag_img, interpol_factor)
            complex_rescaled_image = real_img_scaled + 1j*imag_img_scaled
            
        return complex_rescaled_image

    #
    '''
    def clean_image(self, n_comps = 4):
        self.mode = 'image'
        self.interpol_factor = 1
        self.zoom_factor = 1
        self.window_size_final_x = self.window_size_x
        self.window_size_final_y = self.window_size_y
        new_windows = self.MakeWindows(self.dataset)
        
        from ..learn.ml import MatrixFactor
        svd_results = MatrixFactor(new_windows, method='svd', n_components = n_comps)

        return svd_results'''




