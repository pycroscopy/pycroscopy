# Image windowing class

import numpy as np
import sidpy
from scipy import fftpack
from scipy.ndimage import zoom
from scipy.signal import hanning, blackman
from skimage.transform import rescale

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
            - 'window_step_x' (integer) (required): step size of the window across the x-axis. Sometimes referred to as 'strides'
            - 'window_step_y' (integer) (required): step size of the window across the y-axis. Sometimes referred to as 'strides'
            - 'mode' (string) (Optional, default is 'image'): One of 'image' or 'fft' which defines the processing to be performed for each window.
                The choice of 'fft' will perform 2D fast Fourier transforms on each image whereas 'image' will not perform any operation on the window
            - 'fft_mode' (string) (Optional, default is 'abs'): If mode is 'fft', choose whether to look at amplitude or phase. Options are 'abs', 'phase'.
            - 'interpol_factor' (float) (Optional, default is 1.0): Interpolation factor for windows to increase or decrease size of the windows.
            - 'zoom_factor' (float) (Optional, default is 1.0): Zoom the window by this factor, typically done for 'fft' mode to observe higher frequencies clearly
            - 'filter' (string) (Optional, default is None): Filtering to use for the image window. Options are 'blackman', 'hanning'.
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
            parms_dict['interpol_facor']=1

        if 'zoom_factor' in parms_dict.keys(): self.zoom_factor = parms_dict['zoom_factor']
        else:
            self.zoom_factor = 1
            parms_dict['zoom_facor'] = 1

        # Based on the zoom and interpolation factors we need to figure out the final size of the window
        self.window_size_final_x, self.window_size_final_y = self._get_window_size()
        #Setup the filter for the window
        self.filter = 'None'
        self.filter_mat = np.ones((self.window_size_final_x, self.window_size_final_y))

        if self.mode=='fft':
            #load FFT options
            if 'filter' in parms_dict.keys():
                if parms_dict['filter'] not in ['blackman', 'hanning']:
                    raise ValueError("Parameter 'filter' must be one of 'hanning', 'blackman'")
                else:
                    self.filter = parms_dict['filter']
                    if self.filter=='hanning':
                        filter_x = hanning(self.window_size_final_x)
                        filter_y = hanning(self.window_size_final_y)
                        self.filter_mat = np.sqrt(np.outer(filter_x,filter_y))
                    elif self.filter=='blackman':
                        filter_x = blackman(self.window_size_final_x)
                        filter_y = blackman(self.window_size_final_y)
                        self.filter_mat = np.sqrt(np.outer(filter_x,filter_y))
            if 'fft_mode' in parms_dict.keys():
                if parms_dict['fft_mode'] not in ['abs', 'phase']:
                    raise ValueError("Parameter 'fft_mode' must be one of 'abs', 'phase'")
                else:
                    self.fft_mode = parms_dict['fft_mode']
            else:
                self.fft_mode = 'abs' #default to absolute value in case fft mode is not provided
                parms_dict['fft_mode'] = 'abs'
        if self.verbose:
            print('ImageWindowing Object created with parameters {}'.format(parms_dict))

        self.window_parms = parms_dict

        return

    def _get_window_size(self):
        '''
        Computes window size based on zoom and interpolation factors
        '''

        image_test = np.random.uniform(size=(self.window_size_x, self.window_size_y))

        image_zoomed = zoom(image_test, self.zoom_factor)

        #interpolate it
        zoomed_interpolated = rescale(image_zoomed, self.interpol_factor)

        return zoomed_interpolated.shape[0],zoomed_interpolated.shape[1]

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

        image_shape = image_source.shape
        if self.verbose:
            print('Full image shape is {}'.format(image_shape))

        window_step = [self.window_step_x, self.window_step_y]
        window_size = [self.window_size_x, self.window_size_y]
        window_size_final = [self.window_size_final_x, self.window_size_final_y]

        dim_vec = []
        for i in range(2):
            dim_vec.append(np.arange(0, image_shape[i] - window_size[i], window_step[i]))
        print("dim vec is {}".format(dim_vec))

        _, pos_vec = self.build_ind_val_matrices(dim_vec)
        if self.verbose:
            print("Pos vec is {}".format(pos_vec))

        pca_mat = np.zeros(shape=(pos_vec.shape[0], np.prod(window_size_final)), dtype=np.complex64)
        pos_vec = np.int32(pos_vec)

        for ind, pos in enumerate(pos_vec):
            start_stop = [slice(x, x + y, 1) for x, y in zip(pos, window_size)]
            full_slice = image_source[tuple(start_stop)]
            full_slice = self._return_win_image_processed(full_slice)
            pca_mat[ind] = full_slice.flatten()

        self.pos_vec = pos_vec

        # Get the positions and make them dimensions
        new_x_vals = np.linspace(dataset._axes[image_dims[0]].values.min(),
                                 dataset._axes[image_dims[0]].values.max(), len(np.unique(pos_vec[:, 0])))

        new_y_vals = np.linspace(dataset._axes[image_dims[1]].values.min(),
                                 dataset._axes[image_dims[1]].values.max(), len(np.unique(pos_vec[:, 1])))
        if self.verbose:
            print("position values x {} and y {}".format(new_x_vals, new_y_vals))
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

        window_size_fraction_x = window_size[0]/image_shape[0]
        window_size_fraction_y = window_size[1] / image_shape[1]

        window_extent_x = (dataset._axes[image_dims[0]].values.max() -
                           dataset._axes[image_dims[0]].values.min())*window_size_fraction_x

        window_extent_y = (dataset._axes[image_dims[1]].values.max() -
                           dataset._axes[image_dims[1]].values.min()) * window_size_fraction_y

        if self.mode =='fft':
            #to check if this is correct
            z_dimx = np.linspace(0, 1.0/(window_extent_x / self.zoom_factor), data_set.shape[2])
            z_dimy = np.linspace(0, 1.0/(window_extent_y / self.zoom_factor), data_set.shape[3])
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
                                                  units='m', quantity='kx',
                                                  dimension_type='spectral'))

        data_set.set_dimension(3, sidpy.Dimension(z_dimy,
                                                  name='WindowY',
                                                  units='m', quantity='ky',
                                                  dimension_type='spectral'))

        # append metadata
        data_set.metadata = self._merge_dictionaries(dataset.metadata, self.window_parms)

        return data_set
    #TODO: After next release of sidpy, remove this method and use sidpy.base.num_utils copy
    def build_ind_val_matrices(self, unit_values):
        """
        Builds indices and values matrices using given unit values for each dimension.
        This function is originally from pyUSID.io
        Unit values must be arranged from fastest varying to slowest varying

        Parameters
        ----------
        unit_values : list / tuple
            Sequence of values vectors for each dimension

        Returns
        -------
        ind_mat : 2D numpy array
            Indices matrix
        val_mat : 2D numpy array
            Values matrix
        """
        if not isinstance(unit_values, (list, tuple)):
            raise TypeError('unit_values should be a list or tuple')
        if not np.all([np.array(x).ndim == 1 for x in unit_values]):
            raise ValueError('unit_values should only contain 1D array')
        lengths = [len(x) for x in unit_values]
        tile_size = [np.prod(lengths[x:]) for x in range(1, len(lengths))] + [1]
        rep_size = [1] + [np.prod(lengths[:x]) for x in range(1, len(lengths))]
        val_mat = np.zeros(shape=(len(lengths), np.prod(lengths)))
        ind_mat = np.zeros(shape=val_mat.shape, dtype=np.uint32)
        for ind, ts, rs, vec in zip(range(len(lengths)), tile_size, rep_size, unit_values):
            val_mat[ind] = np.tile(np.repeat(vec, rs), ts)
            ind_mat[ind] = np.tile(np.repeat(np.arange(len(vec)), rs), ts)

        val_mat = val_mat.T
        ind_mat = ind_mat.T
        return ind_mat, val_mat

    def _return_win_image_processed(self, img_window):
        #Real image slice, returns it back with image processed
        if self.zoom_factor==1 and self.interpol_factor==1 and self.filter == 'None':
            #simply skip this function if there is no zooming, interpolation to be done.
            return img_window
        else:
            img_window = zoom(img_window, self.zoom_factor) #Zoom it
            img_window = rescale(img_window, self.interpol_factor) #Rescale
            img_window *= self.filter_mat  # Apply filter
            if self.mode == 'fft': img_window = np.fft.fftshift(np.fft.fft2(img_window))
        return img_window

    def _merge_dictionaries(self, dict1, dict2):
        #given two dictionaries, merge them into one
        merged_dict = {**dict1, **dict2}
        return merged_dict



