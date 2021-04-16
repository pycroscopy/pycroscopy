# Image windowing class


import numpy as np
from scipy import fftpack
from scipy import ndimage
import sidpy

class ImageWindowing:

    def __init__(self, parms_dict):
        '''Sliding Window Class.
       This class will generate windows from 2D sidpy dataset objects
       '''

        self.window_step_x = parms_dict['window_step_x']
        self.window_step_y = parms_dict['window_step_y']
        self.window_size_x = parms_dict['window_size_x']
        self.window_size_y = parms_dict['window_size_y']

        #self.interpol_factor = parms_dict['interpolation_factor']
        #self.zoom_factor = parms_dict['zoom_factor']
        #self.hamming_filter = parms_dict['hamming_filter']
        return

    def MakeWindows(self, dataset, dim_slice = None):
        '''

        image should be a sidpy dataset object
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
            if num_dimensions>2:
                raise ValueError('You have specified windowing on a sidpy dataset '
                      'with more than 2 dimensions without specifying slices')
            else:
                image_source = dataset[:]
        elif dim_slice is not None:
            """Get all spatial dimensions"""
            image_dims = []
            for dim, axis in dataset._axes.items():
                if axis.dimension_type == sidpy.DimensionType.SPATIAL:
                    image_dims.append(dim)
            all_dims = np.arange(0, num_dimensions)
            image_source = dataset[image_dims]

        image_shape = image_source.shape

        window_step = [self.window_step_x, self.window_step_y]
        window_size = [self.window_size_x, self.window_size_y]

        dim_vec = []
        for i in range(num_dimensions):
            dim_vec.append(np.arange(0, image_shape[i] - window_size[i], window_step[i]))

        _, pos_vec = self.build_ind_val_matrices(dim_vec)
        pca_mat = np.zeros(shape=(pos_vec.shape[0], np.prod(window_size)), dtype=np.complex64)
        pos_vec = np.int32(pos_vec)

        for ind, pos in enumerate(pos_vec):
            start_stop = [slice(x, x + y, 1) for x, y in zip(pos, window_size)]
            full_slice = image[start_stop]
            pca_mat[ind] = full_slice.flatten()

        self.pos_vec = pos_vec

        return pca_mat.reshape(-1, self.window_size_x, self.window_size_y)

    def ApplyFilter(self, imgsrc, filter_type = 'Hamming'):
        # Applies a Hamming window to the input imgsec,
        # returns the window after filter applied.
        bw2d = np.outer(np.hamming(self.window_size_x), np.ones(self.window_size_y))
        bw2d = np.sqrt(bw2d * bw2d.T)
        imgsrc *= bw2d
        return imgsrc

    def zoom_interpol(self, FFT_image):

        # Accepts an image, returns zoomed image
        zoom_size = (FFT_image.shape[0] / self.zoom_factor) / 2

        if np.mod(FFT_image.shape[0] / self.zoom_factor, 2) == 0:
            F2_zoomed = FFT_image[int(self.window_size_x / 2 - zoom_size):int(self.window_size_x / 2 + zoom_size),
                        int(self.window_size_y / 2 - zoom_size):int(self.window_size_y / 2 + zoom_size)]
        else:
            F2_zoomed = FFT_image[int(self.window_size_x / 2 - zoom_size):int(self.window_size_x / 2 + 1 + zoom_size),
                        int(self.window_size_y / 2 - zoom_size):int(self.window_size_y / 2 + 1 + zoom_size)]

        zoomed_interpolated_image = ndimage.zoom(F2_zoomed, self.interpol_factor)
        self.window_size_final_x = zoomed_interpolated_image.shape[0]
        self.window_size_final_y = zoomed_interpolated_image.shape[1]
        return ndimage.zoom(F2_zoomed, self.interpol_factor)

    def Do_Sliding_FFT(self, windows):

        # Carries out the FFT on the windows
        FFT_mat4 = []

        for i in range(windows.shape[0]):
            img_window = windows[i, :, :]

            if self.hamming_filter:  # Apply filter if requested
                img_window_filtered = self.ApplyHamming(np.copy(img_window))
            else:
                img_window_filtered = (np.copy(img_window))

            # Take the fourier transform of the image.
            F1 = fftpack.fft2((img_window_filtered))

            # Now shift so that low spatial frequencies are in the center.
            F2 = (fftpack.fftshift((F1)))

            final_FFT = self.zoom_interpol(np.abs(F2))

            FFT_mat4.append(final_FFT)

        return np.array(FFT_mat4)

    def build_ind_val_matrices(self, unit_values):
        """
        Builds indices and values matrices using given unit values for each dimension.
        This function is taken from pyUSID.io
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


