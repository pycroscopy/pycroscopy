import array
from typing import List, Optional, Tuple, Union
import dask.array
import numpy as np
from tensorly.decomposition import non_negative_parafac, tucker
import sidpy
import matplotlib.pyplot as plt
import warnings


class TensorFactor():
    def __init__(self, data: Union[np.ndarray, dask.array.Array],
                 rank: Union[int, List[int]],
                 decomposition_type: str = "cp",
                 flat_from: Optional[int] = None,
                 spec_dims: Optional[int] = None
                 ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Tensor decomposition

        Parameters
        ----------
        data
            sidpy.Dataset object
        rank
            Number of components
        decomposition type
            PARAFAC ('cp') or Tucker ('tucker'). Default: 'cp'
        flat_from
            flattens image dimensions starting from the specified integer
        spec_dims:
            List of tuples or list of lists
            dimensions corresponding to the spectral dimensions of the the dataset

        Returns
        -------
        weights and list of factors from the decomposition as sidpy.Dataset objects
        """

        self.data = data
        self.allowed_methods = ['cp', 'tucker']  # Might want to add others...
        if decomposition_type not in self.allowed_methods:
            raise NotImplementedError(
                "Currently available decomposition types are {}".format(self.allowed_methods))

        self.decomposition_type = decomposition_type
        self.data_3d = self._return_3d_dataset(self.data, spec_dims)
        self.dim_order = self.data_3d.metadata['fold_attr']['dim_order']
        self.results_computed = False
        self.rank = rank
        self.weights, self.factors = None, None  # Datasets to be returned

    def _return_3d_dataset(self, data, spec_dims):
        # here goes the code for flattening the dataset into a 3D block
        if spec_dims is not None:
            # Here's where user has specified the spectral dimensions
            dims = set(np.arange(len(data.shape)))
            dim_order = [[], [], []]
            dim_order[1], dim_order[2] = spec_dims[0], spec_dims[1]
            for dim in dims:
                if not (dim in dim_order[1] or dim in dim_order[2]):
                    dim_order[0].extend([dim])

            folded_dset = data.fold(dim_order=dim_order)

        else:
            # We are on our own now
            spa_dims, spec_dims = [], []
            dim_order = [[], [], []]

            for dim, axis in data._axes.items():
                if axis.dimension_type == sidpy.DimensionType.SPATIAL:
                    spa_dims.extend([dim])
                elif axis.dimension_type == sidpy.DimensionType.SPECTRAL:
                    spec_dims.extend([dim])
                else:
                    raise NotImplementedError('Dimension {} is not one of SPATIAL or SPECTRAl'.format(dim))

            if len(spa_dims) == 0:
                raise NotImplementedError("No SPATIAL Dimensions found. Can't reshape the array.Please provide "
                                          "spec_dims to work around this problem.")

            if len(spec_dims) == 1:
                raise NotImplementedError("Only one SPECTRAL Dimension found. "
                                          "Can't reshape the array.Please provide "
                                          "spec_dims to work around this problem.")

            if len(spec_dims) > 2:
                warnings.warn('More than 2 Spectral dimensions are found, all spectral dimensions expect for the'
                              'first will be collapsed into one. Please provide spec_dims if this is not the case')

                dim_order[0] = spa_dims
                dim_order[1] = [spec_dims[0]]
                dim_order[2] = spec_dims[1::]

            if len(spec_dims) == 2:
                dim_order[0] = spa_dims
                dim_order[1] = [spec_dims[0]]
                dim_order[2] = [spec_dims[1]]

            folded_dset = data.fold(dim_order)

        return folded_dset

    def do_fit(self, **kwargs):
        """
        Parameters:
        **kwargs
            additional parameters passed to PARAFAC or tucker decomposition methods

        Returns:
            Sidpy dataset after fit operation. Fit will calculate according to the method chosen.
        """

        if self.decomposition_type == "cp":
            weights, factors = non_negative_parafac(np.array(self.data_3d), self.rank, **kwargs)
        elif self.decomposition_type == "tucker":
            weights, factors = tucker(np.array(self.data_3d), self.rank, **kwargs)

        # Let's create the sidpy datasets from these numpy arrays and lists
        weights_dset = sidpy.Dataset.from_array(weights, title='weights')

        rank_dim = sidpy.Dimension(np.arange(self.rank),
                                   name='components/rank',
                                   units='generic', quantity='generic',
                                   dimension_type='spectral')

        # This corresponds to the spa_dims*rank dset

        fac_dset0 = self.data_3d.like_data(factors[0], title='factors_0',
                                           check_dims=False)
        del fac_dset0.metadata['fold_attr']
        fac_dset0_axes = {}
        for i, dim in enumerate(self.dim_order[0]):
            fac_dset0_axes[i] = self.data_3d.metadata['fold_attr']['_axes'][dim].copy()
        fac_dset0_axes[len(self.dim_order[0])] = rank_dim
        fac_dset0.metadata['fold_attr'] = dict(dim_order_flattened=list(np.arange(len(self.dim_order[0]) + 1)),
                                               shape_transposed=self.data_3d.metadata['fold_attr'][
                                                                    'shape_transposed'][:len(self.dim_order[0])]
                                                                + [self.rank], _axes=fac_dset0_axes)

        fac_dset1 = self.data_3d.like_data(factors[1], title='factors_1',
                                           check_dims=False)
        del fac_dset1.metadata['fold_attr']
        fac_dset1_axes = {}
        for i, dim in enumerate(self.dim_order[1]):
            fac_dset1_axes[i] = self.data_3d.metadata['fold_attr']['_axes'][dim].copy()
        fac_dset1_axes[len(self.dim_order[1])] = rank_dim
        fac_dset1.metadata['fold_attr'] = dict(dim_order_flattened=list(np.arange(len(self.dim_order[1]) + 1)),
                                               shape_transposed=self.data_3d.metadata['fold_attr'][
                                                                    'shape_transposed'][
                                                                len(self.dim_order[0]):len(self.dim_order[0]) + len(
                                                                    self.dim_order[1])]
                                                                + [self.rank], _axes=fac_dset1_axes)

        fac_dset2 = self.data_3d.like_data(factors[2], title='factors_2',
                                           check_dims=False)
        del fac_dset2.metadata['fold_attr']
        fac_dset2_axes = {}
        for i, dim in enumerate(self.dim_order[2]):
            fac_dset2_axes[i] = self.data_3d.metadata['fold_attr']['_axes'][dim].copy()
        fac_dset2_axes[len(self.dim_order[2])] = rank_dim
        fac_dset2.metadata['fold_attr'] = dict(dim_order_flattened=list(np.arange(len(self.dim_order[2]) + 1)),
                                               shape_transposed=self.data_3d.metadata['fold_attr'][
                                                                    'shape_transposed'][-len(self.dim_order[2]):]
                                                                + [self.rank], _axes=fac_dset2_axes)

        self.weights = weights_dset
        self.factors = [fac_dset0.unfold(), fac_dset1.unfold(), fac_dset2.unfold()]
        self.factors[0].data_type = sidpy.DataType.IMAGE_STACK
        self.factors[1].data_type = sidpy.DataType.LINE_PLOT_FAMILY
        self.factors[2].data_type = sidpy.DataType.LINE_PLOT_FAMILY
        return self.factors, self.weights


def tensor_decomposition(data: Union[np.ndarray, dask.array.Array],
                         rank: Union[int, List[int]],
                         decomposition_type: str = "cp",
                         flat_from: Optional[int] = None,
                         **kwargs) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Tensor decomposition

    Parameters
    ----------
    data
        Multidimensional array/tensor
    rank
        Number of components
    decomposition type
        PARAFAC ('cp') or Tucker ('tucker'). Default: 'cp'
    flat_from
        flattens image dimensions starting from the specified integer
    **kwargs
        additional parameters passed to PARAFAC or tucker decomposition methods

    Returns
    -------
    weights and list of factors from the decomposition
    """
    if decomposition_type not in ["cp", "tucker"]:
        raise NotImplementedError(
            "Currently available decomposition types are 'ce' and 'tucker'")
    if flat_from is not None:
        reshape_ = np.product(data.shape[flat_from:])
        keep_dim = data.shape[:flat_from]
        data = data.reshape(*keep_dim, reshape_)
    if decomposition_type == "cp":
        weights, factors = non_negative_parafac(data, rank, **kwargs)
    else:
        weights, factors = tucker(data, rank, **kwargs)
    return weights, factors
