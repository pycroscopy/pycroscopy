from typing import List, Optional, Tuple, Union
import dask.array
import numpy as np
from tensorly.decomposition import non_negative_parafac, tucker
import sidpy
import matplotlib.pyplot as plt

class TensorFactor():
    def __init__(self, data: Union[np.ndarray, dask.array.Array],
                         rank: Union[int, List[int]],
                         decomposition_type: str = "cp",
                         flat_from: Optional[int] = None
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
        

        Returns
        -------
        weights and list of factors from the decomposition as sidpy.Dataset objects
        """
        
        self.data = data
        self.allowed_methods = ['cp', 'parfac'] #Might want to add others...
        if decomposition_type not in self.allowed_methods:
            raise NotImplementedError(
                "Currently available decomposition types are {}".format(self.allowed_methods))
       
        self.decomposition_type = decomposition_type
        self.data_3d = self._return_3d_dataset(self, self.data)
        self.results_computed = False
        self.rank = rank

    def _return_3d_dataset(self, data):
        # here goes the code for flattening the dataset into a 3D block
        # Mani's code will be used here. So ignore this for now.
        return

    def do_fit(self, **kwargs) -> sidpy.Dataset.dataset:
        """
        Parameters:
        **kwargs
            additional parameters passed to PARAFAC or tucker decomposition methods

        Returns: 
            Sidpy dataset after fit operation. Fit will calculate according to the method chosen.
        """

        if self.decomposition_type == "cp":
            weights, factors = non_negative_parafac(self.data_3d, self.rank, **kwargs)
        elif self.decomposition_type == "tucker":
            weights, factors = tucker(self.data_3d, self.rank, **kwargs)
        return weights, factors

    def plot_results(self)->plt.figure:
        """Plots the results"""
        if self.results_computed is False:
            raise RuntimeError("No results are available. Call 'do_fit()' method first")
        fig, axes = plt.subplots()
        return fig



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
