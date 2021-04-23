from typing import List, Optional, Tuple, Union

import dask.array
import numpy as np
from tensorly.decomposition import non_negative_parafac, tucker


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
