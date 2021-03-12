from typing import Tuple, Type, Union

import torch
import numpy as np


def init_dataloaders(X_train: np.ndarray,
                     y_train: np.ndarray,
                     X_test: np.ndarray,
                     y_test: np.ndarray,
                     batch_size: int
                     ) -> Tuple[Type[torch.utils.data.DataLoader]]:
    """
    Initializes train and test data loaders
    """
    test_iterator = None
    test_data = X_test is not None and y_test is not None
    device_ = 'cuda' if torch.cuda.is_available() else 'cpu'
    X_train = tor(X_train).to(device_)
    y_train = tor(y_train).to(device_)
    tensordata_train = torch.utils.data.TensorDataset(X_train, y_train)
    train_iterator = torch.utils.data.DataLoader(
        tensordata_train, batch_size=batch_size, shuffle=True)
    if test_data:
        X_test = tor(X_test).to(device_)
        y_test = tor(y_test).to(device_)
        tensordata_test = torch.utils.data.TensorDataset(X_test, y_test)
        test_iterator = torch.utils.data.DataLoader(
            tensordata_test, batch_size=batch_size)
    return train_iterator, test_iterator


def tor(arr: Union[np.ndarray, torch.Tensor],
        out_type: str = "float") -> torch.Tensor:
    """
    Convertor to PyTorch tensor ('float' or 'long')
    """
    if not isinstance(arr, (np.ndarray, torch.Tensor)):
        raise NotImplementedError("Provide data as numpy array or torch tensor")
    if isinstance(arr, np.ndarray):
        arr = torch.from_numpy(arr)
    if out_type == "float":
        arr = arr.float()
    elif out_type == "long":
        arr = arr.long()
    return arr
