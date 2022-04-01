from typing import Type, Union, Optional
import torch
import numpy as np
from .datautils import init_dataloaders


class Trainer:
    """
    Trainer class for neural networks in PyTorch

    Parameters
    ----------
    model
        PyTorch model to train.
    X_train
        Training data
    y_train
        Training targets
    X_test
        Testing data
    y_test
        Testing data
    batch_size
        Batch size (Default: 32)
    filename
        filename for saving trained weights
    kwargs
        Use kwargs to specify learning rate (e.g. lr=0.005)

    Examples
    --------

    Initialize a trainer and train an autoencoder model for 30 epochs

    >>> # The inputs and targets are identical in this case
    >>> t = Trainer(model, images, images)
    >>> # Train
    >>> t.fit(num_epochs=30)
    """
    def __init__(self,
                 model: Type[torch.nn.Module],
                 X_train: Union[torch.Tensor, np.ndarray],
                 y_train: Union[torch.Tensor, np.ndarray],
                 X_test: Optional[Union[torch.Tensor, np.ndarray]] = None,
                 y_test: Optional[Union[torch.Tensor, np.ndarray]] = None,
                 batch_size: int = 32,
                 filename: str = 'model',
                 **kwargs: Union[int, float]
                 ) -> None:
        """
        Initializes trainer
        """
        seed = kwargs.get("seed", 1)
        set_rng_seed(seed)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model
        self.model.to(device)
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=kwargs.get("lr", 1e-3))
        self.criterion = torch.nn.MSELoss()
        (self.train_iterator,
         self.test_iterator) = init_dataloaders(
             X_train, y_train, X_test, y_test, batch_size)
        self.filename = filename
        self.train_losses = []
        self.test_losses = []

    def train_step(self, feat: torch.Tensor,
                   tar: torch.Tensor) -> torch.Tensor:
        """
        Single training step, which propagates data through a NN
        to get a prediction, compares predicted value with a ground truth,
        and then performs backpropagation to compute gradients
        and optimizes weights.
        """
        self.model.train()
        pred = self.model.forward(feat)
        loss = self.criterion(pred, tar)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()

    def test_step(self, feat: torch.Tensor,
                  tar: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for test data with deactivated autograd engine
        """
        self.model.eval()
        with torch.no_grad():
            pred = self.model.forward(feat)
            loss = self.criterion(pred, tar)
        return loss.item()

    def step(self) -> None:
        """
        Model training and (optionally) evaluation on a single epoch
        """
        c, c_test = 0, 0
        losses, losses_test = 0, 0
        for feature, target in self.train_iterator:
            losses += self.train_step(feature, target)
            c += 1
        self.train_losses.append(losses / c)

        if self.test_iterator is not None:
            for feature, target in self.test_iterator:
                losses_test += self.test_step(feature, target)
                c_test += 1
            self.test_losses.append(losses_test / c_test)

    def fit(self, num_epochs: int = 20) -> None:
        """
        Wraps train and test steps. Print statistics after each epoch
        and saves trained weights at the end
        """
        for e in range(num_epochs):
            self.step()
            self.print_statistics(e)
        self.save_weights()

    def save_weights(self, *args: str) -> None:
        """
        Save model weights
        """
        try:
            filename = args[0]
        except IndexError:
            filename = self.filename
        torch.save(self.model.state_dict(), filename + '.pt')

    def print_statistics(self, e: int) -> None:
        if self.test_iterator is not None:
            template = 'Epoch: {}... Training loss: {:.4f}... Test loss: {:.4f}'
            print(template.format(
                e+1, self.train_losses[-1], self.test_losses[-1]))
        else:
            template = 'Epoch: {}... Training loss: {:.4f}...'
            print(template.format(e+1, self.train_losses[-1]))


def set_rng_seed(seed: int) -> None:
    """
    For reproducibility
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # for GPU
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
