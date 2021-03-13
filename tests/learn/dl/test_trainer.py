import sys

import numpy as np
import torch
from numpy.testing import assert_

sys.path.append("../../../")

from pycroscopy.learn import Trainer, models


def assert_weights_equal(m1, m2):
    eq_w = []
    for p1, p2 in zip(m1.values(), m2.values()):
        eq_w.append(np.array_equal(
            p1.detach().cpu().numpy(),
            p2.detach().cpu().numpy()))
    return all(eq_w)


def test_trainer():
    # Initialize a model
    in_dim = (1, 8, 8)
    model = models.AutoEncoder(
        in_dim, layers_per_block=[1, 2], nfilters=2)
    weights_before = model.state_dict()
    # Create dummy train set
    X_train = torch.randn(10, 1, 8, 8)
    # Initialize trainer
    t = Trainer(model, X_train, X_train, batch_size=4)
    # train and compare model params before and after
    t.fit(num_epochs=2)
    weights_after = model.state_dict()
    assert_(not assert_weights_equal(weights_before, weights_after))


def test_trainer_determenism():
    # Create dummy train set
    torch.manual_seed(0)
    X_train = torch.randn(10, 1, 8, 8)
    # Initialize a model
    in_dim = (1, 8, 8)
    model1 = models.AutoEncoder(
        in_dim, layers_per_block=[1, 2], nfilters=2,
        upsampling_mode="nearest")
    # Initialize trainer
    t = Trainer(model1, X_train, X_train, batch_size=4)
    # train
    t.fit(num_epochs=5)
    # Reininitiaize model and train again
    torch.manual_seed(0)
    X_train = torch.randn(10, 1, 8, 8)
    model2 = models.AutoEncoder(
        in_dim, layers_per_block=[1, 2], nfilters=2,
        upsampling_mode="nearest")
    t = Trainer(model2, X_train, X_train, batch_size=4)
    t.fit(num_epochs=5)
    assert_(assert_weights_equal(model1.state_dict(), model2.state_dict()))
