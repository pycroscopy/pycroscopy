"""
Statistics Library

Could contain
- gaussian process regression
- compressed sensing for image reconstruction
- other relevant statistics tools, e.g. bayesian inference tools
- tree sturcture objects for e.g. functional fit propagation

Submodules
----------
.. autosummary::
    :toctree: _autosummary

"""

from .tree import ClusterTree, Node

__all__ = ['Node', 'ClusterTree']