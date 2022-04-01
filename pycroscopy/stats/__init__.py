"""
Statistics Library

Should eventually contain
- gaussian process regression
- compressed sensing for image reconstruction
- other relevant statistics tools, e.g. bayesian inference tools
- tree sturcture objects for e.g. functional fit propagation
- local crystallography

Submodules
----------
.. autosummary::
    :toctree: _autosummary

"""

from .tree import ClusterTree, Node

__all__ = ['Node', 'ClusterTree']