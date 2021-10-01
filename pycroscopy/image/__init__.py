"""
Image Module

Should contain
- general feature extraction
- geometry feature extraction
- atom finding
- denoising
- windowing
- transforms (e.g., radon, hough)

Submodules
----------
.. autosummary::
    :toctree: _autosummary

"""

from .image_window import ImageWindowing

from .image_utilities import crop_image
from .image_clean import decon_lr, clean_svd
from .image_atoms import find_atoms, atom_refine

__all__ = ['ImageWindowing', 'crop_image', 'decon_lr', 'clean_svd', 'find_atoms', 'atom_refine']

