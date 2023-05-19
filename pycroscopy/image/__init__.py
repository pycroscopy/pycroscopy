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
from .image_utilities import crop_image, flatten_image
from .image_clean import decon_lr, clean_svd
from .image_atoms import find_atoms, atom_refine
from .image_graph import make_structural_units, find_structural_units, get_polygons, add_graph
from .image_registration import complete_registration, rigid_registration, demon_registration

__all__ = ['ImageWindowing', 'crop_image', 'decon_lr', 'clean_svd', 'find_atoms', 'atom_refine', 
           'make_structural_units', 'find_structural_units', 'get_polygons', 'add_graph',
           'complete_registration', 'demon_registration', 'rigid_registration',
           'flatten_image']

