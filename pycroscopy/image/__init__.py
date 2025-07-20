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
from .image_utilities import crop_image, flatten_image, inpaint_image
from .image_clean import decon_lr, clean_svd, background_correction
from .image_atoms import find_atoms, atom_refine
from .image_graph import make_structural_units, find_structural_units, get_polygons, add_graph
from .image_registration import complete_registration, rigid_registration, demon_registration
from .image_fft import power_spectrum, diffractogram_spots, adaptive_fourier_filter, rotational_symmetry_diffractogram


__all__ = ['ImageWindowing', 'crop_image', 'decon_lr', 'clean_svd', 'background_correction', 'find_atoms', 'atom_refine',
           'make_structural_units', 'find_structural_units', 'get_polygons', 'add_graph',
           'complete_registration', 'demon_registration', 'rigid_registration', 'flatten_image', 'inpaint_image',
           'power_spectrum', 'diffractogram_spots', 'adaptive_fourier_filter', 'rotational_symmetry_diffractogram']

