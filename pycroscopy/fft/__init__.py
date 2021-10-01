"""
FFT Module

Should contain
- power spectra
- wavelets
- and potentially others

Submodules
----------

.. autosummary::
    :toctree: _autosummary

"""

from .image_fft import power_spectrum, diffractogram_spots


__all__ = ['power_spectrum', 'diffractogram_spots']
