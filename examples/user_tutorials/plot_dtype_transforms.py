"""
======================================================================================
Walk-through: Data transformation utilities
======================================================================================

**Suhas Somnath**

10/12/2017

This is a short walk-through of useful utilities that simplify data transformations, available in pycroscopy.
"""

from warnings import warn
import numpy as np
try:
    import pycroscopy as px
except ImportError:
    warn('pycroscopy not found.  Will install with pip.')
    import pip
    pip.main(['install', 'pycroscopy'])
    import pycroscopy as px

complex_value = 3 + 4j
stacked_real_value = px.dtype_utils.complex_to_real(complex_value)
print('Complex value: {}. Stacked real value: {} has shape: '
      '{}'.format(complex_value, stacked_real_value, stacked_real_value.shape))

complex_array = np.random.randint(-5, high=5, size=(3)) + 1j* np.random.randint(-5, high=5, size=(3))
stacked_real_array = px.dtype_utils.complex_to_real(complex_array)
print('Complex value: {}. Stacked real value: {} has shape: '
      '{}'.format(complex_array, stacked_real_array, stacked_real_array.shape))

struc_dtype = np.dtype({'names': ['r', 'g', 'b'],
                        'formats': [np.float32, np.uint16, np.float64]})
num_elems = 5
structured_array = np.zeros(shape=(num_elems), dtype=struc_dtype)
structured_array['r'] = np.random.random(size=num_elems)
structured_array['g'] = np.random.randint(0, high=1024, size=(num_elems))
structured_array['b'] = np.random.random(size=num_elems)
real_array = px.dtype_utils.compound_to_real(structured_array)

print('Structured array is of shape {} and have values:'.format(structured_array.shape))
print(structured_array)
print('\nThis array converted to regular scalar matrix has shape: {} and values:'.format(real_array.shape))
print(real_array)
