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

################################################################################################
# The general nature of pycroscopy facilitates the representation of any kind of measurement data.
# This includes:
# 1. Conventional data represented using floating point numbers such as 1.2345
# 2. Integer data (with or without sign) such as 1, 2, 3, 4
# 3. Complex-valued data such as 1.23 + 4.5i
# 4. Multi-valued or compound valued data cells such as ('Frequency': 301.2, 'Amplitude':1.553E-3, 'Phase': 2.14)
#    where a single value or measurement is represented by multiple elements, each with their own names, and data types
#
# While HDF5 datasets are capable of storing all of these kinds of data, many conventional data analysis techniques
# such as decomposition, clustering, etc. are either unable to handle complicated data types such as complex-valued
# datasets and compound valued datasets, or the results from these techniques do not produce physically meaningful
# results. For example, most singular value decomposition algorthms are capable of processing complex-valued datasets.
# Hpwever, while the eigenvectors can have complex values, the resultant complex-valued abundance maps are meaningless.
# To avoid these problems, we need functions that transform the data tp the necessary type (integer, real-value etc.)
#
# The dtype_utils module in .io facilitates two main kinds of data transformations:
#
# 1. Transforming from complex or compound (also known as structured arrays in numpy) valued datasets to real-valued
#    datasets
# 2. And the reverse - Transforming
# ensure that the data of the appropriate type (integer, real-value etc.) is
# supplied to such functions. In the same way, we need to be careful to transform the data back to its original
# complex or compound valued form. have two kinds of data transformation functions in the io.dtype_utils mofileathe results are meaningless. Specifically, complex-valued abundance maps are physically meaningless.  since the abundance maps and it often becomes a little challenging to fe


complex_value = 3 + 4j
stacked_real_value = px.dtype_utils.flatten_complex_to_real(complex_value)
print('Complex value: {}. Stacked real value: {} has shape: '
      '{}'.format(complex_value, stacked_real_value, stacked_real_value.shape))

complex_array = np.random.randint(-5, high=5, size=(3)) + 1j* np.random.randint(-5, high=5, size=(3))
stacked_real_array = px.dtype_utils.flatten_complex_to_real(complex_array)
print('Complex value: {}. Stacked real value: {} has shape: '
      '{}'.format(complex_array, stacked_real_array, stacked_real_array.shape))

struc_dtype = np.dtype({'names': ['r', 'g', 'b'],
                        'formats': [np.float32, np.uint16, np.float64]})
num_elems = 5
structured_array = np.zeros(shape=(num_elems), dtype=struc_dtype)
structured_array['r'] = np.random.random(size=num_elems)
structured_array['g'] = np.random.randint(0, high=1024, size=(num_elems))
structured_array['b'] = np.random.random(size=num_elems)
real_array = px.dtype_utils.flatten_compound_to_real(structured_array)

print('Structured array is of shape {} and have values:'.format(structured_array.shape))
print(structured_array)
print('\nThis array converted to regular scalar matrix has shape: {} and values:'.format(real_array.shape))
print(real_array)
