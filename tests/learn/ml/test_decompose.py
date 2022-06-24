import sys
import pytest
from numpy.testing import assert_equal, assert_
import numpy as np
import sidpy as sid
sys.path.append("../../../")

from pycroscopy.learn import TensorFactor

def return_Vt(t, *parms):
    V,b,c = parms    
    return V*(-b*t/c)

def return_4D_mat(data_shape=(10,10,9,7)):

    time_vec = np.linspace(0.15,1,data_shape[-1])
    voltage_vec = np.linspace(-2,2,data_shape[-2])

    data_mat = np.zeros(data_shape)
    for row in range(data_mat.shape[0]):
        for col in range(data_mat.shape[1]):
            for spec_1 in range(data_mat.shape[2]):
                    a = voltage_vec[spec_1]
                    b = voltage_vec[spec_1]**2
                    c = np.random.normal()
                    data_mat[row,col,spec_1,:] = return_Vt(time_vec, *[a,b,c]) + 2.5*np.random.normal(size=len(time_vec))

    x_dim = np.linspace(0, 1E-6,
                        data_mat.shape[0])
    y_dim = np.linspace(0, 1E-6,
                        data_mat.shape[1])
    z1_dim = voltage_vec
    z2_dim = time_vec

    #Make a sidpy dataset
    data_set = sid.Dataset.from_array(data_mat, name='Piezoresponse_relaxation')

    #Set the data type
    data_set.data_type = sid.DataType.SPECTRAL_IMAGE

    # Add quantity and units
    data_set.units = 'a.u.'
    data_set.quantity = 'Piezoresponse'

    # Add dimension info
    data_set.set_dimension(0, sid.Dimension(x_dim,
                                            name='x',
                                            units='m', quantity='x',
                                            dimension_type='spatial'))
    data_set.set_dimension(1, sid.Dimension(y_dim,
                                            name='y',
                                            units='m', quantity='y',
                                            dimension_type='spatial'))
    data_set.set_dimension(2, sid.Dimension(z1_dim,
                                            name='Voltage',
                                            units='V', quantity='Voltage',
                                            dimension_type='spectral'))

    data_set.set_dimension(3, sid.Dimension(z2_dim,
                                            name='Time',
                                            units='t', quantity='Time',
                                            dimension_type='spectral'))
    parms_dict_expt = {'info_1': np.linspace(0,1,100), 'instrument': "Dan's noisy AFM"}

    # append metadata            
    data_set.metadata = parms_dict_expt
    
    return data_set


@pytest.mark.parametrize("decomp_type", ["cp", "tucker"])
@pytest.mark.parametrize("in_dim", [(3, 5, 7,4), (2, 2, 8, 3)])
def test_input_dim(in_dim, decomp_type):
    x = return_4D_mat(in_dim)
    tf = TensorFactor(x, rank=4,decomposition_type = decomp_type )
    _, factors = tf.do_fit()
    assert_equal(len(factors),3)


@pytest.mark.parametrize("decomp_type", ["cp", "tucker"])
@pytest.mark.parametrize("rank", [1,2,3,4])
@pytest.mark.parametrize("in_dim", [(3, 5, 7,4), (2, 2, 8, 5)])
def test_rank_output(in_dim, rank, decomp_type):
    x = return_4D_mat(in_dim)
    tf = TensorFactor(x, rank=rank,decomposition_type = decomp_type )
    weights, _ = tf.do_fit()
    assert_equal(weights.shape[-1],rank)
