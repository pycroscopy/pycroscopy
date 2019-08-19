"""
Example code that utilizes bayesian_inference and bayesian_utils. Runs test and compute.

Created on Tue August 13, 2019

@author: Alvin Tan
"""

from __future__ import division, print_function, absolute_import, unicode_literals
import os
import subprocess
import sys
import io
import h5py
#from mpi4py import MPI
import time
import random
from shutil import copyfile
import numpy as np

# Helper for importing packages
def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", "--user", package])

try:
    import pyUSID as usid
except ImportError:
    print("pyUSID not found. Will install with pip.")
    import pip
    install("pyUSID")
    import pyUSID as usid

from bayesian_inference import AdaptiveBayesianInference
from pyUSID.io.hdf_utils import write_main_dataset, write_ind_val_dsets
from pyUSID.io.write_utils import Dimension

## Set path name to the desired dataset
h5_path = r"C:/Users/Administrator/Dropbox/GIv Bayesian June 2019/pzt_nanocap_6_split_bayesian_compensation_R_correction (Alvin Tan's conflicted copy 2019-06-25).h5"

## Determine what parameters we want to run with.
# Note: By inspection it was found that M=100 and Ns=7e7 is sufficient for a reasonable resistance inference.
#       However, further testing showed that the convergence rate differs across machines, so a conservative
#       Ns=1e8 is used in this case.
M = 100
Ns = int(1e8)

## Open the file
with h5py.File(h5_path, mode='r+') as h5_f:
    h5_grp = h5_f['Measurement_000/Channel_000']

    # Get attributes of the data to be used for the Bayesian inference
    f = usid.hdf_utils.get_attr(h5_grp, 'excitation_frequency_[Hz]')
    V0 = usid.hdf_utils.get_attr(h5_grp, 'excitation_amplitude_[V]')

    # Grab the filtered and reshaped dataset
    h5_resh = usid.USIDataset(h5_grp['Raw_Data-FFT_Filtering_000/Filtered_Data-Reshape_000/Reshaped_Data'])


    ##### Running test() #####

    ## Instantiate a new Process class and run test() on a random pixel
    abi = AdaptiveBayesianInference(h5_resh, f=f, V0=V0, Ns=Ns, M=M)
    pix_ind, bayesFigure = abi.test()
    # Save the resulting figure for later inspection
    bayesFigure.savefig("testPlotOnPixel{}.png".format(pix_ind))

    # We can also run test() on a particular pixel
    pix_ind = 2338
    pix_ind, bayesFigure = abi.test(pix_ind=pix_ind)
    bayesFigure.savefig("testPlotOnPixel{}.png".format(pix_ind))

    # We can also run test() and save traces that we can inspect for parameter optimization
    pix_ind=64550
    pix_ind, figBois = abi.test(pix_ind=pix_ind, traces=True)
    bayesFigure, traceFigure = figBois
    bayesFigure.savefig("testPlotOnPixel{}.png".format(pix_ind))
    traceFigure.savefig("tracePlotOnPixel{}.png".format(pix_ind))


    ##### Running compute() #####

    ## We can run compute() on the entire dataset, which creates datasets for the results and
    ## stores the results
    bayesResultsGroup = abi.compute()
    print(bayesResultsGroup)


    ## If we only want to run compute() on a subset of pixels, we can create a new h5 file
    ## with a subset of the pixels and go from there.

    # numPixels holds the number of pixels we want to run on
    numPixels = 500
    # Here we randomly select 500 pixels to use, but pixelInds can be any list-like object
    pixelInds = np.random.randint(0, h5_resh[()].shape[0], numPixels)
    print("PixelInds is {} with shape {}".format(pixelInds, pixelInds.shape))

    # Create new h5 file and copy subset to said file
    sub_f_path = 'subsetFile{}.h5'.format(time.time())
    sub_f = h5py.File(sub_f_path, 'a')
    subsetGroup = sub_f.create_group("subsetBoi")
    h5_spec_inds, h5_spec_vals = write_ind_val_dsets(subsetGroup, Dimension("Bias", "V", int(h5_resh.h5_spec_inds.size)), is_spectral=True)
    h5_spec_vals[()] = h5_resh.h5_spec_vals[()]
    h5_pos_inds, h5_pos_vals = write_ind_val_dsets(subsetGroup, Dimension("Position", "m", numPixels), is_spectral=False)
    # Note: the position values are not copied over, so those will be zeros. However, they can be recovered from the original dataset
    # with the pixelInds list if necessary.

    h5_subset = write_main_dataset(subsetGroup, (numPixels, h5_resh.shape[1]), "Measured Current", "Current",
                                   "nA", None, None, dtype=np.float64,
                                   h5_pos_inds = h5_pos_inds,
                                   h5_pos_vals = h5_pos_vals,
                                   h5_spec_inds = h5_spec_inds,
                                   h5_spec_vals = h5_spec_vals)
    h5_subset[()] = h5_resh[()][pixelInds, :]

    # Once our subset file is created, we can close the original dataset and continue processing with the subset file.

# create new Process class and run compute
abi = AdaptiveBayesianInference(h5_subset, f=f, V0=V0, Ns=Ns, M=M)
h5_bayes_group = abi.compute()

# close the subset file
sub_f.close()
os.remove(sub_f_path)

