# This is just to look at the data stored in the h5 file cuz idk what to do with it

import time
import h5py
import numpy as np
import pyUSID as usid
from matplotlib import pyplot as plt
from kpfm_bayesian import KPFMBayesianInference

h5_path = r"C:\Users\Administrator\Dropbox\polynomial approximation paper\G data\GFA0_n1000mV_0003\GFA0_n1000mV_0003.h5"

with h5py.File(h5_path, mode="r+") as h5_file:
    h5_group = h5_file["Measurement_000/Channel_000/Raw_Data-Reshape_000"]
    h5_main = h5_group["Reshaped_Data"]
    '''
    for attr in h5_group.attrs:
        print("attribute {} is {}".format(attr, usid.hdf_utils.get_attr(h5_group, attr)))
    breakpoint()
    print("h5_main has shape {}".format(h5_main.shape))
    '''

    # Remove the previously created results group before trying to compute
    del h5_group["Reshaped_Data-KPFM_Bayesian_000"]

    kbi = KPFMBayesianInference(h5_main)

    startTime = time.time()
    kbi.compute()
    computeTime = time.time() - startTime

    print("Compute took {} seconds".format(computeTime))
    #results = kbi.test(graph=False, verbose=True)
    #breakpoint()
    #input("Pause to inspect variables...")

    '''
    pix_ind, graphBois = kbi.test(verbose=True)

    print("Ran Bayesian inference on pixel number {}".format(pix_ind))
    graphBois[0].savefig("3DplotOn{}.png".format(pix_ind))
    graphBois[1].savefig("OtherPlotsOn{}.png".format(pix_ind))
    input("Pause to inspect graphs...")
    '''


