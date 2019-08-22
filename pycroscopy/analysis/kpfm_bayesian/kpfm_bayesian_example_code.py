# Oak Ridge National Lab
# Center for Nanophase Materials Sciences
# written by Alvin Tan on 08/22/2019
# in collaboration with Rama Vasudevan, Liam Collins, and Kody Law

# This program provides an example use of the kpfm_bayesian and kpfm_bayesian_utils scripts.
# The KPFM Process class is functional, but this code does not process the data correctly, probably
# due to inappropriate inputs. This is meant to act as only an example to use when writing actual code.

import h5py
import numpy as np
import pyUSID as usid
from matplotlib import pyplot as plt
from kpfm_bayesian import KPFMBayesianInference

# This is the path to the h5 file in which the data to be process is stored
h5_path = r"C:\Users\Administrator\Dropbox\polynomial approximation paper\G data\GFA0_n1000mV_0003\GFA0_n1000mV_0003.h5"

with h5py.File(h5_path, mode="r+") as h5_file:
    # We pull out the dataset that holds data reshaped into pixels (or any arrangement where each row is a
    # separate entity to be analyzed)
    h5_group = h5_file["Measurement_000/Channel_000/Raw_Data-Reshape_000"]
    h5_main = h5_group["Reshaped_Data"]

    # Create a dictionary of physical parameters of the measurement setup. See get_default_parameters() in
    # kpfm_bayesian_utils for a sample declaration of this dictionary. In the case of this example code,
    # I have no idea what these values should be, so I will just input None and use the default values.
    p = None

    # Instantiate the KPFM Process class with the reshaped dataset and the parameter dictionary
    kbi = KPFMBayesianInference(h5_main, p=p)

    ##### Running test() #####

    # We can run the Bayesian inference algorithm on a single random pixel, and return the resulting graphs
    pix_ind, resultGraphs = kbi.test(verbose=True)
    resultGraphs[0].savefig("3DplotOn{}.png".format(pix_ind))
    resultGraphs[1].savefig("OtherPlotsOn{}.png".format(pix_ind))

    # Or on a particular pixel if you so choose
    pix_ind = 42
    pix_ind, resultGraphs = kbi.test()
    resultGraphs[0].savefig("3DplotOn{}.png".format(pix_ind))
    resultGraphs[1].savefig("OtherPlotsOn{}.png".format(pix_ind))

    ##### Running compute() #####

    # And if everything is looking good, we can run the Bayesian inference on the entire dataset and save
    # the results. compute() returns a reference to the group that holds the generated data, if you are
    # interested in looking through that
    resultGroup = kbi.compute()


