# Oak Ridge National Lab
# Center for Nanophase Materials Sciences
# written by Alvin Tan on 08/22/2019
# in collaboration with Rama Vasudevan and Kody Law

# This program runs bayesian_inference and bayesian_utils on multiple nodes using mpi4py. It should be used
# as a supplement to bayesian_sample_code.py, which contains more options of things that can be done.
# This code only demonstrates opening the h5 file with mpi4py and lists ways to run the code below.

# Run this code on bash with
#     $ mpiexec -n 4 python bayesian_mpi_sample_code.py
# to run on 4 processors. If you are running on CADES SHCP, load modules PE-gnu, python, and hdf5-parallel.

# Run this code on CADES SHCP with the bayesian_hpc_run.pbs file by running
#     $ qsub bayesian_hpc_run.pbs
# as you likely know how to do already.

from __future__ import division, print_function, absolute_import, unicode_literals
import os
import subprocess
import sys
import io
import h5py
import time
import random
from shutil import copyfile
import numpy as np
from mpi4py import MPI

# Helper for importing packages that are not installed yet
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

# Set path name to the desired dataset. In this code, I use a file that only holds a subset of the
# full dataset, as created by code similar to that in bayesian_sample_code.py.
h5_path = r"currentMeasurements.h5"

# Select parameters we want to run with
M = 100
Ns = int(1e8)

# Open the file with mpi4py functionality
with h5py.File(h5_path, mode='r+', driver='mpio', comm=MPI.COMM_WORLD) as h5_f:
    # Create a new Process class. Note that here we use default values for f and V0 and whatnot.
    # See bayesian_sample_code.py for an example of how to parse those values from the actual dataset.
    abi = AdaptiveBayesianInference(h5_f["subsetBoi/Measured Current"], M=M, Ns=Ns)

    # Print out some cool MPI things
    print("Starting compute on processor {} of {}".format(MPI.COMM_WORLD.rank, MPI.COMM_WORLD.size))
    # Take timing measurements for fun
    startTime = time.time()
    h5_bayes_group = abi.compute()
    totalTime = time.time() - startTime
    print("Compute ended. took {} seconds".format(totalTime))

# Record the timing results
outputFile = open("timings.txt", "a")
outputFile.write("{}\n".format(totalTime))
outputFile.close()