# Oak Ridge National Lab
# Center for Nanophase Materials Sciences
# written by Alvin Tan on 08/22/2019
# in collaboration with Rama Vasudevan and Kody Law

# This program goes through the result database of a dataset processed by the adaptive metropolis
# and graphs the results to visually inspect how well the algorithm worked. Hopefully this will be
# used only for parameter refinement and initial troubleshooting instead of after compute() had been
# run on the entire dataset just to find out that it's gone poorly. But that's up you as the user.

import h5py
import io
import time
from bayesian_inference import AdaptiveBayesianInference
from matplotlib import pyplot as plt
import pyUSID as pyUSID
import numpy as np 
import pyUSID as usid

# The path to the h5 file that holds the processed dataset. In this code, I use a file that only
# holds a subset of the full dataset, as created by code similar to that in bayesian_sample_code.py.
h5_path = r"C:\Users\Administrator\Documents\29TSummer2019\subsetFile1565658168.7634015.h5"

with h5py.File(h5_path, mode="r+") as h5_f:
	# h5_main is the filtered and reshaped dataset
	h5_main = h5_f["subsetBoi/Measured Current"]

	# h5_results is the group created and returned by the compute() function in bayesian_inference
	h5_results_grp = h5_f["subsetBoi/Measured Current-Adaptive_Bayesian_000"]

	# Create a Process class to use the plotPixel() function in it
	abi = AdaptiveBayesianInference(h5_main)

	# Iterate through all of the pixels and generate graphs for each one
	for i in range(h5_main.shape[0]):
		figBoi = abi.plotPixel(i, h5_results_grp=h5_results_grp)
		plt.show()
		print("Close the current figure to see the next one...")
