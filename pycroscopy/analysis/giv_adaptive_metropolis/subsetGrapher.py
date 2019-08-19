# This makes graphs for the subset results to see if the adaptive Bayesian worked.

import h5py
import io
import time
from bayesian_inference import AdaptiveBayesianInference
from matplotlib import pyplot as plt
import pyUSID as pyUSID
import numpy as np 
import pyUSID as usid

h5_path = r"C:\Users\Administrator\Documents\29TSummer2019\subsetFile1565658168.7634015.h5"

with h5py.File(h5_path, mode="r+") as h5_f:
	h5_main = h5_f["subsetBoi/Measured Current"]
	h5_results = h5_f["subsetBoi/Measured Current-Adaptive_Bayesian_000"]

	abi = AdaptiveBayesianInference(h5_main)

	for i in range(h5_main.shape[0]):
		figBoi = abi.plotPixel(i, h5_results_grp=h5_results)
		figBoi.show()
		input("Press <Enter> for next figure...")
