# Oak Ridge National Lab
# Center for Nanophase Materials Sciences
# written by Alvin Tan on 08/06/2019
# in collaboration with Rama Vasudevan, Liam Collins, and Kody Law

# This program provides the Pycroscopy framework to process messy, noisy
# data streamed in from a fundamentally new data acquisition method
# incepted by Liam and involving Kelvin probe microscopy.

from __future__ import division, print_function, absolute_import, unicode_literals
from pyUSID.processing.process import Process
import numpy as np
import h5py

# Helper function for importing packages if they don't exist
import os
import subprocess
import sys
def install(package):
	subsystem.call([sys.executable, "-m", "pip", "install", "--user", package])

try:
	import pyUSID as usid 
except ImportError:
	print("pyUSID not found. Will install with pip.")
	import pip
	install("pyUSID")
	import pyUSID as usid 

try:
	import pycroscopy as px 
except ImportError:
	print("pycroscopy not found. Will install with pip.")
	import pip
	install("pycroscopy")
	import pycroscopy as px 

from pyUSID.processing.comp_utils import parallel_compute
from pyUSID.io.hdf_utils import create_results_group, write_main_dataset, write_simple_attrs, create_empty_dataset, \
	write_ind_val_dsets
from pyUSID.io.write_utils import Dimension
from kpfm_bayesian_utils import get_default_parameters, process_pixel


class KPFMBayesianInference(Process):
	def __init__(self, h5_main, p=None, **kwargs):
		'''
		Description of functions, inputs, and outputs
		'''
		super(KPFMBayesianInference, self).__init__(h5_main, **kwargs)

		# Name the process
		self.process_name = "KPFM_Bayesian"

		# Check to make sure this data has not already been processed
		self.duplicate_h5_groups, self.partial_h5_groups = self._check_for_duplicates()

		# Instantiate data variable
		self.data = None

		if p is None:
			p = get_default_parameters()
		self.p = p

		# Make list of parameters to be added to the results dataset.
		# I assume we will store all the parameters as attributes. If we do not want to do that, then simply
		# instantiate an empty dict here with self.params_dict = dict()
		self.params_dict = p
		if self.verbose: print("Results attributes dictionary created")

		# Get wd from the parameters
		IO_rate = p["Sim.IOrate"]
		t_max = p["Sim.Tmax"]
		N = t_max*IO_rate
		wd_req = p["Sim.Vfreq"]
		f = np.arange(-IO_rate/2, IO_rate/2, IO_rate/N)

		# wd_ind should be a tuple, I believe...
		wd_ind = np.where(np.absolute(wd_req - f) == np.amin(np.absolute(wd_req - f)))
		self.wd = f[wd_ind[0]][0]

		## Set up variables to store the results of the Bayesian algorithm

		# These will be the results from each processed chunk to transfer to the datasets
		self.rrmse = None
		self.V_B = None
		self.R_seg = None
		self.residual = None
		self.polynomialFit = None
		self.rsq = None

		# These are the actual result datasets
		self.h5_rrmse = None
		self.h5_V_B = None
		self.h5_R_seg = None
		self.h5_residual = None
		self.h5_polynomialFit = None
		self.h5_rsq = None
		if self.verbose: print("Empty result variables are set up")

		# We should be done with setup.
		if self.verbose: print("Finished initialization of KPFMBayesianInference class")


	def test(self, pix_ind=None, graph=True, verbose=False):
		'''
		Description
		'''
		if self.mpi_rank > 0:
			return
		if pix_ind is None:
			pix_ind = np.random.randint(self.h5_main.shape[0])

		if verbose: print("Running test on pixel number {}".format(pix_ind))

		return pix_ind, process_pixel(self.h5_main[pix_ind, :], self.wd, self.p, graph=graph, verbose=verbose)

	def _create_results_datasets(self):
		'''
		Creates all the datasets necessary for holding all parameters and data
		'''
		self.h5_results_grp = create_results_group(self.h5_main, self.process_name)

		self.params_dict.update({'last_pixel': 0, 'algorithm': 'pycroscopy_KPFMBayesianInference'})

		write_simple_attrs(self.h5_results_grp, self.params_dict)

		assert isinstance(self.h5_results_grp, h5py.Group)

		# Grab some values we will use as dimensions for the datasets
		plotLength = self.h5_main.shape[1]//self.p["Bayes.fac"]
		polyCoeffNum = self.p["Bayes.Npoly"] + 1

		# Make some vectors that we will use as spectroscopic values
		Tsec = self.p["Sim.Tmax"]/self.p["Bayes.fac"]
		res_spect = np.arange(Tsec, step=Tsec/plotLength) # same as tt1 in utils
		V_B_spect = self.p["Sim.VAmp"]*np.sin(self.wd*2*np.pi*res_spect + self.p["Sim.Phasshift"])

		# Make spectroscopic datasets that hold the above vectors 
		h5_spec_inds_res, h5_spec_vals_res = write_ind_val_dsets(self.h5_results_grp, Dimension("Time", "sec", plotLength), is_spectral=True, base_name="Spectroscopic_Res")
		h5_spec_vals_res[()] = res_spect

		h5_spec_inds_V_B, h5_spec_vals_V_B = write_ind_val_dsets(self.h5_results_grp, Dimension("Bias", "V", plotLength), is_spectral=True, base_name="Spectroscopic_V_B")
		h5_spec_vals_V_B[()] = V_B_spect

		# Note that RRMSE and RSQ are both constants, so do not have a spectroscopic aspect to them. This one is for those.
		h5_spec_inds_const, h5_spec_vals_const = write_ind_val_dsets(self.h5_results_grp, Dimension("None", "None", 1), is_spectral=True, base_name="Spectroscopic_Const")

		#breakpoint()
		#input("Pause here to inspect constructed datasets...")

		## Initialize our datasets

		# Note that RRMSE doesn't really have a spectroscopic aspect, so that's just nothing useful.
		self.h5_rrmse = write_main_dataset(self.h5_results_grp, (self.h5_main.shape[0], 1), "RRMSE", "Stat",
										   "None", None, None, dtype=np.float64, 
										   h5_pos_inds=self.h5_main.h5_pos_inds,
										   h5_pos_vals=self.h5_main.h5_pos_vals,
										   h5_spec_inds=h5_spec_inds_const,
										   h5_spec_vals=h5_spec_vals_const)

		self.h5_V_B = write_main_dataset(self.h5_results_grp, (self.h5_main.shape[0], plotLength), "V_B", "Voltage",
										 "V", None, None, dtype=np.float64, 
										 h5_pos_inds=self.h5_main.h5_pos_inds,
										 h5_pos_vals=self.h5_main.h5_pos_vals,
										 h5_spec_inds=h5_spec_inds_V_B,
										 h5_spec_vals=h5_spec_vals_V_B)

		# Honestly not too sure what R_seg is, but it seems to be from simulated data, so this dataset will be full of Nan
		# when processing actual data, so we probably don't want to keep this haha.
		self.h5_R_seg = write_main_dataset(self.h5_results_grp, (self.h5_main.shape[0], plotLength), "R_seg", "Voltage",
										   "V", None, None, dtype=np.float64, 
										   h5_pos_inds=self.h5_main.h5_pos_inds,
										   h5_pos_vals=self.h5_main.h5_pos_vals,
										   h5_spec_inds=h5_spec_inds_res,
										   h5_spec_vals=h5_spec_vals_res)
		
		self.h5_residual = write_main_dataset(self.h5_results_grp, (self.h5_main.shape[0], plotLength), "Residual", "Residual",
											  "None", None, None, dtype=np.float64, 
											  h5_pos_inds=self.h5_main.h5_pos_inds,
											  h5_pos_vals=self.h5_main.h5_pos_vals,
											  h5_spec_inds=h5_spec_inds_res,
											  h5_spec_vals=h5_spec_vals_res)
		
		self.h5_polynomialFit = write_main_dataset(self.h5_results_grp, (self.h5_main.shape[0], 3), "Polynomial_Coefficients", "Coefficients",
												   "None", None, Dimension("Power", "None", 3), dtype=np.float64, 
												   h5_pos_inds=self.h5_main.h5_pos_inds,
												   h5_pos_vals=self.h5_main.h5_pos_vals,
												   aux_spec_prefix="Spectroscopic_Poly")
		
		# The RSQ dataset has the same metadata and structure as the RRMSE dataset
		self.h5_rsq = create_empty_dataset(self.h5_rrmse, np.float64, "RSQ")

		if self.verbose: print("Results datasets created.")
		self.h5_main.file.flush()


	def _get_existing_datasets(self):
		'''
		Extracts references to the existing datasets that hold the results
		'''

		self.h5_rrmse = self.h5_results_grp["RRMSE"]
		self.h5_V_B = self.h5_results_grp["V_B"]
		self.h5_R_seg = self.h5_results_grp["R_seg"]
		self.h5_residual = self.h5_results_grp["Residual"]
		self.h5_polynomialFit = self.h5_results_grp["Polynomial_Coefficients"]
		self.h5_rsq = self.h5_results_grp["RSQ"]
		if self.verbose: print("Got existing datasets.")

	def _write_results_chunk(self, pos_in_batch=None):
		'''
		Writes data chunks back to the file
		'''

		if pos_in_batch is None:
			pos_in_batch = self._get_pixels_in_current_batch()

		#breakpoint()
		#input("Pause to inspect attempts at broadcasting...")

		# Write data into results datasets
		self.h5_rrmse[pos_in_batch, :] = self.rrmse[np.newaxis].T # Since self.rrmse is a row vector
		self.h5_V_B[pos_in_batch, :] = self.V_B
		self.h5_R_seg[pos_in_batch, :] = self.R_seg
		self.h5_residual[pos_in_batch, :] = self.residual
		self.h5_polynomialFit[pos_in_batch, :] = self.polynomialFit
		self.h5_rsq[pos_in_batch, :] = self.rsq[np.newaxis].T # since self.rsq is a row vector
		if self.verbose: print("Results written back to file.")

	def _unit_computation(self, pos_in_batch=None, *args, **kwargs):
		'''
		Processing per chunk of the dataset
		'''

		# Grab the chunk of data that we want to process from the h5_main dataset
		if pos_in_batch is not None:
			self.data=self.h5_main[()][pos_in_batch, :]

		# Run Bayesian inference on each pixel
		all_data = parallel_compute(self.data, process_pixel, cores=self._cores, func_args=[self.wd, self.p])

		# Since process_pixel returns a tuple, parse the list of tuples into individual lists
		# Note: V_B, residual, and polynomialFit are returned as column vectors, which are funky to work with
		self.rrmse = np.array([result[0] for result in all_data]).astype(np.float64)
		self.V_B = np.array([result[1].T[0] for result in all_data]).astype(np.float64)
		self.R_seg = np.array([result[2] for result in all_data]).astype(np.float64)
		self.residual = np.array([result[3].T[0] for result in all_data]).astype(np.float64)
		self.polynomialFit = np.array([result[4].T[0] for result in all_data]).astype(np.float64)
		self.rsq = np.array([result[5] for result in all_data]).astype(np.float64)

		if self.verbose: print("Finished computing chunk.")






