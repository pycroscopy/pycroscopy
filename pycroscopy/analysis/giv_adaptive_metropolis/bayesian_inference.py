# -*- coding: utf-8 -*-
"""
Runs a full Bayesian inference algorithm on spectroscopic data for ultrafast current imaging (see paper below)
https://www.nature.com/articles/s41467-017-02455-7
Used in conjunction with bayesian_utils.

Created on Tue July 02, 2019

@author: Alvin Tan, Emily Costa
"""

from __future__ import division, print_function, absolute_import, unicode_literals
import h5py
import numpy as np

# Helper function for importing packages
import os
import subprocess
import sys
def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", "--user", package])

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

from pyUSID.processing.process import Process
# Set up parallel compute later to run through supercomputer or cluster
from pyUSID.processing.comp_utils import parallel_compute
from pyUSID.io.hdf_utils import create_results_group, write_main_dataset, write_simple_attrs, create_empty_dataset, \
	write_ind_val_dsets
from pyUSID.io.write_utils import Dimension

import time
import math
import scipy.linalg as spla 

from bayesian_utils import get_shift_and_split_indices, process_pixel, get_shifted_response, get_unshifted_response, get_M_dx_x, publicGetGraph


class AdaptiveBayesianInference(Process):
	def __init__(self, h5_main, f=200, V0=None, Ns=int(1e8), M=101, parse_mod=1, **kwargs):
		"""
		Bayesian inference is done on h5py dataset object that has already been filtered
			and reshaped.

		Parameters
		----------
		h5_main 	: h5py.Dataset object
			Dataset to process. Must be a USID main dataset. Contains filtered measured current.
		f 			: (Optional) number
			Excitation frequency (Hz)
		V0			: (Optional) number
			Excitation amplitude (V). If left empty, the maximum value of the excitation wave is used.
		Ns 			: (Optional) integer
			Number of iterations of the adaptive metropolis to conduct. Must be sufficiently large to converge.
		M 			: (Optional) number
			Target number of points to use in the piecewise linear basis (actual number may be slightly off i.e. +- 1)
		parse_mod	: (Optional) integer
			Determines the number of datapoints we use to fit the model. If it's 1, then we use all available datapoints.
			If it's n > 1, we use every nth datapoint from the h5_main dataaset. 
		kwargs		: (Optional) dictionary
			See Process class for additional inputs
		"""

		super(AdaptiveBayesianInference, self).__init__(h5_main, **kwargs)

		## Name the process
		self.process_name = 'Adaptive_Bayesian'

		## Set some constants and vectors for the Bayesian analysis.

		# This determines how to parse down the data (i.e. used_data = actual_data[::parse_mod]).
		self.parse_mod = parse_mod

		# Grab the excitation waveform and take its derivative dv/dt
		ex_wave = self.h5_main.h5_spec_vals[()]
		dt = 1.0/(f*ex_wave.size)
		self.dvdt = np.diff(ex_wave)/dt
		self.dvdt = np.append(self.dvdt, self.dvdt[0][-1])
		# Parse out what values we want given parse_mod
		self.dvdt = self.dvdt[::self.parse_mod][np.newaxis].T
		self.full_V = ex_wave[0][::self.parse_mod]
		if self.verbose: print("Excitation wave form and derivative acquired.")

		# This line checks to make sure this data has not already been processed
		self.duplicate_h5_groups, self.partial_h5_groups = self._check_for_duplicates()

		# This will hold the chunks of data from the h5_main dataset to be used in compute()
		self.data = None

		# The data is currently ordered as a sine excitation wave. We want to feed the data into
		# Bayesian inference algorithm as forward (-6 V to 6 V) and reverse (6 V to -6 V) sweeps,
		# so we have to shift the data and split it in half.
		self.full_V, self.shift_index, self.split_index = get_shift_and_split_indices(self.full_V)
		self.dvdt = get_shifted_response(self.dvdt, self.shift_index)

		# Grab the excitation amplitude if we don't have it
		if V0 is None:
			V0 = max(self.full_V)

		# Grab the voltage points we will predict resistances for in the piecewise basis
		self.M, self.dx, self.x = get_M_dx_x(V0=V0, M=M)

		# Store a couple constants
		self.V0 = V0
		self.f = f
		self.Ns = Ns

		if self.verbose: print("Bayesian inference parameters are set up")
		
		## Set up variables to store the results of the Bayesian algorithm

		# These will be the results from each processed chunk to transfer to the datasets
		self.R = None
		self.R_sig = None
		self.capacitance = None
		self.i_recon = None
		self.i_corrected = None
	
		# These are the actual result databases
		self.h5_R = None
		self.h5_R_sig = None
		self.h5_capacitance = None
		self.h5_i_recon = None
		self.h5_i_corrected = None
		if self.verbose: print("Empty result variables are set up")

		# Create a dictionary for the attributes of the results group
		self.params_dict = dict()
		self.params_dict["Ns"] = self.Ns
		self.params_dict["num_pixels"] = self.h5_main[()].shape[0]
		self.params_dict["dvdt"] = self.dvdt
		self.params_dict["full_V"] = self.full_V
		self.params_dict["V0"] = self.V0
		self.params_dict["freq"] = self.f
		self.params_dict["shift_index"] = self.shift_index
		self.params_dict["split_index"] = self.split_index
		self.params_dict["M"] = self.M
		self.params_dict["dx"] = self.dx
		self.params_dict["x"] = self.x
		self.params_dict["parse_mod"] = self.parse_mod
		if self.verbose: print("Results attributes dictionary created.")

		# Yay we are done with setup!
		if self.verbose: print("Finished initialization of AdaptiveBayesianInference class.")

	def test(self, pix_ind=None, traces=False):
		"""
		Tests the Bayesian inference on a single pixel (randomly chosen unless manually specified) worth of data.
		Returns the resulting figure that has the forward and reverse resistances and the corrected current.

		Parameters
		----------
		pix_ind	: (Optional) integer
			Index of the pixel whose data will be used to test Bayesian inference. If not provided,
				a random pixel will be selected.
		traces	: (Optional) boolean
			Returns both a final figure and a figure of traces if True

		Returns
		-------
		pix_ind : integer
			Index of the pixel whose data was used to test Bayesian inference.
		figure	: pyplot.figure
			Plot of the inferred resistances and corrected current that can be saved or displayed.
		"""

		# Only run this once if doing distributed processing
		if self.mpi_rank > 0:
			return

		# Get a random pixel index if one is not provided
		if pix_ind is None:
			pix_ind = np.random.randint(1, high=self.h5_main.shape[0])

		# Shift the data to get the forward and reverse sweeps
		full_i_meas = get_shifted_response(self.h5_main[pix_ind, ::self.parse_mod], self.shift_index)

		# Process the pixel and return the resulting figure.
		return pix_ind, process_pixel(full_i_meas, self.full_V, self.split_index, self.M, self.dx, self.x, self.shift_index, self.f, self.V0, self.Ns, self.dvdt, pix_ind=pix_ind, traces=traces, graph=True, verbose=True)

	def plotPixel(self, pix_ind, h5_results_grp=None):
		"""
		Reads the resulting data of one of the pixels, plots out the forward and reverse resistances
			and the corrected current, and returns the plot.

		Parameters
		----------
		pix_ind			: integer
			Index of the pixel whose data we want to plot.
		h5_results_grp 	: H5 group
			Group that contains the data from the computations. Used when inspecting the generated results
			if substantial time has passed after its initial generation (i.e. when self.h5_results_grp is None).

		Returns
		-------
		figure	: pyplot.figure
			Plot of the inferred resistances and corrected current that can be saved or displayed.
		"""

		if h5_results_grp is not None:
			self.h5_results_grp = h5_results_grp

		# Need x, R, R_sig, V, i_meas, i_recon, i_corrected to graph against .h5_spec_vals[()]
		# Get these values from the results dataset that we presumably already created.
		x = usid.USIDataset(self.h5_results_grp["Resistance"]).h5_spec_vals[()][0]
		R = self.h5_results_grp["Resistance"][()][pix_ind, :]
		R_sig = self.h5_results_grp["R_sig"][()][pix_ind, :]
		V = usid.USIDataset(self.h5_results_grp["Reconstructed_Current"]).h5_spec_vals[()][0]
		i_meas = self.h5_main[()][pix_ind, ::self.parse_mod]
		i_recon = self.h5_results_grp["Reconstructed_Current"][()][pix_ind, :]
		i_corrected = self.h5_results_grp["Corrected_Current"][()][pix_ind, :]

		# Plot and return the plot of all these values.
		return publicGetGraph(self.Ns, pix_ind, self.shift_index, self.split_index, x, R, R_sig, V, i_meas, i_recon, i_corrected)

	def _create_results_datasets(self):
		"""
		Creates all the datasets necessary for holding the parameters and resulting inference data.
		"""

		## Create a group with the relevant attributes.

		self.h5_results_grp = create_results_group(self.h5_main, self.process_name)

		self.params_dict.update({'last_pixel': 0, 'algorithm': 'pycroscopy_AdaptiveBayesianInference'})

		# Write in the attributes using the dictionary created during initialization of the class.
		write_simple_attrs(self.h5_results_grp, self.params_dict)
		assert isinstance(self.h5_results_grp, h5py.Group)

		# If we ended up parsing down the data, create new spectral datasets (i.e. smaller full_V's)
		# By convention, we convert the full_V back to a sine wave.
		if self.parse_mod != 1:
			h5_spec_inds_new, h5_spec_vals_new = write_ind_val_dsets(self.h5_results_grp, Dimension("Bias", "V", self.full_V.size), is_spectral=True)
			h5_spec_vals_new[()] = get_unshifted_response(self.full_V, self.shift_index)
		else:
			h5_spec_inds_new = self.h5_main.h5_spec_inds
			h5_spec_vals_new = self.h5_main.h5_spec_vals

		# Make some new spectroscopic datasets for R and R_sig
		h5_spec_inds_R, h5_spec_vals_R = write_ind_val_dsets(self.h5_results_grp, Dimension("Bias", "V", 2*self.M), is_spectral=True, base_name="Spectroscopic_R")
		h5_spec_vals_R[()] = np.concatenate((self.x, self.x)).T

		# Initialize our datasets
		# Note by convention, the spectroscopic values are stored as a sine wave
		# so i_recon and i_corrected are shifted at the end of bayesian_utils.process_pixel
		# accordingly.
		self.h5_R = write_main_dataset(self.h5_results_grp, (self.h5_main.shape[0], 2*self.M), "Resistance", "Resistance",
									   "GOhms", None, None, dtype=np.float64,
									   h5_pos_inds=self.h5_main.h5_pos_inds,
									   h5_pos_vals=self.h5_main.h5_pos_vals,
									   h5_spec_inds=h5_spec_inds_R,
									   h5_spec_vals=h5_spec_vals_R)

		assert isinstance(self.h5_R, usid.USIDataset) # Quick sanity check
		self.h5_R_sig = create_empty_dataset(self.h5_R, np.float64, "R_sig")

		self.h5_capacitance = write_main_dataset(self.h5_results_grp, (self.h5_main.shape[0], 2), "Capacitance", "Capacitance",
												 "pF", None, Dimension("Direction", "", 2),
												 h5_pos_inds=self.h5_main.h5_pos_inds,
												 h5_pos_vals=self.h5_main.h5_pos_vals,
												 dtype=np.float64, aux_spec_prefix="Cap_Spec_")

		self.h5_i_recon = write_main_dataset(self.h5_results_grp, (self.h5_main.shape[0], self.full_V.size), "Reconstructed_Current", "Current",
											 "nA", None, None, dtype=np.float64,
											 h5_pos_inds=self.h5_main.h5_pos_inds,
											 h5_pos_vals=self.h5_main.h5_pos_vals,
											 h5_spec_inds=h5_spec_inds_new,
											 h5_spec_vals=h5_spec_vals_new)
		self.h5_i_corrected = create_empty_dataset(self.h5_i_recon, np.float64, "Corrected_Current")

		if self.verbose: print("Results datasets created.")
		self.h5_main.file.flush()

	def _get_existing_datasets(self):
		"""
		Extracts references to the existing datasets that hold the results
		"""
		
		self.h5_R = self.h5_results_grp["Resistance"]
		self.h5_R_sig = self.h5_results_grp["R_sig"]
		self.h5_capacitance = self.h5_results_grp["Capacitance"]
		self.h5_i_recon = self.h5_results_grp["Reconstructed_Current"]
		self.h5_i_corrected = self.h5_results_grp["Corrected_Current"]

	def _write_results_chunk(self, pos_in_batch=None):
		"""
		Writes data chunks back to the file

		Parameter
		---------
		pos_in_batch	: list-type object
			Holds the list of pixel indices that we have just run Bayesian inference and
				to store the results of.
		"""

		# Get the pixels that we just ran the Bayesian inference on.
		if pos_in_batch is None:
			pos_in_batch = self._get_pixels_in_current_batch()

		# Write the results generated by _unit_computation() into the corresponding datasets
		self.h5_R[pos_in_batch, :] = self.R
		self.h5_R_sig[pos_in_batch, :] = self.R_sig
		self.h5_capacitance[pos_in_batch, :] = self.capacitance
		self.h5_i_recon[pos_in_batch, :] = self.i_recon
		self.h5_i_corrected[pos_in_batch, :] = self.i_corrected

		if self.verbose: print("Results written back to file.")

		# Note: the Process class handles checkpointing.

	def _unit_computation(self, pos_in_batch=None, *args, **kwargs):
		"""
		Processes a chunk of pixels (as given by pos_in_batch) in parallel and stores the results in local variables.

		Parameters
		----------
		pos_in_batch	: list-type object
			Holds the list of pixel indices that we want to run Bayesian inference on.
		args 			: list
			Not used
		kwargs 			: dictionary
			Not used
		"""

		# Grab the chunk of data that we want to process from the h5_main dataset
		if pos_in_batch is not None:
			self.data = self.h5_main[()][pos_in_batch, :]

		# Shift and parse the data for each pixel so that we get our forward and reverse responses.
		shifted_i_meas = parallel_compute(self.data[:, ::self.parse_mod], get_shifted_response, cores=self._cores, func_args=[self.shift_index])

		# Run Bayesian inference on each pixel.
		all_data = parallel_compute(np.array(shifted_i_meas), process_pixel, cores=self._cores,
									func_args=[self.full_V, self.split_index, self.M, self.dx, self.x, self.shift_index, self.f, self.V0, self.Ns, self.dvdt])

		# Since process_pixel returns a tuple, parse the list of tuples into individual lists
		# Note, results are (R, R_sig, capacitance, i_recon, i_corrected)
		# and R, R_sig, i_recon, and i_corrected are column vectors, which are funky to work with
		self.R = np.array([result[0].T[0] for result in all_data]).astype(np.float64)
		self.R_sig = np.array([result[1].T[0] for result in all_data]).astype(np.float64)
		self.capacitance = np.array([result[2] for result in all_data]).astype(np.float64)
		self.i_recon = np.array([result[3].T[0] for result in all_data]).astype(np.float64)
		self.i_corrected = np.array([result[4].T[0] for result in all_data]).astype(np.float64)

		if self.verbose: print("Finished computing chunk.")





















