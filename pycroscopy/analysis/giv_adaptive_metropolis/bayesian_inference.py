# -*- coding: utf-8 -*-
"""
Runs a full Bayesian inference algorithm on spectroscopic data for ultrafast current imaging (see paper below)
https://www.nature.com/articles/s41467-017-02455-7

Created on Tue July 02, 2019

@author: Alvin Tan, Emily Costa

"""

from __future__ import division, print_function, absolute_import, unicode_literals
import h5py
import numpy as np
from pyUSID.processing.process import Process
# Set up parallel compute later to run through supercomputer or cluster
from pyUSID.processing.comp_utils import parallel_compute
from pyUSID.io.hdf_utils import create_results_group, write_main_dataset, write_simple_attrs, create_empty_dataset, \
	write_ind_val_dsets
from pyUSID.io.write_utils import Dimension

import time
import math
import scipy.linalg as spla 
import pycroscopy as px 
import pyUSID as usid

from bayesian_utils import get_shift_and_split_indices, process_pixel, get_shifted_response, get_unshifted_response, get_M_dx_x, publicGetGraph


class AdaptiveBayesianInference(Process):
	def __init__(self, h5_main, f=200, V0=None, Ns=int(7e7), M=101, parse_mod=1, **kwargs):
		"""
		Bayesian inference is done on h5py dataset object that has already been filtered
		and reshaped.
		----------
		h5_main : h5py.Dataset object
			Dataset to process
		kwargs : (Optional) dictionary
			Please see Process class for additional inputs
		"""
		super(AdaptiveBayesianInference, self).__init__(h5_main, **kwargs)

		#now make sure all parameters were inputted correctly
		# Ex. if frequency_filters is None and noise_threshold is None:
		#    raise ValueError('Need to specify at least some noise thresholding / frequency filter')
		# This determines how to parse down the data (i.e. used_data = actual_data[::parse_mod]).
		# If parse_mod == 1, then we use the entire dataset. We may be able to input this as an
		# argument, but for now this is just to improve maintainability.
		self.parse_mod = parse_mod
		if self.verbose: print("parsed data")
		# Now do some setting of the variables
		# Ex. self.frequency_filters = frequency_filters
		#breakpoint()
		ex_wave = self.h5_main.h5_spec_vals[()]
		dt = 1.0/(f*ex_wave.size)
		self.dvdt = np.diff(ex_wave)/dt
		self.dvdt = np.append(self.dvdt, self.dvdt[0][-1])
		self.dvdt = self.dvdt[::self.parse_mod][np.newaxis].T
		self.full_V = ex_wave[0][::self.parse_mod]
		if self.verbose: print("V set up")
		# Name the process
		# Ex. self.process_name = 'FFT_Filtering'
		self.process_name = 'Adaptive_Bayesian'

		# Honestly no idea what this line does
		# This line checks to make sure this data has not already been processed
		self.duplicate_h5_groups, self.partial_h5_groups = self._check_for_duplicates()

		self.data = None
		# Add other datasets needs
		# Ex. self.filtered_data = None

		# A couple constants and vectors we will be using
		self.full_V, self.shift_index, self.split_index = get_shift_and_split_indices(self.full_V)
		self.dvdt = get_shifted_response(self.dvdt, self.shift_index)
		if V0 is None:
			V0 = max(self.full_V)
		self.M, self.dx, self.x = get_M_dx_x(V0=V0, M=M)
		self.V0 = V0
		self.f = f
		self.Ns = Ns
		if self.verbose: print("data and variables set up")
		# These will be the results from the processed chunks
		self.R = None
		self.R_sig = None
		self.capacitance = None
		self.i_recon = None
		self.i_corrected = None
	
		# These are the actual databases
		self.h5_R = None
		self.h5_R_sig = None
		self.h5_capacitance = None
		self.h5_i_recon = None
		self.h5_i_corrected = None
		if self.verbose: print("empty results set up")

		# Make full_V and num_pixels attributes
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
		if self.verbose: print("attributes set up")

		#print("Ran init. self.params_dict is {}".format(self.params_dict))

	def test(self, pix_ind=None):
		"""
		Tests the Bayesian inference on a single pixel (randomly chosen unless manually specified) worth of data.
		Displays the resulting figure (resistances with variance, and reconstructed current)
			before returning the same figure.
		Parameters
		----------
		pix_ind : int, optional. default = random
			Index of the pixel whose data will be used for inference
		Returns
		-------
		fig, axes
		"""
		if self.mpi_rank > 0:
			return
		if pix_ind is None:
			pix_ind = np.random.randint(1, high=self.h5_main.shape[0])

		full_i_meas = get_shifted_response(self.h5_main[pix_ind, ::self.parse_mod], self.shift_index)

		# Return from test function you built seperately (see gmode_utils.test_filter for example)
		return pix_ind, process_pixel(full_i_meas, self.full_V, self.split_index, self.M, self.dx, self.x, self.shift_index, self.f, self.V0, self.Ns, self.dvdt, pix_ind=pix_ind, graph=True, verbose=True)

	def plotPixel(self, pix_ind=None):
		if pix_ind is None:
			return None

		# Need x, R, R_sig, V, i_meas, i_recon, i_corrected to graph against .h5_spec_vals[()]
		x = usid.USIDataset(self.h5_results_grp["Resistance"]).h5_spec_vals[()][0]
		R = self.h5_results_grp["Resistance"][()][pix_ind, :]
		R_sig = self.h5_results_grp["R_sig"][()][pix_ind, :]
		V = usid.USIDataset(self.h5_results_grp["Reconstructed_Current"]).h5_spec_vals[()][0]
		i_meas = self.h5_main[()][pix_ind, ::self.parse_mod]
		i_recon = self.h5_results_grp["Reconstructed_Current"][()][pix_ind, :]
		i_corrected = self.h5_results_grp["Corrected_Current"][()][pix_ind, :]

		return publicGetGraph(self.Ns, pix_ind, self.shift_index, self.split_index, x, R, R_sig, V, i_meas, i_recon, i_corrected)


	def _create_results_datasets(self):
		"""
		Creates all the datasets necessary for holding all parameters + data.
		"""

		self.h5_results_grp = create_results_group(self.h5_main, self.process_name)

		self.params_dict.update({'last_pixel': 0, 'algorithm': 'pycroscopy_AdaptiveBayesianInference'})

		# Write in our full_V and num_pixels as attributes to this new group
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

		# Also make some new spectroscopic datasets for R and R_sig
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

		# Not sure what units this should be so tentatively storing it as amps
		self.h5_i_recon = write_main_dataset(self.h5_results_grp, (self.h5_main.shape[0], self.full_V.size), "Reconstructed_Current", "Current",
											 "nA", None, None, dtype=np.float64,
											 h5_pos_inds=self.h5_main.h5_pos_inds,
											 h5_pos_vals=self.h5_main.h5_pos_vals,
											 h5_spec_inds=h5_spec_inds_new,
											 h5_spec_vals=h5_spec_vals_new)
		self.h5_i_corrected = create_empty_dataset(self.h5_i_recon, np.float64, "Corrected_Current")

		'''
		# Initialize our datasets
		# Note, each pixel of the datasets will hold the forward and reverse sweeps concatenated together.
		# R and R_sig are plotted against [x, -x], and i_recon and i_corrected are plotted against full_V.
		self.h5_R = h5_results_grp.create_dataset("R", shape=(self.h5_main.shape[0], 2*self.M), dtype=np.float)
		self.h5_R_sig = h5_results_grp.create_dataset("R_sig", shape=(self.h5_main.shape[0], 2*self.M), dtype=np.float)
		self.h5_capacitance = h5_results_grp.create_dataset("capacitance", shape=(self.h5_main.shape[0], 2), dtype=np.float)
		self.h5_i_recon = h5_results_grp.create_dataset("i_recon", shape=(self.h5_main.shape[0], self.full_V.size), dtype=np.float)
		self.h5_i_corrected = h5_results_grp.create_dataset("i_corrected", shape=(self.h5_main.shape[0], self.full_V.size), dtype=np.float)
		'''

		if self.verbose: print("results datasets set up")
		self.h5_main.file.flush()

	def _get_existing_datasets(self):
		"""
		Extracts references to the existing datasets that hold the results
		"""
		# Do not worry to much about this now 
		self.h5_R = self.h5_results_grp["R"]
		self.h5_R_sig = self.h5_results_grp["R_sig"]
		self.h5_capacitance = self.h5_results_grp["capacitance"] * 1000 # convert from nF to pF
		self.h5_i_recon = self.h5_results_grp["i_recon"]
		self.h5_i_corrected = self.h5_results_grp["i_corrected"]

	def _write_results_chunk(self, pos_in_batch=None):
		"""
		Writes data chunks back to the file
		"""
		# Get access to the private variable:
		if pos_in_batch is None:
			pos_in_batch = self._get_pixels_in_current_batch()

		# Ex. if self.write_filtered:
		#    self.h5_filtered[pos_in_batch, :] = self.filtered_data
		self.h5_R[pos_in_batch, :] = self.R
		self.h5_R_sig[pos_in_batch, :] = self.R_sig
		self.h5_capacitance[pos_in_batch, :] = self.capacitance
		self.h5_i_recon[pos_in_batch, :] = self.i_recon
		self.h5_i_corrected[pos_in_batch, :] = self.i_corrected
		if self.verbose: print("results written back to file")
		# Process class handles checkpointing.

	def _unit_computation(self, pos_in_batch=None, *args, **kwargs):
		"""
		Processing per chunk of the dataset
		Parameters
		----------
		args : list
			Not used
		kwargs : dictionary
			Not used
		"""

		if pos_in_batch is not None:
			self.data = self.h5_main[()][pos_in_batch, :]

		# This is where you add the Matlab code you translated. You can add stuff to other Python files in processing, 
		#like gmode_utils or comp_utils, depending on what your function does. Just add core code here.
		shifted_i_meas = parallel_compute(self.data[:, ::self.parse_mod], get_shifted_response, cores=self._cores, func_args=[self.shift_index])

		#print("Shifted_i_meas is {}".format(shifted_i_meas))

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
		#print("capacitances are {} with shape {}".format(self.capacitance, self.capacitance.shape))
		if self.verbose: print("chunk computed")





















