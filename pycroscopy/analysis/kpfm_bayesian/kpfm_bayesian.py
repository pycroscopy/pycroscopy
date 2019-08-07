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

from pyUSID.processing.process import Process

class KPFMBayesianInference(Process):
	def __init__(self, h5_main, **kwargs):
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

		# Make list of parameters to be added to the results dataset
		self.params_dict = dict()

	def test(self, pix_ind=None):
		'''
		Description
		'''
		if self.mpi_rank > 0:
			return
		if pix_ind is None:
			pix_ind = np.random.randint(self.h5_main.shape[0])

		return process_pixel() # TODO

	def _create_results_datasets(self):
		'''
		Creates all the datasets necessary for holding all parameters and data
		'''
		self.h5_results_grp = create_results_group(self.h5_main, self.process_name)

		self.params_dict.update({'last_pixel': 0, 'algorithm': 'pycroscopy_KPFMBayesianInference'})

		write_simple_attrs(self.h5_results_grp, self.params_dict)

		assert isinstance(self.h5_results_grp, h5py.Group)

		# TODO

	def _get_existing_datasets(self):
		'''
		Extracts references to the existing datasets that hold the results
		'''

		# TODO

	def _write_results_chunk(self, pos_in_batch=None):
		'''
		Writes data chunks back to the file
		'''

		if pos_in_batch is None:
			pos_in_batch = self._get_pixels_in_current_batch()

		# Write data into results datasets
		# TODO

	def _unit_computation(self, pos_in_batch=None, *args, **kwargs):
		'''
		Processing per chunk of the dataset
		'''

		if pos_in_batch is not None:
			self.data=self.h5_main[()][pos_in_batch, :]

		result = parallel_compute(self.data, process_pixel, cores=self._cores, func_args=[])

		# TODO






