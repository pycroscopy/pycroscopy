"""
=================================================================
Tutorial 5: Formalizing Data Processing
=================================================================

**Suhas Somnath**

9/8/2017


This set of tutorials will serve as examples for developing end-to-end workflows for and using pycroscopy.

**In this example, we will learn how to write a simple yet formal pycroscopy class for processing data.**

Introduction
============

Data processing / analysis typically involves a few basic tasks:
1. Reading data from file
2. Computation
3. Writing results to disk

This example is based on the parallel computing example where we fit a dataset containing spectra at each location to a
function. While the previous example focused on comparing serial and parallel computing, we will focus on the framework
that needs to be built around a computation for robust data processing. As the example will show below, the framework
essentially deals with careful file reading and writing.

The majority of the code for this example is based on the BESHOModel Class under pycroscopy.analysis
"""

#########################################################################
# Import necessary packages
# =========================
#
# Ensure python 3 compatibility:
from __future__ import division, print_function, absolute_import, unicode_literals

# The package for accessing files in directories, etc.:
import os

# Warning package in case something goes wrong
from warnings import warn

# Package for downloading online files:
try:
    # This package is not part of anaconda and may need to be installed.
    import wget
except ImportError:
    warn('wget not found.  Will install with pip.')
    import pip
    pip.main(['install', 'wget'])
    import wget

# The mathematical computation package:
import numpy as np
from numpy import exp, abs, sqrt, sum, real, imag, arctan2, append

# The package used for creating and manipulating HDF5 files:
import h5py

# Packages for plotting:
import matplotlib.pyplot as plt

# Finally import pycroscopy for certain scientific analysis:
try:
    import pycroscopy as px
except ImportError:
    warn('pycroscopy not found.  Will install with pip.')
    import pip
    pip.main(['install', 'pycroscopy'])
    import pycroscopy as px


field_names = ['Amplitude [V]', 'Frequency [Hz]', 'Quality Factor', 'Phase [rad]']
sho32 = np.dtype({'names': field_names,
                  'formats': [np.float32 for name in field_names]})

#########################################################################
# Build the class
# ===============
#
# Every process class consists of the same basic functions:
# 1. __init__ - instantiates a 'Process' object of this class after validating the inputs
# 2. _create_results_datasets - creates the HDF5 datasets and datagroups to store the results.
# 3. _unit_function - this is the operation that will per be performed on each element in the dataset.
# 4. compute - This function essentially applies the unit function to every single element in the dataset.
# 5. _write_results_chunk - writes the computed results back to the file
#
# Note that:
#
# * Only the code specific to this process needs to be implemented. However, the generic portions common to most
#   Processes will be handled by the Process class.
# * The other functions such as the sho_function, sho_fast_guess function are all specific to this process. These have
#   been inherited directly from the BE SHO model.
# * While the class appears to be large, remember that the majority of it deals with the creation of the datasets to store
#   the results and the actual function that one would have anyway regardless of serial / parallel computation of the
#   function. The additional code to turn this operation into a Pycroscopy Process is actually rather minimal. As
#   described earlier, the goal of the Process class is to modularize and compartmentalize the main sections of the code
#   in order to facilitate faster and more robust implementation of data processing algorithms.
#

class ShoGuess(px.Process):

    def __init__(self, h5_main, cores=None):
        """
        Validate the inputs and set some parameters

        Parameters
        ----------
        h5_main - dataset to compute on
        cores - Number of CPU cores to use for computation - Optional
        """
        super(ShoGuess, self).__init__(h5_main, cores)

        # find the frequency vector
        h5_spec_vals = px.hdf_utils.getAuxData(h5_main, 'Spectroscopic_Values')[-1]
        self.freq_vec = np.squeeze(h5_spec_vals.value) * 1E-3

    def _create_results_datasets(self):
        """
        Creates the datasets an datagroups necessary to store the results.
        Just as the raw data is stored in the pycroscopy format, the results also need to conform to the same
        standards. Hence, the create_datasets function can appear to be a little longer than one might expect.
        """
        h5_spec_inds = px.hdf_utils.getAuxData(self.h5_main, auxDataName=['Spectroscopic_Indices'])[0]
        h5_spec_vals = px.hdf_utils.getAuxData(self.h5_main, auxDataName=['Spectroscopic_Values'])[0]

        self.step_start_inds = np.where(h5_spec_inds[0] == 0)[0]
        self.num_udvs_steps = len(self.step_start_inds)
        
        ds_guess = px.MicroDataset('Guess', data=[],
                                             maxshape=(self.h5_main.shape[0], self.num_udvs_steps),
                                             chunking=(1, self.num_udvs_steps), dtype=sho32)

        not_freq = px.hdf_utils.get_attr(h5_spec_inds, 'labels') != 'Frequency'

        ds_sho_inds, ds_sho_vals = px.hdf_utils.buildReducedSpec(h5_spec_inds, h5_spec_vals, not_freq,
                                                                 self.step_start_inds)

        dset_name = self.h5_main.name.split('/')[-1]
        sho_grp = px.MicroDataGroup('-'.join([dset_name, 'SHO_Fit_']), self.h5_main.parent.name[1:])
        sho_grp.addChildren([ds_guess, ds_sho_inds, ds_sho_vals])
        sho_grp.attrs['SHO_guess_method'] = "pycroscopy BESHO"

        h5_sho_grp_refs = self.hdf.writeData(sho_grp)

        self.h5_guess = px.hdf_utils.getH5DsetRefs(['Guess'], h5_sho_grp_refs)[0]
        self.h5_results_grp = self.h5_guess.parent
        h5_sho_inds = px.hdf_utils.getH5DsetRefs(['Spectroscopic_Indices'],
                                                 h5_sho_grp_refs)[0]
        h5_sho_vals = px.hdf_utils.getH5DsetRefs(['Spectroscopic_Values'],
                                                 h5_sho_grp_refs)[0]

        # Reference linking before actual fitting
        px.hdf_utils.linkRefs(self.h5_guess, [h5_sho_inds, h5_sho_vals])
        # Linking ancillary position datasets:
        aux_dsets = px.hdf_utils.getAuxData(self.h5_main, auxDataName=['Position_Indices', 'Position_Values'])
        px.hdf_utils.linkRefs(self.h5_guess, aux_dsets)
        print('Finshed creating datasets')

    def compute(self, *args, **kwargs):
        """
        Apply the unit_function to the entire dataset. Here, we simply extend the existing compute function and only
        pass the parameters for the unit function. In this case, the only parameter is the frequency vector.

        Parameters
        ----------
        args
        kwargs

        Returns
        -------

        """
        return super(ShoGuess, self).compute(w_vec=self.freq_vec)

    def _write_results_chunk(self):
        """
        Write the computed results back to the H5 file
        """
        # converting from a list to a 2D numpy array
        self._results = np.array(self._results, dtype=np.float32)
        self.h5_guess[:, 0] = px.io_utils.realToCompound(self._results, sho32)

        # Now update the start position
        self._start_pos = self._end_pos
        # this should stop the computation.

    @staticmethod
    def _unit_function():

        return px.be_sho.SHOestimateGuess


#########################################################################
# Load the dataset
# ================
#
# For this example, we will be working with a Band Excitation Piezoresponse Force Microscopy (BE-PFM) imaging dataset
# acquired from advanced atomic force microscopes. In this dataset, a spectra was collected for each position in a two
# dimensional grid of spatial locations. Thus, this is a three dimensional dataset that has been flattened to a two
# dimensional matrix in accordance with the pycroscopy data format.

# download the raw data file from Github:
h5_path = 'temp.h5'
url = 'https://raw.githubusercontent.com/pycroscopy/pycroscopy/master/data/BELine_0004.h5'
if os.path.exists(h5_path):
    os.remove(h5_path)
_ = wget.download(url, h5_path, bar=None)

########################################################################

# Open the file in read-only mode
h5_file = h5py.File(h5_path, mode='r+')

# Get handles to the the raw data along with other datasets and datagroups that contain necessary parameters
h5_meas_grp = h5_file['Measurement_000']
num_rows = px.hdf_utils.get_attr(h5_meas_grp, 'grid_num_rows')
num_cols = px.hdf_utils.get_attr(h5_meas_grp, 'grid_num_cols')

# Getting a reference to the main dataset:
h5_main = h5_meas_grp['Channel_000/Raw_Data']

# Extracting the X axis - vector of frequencies
h5_spec_vals = px.hdf_utils.getAuxData(h5_main, 'Spectroscopic_Values')[-1]
freq_vec = np.squeeze(h5_spec_vals.value) * 1E-3

########################################################################
# Use the ShoGuess class, defined earlier, to calculate the four
# parameters of the complex gaussian.

fitter = ShoGuess(h5_main, cores=1)
h5_results_grp = fitter.compute()
h5_guess = h5_results_grp['Guess']

row_ind, col_ind = 103, 19
pix_ind = col_ind + row_ind * num_cols
resp_vec = h5_main[pix_ind]
norm_guess_parms = h5_guess[pix_ind]

# Converting from compound to real:
norm_guess_parms = px.io_utils.compound_to_scalar(norm_guess_parms)
print('Functional fit returned:', norm_guess_parms)
norm_resp = px.be_sho.SHOfunc(norm_guess_parms, freq_vec)

########################################################################
# Plot the Amplitude and Phase of the gaussian versus the raw data.

fig, axes = plt.subplots(nrows=2, sharex=True, figsize=(5, 10))
for axis, func, title in zip(axes.flat, [np.abs, np.angle], ['Amplitude (a.u.)', 'Phase (rad)']):
    axis.scatter(freq_vec, func(resp_vec), c='red', label='Measured')
    axis.plot(freq_vec, func(norm_resp), 'black', lw=3, label='Guess')
    axis.set_title(title, fontsize=16)
    axis.legend(fontsize=14)

axes[1].set_xlabel('Frequency (kHz)', fontsize=14)
axes[0].set_ylim([0, np.max(np.abs(resp_vec)) * 1.1])
axes[1].set_ylim([-np.pi, np.pi])

########################################################################
# **Delete the temporarily downloaded file**

h5_file.close()
os.remove(h5_path)
