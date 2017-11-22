"""
=================================================================
Tutorial 4: Parallel Computing
=================================================================

**Suhas Somnath, Chris R. Smith**

9/8/2017


This set of tutorials will serve as examples for developing end-to-end workflows for and using pycroscopy.

**In this example, we will learn how to compute using all available cores on a computer.** Note, that this is
applicable only for a single CPU. Please refer to another advanced example for multi-CPU computing.

Quite often, we need to perform the same operation on every single component in our data. One of the most popular
examples is functional fitting applied to spectra collected at each location on a grid. While, the operation itself
may not take very long, computing this operation thousands of times, once per location, using a single CPU core can
take a long time to complete. Most personal computers today come with at least two cores, and in many cases, each of
these cores is represented via two logical cores, thereby summing to a total of at least four cores. Thus, it is
prudent to make use of these unused cores whenever possible. Fortunately, there are a few python packages that
facilitate the efficient use of all CPU cores with minimal modifications to the existing code.

**Here, we show how one can fit the thousands of spectra, one at each location, using multiple CPU cores.**

"""

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

# The package used for creating and manipulating HDF5 files:
import h5py

# Packages for plotting:
import matplotlib.pyplot as plt

# Parallel computation library:
try:
    import joblib
except ImportError:
    warn('joblib not found.  Will install with pip.')
    import pip
    pip.main(['install', 'joblib'])
    import joblib

# Timing
import time

# Finally import pycroscopy for certain scientific analysis:
try:
    import pycroscopy as px
except ImportError:
    warn('pycroscopy not found.  Will install with pip.')
    import pip
    pip.main(['install', 'pycroscopy'])
    import pycroscopy as px

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
_ = wget.download(url, h5_path)

#########################################################################

# Open the file in read-only mode
h5_file = h5py.File(h5_path, mode='r')

# Get handles to the the raw data along with other datasets and datagroups that contain necessary parameters
h5_meas_grp = h5_file['Measurement_000']

# Getting a reference to the main dataset:
h5_main = h5_meas_grp['Channel_000/Raw_Data']
print('\nThe main dataset:\n------------------------------------')
print(h5_main)

num_rows = px.hdf_utils.get_attr(h5_meas_grp, 'grid_num_rows')
num_cols = px.hdf_utils.get_attr(h5_meas_grp, 'grid_num_cols')

# Extracting the X axis - vector of frequencies
h5_spec_vals = px.hdf_utils.getAuxData(h5_main, 'Spectroscopic_Values')[-1]
freq_vec = np.squeeze(h5_spec_vals.value) * 1E-3

#########################################################################
# Visualize the data
# ==================
#
# Visualize the spectra at each of the locations using the interactive jupyter widgets below:

px.viz.be_viz_utils.jupyter_visualize_be_spectrograms(h5_main)


#########################################################################
# The operation
# =============
# We will be computing the parameters that would best describe these complex-valued spectra using a simple harmonic
# oscillator model in the functions below. These functions have been taken from the BESHOFit submodule available in
# pycroscopy.analysis.
#
# The specifics of the functions are not of interest for this example. Instead, all we need to know
# is that we need to apply a function (SHOestimateGuess in our case) on each element in our dataset.
# the functions below
#
# .. code-block:: python
#
#     def SHOfunc(parms, w_vec):
#         """
#         Generates the SHO response over the given frequency band
#
#         Parameters
#         -----------
#         parms : list or tuple
#             SHO parameters=(Amplitude, frequency ,Quality factor, phase)
#         w_vec : 1D numpy array
#             Vector of frequency values
#         """
#         return parms[0] * exp(1j * parms[3]) * parms[1] ** 2 /         (w_vec ** 2 - 1j * w_vec * parms[1] / parms[2] - parms[1] ** 2)
#
#
#     def SHOestimateGuess(resp_vec, w_vec=None, num_points=5):
#         """
#         Generates good initial guesses for fitting
#
#         Parameters
#         ------------
#         resp_vec : 1D complex numpy array or list
#             BE response vector as a function of frequency
#         w_vec : 1D numpy array or list, Optional
#             Vector of BE frequencies
#         num_points : (Optional) unsigned int
#             Quality factor of the SHO peak
#
#         Returns
#         ---------
#         retval : tuple
#             SHO fit parameters arranged as amplitude, frequency, quality factor, phase
#         """
#         if w_vec is None:
#             # Some default value
#             w_vec = np.linspace(300E+3, 350E+3, resp_vec.size)
#
#         ii = np.argsort(abs(resp_vec))[::-1]
#
#         a_mat = np.array([])
#         e_vec = np.array([])
#
#         for c1 in range(num_points):
#             for c2 in range(c1 + 1, num_points):
#                 w1 = w_vec[ii[c1]]
#                 w2 = w_vec[ii[c2]]
#                 X1 = real(resp_vec[ii[c1]])
#                 X2 = real(resp_vec[ii[c2]])
#                 Y1 = imag(resp_vec[ii[c1]])
#                 Y2 = imag(resp_vec[ii[c2]])
#
#                 denom = (w1 * (X1 ** 2 - X1 * X2 + Y1 * (Y1 - Y2)) + w2 * (-X1 * X2 + X2 ** 2 - Y1 * Y2 + Y2 ** 2))
#                 if denom > 0:
#                     a = ((w1 ** 2 - w2 ** 2) * (w1 * X2 * (X1 ** 2 + Y1 ** 2) - w2 * X1 * (X2 ** 2 + Y2 ** 2))) / denom
#                     b = ((w1 ** 2 - w2 ** 2) * (w1 * Y2 * (X1 ** 2 + Y1 ** 2) - w2 * Y1 * (X2 ** 2 + Y2 ** 2))) / denom
#                     c = ((w1 ** 2 - w2 ** 2) * (X2 * Y1 - X1 * Y2)) / denom
#                     d = (w1 ** 3 * (X1 ** 2 + Y1 ** 2) -
#                          w1 ** 2 * w2 * (X1 * X2 + Y1 * Y2) -
#                          w1 * w2 ** 2 * (X1 * X2 + Y1 * Y2) +
#                          w2 ** 3 * (X2 ** 2 + Y2 ** 2)) / denom
#
#                     if d > 0:
#                         a_mat = append(a_mat, [a, b, c, d])
#
#                         A_fit = abs(a + 1j * b) / d
#                         w0_fit = sqrt(d)
#                         Q_fit = -sqrt(d) / c
#                         phi_fit = arctan2(-b, -a)
#
#                         H_fit = A_fit * w0_fit ** 2 * exp(1j * phi_fit) / (
#                             w_vec ** 2 - 1j * w_vec * w0_fit / Q_fit - w0_fit ** 2)
#
#                         e_vec = append(e_vec,
#                                        sum((real(H_fit) - real(resp_vec)) ** 2) +
#                                        sum((imag(H_fit) - imag(resp_vec)) ** 2))
#         if a_mat.size > 0:
#             a_mat = a_mat.reshape(-1, 4)
#
#             weight_vec = (1 / e_vec) ** 4
#             w_sum = sum(weight_vec)
#
#             a_w = sum(weight_vec * a_mat[:, 0]) / w_sum
#             b_w = sum(weight_vec * a_mat[:, 1]) / w_sum
#             c_w = sum(weight_vec * a_mat[:, 2]) / w_sum
#             d_w = sum(weight_vec * a_mat[:, 3]) / w_sum
#
#             A_fit = abs(a_w + 1j * b_w) / d_w
#             w0_fit = sqrt(d_w)
#             Q_fit = -sqrt(d_w) / c_w
#             phi_fit = np.arctan2(-b_w, -a_w)
#
#             H_fit = A_fit * w0_fit ** 2 * exp(1j * phi_fit) / (w_vec ** 2 - 1j * w_vec * w0_fit / Q_fit - w0_fit ** 2)
#
#             if np.std(abs(resp_vec)) / np.std(abs(resp_vec - H_fit)) < 1.2 or w0_fit < np.min(w_vec) or w0_fit > np.max(
#                     w_vec):
#                 p0 = sho_fast_guess(w_vec, resp_vec)
#             else:
#                 p0 = np.array([A_fit, w0_fit, Q_fit, phi_fit])
#         else:
#             p0 = sho_fast_guess(resp_vec, w_vec)
#
#         return p0
#
#
#     def sho_fast_guess(resp_vec, w_vec, qual_factor=200):
#         """
#         Default SHO guess from the maximum value of the response
#
#         Parameters
#         ------------
#         resp_vec : 1D complex numpy array or list
#             BE response vector as a function of frequency
#         w_vec : 1D numpy array or list
#             Vector of BE frequencies
#         qual_factor : float
#             Quality factor of the SHO peak
#
#         Returns
#         -------
#         retval : 1D numpy array
#             SHO fit parameters arranged as [amplitude, frequency, quality factor, phase]
#         """
#         amp_vec = abs(resp_vec)
#         i_max = int(len(resp_vec) / 2)
#         return np.array([np.mean(amp_vec) / qual_factor, w_vec[i_max], qual_factor, np.angle(resp_vec[i_max])])

#########################################################################
# Testing the function
# ====================
# Let's see what the operation on an example spectra returns. The function essentially returns four parameters that can
# capture the the shape of the spectra.
# 
# A single call to the function does not take substantial time. However, performing the same operation on each of the
# 16,384 pixels can take substantial time

row_ind, col_ind = 103, 19
resp_vec = h5_main[col_ind + row_ind*num_cols]
norm_guess_parms = px.analysis.be_sho.SHOestimateGuess(resp_vec, freq_vec)
print('Functional fit returned:', norm_guess_parms)
norm_resp = px.analysis.be_sho.SHOfunc(norm_guess_parms, freq_vec)


fig, axes = plt.subplots(ncols=2, figsize=(10, 5))
for axis, func, title in zip(axes.flat, [np.abs, np.angle], ['Amplitude (a.u.)', 'Phase (rad)']):
    axis.scatter(freq_vec, func(resp_vec), c='red', label='Measured')
    axis.plot(freq_vec, func(norm_resp), 'black', lw=3, label='Guess')
    axis.set_title(title, fontsize=16)
    axis.legend(fontsize=14)
    axis.set_xlabel('Frequency (kHz)', fontsize=14)

axes[0].set_ylim([0, np.max(np.abs(resp_vec))*1.1])
axes[1].set_ylim([-np.pi, np.pi])

#########################################################################
# Applying the function to the entire dataset
# ===========================================
#
# We will be comparing the:
# 1. Traditional - serial computation approach
# 2. Parallel computation
# 
# In an effort to avoid reading / writing to the data files, we will read the entire dataset to memory.

raw_data = h5_main[()]

serial_results = np.zeros((raw_data.shape[0], 4), dtype=np.float)

#########################################################################
# 1. Serial Computation
# ---------------------
# The simplest method to compute the paramters for each spectra in the dataset is by looping over each position using
# a simple for loop.
t_0 = time.time()
for pix_ind in range(raw_data.shape[0]):
    serial_results[pix_ind] = px.analysis.be_sho.SHOestimateGuess(raw_data[pix_ind], freq_vec)
print('Serial computation took', np.round(time.time()-t_0, 2), ' seconds')

#########################################################################
# 2. Parallel Computation
# -----------------------
#
# There are several libraries that can utilize multiple CPU cores to perform the same computation in parallel. Popular
# examples are **Multiprocessing**, **Mutiprocess**, **Dask**, **Joblib** etc. Each of these has their own
# strengths and weaknesses. An installation of **Anaconda** comes with **Multiprocessing** by default and could be
# the example of choice. However, in our experience we found **Joblib** to offer the best balance of efficiency,
# simplicity, portabiity, and ease of installation.
# 
# For illustrative purposes, we will only be demonstrating how the above serial computation can be made parallel using
# **Joblib**. We only need two lines to perform the parallel computation. The first line sets up the computational
# jobs while the second performs the computation.
# 
# Note that the first argument to the function **MUST** be the data vector itself. The other arguments (parameters),
# such as the frequency vector in this case, must come after the data argument. This approach allows the specification
# of both required arguments and optional (keyword) arguments.
#
# Parallel computing has been made more accessible via the parallel_compute() function in the `process` module in
# pycroscopy. The below parallel computation is reduced to a single line with this function.

func = px.analysis.be_sho.SHOestimateGuess
cores = 4
args = freq_vec

t_0 = time.time()
values = [joblib.delayed(func)(x, args) for x in raw_data]
parallel_results = joblib.Parallel(n_jobs=cores)(values)
print('Parallel computation took', np.round(time.time()-t_0, 2), ' seconds')

#########################################################################
# Compare the results
# -------------------
#
# By comparing the run-times for the two approaches, we see that the parallel computation is substantially faster than
# the serial computation. Note that the numbers will differ between computers. Also, the computation was performed on
# a relatively small dataset for illustrative purposes. The benefits of using such parallel computation will be far
# more apparent for much larger datasets.
#
# Let's compare the results from both the serial and parallel methods to ensure they give the same results:

row_ind, col_ind = 103, 19
pix_ind = col_ind + row_ind * num_cols
print('Parallel and serial computation results matching:',
      np.all(np.isclose(serial_results[pix_ind], parallel_results[pix_ind])))

#########################################################################
# Best practices for parallel computing
# =====================================
#
# While it may seem tempting to do everything in parallel, it is important to be aware of some of the trade-offs and
# best-practices for parallel computing (multiple CPU cores) when compared to traditional serial computing (single
# CPU core):
# * There is noticable time overhead involved with setting up each parallel computing job. For very simple or small
# computations, this overhead may outweigh the speed-up gained with using multiple cores.
# * Parallelizing computations that read and write to files at each iteration may be actually be noticably __slower__
# than serial computation since each core will compete with all other cores for rights to read and write to the file(s)
# and these input/output operations are by far the slowest components of the computation. Instead, it makes sense to
# read large amounts of data from the necessary files once, perform the computation, and then write to the files once
# after all the computation is complete. In fact, this is what we automatically do in the **Analysis** and
# **Process** class in **pycroscopy**
#
# Process class - Formalizing data processing
# -------------------------------------------
#
# Data processing / analysis typically involves a few basic operations:
# 1. Reading data from file
# 2. Parallel computation
# 3. Writing results to disk
#
# The Process class in pycroscopy aims to modularize these operations for faster development of standardized,
# easy-to-debug code. Common operations can be inherited from this class and only the operation-specific functions
# need to be extended in your class.
# Please see another example on how to write a Process class for Pycroscopy


#########################################################################
# **Delete the temporarily downloaded file**


h5_file.close()
os.remove(h5_path)
