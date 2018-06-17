"""
================================================================================
7. Speed up computations with parallel_compute()
================================================================================

**Suhas Somnath, Chris R. Smith**

9/8/2017

**This document will demonstrate how ``pycroscopy.parallel_compute()`` can significantly speed up data processing by
using all available CPU cores in a computer**
"""
########################################################################################################################
# Introduction
# -------------
# Quite often, we need to perform the same operation on every single component in our data. One of the most popular
# examples is functional fitting applied to spectra collected at each location on a grid. While, the operation itself
# may not take very long, computing this operation thousands of times, once per location, using a single CPU core can
# take a long time to complete. Most personal computers today come with at least two cores, and in many cases, each of
# these cores is represented via two logical cores, thereby summing to a total of at least four cores. Thus, it is
# prudent to make use of these unused cores whenever possible. Fortunately, there are a few python packages that
# facilitate the efficient use of all CPU cores with minimal modifications to the existing code.
#
# ``pycroscopy.parallel_compute()`` is a very handy function that simplifies parallel computation significantly to a
# ``single function call`` and will be discussed in this document.
#
# Example scientific problem
# ---------------------------
# For this example, we will be working with a ``Band Excitation Piezoresponse Force Microscopy (BE-PFM)`` imaging dataset
# acquired from advanced atomic force microscopes. In this dataset, a spectra was collected for each position in a two
# dimensional grid of spatial locations. Thus, this is a three dimensional dataset that has been flattened to a two
# dimensional matrix in accordance with the pycroscopy data format.
#
# Each spectra in this dataset is expected to have a single peak. The goal is to find the positions of the peaks in each
# spectra. Clearly, the operation of finding the peak in one spectra is independent of the same operation on another
# spectra. Thus, we could in theory divide the dataset in to N parts and use N CPU cores to compute the results much
# faster than it would take a single core to compute the results. There is an important caveat to this statement and it
# will be discussed at the end of this document.
#
# ``Here, we will learn how to fit the thousands of spectra using all available cores on a computer.``
# Note, that this is applicable only for a single CPU. Please refer to another advanced example for multi-CPU computing.

# Ensure python 3 compatibility:
from __future__ import division, print_function, absolute_import, unicode_literals

# The package for accessing files in directories, etc.:
import os

# Warning package in case something goes wrong
from warnings import warn
import subprocess
import sys

def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])
# Package for downloading online files:
try:
    # This package is not part of anaconda and may need to be installed.
    import wget
except ImportError:
    warn('wget not found.  Will install with pip.')
    import pip
    install(wget)
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
    install('joblib')
    import joblib

# Timing
import time

# A handy python utility that allows us to preconfigure parts of a function
from functools import partial

# Function for reading the number of CPU cores on this computer:
from multiprocessing import cpu_count

# Finally import pycroscopy for certain scientific analysis:
try:
    import pycroscopy as px
except ImportError:
    warn('pycroscopy not found.  Will install with pip.')
    import pip
    install('pycroscopy')
    import pycroscopy as px

########################################################################################################################
# Load the dataset
# --------------------
# In order to demonstrate parallel computing, we will be using a real experimental dataset that is available on the
# pycroscopy GitHub project. First, lets download this file from Github:

h5_path = 'temp.h5'
url = 'https://raw.githubusercontent.com/pycroscopy/pycroscopy/master/data/BELine_0004.h5'
if os.path.exists(h5_path):
    os.remove(h5_path)
_ = wget.download(url, h5_path, bar=None)

########################################################################################################################
# Now, lets open this HDF5 file. The focus of this example is not on the data storage or arrangement but rather on
# demonstrating parallel computation so lets dive straight into the main dataset that requires fitting of the spectra:

# Open the file in read-only mode
h5_file = h5py.File(h5_path, mode='r')
# Get handle to the the raw data
h5_meas_grp = h5_file['Measurement_000']

# Accessing the dataset of interest:
h5_main = px.PycroDataset(h5_meas_grp['Channel_000/Raw_Data'])
print('\nThe main dataset:\n------------------------------------')
print(h5_main)

num_rows, num_cols = h5_main.pos_dim_sizes
cores_vec = list()
times_vec = list()

########################################################################################################################
# The operation
# -------------
# Pycroscopy already has a function called wavelet_peaks() that facilitates the seach for one or more peaks in a spectra
# located in pycroscopy.analysis.guess_methods. For the purposes of this example, we do not be concerned with how this
# function works. All we need to know is that this function takes 3 inputs:
#
# * ``vector`` - a 1D array containing the spectra at a single location
# * ``peak_widths`` - something like [20, 50] that instructs the function to look for peaks that are 20-50 units wide.
#   The function will look for a peak with width of 20, then again for a peak of width - 21 and so on.
# * ``peak_step`` - The number of steps within the possiple widths [20, 50], that the search must be performed
#
# The function has one output:
#
# * ``peak_indices`` - an array of the positions at which peaks were found.
#
# .. code-block:: python
#
#     @staticmethod
#     def wavelet_peaks(vector, *args, **kwargs):
#         """
#         This is the function that will be mapped by multiprocess. This is a wrapper around the scipy function.
#         It uses a parameter - wavelet_widths that is configured outside this function.
#
#         Parameters
#         ----------
#         vector : 1D numpy array
#             Feature vector containing peaks
#
#         Returns
#         -------
#         peak_indices : list
#             List of indices of peaks within the prescribed peak widths
#         """
#         try:
#             peak_width_bounds = kwargs.get('peak_widths')
#             kwargs.pop('peak_widths')
#             peak_width_step = kwargs.get('peak_step', 20)
#             kwargs.pop('peak_step')
#             # The below numpy array is used to configure the returned function wpeaks
#             wavelet_widths = np.linspace(peak_width_bounds[0], peak_width_bounds[1], peak_width_step)
#
#             peak_indices = find_peaks_cwt(np.abs(vector), wavelet_widths, **kwargs)
#
#             return peak_indices
#
#         except KeyError:
#             warn('Error: Please specify "peak_widths" kwarg to use this method')
#
# For simplicity, lets get a shortcut to this function:


wavelet_peaks = px.analysis.guess_methods.GuessMethods.wavelet_peaks

########################################################################################################################
# Testing the function
# ----------------------
# Let’s see what the operation on an example spectra returns.

row_ind, col_ind = 103, 19
pixel_ind = col_ind + row_ind * num_cols
spectra = h5_main[pixel_ind]

peak_inds = wavelet_peaks(spectra, peak_widths=[20, 60], peak_step=30)

fig, axis = plt.subplots()
axis.scatter(np.arange(len(spectra)), np.abs(spectra), c='black')
axis.axvline(peak_inds[0], color='r', linewidth=2)
axis.set_ylim([0, 1.1 * np.max(np.abs(spectra))]);
axis.set_title('wavelet_peaks found peaks at index: {}'.format(peak_inds), fontsize=16)

########################################################################################################################
# Before we apply the function to the entire dataset, lets load the dataset to memory so that file-loading time is not a
# factor when comparing the times for serial and parallel computing times:

raw_data = h5_main[()]

########################################################################################################################
# Serial computing
# ----------------
# A single call to the function does not take substantial time. However, performing the same operation on each of the
# 16,384 pixels sequentially can take substantial time. The simplest way to find all peak positions is to simply loop
# over each position in the dataset:

serial_results = list()

t_0 = time.time()
for ind, vector in enumerate(raw_data):
    serial_results.append(wavelet_peaks(vector, peak_widths=[20, 60], peak_step=30))
times_vec.append(time.time()-t_0)
cores_vec.append(1)
print('Serial computation took', np.round(times_vec[-1], 2), ' seconds')

########################################################################################################################
# pycroscopy.parallel_compute()
# -----------------------------
#
# There are several libraries that can utilize multiple CPU cores to perform the same computation in parallel. Popular
# examples are ``Multiprocessing``, ``Mutiprocess``, ``Dask``, ``Joblib`` etc. Each of these has their own
# strengths and weaknesses. Some of them have painful caveats such as the inability to perform the parallel computation
# within a jupyter notebook. In order to lower the barrier to parallel computation, we have developed a very handy
# function called ``pycroscopy.parallel_compute()`` that simplifies the process to a single function call.
#
# It is a lot **more straightforward** to provide the arguments and keyword arguments of the function that needs to be
# applied to the entire dataset. Furthermore, this function intelligently assigns the number of CPU cores for the
# parallel computation based on the size of the dataset and the computational complexity of the unit computation.
# For instance, it scales down the number of cores for small datasets if each computation is short. It also ensures that
# 1-2 cores fewer than all available cores are used by default so that the user can continue using their computer for
# other purposes while the computation runs.
#
# Lets apply this ``parallel_compute`` to this problem:

cpu_cores = 2
kwargs = {'peak_widths': [20, 60], 'peak_step': 30}

t_0 = time.time()

# Execute the parallel computation
parallel_results = px.parallel_compute(raw_data, wavelet_peaks, cores=cpu_cores, func_kwargs=kwargs)

cores_vec.append(cpu_cores)
times_vec.append(time.time()-t_0)
print('Parallel computation with {} cores took {} seconds'.format(cpu_cores, np.round(times_vec[-1], 2)))

########################################################################################################################
# Compare the results
# -------------------
# By comparing the run-times for the two approaches, we see that the parallel computation is substantially faster than
# the serial computation. Note that the numbers will differ between computers. Also, the computation was performed on
# a relatively small dataset for illustrative purposes. The benefits of using such parallel computation will be far
# more apparent for much larger datasets.
#
# Let's compare the results from both the serial and parallel methods to ensure they give the same results:

print('Result from serial computation: {}'.format(serial_results[pixel_ind]))
print('Result from parallel computation: {}'.format(parallel_results[pixel_ind]))

########################################################################################################################
# Simplifying the function
# --------------------------
# Note that the ``peak_widths`` and ``peak_step`` arguments will not be changed from one pixel to another. It would be
# great if we didn't have to keep track of these constant arguments. We can use a very handy python tool called
# ``partial()`` to do just this. Below, all we are doing is creating a new function that always passes our preferred
# values for ``peak_widths`` and ``peak_step`` arguments to wavelet_peaks. While it may seem like this is unimportant,
# it is very convenient when setting up the parallel computing:

find_peaks = partial(wavelet_peaks, peak_widths=[20, 60], peak_step=30)

########################################################################################################################
# Let's try calling our simplified function, ``find_peak()`` to make sure that it results in the same peak index for the
# aforementioned chosen spectra:

print('find_peaks found peaks at index: {}'.format(find_peaks(h5_main[pixel_ind])))

########################################################################################################################
# More cores!
# -----------
# Lets use ``find_peaks()`` instead of ``wavelet_peaks`` on the entire dataset but increase the number of cores to 3.
# Note that we do not need to specify ``func_kwargs`` anymore. Also note that this is a very simple function and the
# benefits of ``partial()`` will be greater for more complex problems.

cpu_cores = 3

t_0 = time.time()

# Execute the parallel computation
parallel_results = px.parallel_compute(raw_data, find_peaks, cores=cpu_cores)

cores_vec.append(cpu_cores)
times_vec.append(time.time()-t_0)
print('Parallel computation with {} cores took {} seconds'.format(cpu_cores, np.round(times_vec[-1], 2)))

########################################################################################################################
# Scalability
# -----------
# Now lets see how the computational time relates to the number of cores.
# Depending on your computer (and what was running on your computer along with this computation), you are likely to see
# diminishing benefits of additional cores beyond 2 cores for this specific problem in the plot below. This is because
# the dataset is relatively small and each peak-finding operation is relatively quick. The overhead of adding additional
# cores quickly outweighs the speedup in distributing the work among multiple CPU cores.

fig, axis = plt.subplots(figsize=(3.5, 3.5))
axis.scatter(cores_vec, times_vec)
axis.set_xlabel('CPU cores', fontsize=14)
axis.set_ylabel('Compute time (sec)', fontsize=14)
fig.tight_layout()

########################################################################################################################
# Best practices for parallel computing
# --------------------------------------
# While it may seem tempting to do everything in parallel, it is important to be aware of some of the trade-offs and
# best-practices for parallel computing (multiple CPU cores) when compared to traditional serial computing (single
# CPU core):
#
# * There is noticeable time overhead involved with setting up each parallel computing job. For very simple or small
#   computations, this overhead may outweigh the speed-up gained with using multiple cores.
# * Parallelizing computations that read and write to files at each iteration may be actually be noticeably *slower*
#   than serial computation since each core will compete with all other cores for rights to read and write to the file(s)
#   and these input/output operations are by far the slowest components of the computation. Instead, it makes sense to
#   read large amounts of data from the necessary files once, perform the computation, and then write to the files once
#   after all the computation is complete. In fact, this is what we automatically do in the ``Fitter`` and
#   ``Process`` classes in pycroscopy
#
########################################################################################################################
# Formalizing data processing and pycroscopy.Process
# --------------------------------------------------
# Data processing / analysis typically involves a few basic operations:
#
# 1. Reading data from file
# 2. Parallel computation
# 3. Writing results to disk
#
# The Process class in pycroscopy has modularized these operations for simpler and faster development of standardized,
# easy-to-debug code. In the case of this example, one would only need to write the wavelet_peaks() function along with
# the appropriate data reading and data writing functions. Other common operations can be inherited from
# pycroscopy.Process.
#
# Please see another example on how to write a Process class for Pycroscopy based on this example

########################################################################################################################
# Lets not forget to close and delete the temporarily downloaded file:

h5_file.close()
os.remove(h5_path)
