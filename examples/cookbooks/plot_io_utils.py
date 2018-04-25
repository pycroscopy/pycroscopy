"""
======================================================================================
Input / Output / Computing utilities
======================================================================================

**Suhas Somnath**

8/12/2017

Introduction
------------
This is a short walk-through of useful utilities in pycroscopy.io_utils that simplify common i/o and computation tasks.
"""

from __future__ import print_function, division, unicode_literals
from multiprocessing import cpu_count
try:
    import pycroscopy as px
except ImportError:
    print('pycroscopy not found.  Will install with pip.')
    import pip
    pip.main(['install', 'pycroscopy'])
    import pycroscopy as px

########################################################################################################################
# Computation related utilities
# ============================
# recommend_cpu_cores()
# ---------------------
# Time is of the essence and every developer wants to make the best use of all available cores in a CPU for massively
# parallel computations. recommend_cpu_cores() is a popular function that looks at the number of parallel operations,
# available CPU cores, duration of each computation to recommend the number of cores that should be used for any
# computation. If the developer / user requests the use of N CPU cores, this function will validate this number against
# the number of available cores and the nature (lengthy / quick) of each computation. Unless, a suggested number of
# cores is specified, recommend_cpu_cores() will always recommend the usage of N-2 CPU cores, where N is the total
# number of logical cores (Intel uses hyperthreading) on the CPU to avoid using up all computational resources and
# preventing the computation from making the computer otherwise unusable until the computation is complete
# Here, we demonstrate this function being used in a few use cases:

print('This CPU has {} cores available'.format(cpu_count()))

########################################################################################################################
# 1. several independent computations or jobs, each taking far less than 1 second. The number of desired cores is not
# specified. The function will return 2 lesser than the total number of cores on the CPU
num_jobs = 14035
recommeded_cores = px.io_utils.recommend_cpu_cores(num_jobs, lengthy_computation=False)
print('Recommended number of CPU cores for {} independent, FAST, and parallel '
      'computations is {}\n'.format(num_jobs, recommeded_cores))

########################################################################################################################
# 2. Several independent and fast computations, and the function is asked if 3 cores is OK. In this case, the function
# will allow the usage of the 3 cores so long as the CPU actually has 3 or more cores
requested_cores = 3
recommeded_cores = px.io_utils.recommend_cpu_cores(num_jobs, requested_cores=requested_cores, lengthy_computation=False)
print('Recommended number of CPU cores for {} independent, FAST, and parallel '
      'computations using the requested {} CPU cores is {}\n'.format(num_jobs, requested_cores, recommeded_cores))

########################################################################################################################
# 3. Far fewer independent and fast computations, and the function is asked if 3 cores is OK. In this case, configuring
# multiple cores for parallel computations will probably be slower than serial computation with a single core. Hence,
# the function will recommend the use of only one core in this case.
num_jobs = 13
recommeded_cores = px.io_utils.recommend_cpu_cores(num_jobs, requested_cores=requested_cores, lengthy_computation=False)
print('Recommended number of CPU cores for {} independent, FAST, and parallel '
      'computations using the requested {} CPU cores is {}\n'.format(num_jobs, requested_cores, recommeded_cores))

########################################################################################################################
# 4. The same number of a few independent computations but eahc of these computations are expected to be lengthy. In
# this case, the overhead of configuring the CPU core for parallel computing is worth the benefit of parallel
# computation. Hence, the function will allow the use of the 3 cores even though the number of computations is small.
recommeded_cores = px.io_utils.recommend_cpu_cores(num_jobs, requested_cores=requested_cores, lengthy_computation=True)
print('Recommended number of CPU cores for {} independent, SLOW, and parallel '
      'computations using the requested {} CPU cores is {}'.format(num_jobs, requested_cores, recommeded_cores))

########################################################################################################################
# get_available_memory()
# ----------------------
# Among the many best-practices we follow when developing a new data analysis or processing class is memory-safe
# computation. This handy function helps us quickly get the available memory. Note that this function returns the
# available memory in bytes. So, we have converted it to gigabytes here:
print('Available memory in this machine: {} GB'.format(px.io_utils.get_available_memory()/1024**3))

########################################################################################################################
# String formatting utilities
# ===========================
# Frequently, there is a need to print out logs on the console to inform the user about the size of files, or estimated
# time remaining for a computation to complete, etc. pycroscopy.io_utils has a few handy functions that help in
# formatting quanties in a human readable format.
# format_size()
# -------------
# One function that uses this functionality to print the size of files etc. is format_size(). While one can manually
# print the available memory in gibibytes (see above), format_size() simplifies this substantially:
print('Available memory in this machine: {}'.format(px.io_utils.format_size(px.io_utils.get_available_memory())))

########################################################################################################################
# format_time()
# -------------
# On the same lines, format_time() is another handy function that is great at formatting time and is often used in
# Process and Fitter to print the remaining time
print('{} seconds = {}'.format(14497.34, px.io_utils.format_time(14497.34)))

########################################################################################################################
# format_quantity()
# -----------------
# You can generate your own formatting function based using the generic function: format_quantity().
# For example, if format_time() were not available, we could get the same functionality via:
units = ['msec', 'sec', 'mins', 'hours']
factors = [0.001, 1, 60, 3600]
time_value = 14497.34
print('{} seconds = {}'.format(14497.34, px.io_utils.format_quantity(time_value, units, factors)))

########################################################################################################################
# formatted_str_to_number()
# -------------------------
# Pycroscopy also has a handy function for the inverse problem of getting a numeric value from a formatted string:
unit_names = ["MHz", "kHz"]
unit_magnitudes = [1E+6, 1E+3]
str_value = "4.32 MHz"
num_value = px.io_utils.formatted_str_to_number(str_value, unit_names, unit_magnitudes, separator=' ')
print('formatted_str_to_number says: {} = {}'.format(str_value, num_value))

########################################################################################################################
# get_time_stamp()
# ----------------
# We try to use a standardized format for storing time stamps in HDF5 files. The function below generates the time
# as a string that can be easily parsed if need be
print('Current time is: {}'.format(px.io_utils.get_time_stamp()))

########################################################################################################################
# Communication utilities
# ========================
# check_ssh()
# -----------
# When developing workflows that need to work on remote or virtual machines in addition to one's own personal computer
# such as a laptop, this function is handy at letting the developer know where the code is being executed
print('Running on remote machine: {}'.format(px.io_utils.check_ssh()))

########################################################################################################################
# file_dialog()
# -------------
# This handy function generates a file window to select files. We encourage you to try this function out since it cannot
# demonstrated within this static document.