"""
=================================================================
Spectral Unmixing
=================================================================

S. Somnath\ :sup:`1,2`,  R. K. Vasudevan\ :sup:`1,3`\

* :sup:`1` Institute for Functional Imaging of Materials
* :sup:`2` Advanced Data and Workflows Group
* :sup:`3` Center for Nanophase Materials Sciences

Oak Ridge National Laboratory, Oak Ridge TN 37831, USA

In this notebook we load some spectral data, and perform basic data analysis, including:
========================================================================================
* KMeans Clustering
* Non-negative Matrix Factorization
* Principal Component Analysis

Software Prerequisites:
=======================
* Standard distribution of **Anaconda** (includes numpy, scipy, matplotlib and sci-kit learn)
* **pycroscopy** : Though pycroscopy is mainly used here for plotting purposes only, it's true capabilities
  are realized through the ability to seamlessly perform these analyses on any imaging dataset (regardless
  of origin, size, complexity) and storing the results back into the same dataset among other things

"""

#Import packages

# Ensure that this code works on both python 2 and python 3
from __future__ import division, print_function, absolute_import, unicode_literals

# basic numeric computation:
import numpy as np

# The package used for creating and manipulating HDF5 files:
import h5py

# Plotting and visualization:
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# for downloading files:
import wget
import os

# multivariate analysis:
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF

# finally import pycroscopy:
import pycroscopy as px

"""
  
"""

#####################################################################################
# The Data
# ========
#
# In this example, we will work on a **Band Excitation Piezoresponse Force Microscopy (BE-PFM)** imaging dataset
# acquired from advanced atomic force microscopes. In this dataset, a spectra was colllected for each position in a two
# dimensional grid of spatial locations. Thus, this is a three dimensional dataset that has been flattened to a two
# dimensional matrix in accordance with the pycroscopy data format.
#
# Fortunately, all statistical analysis, machine learning, spectral unmixing algorithms, etc. only accept data that is
# formatted in the same manner of [position x spectra] in a two dimensional matrix.
#
# We will begin by downloading the BE-PFM dataset from Github
#

data_file_path = 'temp_um.h5'
# download the data file from Github:
url = 'https://raw.githubusercontent.com/pycroscopy/pycroscopy/master/data/BELine_0004.h5'
_ = wget.download(url, data_file_path, bar=None)

hdf = px.ioHDF5(data_file_path)
h5_file = hdf.file

print('Contents of data file:')
print('----------------------')
px.hdf_utils.print_tree(h5_file)
print('----------------------')

h5_meas_grp = h5_file['Measurement_000']

# Extracting some basic parameters:
num_rows = px.hdf_utils.get_attr(h5_meas_grp,'grid_num_rows')
num_cols = px.hdf_utils.get_attr(h5_meas_grp,'grid_num_cols')

# Getting a reference to the main dataset:
h5_main = h5_meas_grp['Channel_000/Raw_Data']

# Extracting the X axis - vector of frequencies
h5_spec_vals = px.hdf_utils.getAuxData(h5_main,'Spectroscopic_Values')[-1]
freq_vec = np.squeeze(h5_spec_vals.value) * 1E-3

print('Data currently of shape:', h5_main.shape)

x_label = 'Frequency (kHz)'
y_label = 'Amplitude (a.u.)'

#####################################################################################
# Visualize the Amplitude Data
# ============================
# Note that we are not hard-coding / writing any tick labels / axis labels by hand.
# All the necessary information was present in the H5 file

px.viz.be_viz_utils.jupyter_visualize_be_spectrograms(h5_main)

#####################################################################################
# 1. Singular Value Decomposition (SVD)
# =====================================
#
# SVD is an eigenvector decomposition that is defined statistically, and therefore typically produces
# non-physical eigenvectors. Consequently, the interpretation of eigenvectors and abundance maps from
# SVD requires care and caution in interpretation. Nontheless, it is a good method for quickly
# visualizing the major trends in the dataset since the resultant eigenvectors are sorted in descending
# order of variance or importance. Furthermore, SVD is also very well suited for data cleaning through
# the reconstruction of the dataset using only the first N (most significant) components.
#
# SVD results in three matrices:
# * V - Eigenvectors sorted by variance in descending order
# * U - corresponding bundance maps
# * S - Variance or importance of each of these components
#
# Advantage of pycroscopy:
# ------------------------
# Notice that we are working with a complex valued dataset. Passing the complex values as is to SVD would result in
# complex valued eigenvectors / endmembers as well as abundance maps. Complex valued abundance maps are not physical.
# Thus, one would need to restructure the data such that it is real-valued only.
#
# One solution is to stack the real value followed by the magnitude of the imaginary component before passing to SVD.
# After SVD, the real-valued eigenvectors would need to be treated as the concatenation of the real and imaginary
# components. So, the eigenvectors would need to be restructured to get back the complex valued eigenvectors.
#
# **Pycroscopy handles all these data transformations (both for the source dataset and the eigenvectors)
# automatically.**  In general, pycroscopy handles compund / complex valued datasets everywhere possible
#
# Furthermore, while it is not discussed in this example, pycroscopy also writes back the results from SVD back to
# the same source h5 file including all relevant links to the source dataset and other ancillary datasets

h5_svd_group = px.doSVD(h5_main, num_comps=256)

h5_u = h5_svd_group['U']
h5_v = h5_svd_group['V']
h5_s = h5_svd_group['S']

# Since the two spatial dimensions (x, y) have been collapsed to one, we need to reshape the abundance maps:
abun_maps = np.reshape(h5_u[:,:25], (num_rows, num_cols, -1))

# Visualize the variance / statistical importance of each component:
px.plot_utils.plot_scree(h5_s, title='Note the exponential drop of variance with number of components')

# Visualize the eigenvectors:
first_evecs = h5_v[:9, :]

px.plot_utils.plot_loops(freq_vec, np.abs(first_evecs), x_label=x_label, y_label=y_label, plots_on_side=3,
                         subtitle_prefix='Component', title='SVD Eigenvectors (Amplitude)', evenly_spaced=False)
px.plot_utils.plot_loops(freq_vec, np.angle(first_evecs), x_label=x_label, y_label='Phase (rad)', plots_on_side=3,
                         subtitle_prefix='Component', title='SVD Eigenvectors (Phase)', evenly_spaced=False)

# Visualize the abundance maps:
px.plot_utils.plot_map_stack(abun_maps, num_comps=9, heading='SVD Abundance Maps',
                             color_bar_mode='single', cmap='inferno')

#####################################################################################
# 2. KMeans Clustering
# ====================
#
# KMeans clustering is a quick and easy method to determine the types of spectral responses present in the
# data. It is not a decomposition method, but a basic clustering method. The user inputs the number of
# clusters (sets) to partition the data into. The algorithm proceeds to find the optimal labeling
# (ie., assignment of each spectra as belonging to the k<sup>th</sup> set) such that the within-cluster
# sum of squares is minimized.
#
# Set the number of clusters below

num_clusters = 4

estimators = px.Cluster(h5_main, 'KMeans', n_clusters=num_clusters)
h5_kmeans_grp = estimators.do_cluster(h5_main)
h5_kmeans_labels = h5_kmeans_grp['Labels']
h5_kmeans_mean_resp = h5_kmeans_grp['Mean_Response']

px.plot_utils.plot_cluster_h5_group(h5_kmeans_grp)

#####################################################################################
# 3. Non-negative Matrix Factorization (NMF)
# ===========================================
#
# NMF, or non-negative matrix factorization, is a method that is useful towards unmixing of spectral
# data. It only works on data with positive real values. It operates by approximate determination of
# factors (matrices) W and H, given a matrix V, as shown below
#
# .. image:: https://upload.wikimedia.org/wikipedia/commons/f/f9/NMF.png
#
# Unlike SVD and k-Means that can be applied to complex-valued datasets, NMF only works on non-negative datasets.
# For illustrative purposes, we will only take the amplitude component of the spectral data

num_comps = 4

# get the non-negative portion of the dataset
data_mat = np.abs(h5_main)

model = NMF(n_components=num_comps, init='random', random_state=0)
model.fit(data_mat)

fig, axis = plt.subplots(figsize=(5.5, 5))
px.plot_utils.plot_line_family(axis, freq_vec, model.components_, label_prefix='NMF Component #')
axis.set_xlabel(x_label, fontsize=12)
axis.set_ylabel(y_label, fontsize=12)
axis.set_title('NMF Components', fontsize=14)
axis.legend(bbox_to_anchor=[1.0, 1.0], fontsize=12)

#####################################################################################

# Close and delete the h5_file
h5_file.close()
os.remove(data_file_path)
