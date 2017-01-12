# -*- coding: utf-8 -*-
"""
Created on Sun May 29 19:09:34 2016

@author: Suhas Somnath
"""

#%% Translate data first 

from pycroscopy.io.io_utils import uiGetFile
from pycroscopy.io.translators.gmode_iv import GIVTranslator
parm_path = uiGetFile('parms.mat')
translator = GIVTranslator()
h5_path = translator.translate(parm_path)

#%% Begin processing a translated (.h5) file
import h5py
import matplotlib.pyplot as plt
from os import path
import numpy as np
from pycroscopy.analysis.PCAutils import doKMeans, doPCA, plotScree, plotLoadingMaps, fastSVD, plotKMeansResults
from pycroscopy.io.io_hdf5 import ioHDF5
from pycroscopy.io.hdf_utils import getH5DsetRefs, findH5group
from pycroscopy.io.translators.utils import makePositionMat, getPositionSlicing
from pycroscopy.io.microdata import MicroDataGroup, MicroDataset
from pycroscopy.processing.gmode_utils import test_filter, fft_filter_raw_data
from pycroscopy.viz.plot_utils import plot_loops

#%% Load data
#h5_path = uiGetFile('.h5')
folder_path, filename = path.split(h5_path)
h5_f = h5py.File(h5_path,'r+')
h5_grp = h5_f['Measurement_000']['Channel_000']
h5_main = h5_grp['Raw_Data']
excit_wfm = h5_grp['Excitation_Waveform'].value
samp_rate = h5_grp.attrs['IO_samp_rate_[Hz]']
num_pts = h5_main.shape[1]
num_lines = h5_main.shape[0]
w_vec = 1E-3*np.linspace(-0.5*samp_rate, 0.5*samp_rate - samp_rate/num_pts, num_pts)
ex_freq = h5_grp.attrs['excitation_frequency_[Hz]']
pts_per_cycle = int(np.round(samp_rate/ex_freq))
single_AO = excit_wfm[:pts_per_cycle]   

#%% Visualize data in frequency domain and think about filter here:

row_ind = 20
F_resp = np.fft.fftshift(np.fft.fft(h5_main[row_ind]))

fig,ax = plt.subplots()
ax.plot(w_vec[int(0.5*len(w_vec)):], np.log10(np.abs(F_resp[int(0.5*len(w_vec)):])), label='Response')
ax.plot([1E-3*ex_freq, 1E-3*ex_freq], [-1,5],'r',label='Excitation')
ax.set_xlabel('Frequency (kHz)')
ax.set_ylabel('Amplitude (a.u.)')
ax.legend()
ax.set_xscale('log')
ax.set_xlim(1E-2,samp_rate*0.5E-3)
ax.set_title('Noise Spectrum log-log')
fig.savefig(path.join(folder_path,'noise_spectrum.png'), format='png', dpi=150)

#%% try out certain filter parameters:

filter_parms = dict()
filter_parms['noise_threshold'] = 1E-6
filter_parms['comb_[Hz]'] = -1
filter_parms['LPF_cutOff_[Hz]'] = 45E+3
# Noise frequencies - 15.6 kHz ~ 14-17.5, 7.8-8.8, 45-49.9 ~ 48.9414 kHz
filter_parms['band_filt_[Hz]'] = [[8.3E+3,15.6E+3,48.9414E+3],[1E+3, 0.5E+3, 0.1E+3]]
filter_parms['phase_[rad]'] = 0
filter_parms['samp_rate_[Hz]'] = samp_rate
filter_parms['num_pix'] = 1

#%% Test filter on a single line:

row_ind = 50
filt_line, fig_filt, axes_filt = test_filter(h5_main[row_ind], filter_parms, samp_rate, show_plots=True)
fig_filt.savefig(path.join(folder_path,'FFT_filter.png'), format='png', dpi=300)

raw_row = np.reshape(h5_main[row_ind], (-1,pts_per_cycle))
filt_row = filt_line.reshape(-1,pts_per_cycle)

fig, axes = plt.subplots(nrows=5,ncols=5, figsize=(18, 13))
for ax, col_ind in zip(axes.flat,np.linspace(0,filt_row.shape[0]-1,25, dtype=int)):
    ax.set_title('Row: ' + str(row_ind) + ' Col: ' + str(col_ind))    
    ax.plot(single_AO,raw_row[col_ind,:],'r')
    ax.plot(single_AO,filt_row[col_ind,:],'b')
fig.suptitle('Filtered data', fontsize=14)
fig.tight_layout()
fig.savefig(path.join(folder_path,'FFT_filtering_examples.png'), format='png', dpi=300)

#%% Apply filter to entire dataset using external script

hdf = ioHDF5(h5_f)

'''if __name__=='__main__':
    h5_filt_grp = fft_filter_raw_data(hdf, h5_main, filter_parms, write_filtered=True)'''

#%% Now break up the filtered data into individual loops

h5_filt_grp = findH5group(h5_main,'FFT_Filtering')[-1]
h5_filt = h5_filt_grp['Filtered_Data']

AI_mat_2d = np.reshape(h5_filt.value,(-1,pts_per_cycle))
AI_mat_3d = np.reshape(AI_mat_2d,(num_lines,-1,pts_per_cycle))
num_cols = AI_mat_3d.shape[1]

pos_ind_mat = makePositionMat([num_cols, num_lines])
pos_labs = ['X','Y']
pos_slices = getPositionSlicing(pos_labs, AI_mat_2d.shape[0])

scan_width_nm = np.round(100*h5_grp.attrs['grid_scan_width_[m]']*1E+9)/100
scan_height_nm = np.round(100*h5_grp.attrs['grid_scan_height_[m]']*1E+9)/100
slope_x = scan_width_nm/num_cols
slope_y = scan_height_nm/num_lines

pos_val_mat = np.copy(pos_ind_mat)
pos_val_mat[:,0] = pos_val_mat[:,0]*slope_x
pos_val_mat[:,1] = pos_val_mat[:,1]*slope_y

spec_slices = {'Bias':(slice(0,1),slice(None))}

ds_pos_ind = MicroDataset('Position_Indices', np.uint32(pos_ind_mat))
ds_pos_ind.attrs['labels'] = pos_slices
ds_pos_val = MicroDataset('Position_Values', np.float32(pos_val_mat))
ds_pos_val.attrs['labels'] = pos_slices
ds_pos_val.attrs['units'] = [ 'nm' for _ in xrange(len(pos_slices))]

ds_spec_ind = MicroDataset('Spectroscopic_Indices', np.atleast_2d(np.arange(len(single_AO), dtype=np.uint32)))
ds_spec_ind.attrs['labels'] = spec_slices
ds_spec_val = MicroDataset('Spectroscopic_Values', np.atleast_2d(np.float32(single_AO)))
ds_spec_val.attrs['labels'] = spec_slices
ds_spec_val.attrs['units'] = ['V']

ds_filt_data = MicroDataset('Reshaped_Data',data=AI_mat_2d, compression='gzip',chunking=(10,AI_mat_2d.shape[1]))

# write this to H5 as some form of filtered data. 
resh_grp = MicroDataGroup(h5_filt.name.split('/')[-1] + '-Reshape_', parent=h5_filt.parent.name)
resh_grp.addChildren([ds_filt_data, ds_pos_ind, ds_pos_val, ds_spec_ind, ds_spec_val])

h5_refs = hdf.writeData(resh_grp)

h5_resh = getH5DsetRefs(['Reshaped_Data'], h5_refs)[0]
# Link everything:
hdf.linkRefs(h5_resh, getH5DsetRefs(['Position_Indices','Position_Values','Spectroscopic_Indices','Spectroscopic_Values'], h5_refs))

#%% Do k-means to extract that component that contains minimal response = only capacitance
num_clusts = 16
k5_km_raw_grp = doKMeans(hdf, h5_resh, num_clusts)
labels_2d = np.reshape(k5_km_raw_grp['Labels'].value,(num_lines, num_cols))
fig_km, ax_km = plotKMeansResults(labels_2d,k5_km_raw_grp['Centroids'].value, spec_val=single_AO)
fig_km.savefig(path.join(folder_path,'Kmeans_Raw_' + str(num_clusts) + '_clusters.png'), format='png', dpi=300)

#%% Try subtracting the first (lowest response) cluster from the raw data:

# we expect the capacitance component to have the smalleset extents for max and mins
centroids = k5_km_raw_grp['Centroids'].value
min_resp_centroid = np.argmin(np.max(np.abs(centroids),axis=1))
print('Minimum response found at centroid ' + str(min_resp_centroid))

capacitance = centroids[min_resp_centroid]
capacitance_compensated = AI_mat_3d - capacitance

#%% Check to see if capacitance has indeed been compensated correctly:

fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(18, 13))
for ax in axes.flat:
    row_ind = np.random.randint(low=0,high=num_lines)
    col_ind = np.random.randint(low=0, high=num_cols)
    ax.plot(single_AO, AI_mat_3d[row_ind,col_ind],color='b',label='Raw')
    ax.plot(single_AO, capacitance_compensated[row_ind,col_ind], color='r', label='Corrected')
    ax.set_title('Row: ' + str(row_ind) + ' Col: ' + str(col_ind))
ax.legend(loc='best')
fig.suptitle('Capacitance Compensation examples')
fig.tight_layout()
fig.savefig(path.join(folder_path,'Capacitance_compensation_verification.png'), format='png', dpi=150)

#%% Write back capacitance compensated data:

ds_comp_data = MicroDataset('Normalized_Data',data=np.float32(AI_mat_2d-capacitance), 
                            compression='gzip',chunking=(10,h5_resh.shape[1]))
ds_capacitance = MicroDataset('Reference', data=np.float32(capacitance))
comp_grp = MicroDataGroup(h5_resh.name.split('/')[-1] + '-Normalization_', parent=h5_resh.parent.name)
comp_grp.addChildren([ds_comp_data, ds_capacitance])

h5_refs_norm = hdf.writeData(comp_grp)
h5_norm = getH5DsetRefs(['Normalized_Data'], h5_refs_norm)[0]
h5_capac = getH5DsetRefs(['Reference'], h5_refs_norm)[0]
# Link everything:
hdf.linkRefs(h5_norm, getH5DsetRefs(['Position_Indices','Position_Values','Spectroscopic_Indices','Spectroscopic_Values'], h5_refs))
hdf.linkRefs(h5_norm, [h5_capac])
hdf.linkRefs(h5_capac, [k5_km_raw_grp['Centroids']])

#%% What does k-means after compensation show:

h5_km_norm_grp = doKMeans(hdf, h5_norm, 8)
labels_norm_2d = np.reshape(h5_km_norm_grp['Labels'].value,(num_lines, num_cols))
fig_km, ax_km = plotKMeansResults(labels_norm_2d,h5_km_norm_grp['Centroids'].value, spec_val=single_AO,
                                  spec_label='Bias (V)', resp_label='Current (a.u)')
fig_km.savefig(path.join(folder_path,'Kmeans_Compensated.png'), format='png', dpi=300)

#%% What does PCA after compensation show?

h5_pca_norm_grp = doPCA(hdf, h5_norm, num_comps=128)

#%% Plot PCA results:

fig_S,ax_S = plotScree(h5_pca_norm_grp['S'])
fig_S.savefig(path.join(folder_path,'PCA_S.png'), format='png', dpi=300)
loadings = np.reshape(h5_pca_norm_grp['U'].value,(num_lines,int(h5_pca_norm_grp['U'].shape[0]/num_lines),-1))
#loadings = np.transpose(loadings, axes=(1,2,0))
fig_U,ax_U = plotLoadingMaps(loadings, num_comps=16)
fig_U.savefig(path.join(folder_path,'PCA_U.png'), format='png', dpi=100) 

fig_loops, ax_loops = plot_loops(single_AO, h5_pca_norm_grp['V'], evenly_spaced=False,
                                 plots_on_side=4, rainbow_plot=True,
                                 x_label='Bias (V)', y_label='Current (a.u.)', title='Eigenvectors')
fig_loops.tight_layout()
fig_loops.savefig(path.join(folder_path,'PCA_V.png'), format='png', dpi=300)

F_eigvecs = np.fft.fftshift(np.fft.fft(h5_pca_norm_grp['V'].value,axis=1),axes=1)
w_vec_single = 1E-3*np.linspace(-0.5*samp_rate, 0.5*samp_rate - samp_rate/F_eigvecs.shape[1], F_eigvecs.shape[1])
F_eigvecs = np.log10(np.abs(F_eigvecs[:,int(0.5*F_eigvecs.shape[1]):]))
w_vec_single = w_vec_single[int(0.5*w_vec_single.shape[0]):]

fig_F,axes_F = plt.subplots(6,6, figsize=(18, 13))
for ind, ax in enumerate(axes_F.flat):
    ax.plot(w_vec_single,F_eigvecs[ind])
    ax.set_title('Comp #' + str(ind))
for ind in xrange(6):
    axes_F.flat[30+ind].set_xlabel('Frequency (kHz)')
    axes_F.flat[6*ind].set_ylabel('Amplitude (a.u.)')
fig_F.suptitle('Power density of V')
fig_F.tight_layout()
fig_F.savefig(path.join(folder_path,'PCA_V_F.png'), format='png', dpi=200)

#%% Done processing the data. Close the file for now. Open later in read only mode for visualization
h5_f.close()