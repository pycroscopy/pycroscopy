import os
import sys
sys.path.insert(0, r'/Users/rvv/PycharmProjects/pycroscopy-2018')
import pycroscopy as px
import h5py
from scipy.io.matlab import loadmat, savemat
import numpy as np

folder_path = r'/Users/rvv/Downloads/FORC_2um_5x5_0031_d'
file_name = r'FORC_2um_5x5_0031.mat'

a = loadmat(os.path.join(folder_path, file_name))
SS_parm_vec = a['SS_parm_vec']

SS_parm_vec[0][1] = int(1)
a['SS_parm_vec'] = SS_parm_vec
savemat(os.path.join(folder_path, file_name),a)

path_to_file = os.path.join(folder_path, file_name)
tl = px.io.translators.BEodfTranslator()
input_file_path = tl.translate(path_to_file, show_plots=False)

# input_file_path = r'/Users/rvv/Downloads/BEline_Sample1_3um__0025.h5'

(data_dir, filename) = os.path.split(input_file_path)

max_mem         = 1024*8  # Maximum memory to use, in Mbs. Default = 1024
max_cores       = None    # Number of logical cores to use in fitting.  None uses all but 2 available cores.

# No translation here
h5_path = input_file_path
#force = True # Set this to true to force patching of the datafile.
#tl = px.io.translators.LabViewH5Patcher()
#tl.translate(h5_path, force_patch=force)

h5_file = h5py.File(h5_path, 'r+')
print('Working on:\n' + h5_path)

h5_main = px.hdf_utils.find_dataset(h5_file, 'Raw_Data')[0]

h5_main.attrs['quantity'] = 'a.u.'
h5_main.attrs['units'] = 'a.u.'
h5_main.parent.attrs['channel_type']='BE'
h5_main.parent.parent.attrs['last_pixel']=256
h5_main = px.PycroDataset(h5_main)

h5_file = h5py.File(h5_path, 'r+')
print('Working on:\n' + h5_path)

h5_main = px.hdf_utils.find_dataset(h5_file, 'Raw_Data')[0]

sho_fit_points = 5 # The number of data points at each step to use when fitting
sho_fitter = px.analysis.BESHOfitter(h5_main, parallel=True, verbose=True)
h5_sho_guess = sho_fitter.do_guess(strategy='complex_gaussian', options={'num_points':sho_fit_points},
                                   processors=max_cores, max_mem=max_mem)
h5_sho_fit = sho_fitter.do_fit(processors=max_cores, max_mem=max_mem)
