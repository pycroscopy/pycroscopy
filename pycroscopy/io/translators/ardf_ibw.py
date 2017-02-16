# -*- coding: utf-8 -*-
"""
Created on Thu Dec 08 12:00:43 2016

@author: Suhas Somnath

Note that there is no way to read the ARDF file in any language reliably.
Asylum engineers themselves say that the only way is to export the file as a set of ibw files
and then read the ibw files.
"""

from igor import binarywave as bw
import numpy as np
import os
import matplotlib.pyplot as plt

# %% Specify the folder:
folder_path = r'\\ornldata.ornl.gov\home\Documents\Asylum Research Data\161207\ForceMap00'

# Figure out the number of rows and columns in the grid

file_list = np.array(os.listdir(folder_path))
file_list.sort()

last_file = file_list[-1]
last_file = last_file[:-4]  # remove the .ibw
[last_file, num_cols] = last_file.split('Point')
num_cols = int(num_cols) + 1
num_rows = int(last_file.replace('Line', '')) + 1

# %% Find the (maximum) number of Z points:

wave_sizes = np.zeros((num_rows, num_cols), dtype=np.uint16)

for row_ind in range(num_rows):
    file_prefix = '{:s}{:04d}'.format('Line', row_ind)
    for col_ind in range(num_cols):
        file_name = file_prefix + '{:s}{:04d}'.format('Point', col_ind) + '.ibw'
        obj = bw.load(os.path.join(folder_path, file_name))
        wave_sizes[row_ind, col_ind] = obj.get('wave').get('wData').shape[0]

num_chans = obj.get('wave').get('wData').shape[1]
max_z_points = np.max(wave_sizes)

# %% Prepare parameters:
last_file = file_list[-1]

obj = bw.load(os.path.join(folder_path, last_file))  # dictionary with keys = ['version', 'wave']
wave = obj.get('wave')
# Read the note to get parameters
parm_string = wave.get('note')
parm_string = parm_string.rstrip('\r')
parms_list = parm_string.split('\r')
parm_dict = dict()
for pair_string in parms_list:
    temp = pair_string.split(':')
    if len(temp) == 2:
        temp = [item.strip() for item in temp]
        try:
            num = float(temp[1])
            parm_dict[temp[0]] = num
            try:
                if num == int(num):
                    parm_dict[temp[0]] = int(num)
            except OverflowError:
                pass
        except ValueError:
            parm_dict[temp[0]] = temp[1]

# Grab the creation and modification times:
other_parms = wave.get('wave_header')
for key in ['creationDate', 'modDate', 'bname']:
    parm_dict[key] = other_parms[key]

# Get the channel names
temp = wave.get('labels')
labels = []
for item in temp:
    if len(item) > 0:
        labels += item
for item in labels:
    if item == '':
        labels.remove(item)

default_units = list()
for chan in labels:
    if chan.startswith('Phase'):
        default_units.append('deg')
    else:
        default_units.append('m')

# %% Prepare the data matrix:

force_map = np.zeros(shape=(num_rows, num_cols, max_z_points, num_chans), dtype=np.float32)
print('Force map of shape'.format(force_map.shape))

# load all the data:

for row_ind in range(num_rows):
    file_prefix = '{:s}{:04d}'.format('Line', row_ind)
    for col_ind in range(num_cols):
        file_name = file_prefix + '{:s}{:04d}'.format('Point', col_ind) + '.ibw'
        obj = bw.load(os.path.join(folder_path, file_name))  # dictionary with keys = ['version', 'wave']
        data = np.copy(obj.get('wave').get('wData'))

        # Normalizing the deflection:
        extent = int(data.shape[0] * 0.2)
        pol = np.polyfit(data[:extent, 0], data[:extent, 1], 0)
        data[:, 1] = data[:, 1] - np.polyval(pol, data[:, 0])

        force_map[row_ind, col_ind, :data.shape[0], :] = np.float32(data)
    print('Completed reading line {} of {}'.format(row_ind + 1, num_rows))

force_map = np.transpose(force_map, (3, 0, 1, 2))

# %% Plot a few normalized curves:
fig, ax = plt.subplots()
for row_ind in range(2):
    for col_ind in range(2):
        ax.plot(force_map[1, row_ind, col_ind, :])