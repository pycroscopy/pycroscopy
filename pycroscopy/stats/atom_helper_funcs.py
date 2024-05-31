
import os
import numpy as np
import matplotlib.pyplot as plt
import SciFiReaders as sr
from copy import deepcopy as dc
import matplotlib.gridspec as gridspec
from skimage import color
import pyTEMlib
import pyTEMlib.file_tools as ft     # File input/ output library
import pyTEMlib.image_tools as it
import pyTEMlib.probe_tools
import skimage

def return_good_atoms(mask_image, atoms):
    """
    Given a mask image (where forbidden locations have intensity of 0), and a list of atomic coordinates, remove any atoms
    that are in the forbidden locations.
    Inputs: - mask_image: (sidpy dataset or numpy array) Masked image
            - atoms: (Nx2 numpy array) atomic coordinates 
    Outputs: - atoms: numpy array of new atomic coordinates
    """
    
    good_atoms = []
    for atom_x, atom_y in atoms:
        if mask_image[int(atom_x), int(atom_y)]!=0:
            good_atoms.append((atom_x,atom_y))
    return np.array(good_atoms)


def return_atoms_LR(sidpy_dset, atom_size=0.09, threshold = 0.03):
    """
    Find the atoms in a sidpy dataset, using Lucy-Richardson deconvolution method.
    
    Inputs: - sidpy_dset (sidpy.Dataset) (dataset with image)
            - atoms_size (float) (size in nm of atoms) Default = 0.02
            - threshold = 0.03 #usally between 0.01 and 0.9  the smaller\ the more atoms

    Outputs: - Nx2 numpy array of atom positions
    
    """
   
    dataset = sidpy_dset
    if dataset.ndim >2:
        image = dataset.sum(axis=0)
    else:
        image = dataset
    image.data_type = 'image'
    image.title = 'registered'
    out_tags = {}
    image.metadata['experiment']= {'convergence_angle': 30, 'acceleration_voltage': 200000.}
    
    print(f'Contrast = {np.std(np.array(image))/np.average(np.array(image)):.2f}')
    
    scale_x = ft.get_slope(image.dim_0)
    gauss_diameter = atom_size/scale_x
    gauss_probe = pyTEMlib.probe_tools.make_gauss(image.shape[0], image.shape[1], gauss_diameter)
    
    print('Deconvolution of ', dataset.title)
    LR_dataset = it.decon_lr(image, gauss_probe, verbose=False)
    
    extent = LR_dataset.get_extent([0,1])
    fig, ax = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
    ax[0].imshow(image[:500,:500].T, extent = extent,vmax=np.median(np.array(image))+3*np.std(np.array(image)))
    ax[1].imshow(LR_dataset[:500,:500].T, extent = extent, vmax=np.median(np.array(LR_dataset))+3*np.std(np.array(LR_dataset)));
    plt.figure(figsize=(10,10))
    plt.imshow(LR_dataset)
    # ------- Input ------
    
    # ----------------------
    scale_x = ft.get_slope(image.dim_1)
    blobs =  skimage.feature.blob_log(LR_dataset, max_sigma=atom_size/scale_x, threshold=threshold)
    print(len(blobs))
    
    fig1, ax = plt.subplots(1, 1,figsize=(8,7), sharex=True, sharey=True)
    plt.title("blob detection ")
    blobs = blobs[:,:2]
    plt.imshow(image.T, interpolation='nearest',cmap='gray', vmax=np.median(np.array(image))+3*np.std(np.array(image)))
    plt.scatter(blobs[:, 0], blobs[:, 1], c='r', s=10, alpha = .5);
    
    return blobs


