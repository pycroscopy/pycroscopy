# -*- coding: utf-8 -*-
"""
Created on January 23 2021

@author: Gerd Duscher
"""
import sys
sys.path.insert(0, "../../")
import unittest
import numpy as np
import sidpy
import SciFiReaders as sr

from pycroscopy.stats.atom_stats import LocalCrystallography as lc
import pywget
import pyTEMlib
import pyTEMlib.file_tools as ft     # File input/ output library
import pyTEMlib.image_tools as it
import pyTEMlib.probe_tools
import skimage
import os

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
    
    scale_x = ft.get_slope(image.dim_0)
    gauss_diameter = atom_size/scale_x
    gauss_probe = pyTEMlib.probe_tools.make_gauss(image.shape[0], image.shape[1], gauss_diameter)
    
    LR_dataset = it.decon_lr(image, gauss_probe, verbose=False)
    
    extent = LR_dataset.get_extent([0,1])
   
    # ------- Input ------
    
    # ----------------------
    scale_x = ft.get_slope(image.dim_1)
    blobs =  skimage.feature.blob_log(LR_dataset, max_sigma=atom_size/scale_x, threshold=threshold)
 
    
    return blobs



data_file_name = r'my_image_mos2.tiff'
if os.path.exists(data_file_name): 
    os.remove(data_file_name)

pywget.download(url='https://github.com/pycroscopy/SciFiDatasets/raw/main/data/generic/highres2.tif', out = data_file_name)

ir = sr.ImageReader(data_file_name)
img_data = ir.read()
if len(img_data.shape)==3:
    img_data = img_data[:,:,0]
dataset = img_data

class TestUtilityFunctions(unittest.TestCase):

    def test_local_crystallography(self):
        #Find atoms
        atoms_found = return_atoms_LR(dataset, atom_size = 0.09, threshold = 0.03)
        updated_image = np.array(dataset).T
        atom_types = np.zeros((atoms_found.shape[0]))
        #atom_types[len(atom_types)//2:] = 1

        atom_stats = lc(updated_image, atom_positions = atoms_found, atom_descriptors = {'Mo':0}, 
                    window_size = 30, atom_types = atom_types, comp = '0')

        atom_stats.refine_atomic_positions()

        atom_stats.compute_neighborhood(num_neighbors = 6);

        print(atoms_found)
        #Do PCA and KMeans
        pca_results = atom_stats.compute_pca_of_neighbors()
        km_results = atom_stats.compute_kmeans_neighbors()

        os.remove(data_file_name)