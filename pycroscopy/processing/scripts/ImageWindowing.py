"""
Created on Jun 22, 2016

@author: Chris Smith -- csmith55@utk.edu
"""

import sys
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import measure
from scipy.optimize import least_squares
from time import time
sys.path.append('../../../')
import pycroscopy as px
from pycroscopy import ImageTranslator
from pycroscopy.processing.image_processing import ImageWindow
from pycroscopy.processing.svd_utils import doSVD
from pycroscopy import Cluster
from pycroscopy.viz.plot_utils import plotScree, plot_map_stack, plot_cluster_results_together
from pycroscopy.io.io_utils import uiGetFile
from pycroscopy.io.hdf_utils import reshape_to_Ndims

if __name__ == '__main__':
    '''
    Select the image file
    '''
    imagepath = uiGetFile(filter='Image File (*.tiff *.tif *.jpeg *.jpg *.png);;' +
                                 'DM3 or DM4 File (*.dm3 *.dm4);;' +
                                 'Image Text File (*.txt)',
                          caption='Select Image File')

    '''
    Set up parameters
    '''
    save_plots      = True
    show_plots      = False
    profiling       = False
    bin_factor      = 2
    win_fft         = 'data+abs'     # Options are None, 'abs', 'data+abs', or 'complex'
    num_peaks       = 2         # Number of Peaks to use in window size fitting
    max_mem         = 1024*8    # Maximum memory to use in calculations, in Mb
    num_comp        = 128       # Number of Components to generate
    plot_comps      = 49        # Number of Components to plot, plot_comps<=num_comps
    clean_comps     = [1, 128]        # Components to use for image cleaning.  Can be integer, slice, or list of integers
    clean_method    = 'components'   # Options are 'normal', 'batch', or 'components'
    cluster_method  = 'KMeans'  # See Cluster documentation for options
    num_cluster     = 32        # Number of Clusters to use
    plot_clust      = 16         # Number of clusters to plot, plot_clust<=num_cluster

    '''
    Parse the path to the image file
    '''
    folder, filename = os.path.split(os.path.abspath(imagepath))
    basename, _ = os.path.splitext(filename)

    '''
    Read the image into the hdf5 file
    '''
    h5_path = os.path.join(folder, basename+'.h5')

    tl = ImageTranslator()
    h5_raw = tl.translate(imagepath, bin_factor=bin_factor)
    h5_file = h5_raw.file

    '''
    Initialize the windowing class
    '''
    iw = ImageWindow(h5_raw, max_RAM_mb=max_mem)

    '''
    Extract an optimum window size from the image.  This step is optional.
    If win_x and win_y are not given, this step will be automatically done.
    '''
    win_size, psf_width = iw.window_size_extract(num_peaks,
                                                 save_plots=save_plots,
                                                 show_plots=show_plots)

    '''
    Plot a window
    '''
    raw_image, _ = reshape_to_Ndims(h5_raw)
    raw_image = raw_image.squeeze()
    start_x, start_y = [np.random.randint(raw_image.shape[0] - win_size),
                        np.random.randint(raw_image.shape[1] - win_size)]
    plt.imshow(raw_image[start_x:start_x + win_size, start_y:start_y + win_size])

    '''
    Create the windows.
    '''
    t0 = time()
    h5_wins = iw.do_windowing(win_x=win_size,
                              win_y=win_size,
                              save_plots=save_plots,
                              show_plots=show_plots,
                              win_fft=win_fft)

    print 'Windowing took {} seconds.'.format(time() - t0)

    '''
    Do SVD on the windowed image
    '''
    h5_svd = doSVD(h5_wins, num_comps=num_comp)

    h5_U = h5_svd['U']
    h5_S = h5_svd['S']
    h5_V = h5_svd['V']

    h5_pos = iw.hdf.file[h5_wins.attrs['Position_Indices']]
    num_rows = len(np.unique(h5_pos[:, 0]))
    num_cols = len(np.unique(h5_pos[:, 1]))

    svd_name = '_'.join([basename, h5_svd.name.split('/')[-1]])

    print 'Plotting Scree'
    fig203, axes203 = plotScree(h5_S[()])
    fig203.savefig(os.path.join(folder, svd_name+'_PCA_scree.png'), format='png', dpi=300)
    plt.close('all')
    del fig203, axes203

    stdevs = 2
    plot_comps = min(plot_comps, h5_S.size)

    print 'Plotting Loading Maps'
    fig202, axes202 = plot_map_stack(np.reshape(h5_U[:, :plot_comps], [num_rows, num_cols, -1]),
                                     num_comps=plot_comps, stdevs=2, color_bar_mode='each')
    fig202.savefig(os.path.join(folder, svd_name+'_PCA_Loadings.png'), format='png', dpi=300)
    plt.close('all')
    del fig202, axes202

    print 'Plotting Eigenvectors'
    num_x = int(np.sqrt(h5_V.shape[1]))
    for field in h5_V.dtype.names:
        fig201, axes201 = plot_map_stack(np.transpose(np.reshape(h5_V[:plot_comps, :][field],
                                                                 [-1, num_x, num_x]), [1, 2, 0]),
                                         num_comps=plot_comps, heading='Eigenvectors')
        fig201.savefig(os.path.join(folder, svd_name+'_'+field+'_PCA_Eigenvectors.png'), format='png', dpi=300)
        plt.close('all')
        del fig201, axes201

    '''
    Build a cleaned image from the cleaned windows
    '''
    im_x = h5_wins.parent.attrs['image_x']
    im_y = h5_wins.parent.attrs['image_y']

    if clean_method == 'normal':
        h5_wins = h5_raw.parent['Raw_Data-Windowing_000']['Image_Windows']
        t0 = time()
        h5_clean_image = iw.clean_and_build(h5_win=h5_wins, components=clean_comps)
        print 'Cleaning and rebuilding image took {} seconds.'.format(time()-t0)
    elif clean_method == 'batch':

        t0 = time()
        h5_clean_image = iw.clean_and_build_batch(h5_win=h5_wins, components=clean_comps)

        print 'Batch cleaning and rebuilding image took {} seconds.'.format(time() - t0)

        iw.plot_clean_image(h5_clean_image)

    elif clean_method == 'components':

        t0 = time()
        h5_clean_image = iw.clean_and_build_separate_components(h5_win=h5_wins, components=None)

        print 'Batch cleaning and rebuilding image took {} seconds.'.format(time() - t0)

        clean_name = '_'.join([basename, h5_clean_image.name.split('/')[-1]])

        plot_comps = min(plot_comps, h5_clean_image.shape[1])

        fig202, axes202 = plot_map_stack(np.reshape(h5_clean_image[:, :plot_comps], [im_x, im_y, plot_comps]),
                                         num_comps=plot_comps, stdevs=2, color_bar_mode='each')
        fig202.savefig(os.path.join(folder, clean_name+'_Components.png'), format='png', dpi=300)
        plt.close('all')
        del fig202, axes202

        # Calculate a 'best' cleaned image for using in clustering
        best_comps = range(2, 32)
        num_best_comps = len(best_comps)
        best_image = np.sum(h5_clean_image[:, best_comps], axis=1).reshape(raw_image.shape)
        fig_results, axes_results = px.viz.plot_utils.plot_image_cleaning_results(raw_image, best_image,
                                                                                  fig_mult=(8, 8),
                                                                                  cmap=px.plot_utils.cmap_jet_white_center())
        fig_results.savefig(os.path.join(folder, 'best_image_results.png'), format='png')
        plt.close('all')
        print best_comps

    if cluster_method is not None:
        '''
        Do KMeans on the U of Windows
        '''
        km_name = '_'.join([basename, 'Image_Windows_U'])

        cluster = Cluster(h5_U, method_name=cluster_method, num_comps=None, n_clusters=num_cluster)

        h5_kmeans = cluster.do_cluster(rearrange_clusters=True)

        h5_labels = h5_kmeans['Labels']
        h5_centroids = h5_kmeans['Mean_Response']
        h5_km_spec = h5_file[h5_centroids.attrs['Spectroscopic_Values']]

        fig601, ax601 = plot_cluster_results_together(h5_labels, h5_centroids, spec_val=h5_km_spec)
        fig601.savefig(os.path.join(folder, km_name + '_KMeans.png'), format='png', dpi=300)
        del fig601, ax601
        plt.close('all')

    '''
    generate a cropped image that was effectively the area that was used for pattern searching
    Need to get the math righ on the counting
    '''
    clean_image_mat = np.reshape(h5_clean_image[()], [im_x, im_y])
    labels_mat = np.reshape(h5_labels, [num_rows, num_cols])

    # select motifs from the cluster labels using the component list:
    motif_win_centers = [(170,80), (162, 91), (147,91), (154, 79),(156,105),(162,116),(140,105),(147,116)]
    motif_win_size = win_size
    # motif_win_size = 15  # Perhaps the motif should be smaller than the original window
    num_motifs = len(motif_win_centers)
    motifs = list()
    fig, axes = plt.subplots(ncols=3, nrows=num_motifs, figsize=(14, 6 * num_motifs))

    for window_center, ax_row in zip(motif_win_centers, np.atleast_2d(axes)):
        indices = (slice(window_center[0] - int(0.5 * motif_win_size), window_center[0] + int(0.5 * motif_win_size)),
                   slice(window_center[1] - int(0.5 * motif_win_size), window_center[1] + int(0.5 * motif_win_size)))
        motifs.append(labels_mat[indices])

        ax_row[0].hold(True)
        ax_row[0].imshow(clean_image_mat, interpolation='none', cmap=px.plot_utils.cmap_jet_white_center())
        ax_row[0].add_patch(patches.Rectangle((window_center[1] - int(0.5 * motif_win_size),
                                               window_center[0] - int(0.5 * motif_win_size)),
                                              motif_win_size, motif_win_size, fill=False,
                                              color='black', linewidth=2))
        ax_row[0].hold(False)
        ax_row[1].hold(True)
        ax_row[1].imshow(clean_image_mat[indices], interpolation='none', cmap=px.plot_utils.cmap_jet_white_center())
        ax_row[1].plot([0, motif_win_size - 2], [int(0.5 * motif_win_size), int(0.5 * motif_win_size)], 'k--')
        ax_row[1].plot([int(0.5 * motif_win_size), int(0.5 * motif_win_size)], [0, motif_win_size - 2], 'k--')
        # ax_row[1].axis('tight')
        ax_row[1].set_title(
            'Selected window for motif around (row {}, col {})'.format(window_center[0], window_center[1]))
        ax_row[1].hold(False)
        ax_row[2].imshow(labels_mat[indices], interpolation='none', cmap=px.plot_utils.cmap_jet_white_center())
        ax_row[2].set_title('Motif from K-means labels');

    half_wind = int(motif_win_size * 0.5)
    motif_match_coeffs = list()

    for motif_mat in motifs:

        match_mat = np.zeros(shape=(num_rows - motif_win_size, num_cols - motif_win_size))
        for row_count, row_pos in enumerate(range(half_wind, num_rows - half_wind - 1, 1)):
            for col_count, col_pos in enumerate(range(half_wind, num_cols - half_wind - 1, 1)):
                local_cluster_mat = labels_mat[row_pos - half_wind: row_pos + half_wind,
                                    col_pos - half_wind: col_pos + half_wind]
                match_mat[row_count, col_count] = np.sum(local_cluster_mat == motif_mat)
        # Normalize the dataset:
        match_mat = match_mat / np.max(match_mat)

        motif_match_coeffs.append(match_mat)

    fig, axes = plt.subplots(nrows=int(np.ceil(num_motifs / 2.)), ncols=2,
                             figsize=(14, 7 * int(np.ceil(num_motifs / 2.))))
    for motif_ind, match_mat, ax_col in zip(range(num_motifs), motif_match_coeffs, axes.flatten()[:num_motifs]):
        ax_col.imshow(match_mat, interpolation='none', cmap="Greys")
        ax_col.set_title('Pattern matches for motif #{}'.format(motif_ind));

    thresholds = [0.1]
    thresholded_maps = list()
    fig, axes = plt.subplots(nrows=int(np.ceil(num_motifs / 2.)), ncols=2,
                             figsize=(14, 7 * int(np.ceil(num_motifs / 2.))))
    for motif_ind, match_mat, t_hold, ax_col in zip(range(num_motifs), motif_match_coeffs, thresholds,
                                                    axes.flatten()[:num_motifs]):
        bin_map = np.where(match_mat > t_hold,
                           np.ones(shape=match_mat.shape, dtype=np.uint8),
                           np.zeros(shape=match_mat.shape, dtype=np.uint8))
        thresholded_maps.append(bin_map)
        ax_col.imshow(bin_map, interpolation='none', cmap="Greys")
        ax_col.set_title('Motif #{} thresholded to {}'.format(motif_ind, t_hold));

    # find the centers of the patches in the binarized images above
    atom_labels = list()
    for thresh_map in thresholded_maps:
        labled_atoms = measure.label(thresh_map, background=0)
        map_props = measure.regionprops(labled_atoms)
        atom_centroids = np.zeros(shape=(len(map_props), 2))
        for atom_ind, atom in enumerate(map_props):
            atom_centroids[atom_ind] = np.array(atom.centroid)
        atom_labels.append(atom_centroids)

    # plot results
    fig, axis = plt.subplots(ncols=2, figsize=(14, 7))
    axis[0].imshow(clean_image_mat, interpolation='none', cmap=px.plot_utils.cmap_jet_white_center())
    axis[0].set_title('Cleaned image')
    axis[1].hold(True)
    col_map = plt.cm.jet
    for atom_type_ind, atom_centroids in enumerate(atom_labels):
        axis[1].scatter(atom_centroids[:, 1], atom_centroids[:, 0], color=col_map(255 * atom_type_ind / num_motifs))
    axis[1].hold(True)
    axis[1].invert_xaxis()
    axis[1].axis('tight')
    axis[1].set_title('Positions of distinct atoms');

    # generate a cropped image that was effectively the area that was used for pattern searching
    cropped_clean_image = clean_image_mat[half_wind:-half_wind, half_wind:-half_wind]
    # Normalize it to ensure that it is within 0 and 1
    cropped_clean_image -= np.min(cropped_clean_image)
    cropped_clean_image = cropped_clean_image / np.max(cropped_clean_image)

    # generate a compound dataset to store the guesses
    num_tot_atoms = np.sum([family.shape[0] for family in atom_labels])
    atom_dtype = np.dtype([('x', np.float32),
                           ('y', np.float32),
                           ('type', np.uint32)])
    atom_guess_mat = np.zeros(shape=num_tot_atoms, dtype=atom_dtype)
    last_ind = 0
    for family_ind, family in enumerate(atom_labels):
        atom_guess_mat[last_ind:last_ind + family.shape[0]]['x'] = family[:, 0]
        atom_guess_mat[last_ind:last_ind + family.shape[0]]['y'] = family[:, 1]
        atom_guess_mat[last_ind:last_ind + family.shape[0]]['type'] = np.ones(shape=family.shape[0],
                                                                              dtype=np.uint32) * family_ind
        last_ind += family.shape[0]

    # Write these guesses to disk!
    ds_cropped_image = px.io.MicroDataset('Cropped_Clean_Image', data=cropped_clean_image, dtype=np.float32)
    ds_thresholds = px.io.MicroDataset('Thresholds', data=np.array(thresholds), dtype=np.float32)
    ds_motif_centers = px.io.MicroDataset('Motif_Centers', data=np.atleast_2d(motif_win_centers), dtype=np.uint32)
    ds_atom_guesses = px.io.MicroDataset('Guess', data=atom_guess_mat)
    h5_labels = h5_kmeans['Labels']
    dgrp_atom_finding = px.io.MicroDataGroup(h5_labels.name.split('/')[-1] + '-Atom_Finding_', parent=h5_kmeans.name)
    dgrp_atom_finding.addChildren([ds_cropped_image, ds_thresholds, ds_motif_centers, ds_atom_guesses])
    dgrp_atom_finding.attrs['psf_width'] = psf_width
    dgrp_atom_finding.attrs['motif_win_size'] = motif_win_size

    hdf = px.ioHDF5(h5_labels.file)
    h5_atom_refs = hdf.writeData(dgrp_atom_finding)

    # overlay atom positions on original image
    fig, axis = plt.subplots(figsize=(14, 14))
    axis.hold(True)
    col_map = plt.cm.jet
    axis.imshow(cropped_clean_image, interpolation='none', cmap="gray")
    for atom_type_ind, atom_centroids in enumerate(atom_labels):
        axis.scatter(atom_centroids[:, 1], atom_centroids[:, 0], color=col_map(255 * atom_type_ind / num_motifs))

    # fitting parameters
    fit_region_size = win_size * 0.80  # region to consider when fitting
    num_nearest_neighbors = 6  # to consider when fitting
    fitting_tolerance = 1E-4
    max_function_evals = 100
    min_amplitude = 0  # min amplitude limit for gauss fit
    max_amplitude = 2  # max amplitude limit for gauss fit
    position_range = win_size / 2  # range that the fitted position can go from initial guess position[pixels]
    gauss_width_guess = psf_width * 2
    min_gauss_width_ratio = 0.5  # min width of gauss fit ratio
    max_gauss_width_ratio = 2  # max width of gauss fit ratio


    def multi_gauss_surface_fit(coef_mat, s_mat):

        x = s_mat[:, :, 0]
        y = s_mat[:, :, 1]
        num_peaks = coef_mat.shape[0];
        multi_gauss = np.zeros(shape=x.shape, dtype=np.float32);

        for peak_ind in range(num_peaks):
            amp = coef_mat[peak_ind, 0]
            x_val = coef_mat[peak_ind, 1]
            y_val = coef_mat[peak_ind, 2]
            sigma = coef_mat[peak_ind, 3]
            gauss = amp * np.exp(-((x - x_val) ** 2 + (y - y_val) ** 2) / sigma ** 2)
            multi_gauss += gauss

        return multi_gauss


    def gauss_2d_residuals_new(parms_vec, orig_data_mat, x_data_mat):

        # Only need to reshape the parms from 1D to 2D
        parms_mat = np.reshape(parms_vec, (-1, 4))

        err = orig_data_mat - multi_gauss_surface_fit(parms_mat, x_data_mat)
        return err.ravel()


    all_atom_guesses = np.hstack(tuple(atom_labels))
    all_atom_fits = np.zeros(shape=all_atom_guesses.shape, dtype=np.float32)
    num_f_evals = np.zeros(shape=all_atom_guesses.shape[0], dtype=np.uint32)

    N_atoms = all_atom_guesses.shape[0]  # number of atoms

    # build distance matrix
    pos_vec = all_atom_guesses[:, 0] + 1j * all_atom_guesses[:, 1]

    pos_mat1 = np.tile(np.transpose(np.atleast_2d(pos_vec)), [1, N_atoms])
    pos_mat2 = np.transpose(pos_mat1)

    d_mat = np.abs(pos_mat2 - pos_mat1)  # matrix of distances between all atoms

    # sort the distance matrix and keep only the atoms within the nearest neighbor limit
    neighbor_dist_order = np.argsort(d_mat)
    # neighbor dist order has the (indices of the) neighbors for each atom sorted by distance
    closest_neighbors_mat = neighbor_dist_order[:, 1:num_nearest_neighbors + 1]

    # example of set up and procedure for a single fit:
    atom_ind = int(np.round(N_atoms * (9. / 24)))  # chosen atom index
    print('Atom #{} of {} at row {}, col {} selected'.format(atom_ind, N_atoms, round(all_atom_guesses[atom_ind, 0]),
                                                             round(all_atom_guesses[atom_ind, 1])))

    atom_ind = 0

    # for this atom, do:
    for atom_ind in range(25):
        if atom_ind % 10 == 0:
            print('Working on atom {} of {}'.format(atom_ind, N_atoms))

        x_center_atom = all_atom_guesses[atom_ind, 0]
        y_center_atom = all_atom_guesses[atom_ind, 1]
        x_neighbor_atoms = all_atom_guesses[closest_neighbors_mat[atom_ind], 0]
        y_neighbor_atoms = all_atom_guesses[closest_neighbors_mat[atom_ind], 1]
        x_range = slice(max(int(np.round(x_center_atom - fit_region_size)), 0),
                        min(int(np.round(x_center_atom + fit_region_size)), cropped_clean_image.shape[0]))
        y_range = slice(max(int(np.round(y_center_atom - fit_region_size)), 0),
                        min(int(np.round(y_center_atom + fit_region_size)), cropped_clean_image.shape[1]))
        fit_region = cropped_clean_image[x_range, y_range]

        # define x and y fitting range
        s1, s2 = np.meshgrid(range(x_range.start, x_range.stop), range(y_range.start, y_range.stop))
        s_mat = np.dstack((s1.T, s2.T))

        # initial guess values
        x_guess = np.hstack((x_center_atom, x_neighbor_atoms))  # get x-guess from pattern matching results
        y_guess = np.hstack((y_center_atom, y_neighbor_atoms))  # get y-guess from pattern matching results
        a_guess = cropped_clean_image[np.uint32(x_guess), np.uint32(y_guess)]
        sigma_guess = gauss_width_guess * np.ones(num_nearest_neighbors + 1)

        coef_guess_mat = np.transpose(np.vstack((a_guess, x_guess, y_guess, sigma_guess)))

        # Set up upper and lower bounds:
        lb_mat = [min_amplitude * np.ones(num_nearest_neighbors + 1),
                  coef_guess_mat[:, 1] - position_range,
                  coef_guess_mat[:, 2] - position_range,
                  gauss_width_guess * 0.5 * np.ones(num_nearest_neighbors + 1)]

        ub_mat = [max_amplitude * np.ones(num_nearest_neighbors + 1),
                  coef_guess_mat[:, 1] + position_range,
                  coef_guess_mat[:, 2] + position_range,
                  gauss_width_guess * 2 * np.ones(num_nearest_neighbors + 1)]
        lb_mat = np.transpose(lb_mat)
        ub_mat = np.transpose(ub_mat)

        # Now refine the positions!
        plsq = least_squares(gauss_2d_residuals_new, coef_guess_mat.ravel(), args=(fit_region, s_mat),
                             bounds=(lb_mat.ravel(), ub_mat.ravel()), jac='3-point', max_nfev=max_function_evals)
        coeff_fit_mat = np.reshape(plsq.x, (-1, 4))

        # Store the position of the central (first) atom
        all_atom_fits[atom_ind] = coeff_fit_mat[0, 1: 3]
        num_f_evals[atom_ind] = plsq.nfev

    fitting_parms = {'fit_region_size': fit_region_size,
                     'gauss_width_guess': gauss_width_guess,
                     'num_nearest_neighbors': num_nearest_neighbors,
                     'min_amplitude': min_amplitude,
                     'max_amplitude': max_amplitude,
                     'position_range': position_range,
                     'max_function_evals': max_function_evals,
                     'min_gauss_width_ratio': min_gauss_width_ratio,
                     'max_gauss_width_ratio': max_gauss_width_ratio}

    parm_dict = {'atom_pos_guess': all_atom_guesses,
                 'nearest_neighbors': closest_neighbors_mat,
                 'cropped_cleaned_image': cropped_clean_image}


    def fit_atom_pos(single_parm):

        atom_ind = single_parm[0]
        parm_dict = single_parm[1]
        fitting_parms = single_parm[2]

        all_atom_guesses = parm_dict['atom_pos_guess']
        closest_neighbors_mat = parm_dict['nearest_neighbors']
        cropped_clean_image = parm_dict['cropped_cleaned_image']

        fit_region_size = fitting_parms['fit_region_size']
        gauss_width_guess = fitting_parms['gauss_width_guess']
        num_nearest_neighbors = fitting_parms['num_nearest_neighbors']
        min_amplitude = fitting_parms['min_amplitude']
        max_amplitude = fitting_parms['max_amplitude']
        position_range = fitting_parms['position_range']
        max_function_evals = fitting_parms['max_function_evals']
        min_gauss_width_ratio = fitting_parms['min_gauss_width_ratio']
        max_gauss_width_ratio = fitting_parms['max_gauss_width_ratio']
        min_amplitude = fitting_parms['min_amplitude']
        min_amplitude = fitting_parms['min_amplitude']

        x_center_atom = all_atom_guesses[atom_ind, 0]
        y_center_atom = all_atom_guesses[atom_ind, 1]
        x_neighbor_atoms = all_atom_guesses[closest_neighbors_mat[atom_ind], 0]
        y_neighbor_atoms = all_atom_guesses[closest_neighbors_mat[atom_ind], 1]
        x_range = slice(max(int(np.round(x_center_atom - fit_region_size)), 0),
                        min(int(np.round(x_center_atom + fit_region_size)),
                            cropped_clean_image.shape[0]))
        y_range = slice(max(int(np.round(y_center_atom - fit_region_size)), 0),
                        min(int(np.round(y_center_atom + fit_region_size)),
                            cropped_clean_image.shape[1]))
        fit_region = cropped_clean_image[x_range, y_range]

        # define x and y fitting range
        s1, s2 = np.meshgrid(range(x_range.start, x_range.stop),
                             range(y_range.start, y_range.stop))
        s_mat = np.dstack((s1.T, s2.T))

        # initial guess values
        x_guess = np.hstack((x_center_atom, x_neighbor_atoms))
        y_guess = np.hstack((y_center_atom, y_neighbor_atoms))
        a_guess = cropped_clean_image[np.uint32(x_guess), np.uint32(y_guess)]
        sigma_guess = gauss_width_guess * np.ones(num_nearest_neighbors + 1)

        coef_guess_mat = np.transpose(np.vstack((a_guess, x_guess,
                                                 y_guess, sigma_guess)))

        # Set up upper and lower bounds:
        lb_mat = [min_amplitude * np.ones(num_nearest_neighbors + 1),
                  coef_guess_mat[:, 1] - position_range,
                  coef_guess_mat[:, 2] - position_range,
                  min_gauss_width_ratio * gauss_width_guess * np.ones(num_nearest_neighbors + 1)]

        ub_mat = [max_amplitude * np.ones(num_nearest_neighbors + 1),
                  coef_guess_mat[:, 1] + position_range,
                  coef_guess_mat[:, 2] + position_range,
                  max_gauss_width_ratio * gauss_width_guess * np.ones(num_nearest_neighbors + 1)]
        lb_mat = np.transpose(lb_mat)
        ub_mat = np.transpose(ub_mat)

        # Now refine the positions!
        plsq = least_squares(gauss_2d_residuals_new,
                             coef_guess_mat.ravel(),
                             args=(fit_region, s_mat),
                             bounds=(lb_mat.ravel(), ub_mat.ravel()),
                             jac='3-point', max_nfev=max_function_evals)
        coeff_fit_mat = np.reshape(plsq.x, (-1, 4))

        # Store the position of the central (first) atom
        return coeff_fit_mat[0, 1: 3]


    import itertools as itt

    parm_list = itt.izip(range(5), itt.repeat(parm_dict), itt.repeat(fitting_parms))
    fitted_atom_pos = []
    for atom_ind, single_parm in enumerate(parm_list):
        if atom_ind % 10 == 0:
            print('Working on atom {} of {}'.format(atom_ind, N_atoms))
        fitted_atom_pos.append(fit_atom_pos(single_parm))

    for guess0, data1, data2 in zip(all_atom_guesses[:25], all_atom_fits[:25], fitted_atom_pos[:25]):
        print guess0, data1, data2

    print x_range
    print y_range
    print fit_region_size
    print('Center atom position:', str(all_atom_guesses[atom_ind]))
    print('\tAmplitude\tx position\ty position\tsigma')
    print(coef_guess_mat)
    print('-----')
    print(lb_mat)
    print('------')
    print(ub_mat)

    # Visualize the set of atoms that will be fitted
    fig, axes = plt.subplots(ncols=2, figsize=(14, 7))
    axes[0].hold(True)
    col_map = plt.cm.jet
    axes[0].imshow(cropped_clean_image, interpolation='none', cmap="gray")
    axes[0].add_patch(patches.Rectangle((all_atom_guesses[atom_ind, 1] - fit_region_size,
                                         all_atom_guesses[atom_ind, 0] - fit_region_size),
                                        2 * fit_region_size, 2 * fit_region_size, fill=False,
                                        color='orange', linewidth=2))
    axes[0].scatter(all_atom_guesses[:, 1], all_atom_guesses[:, 0], color='yellow')
    axes[0].scatter(all_atom_guesses[atom_ind, 1], all_atom_guesses[atom_ind, 0], color='red')
    axes[0].scatter(coef_guess_mat[1:, 2], coef_guess_mat[1:, 1], color='pink')
    axes[1].imshow(fit_region, interpolation='none', cmap="gray");

    centered_guess_mat = np.copy(coef_guess_mat)
    centered_guess_mat[:, 1] -= all_atom_guesses[atom_ind, 0] - (0.5 * fit_region.shape[0])
    centered_guess_mat[:, 2] -= all_atom_guesses[atom_ind, 1] - (0.5 * fit_region.shape[1])

    gauss_2d_guess = multi_gauss_surface_fit(coef_guess_mat, s_mat)
    fig, axes = plt.subplots(ncols=2, figsize=(14, 6))
    axes[0].hold(True)
    axes[0].imshow(fit_region, cmap="gray")
    axes[0].set_title('Original data')
    axes[0].scatter(centered_guess_mat[:, 2], centered_guess_mat[:, 1], color='orange')
    axes[0].scatter(centered_guess_mat[0, 2], centered_guess_mat[0, 1], color='red')
    axes[1].hold(True)
    axes[1].imshow(gauss_2d_guess, cmap="gray")
    axes[1].scatter(centered_guess_mat[:, 2], centered_guess_mat[:, 1], color='orange')
    axes[1].scatter(centered_guess_mat[0, 2], centered_guess_mat[0, 1], color='red')
    axes[1].set_title('Guess data');

    plsq = least_squares(gauss_2d_residuals_new, coef_guess_mat.ravel(), args=(fit_region, s_mat),
                         bounds=(lb_mat.ravel(), ub_mat.ravel()), jac='3-point', max_nfev=100)
    coeff_fit_mat = np.reshape(plsq.x, (-1, 4))

    print('Number of evaulations of the function: {} and jacobian: {}'.format(plsq.nfev, plsq.njev))
    gauss_2d_fit = multi_gauss_surface_fit(coeff_fit_mat, s_mat)

    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(12, 12))
    for axis, img_mat, coeff_mat, img_title in zip(axes.flat,
                                                   [fit_region, gauss_2d_fit, gauss_2d_guess, fit_region],
                                                   [coef_guess_mat, coeff_fit_mat, coef_guess_mat, coeff_fit_mat],
                                                   ['Original + guess pos', 'Fit', 'Guess', 'Original + fit pos']):
        centered_pos_mat = np.copy(coeff_mat[:, 1:3])
        centered_pos_mat[:, 0] -= all_atom_guesses[atom_ind, 0] - (0.5 * fit_region.shape[0])
        centered_pos_mat[:, 1] -= all_atom_guesses[atom_ind, 1] - (0.5 * fit_region.shape[1])

        axis.hold(True)
        axis.imshow(img_mat, cmap="gray")
        axis.set_title(img_title)
        axis.scatter(centered_pos_mat[1:, 1], centered_pos_mat[1:, 0], color='orange')
        axis.scatter(centered_pos_mat[0, 1], centered_pos_mat[0, 0], color='red')



    pass
