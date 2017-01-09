"""
Created on Jun 22, 2016

@author: Chris Smith -- csmith55@utk.edu
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from time import time
import sys
import cProfile
import pstats
sys.path.append('../../../')
from pycroscopy import ImageTranslator
from pycroscopy.processing.image_processing import ImageWindow
from pycroscopy.processing.svd_utils import doSVD
from pycroscopy import Cluster
from pycroscopy.viz.plot_utils import plotScree, plot_map_stack, plotSpectrograms, plotClusterResults
from pycroscopy.io.io_utils import uiGetFile

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
    win_fft         = 'abs'     # Options are None, 'abs', or 'complex'
    num_peaks       = 2         # Number of Peaks to use in window size fitting
    max_mem         = 1024*2    # Maximum memory to use in calculations, in Mb
    num_comp        = 128       # Number of Components to generate
    plot_comps      = 16        # Number of Components to plot, plot_comps<=num_comps
    # Components to use for image cleaning.  Can be integer, slice, or list of integers
    clean_comps     = [1, 2, 3, 4, 7, 8, 9, 10]
    clean_method    = 'batch'   # Options are 'normal', 'batch', or 'components'
    cluster_method  = 'KMeans'  # See Cluster documentation for options
    num_cluster     = 16        # Number of Clusters to use
    plot_clust      = 16         # Number of clusters to plot, plot_clust<=num_cluster

    '''
    Parse the path to the image file
    '''
    folder, filename = os.path.split(os.path.abspath(imagepath))
    basename, _ = os.path.splitext(filename)

    '''
    Read the image into the hdf5 file
    '''
    tl = ImageTranslator()
    h5_raw = tl.translate(imagepath)
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
    Create the windows.
    '''
    t0 = time()
    h5_wins = iw.do_windowing(win_x=win_size,
                              win_y=win_size,
                              save_plots=save_plots,
                              show_plots=show_plots,
                              win_fft='complex')
    print 'Windowing took {} seconds.'.format(time()-t0)

    if profiling:
        prof_file = os.path.join(folder, 'window_profile.txt')
        run_string = 'h5_wins = iw.do_windowing(win_x=win_size, win_y=win_size, save_plots=save_plots,'+ \
                     ' show_plots=show_plots, win_fft=\'complex\')'
        cProfile.run(run_string, filename=prof_file)
        cistats = pstats.Stats(prof_file)
        print 'Window Creation Image Profiling -- Sorted by total time'
        cistats.sort_stats('time')
        cistats.print_stats(25)
        print 'Window Creation Image Profiling -- Sorted by cumulative time'
        cistats.sort_stats('cumulative')
        cistats.print_stats(25)

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
        fig201, axes201 = plotSpectrograms(np.reshape(h5_V[:plot_comps, :][field],
                                                      [-1, num_x, num_x]),
                                           num_comps=plot_comps,
                                           title='Eigenvectors')
        fig201.savefig(os.path.join(folder, svd_name+'_'+field+'_PCA_Eigenvectors.png'), format='png', dpi=300)
        plt.close('all')
        del fig201, axes201

    '''
    Build a cleaned image from the cleaned windows
    '''
    if clean_method == 'normal':
        h5_wins = h5_raw.parent['Raw_Data-Windowing_000']['Image_Windows']
        t0 = time()
        h5_clean_image = iw.clean_and_build(h5_win=h5_wins, components=clean_comps)
        print 'Cleaning and rebuilding image took {} seconds.'.format(time()-t0)
    elif clean_method == 'batch':
        t0 = time()
        h5_clean_image = iw.clean_and_build_batch(h5_win=h5_wins, components=clean_comps)
        print 'Batch cleaning and rebuilding image took {} seconds.'.format(time()-t0)

        if profiling:
            prof_file = os.path.join(folder, 'clean_profile.txt')
            run_string = 'h5_clean_image = iw.clean_and_build_batch(h5_win=h5_wins, components=clean_comps)'
            cProfile.run(run_string, filename=prof_file)
            cistats = pstats.Stats(prof_file)
            print 'Clean&Rebuild Image Profiling -- Sorted by total time'
            cistats.sort_stats('time')
            cistats.print_stats(25)
            print 'Clean&Rebuild Image Profiling -- Sorted by cumulative time'
            cistats.sort_stats('cumulative')
            cistats.print_stats(25)

        iw.plot_clean_image(h5_clean_image)

    elif clean_method == 'components':
        t0 = time()
        h5_clean_image = iw.clean_and_build_separate_components(h5_win=h5_wins, components=clean_comps)
        print 'Batch cleaning and rebuilding image took {} seconds.'.format(time()-t0)

        if profiling:
            prof_file = os.path.join(folder, 'clean_profile.txt')
            run_string = 'h5_clean_image = iw.clean_and_build_separate_components(h5_win=h5_wins, components=clean_comps)'
            cProfile.run(run_string, filename=prof_file)
            cistats = pstats.Stats(prof_file)
            print 'Clean&Rebuild Image Profiling -- Sorted by total time'
            cistats.sort_stats('time')
            cistats.print_stats(25)
            print 'Clean&Rebuild Image Profiling -- Sorted by cumulative time'
            cistats.sort_stats('cumulative')
            cistats.print_stats(25)

        im_x = h5_wins.parent.attrs['image_x']
        im_y = h5_wins.parent.attrs['image_y']
        clean_name = '_'.join([basename, h5_clean_image.name.split('/')[-1]])

        plot_comps = min(plot_comps, h5_clean_image.shape[1])

        fig202, axes202 = plot_map_stack(np.reshape(h5_clean_image[:, :plot_comps], [im_x, im_y, plot_comps]),
                                         num_comps=plot_comps, stdevs=2, color_bar_mode='each')
        fig202.savefig(os.path.join(folder, clean_name+'_Components.png'), format='png', dpi=300)
        plt.close('all')
        del fig202, axes202

    if cluster_method is not None:
        '''
        Do KMeans on the Windows
        '''
        km_name = '_'.join([basename, 'Image_Windows'])

        cluster = Cluster(h5_wins, method_name=cluster_method, num_comps=None, n_jobs=6, n_clusters=num_cluster)

        h5_kmeans_raw = cluster.do_cluster(rearrange_clusters=True)

        h5_labels_raw = h5_kmeans_raw['Labels']
        h5_centroids_raw = h5_kmeans_raw['Mean_Response']
        h5_km_spec_raw = h5_file[h5_centroids_raw.attrs['Spectroscopic_Values']][1]

        fig601, ax601 = plotClusterResults(h5_labels_raw, h5_centroids_raw, spec_val=h5_km_spec_raw)
        fig601.savefig(os.path.join(folder, km_name + '_KMeans.png'), format='png', dpi=300)

        plt.close('all')
        del fig601, ax601

        fig602, ax602 = plot_map_stack(np.reshape(h5_centroids_raw['Image Data'][:plot_clust, :],
                                                  [-1, win_size, win_size]).T,
                                       num_comps=plot_clust, stdevs=2, color_bar_mode='each')
        fig602.savefig(os.path.join(folder, km_name + '_Mean_Response.png'), format='png', dpi=300)
        plt.close('all')
        del fig602, ax602

        '''
        Do KMeans on the U of Windows
        '''
        km_name = '_'.join([basename, 'Image_Windows_U'])

        cluster = Cluster(h5_U, method_name=cluster_method, num_comps=None, n_jobs=6, n_clusters=num_cluster)

        h5_kmeans_U = cluster.do_cluster(rearrange_clusters=True)

        h5_labels_U = h5_kmeans_U['Labels']
        h5_centroids_U = h5_kmeans_U['Mean_Response']
        h5_km_spec_U = h5_file[h5_centroids_U.attrs['Spectroscopic_Values']]

        fig601, ax601 = plotClusterResults(h5_labels_U, h5_centroids_U, spec_val=h5_km_spec_U)
        fig601.savefig(os.path.join(folder, km_name + '_KMeans.png'), format='png', dpi=300)
        del fig601, ax601
        plt.close('all')

    pass
