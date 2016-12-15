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
from pycroscopy.processing.image_processing import ImageWindow
from pycroscopy.processing.svd_utils import doSVD
from pycroscopy.viz.plot_utils import plotScree, plot_map_stack, plotSpectrograms
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
    save_plots  = True
    show_plots  = False
    fit_win     = True
    num_peaks   = 2
    max_mem     = 1024*2
    num_comp    = None
    plot_comps  = 9
    clean_comps = None

    '''
    Parse the path
    '''
    folder, filename = os.path.split(os.path.abspath(imagepath))
    basename, _ = os.path.splitext(filename)

    h5_path = os.path.join(folder, basename+'.h5')

    '''
    Initialize the windowing
    '''
    iw = ImageWindow(imagepath, h5_path, reset=True, max_RAM_mb=max_mem)

    h5_file = iw.h5_file
    
    h5_raw = iw.h5_raw

    '''
    Extract an optimum window size from the image
    '''
    win_size = iw.window_size_extract(h5_raw,
                                      num_peaks,
                                      save_plots=save_plots,
                                      show_plots=show_plots)

    # win_size = 16
    '''
    Do the windowing
    '''
    t0 = time()
    h5_wins = iw.do_windowing(h5_raw,
                              win_x=win_size,
                              win_y=win_size,
                              # win_step_x=3,
                              # win_step_y=3,
                              save_plots=save_plots,
                              show_plots=show_plots)
    print 'Windowing took {} seconds.'.format(time()-t0)

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

    stdevs = 2
    plot_comps = min(plot_comps, h5_S.size)

    print 'Plotting Loading Maps'
    fig202, axes202 = plot_map_stack(np.reshape(h5_U[:, :plot_comps], [num_rows, num_cols, -1]),
                                     num_comps=plot_comps, stdevs=2, show_colorbar=True)
    fig202.savefig(os.path.join(folder, svd_name+'_PCA_Loadings.png'), format='png', dpi=300)
    plt.close('all')
    del fig203, fig202, axes203, axes202

    print 'Plotting Eigenvectors'
    num_x = int(np.sqrt(h5_V.shape[1]))
    fig201, axes201 = plotSpectrograms(np.reshape(h5_V[:plot_comps, :],
                                                  [-1, num_x, num_x]),
                                       num_comps=plot_comps,
                                       title='Eigenvectors')
    fig201.savefig(os.path.join(folder, svd_name+'_PCA_Eigenvectors.png'), format='png', dpi=300)
    plt.close('all')
    del fig201, axes201

    '''
    Build a cleaned image from the cleaned windows
    '''
    # h5_wins = h5_raw.parent['Raw_Data-Windowing_000']['Image_Windows']
    # t0 = time()
    # h5_clean_image = iw.clean_and_build(h5_win=h5_wins, components=clean_comps)
    # print 'Cleaning and rebuilding image took {} seconds.'.format(time()-t0)

    # t0 = time()
    # h5_clean_image = iw.clean_and_build_batch(h5_win=h5_wins, components=clean_comps)
    # print 'Batch cleaning and rebuilding image took {} seconds.'.format(time()-t0)

    # prof_file = os.path.join(folder, 'clean_profile.txt')
    # run_string = 'h5_clean_image = iw.clean_and_build_batch(h5_win=h5_wins, components=clean_comps)'
    # cProfile.run(run_string, filename=prof_file)
    # cistats = pstats.Stats(prof_file)
    # print 'Clean&Rebuild Image Profiling -- Sorted by total time'
    # cistats.sort_stats('time')
    # cistats.print_stats(25)
    # print 'Clean&Rebuild Image Profiling -- Sorted by cumulative time'
    # cistats.sort_stats('cumulative')
    # cistats.print_stats(25)

    # iw.plot_clean_image(h5_clean_image)

    # t0 = time()
    # h5_clean_image = iw.clean_and_build_separate_components(h5_win=h5_wins, components=clean_comps)
    # print 'Batch cleaning and rebuilding image took {} seconds.'.format(time()-t0)

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
                                     num_comps=plot_comps, stdevs=2, show_colorbar=True)
    fig202.savefig(os.path.join(folder, clean_name+'_Components.png'), format='png', dpi=300)
    plt.close('all')
    del fig202, axes202

    pass
