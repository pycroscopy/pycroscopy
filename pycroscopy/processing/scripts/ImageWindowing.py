'''
Created on Jun 22, 2016

@author: Chris Smith -- csmith55@utk.edu
'''
import os
import numpy as np
import matplotlib.pyplot as plt
from time import time
from pycroscopy.processing.image_processing import ImageWindow
from pycroscopy.analysis.PCAutils import doPCA, plotScree, plotLoadingMaps, plotSpectrograms
from pycroscopy.io.ioUtils import uiGetFile

if __name__ == '__main__':
    imagepath = uiGetFile(filter='Image File (*.tiff, *.jpg, *.png', caption='Select Image File')

    save_plots  = True
    show_plots  = False
    fit_win     = True
    num_peaks   = 2
    max_mem     = 1024*4
    num_comp    = 64
    plot_comps  = 9
    clean_comps = 5
    
    folder, filename = os.path.split(os.path.abspath(imagepath))
    basename, _ = os.path.splitext(filename)

    h5_path = os.path.join(folder, basename+'.h5')
    
    iw = ImageWindow(imagepath, h5_path, reset=True)

    h5_file = iw.h5_file
    
    h5_raw = iw.h5_raw
    
    '''
    Do the normalization
    '''
    t0 = time()
    h5_norm = iw.normalize_image(h5_raw)
    print 'Normalization took {} seconds.'.format(time()-t0)

    '''
    Do the windowing on the normalized image
    '''
    t0 = time()
    h5_wins = iw.do_windowing(h5_norm, num_peaks=num_peaks, fit_win=fit_win, save_plots=save_plots, show_plots=show_plots)
    print 'Windowing took {} seconds.'.format(time()-t0)

    '''
    Do PCA on the windowed image
    '''
    h5_pca = doPCA(iw.hdf, h5_wins, num_comps=num_comp, max_mem=max_mem)

    h5_U = h5_pca['U']
    h5_S = h5_pca['S']
    h5_V = h5_pca['V']

    h5_pos   = iw.hdf.file[h5_wins.attrs['Position_Indices']]
    num_rows = len(np.unique(h5_pos[:,0]))
    num_cols = len(np.unique(h5_pos[:,1]))

    pca_name = '_'.join([basename,h5_pca.name.split('/')[-1]])

    print 'Plotting Scree'
    fig203, axes203 = plotScree(h5_S[()])
    fig203.savefig(os.path.join(folder,pca_name+'_PCA_scree.png'), format='png', dpi=300)
    plt.close(fig203)

    stdevs = 2

    print 'Plotting Loading Maps'
    fig202, axes202 = plotLoadingMaps(np.reshape(h5_U[:,:plot_comps],[num_rows,num_cols,-1]), num_comps=plot_comps, stdevs=2, show_colorbar=True)
    fig202.savefig(os.path.join(folder,pca_name+'_PCA_Loadings.png'), format='png', dpi=300)
    plt.close(fig202)
    del fig203, fig202, axes203,axes202

    print 'Plotting Eigenvectors'
    num_x = int(np.sqrt(h5_V.shape[1]))
    fig201, axes201 = plotSpectrograms(np.reshape(h5_V[:plot_comps,:],[-1,num_x,num_x]), num_comps=plot_comps, title='Eigenvectors')
    fig201.savefig(os.path.join(folder,pca_name+'_PCA_Eigenvectors.png'), format='png', dpi=300)
    plt.close(fig201)
    del fig201,axes201

    '''
    Build a cleaned image from the cleaned windows
    '''
    t0 = time()
    h5_clean_image = iw.clean_and_build(h5_win=h5_wins, components=clean_comps)
    print 'Cleaning and rebuilding image took {} seconds.'.format(time()-t0)

    iw.plot_clean_image(h5_clean_image)

    pass
