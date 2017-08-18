# -*- coding: utf-8 -*-
"""
Created on Thu May 05 13:29:12 2016

@author: Suhas Somnath
"""

from __future__ import division, print_function, absolute_import
import sys
import itertools
from collections import Iterable
from multiprocessing import Pool, cpu_count
from warnings import warn
import matplotlib.pyplot as plt
import numpy as np
from time import time
from .fft import getNoiseFloor, noiseBandFilter, makeLPF, harmonicsPassFilter
from ..io.io_hdf5 import ioHDF5
from ..io.hdf_utils import getH5DsetRefs, linkRefs, getAuxData, link_as_main, copyAttributes, copy_main_attributes
from ..io.io_utils import getTimeStamp
from ..io.microdata import MicroDataGroup, MicroDataset
from ..viz.plot_utils import rainbow_plot
from ..io.translators.utils import build_ind_val_dsets

# TODO: Use filter_parms as a kwargs instead of a required input
# TODO: Phase rotation not implemented correctly. Find and use excitation frequency

###############################################################################


def test_filter(resp_wfm, filter_parms, samp_rate, show_plots=True, use_rainbow_plots=True,
                excit_wfm=None, central_resp_size=None, verbose=False):
    """
    Filters the provided response with the provided filters. Use this only to test filters.
    This function does not care about the file structure etc.
    
    Parameters
    ----------
    resp_wfm : 1D numpy float array
        Raw response waveform in the time domain
    filter_parms : dictionary
        Dictionary that contains all the filtering parameters, see Notes for details.
    samp_rate : unsigned int 
        Sampling rate in Hertz
    show_plots : (Optional) Boolean
        Whether or not to plot FFTs before and after filtering
    use_rainbow_plots : (Optional) Boolean
        Whether or not to plot loops whose color varied as a function of time
    excit_wfm : (Optional) 1D numpy float array
        Excitation waveform in the time domain. This waveform is necessary for plotting loops. 
    central_resp_size : (Optional) unsigned int
        Number of responce sample points from the center of the waveform to show in plots. Useful for SPORC
    verbose : (Optional) string
        Whether or not to print statements
    
    Returns
    -------
    filt_data : 1D numpy float array
        Filtered signal in the time domain

    Notes
    -----
    *Filter Parameters*

    noise_threshold : float
        0<1 eg 1E-4
    comb_[Hz] : Retain harmonics of frequency
        [first frequency, band width, number of harmonics]
    LPF_cutOff_[Hz] : float
        low pass frequency cut off frequency
    band_filt_[Hz] : 2D list
        [0] = center frequency, [1] = band widths
    phase_[rad] : float
        Compensation for instrumentation induced phase offset in radians
    samp_rate_[Hz] : unsigned int
        Sampling rate in Hz
    num_pix : unsigned int
        Number of pixels to filter simultaneously

    """
    num_pts = len(resp_wfm)
    
    show_loops = excit_wfm is not None and show_plots
    
    '''
    Get parameters from the dictionary.
    '''
    noise_band_filter = filter_parms.get('band_filt_[Hz]', 1)
    if isinstance(noise_band_filter, Iterable):
        noise_band_filter = noiseBandFilter(num_pts, samp_rate, noise_band_filter[0],
                                            noise_band_filter[1])
        if verbose and isinstance(noise_band_filter, Iterable):
            print('Calculated valid noise_band_filter')

    low_pass_filter = filter_parms.get('LPF_cutOff_[Hz]', -1)
    if low_pass_filter > 0:
        low_pass_filter = makeLPF(num_pts, samp_rate, low_pass_filter)
        if verbose and isinstance(low_pass_filter, Iterable):
            print('Calculated valid low pass filter')
    else:
        low_pass_filter = 1


    harmonic_filter = filter_parms.get('comb_[Hz]', 1)
    if isinstance(harmonic_filter, Iterable):
        harmonic_filter = harmonicsPassFilter(num_pts, samp_rate, harmonic_filter[0],
                                              harmonic_filter[1], harmonic_filter[2])
        if verbose and isinstance(harmonic_filter, Iterable):
            print('Calculated valid harmonic filter')

    composite_filter = noise_band_filter * low_pass_filter * harmonic_filter

    noise_floor = filter_parms.get('noise_threshold', None)
    fft_pix_data = np.fft.fftshift(np.fft.fft(resp_wfm))
    if 0 < noise_floor < 1:
        noise_floor = getNoiseFloor(fft_pix_data, noise_floor)[0]

    if show_plots:       
        l_ind = int(0.5*num_pts)
        if type(composite_filter) == np.ndarray:
            r_ind = np.max(np.where(composite_filter > 0)[0])
        else:
            r_ind = num_pts
        w_vec = np.linspace(-0.5*samp_rate, 0.5*samp_rate, num_pts)*1E-3
        if central_resp_size:
            sz = int(0.5*central_resp_size)
            l_resp_ind = -sz+l_ind
            r_resp_ind = l_ind+sz
        else:
            l_resp_ind = l_ind
            r_resp_ind = num_pts
        
        fig = plt.figure(figsize=(12, 8))
        lhs_colspan = 2
        if show_loops is False:
            lhs_colspan = 4
        else:
            ax_loops = plt.subplot2grid((2, 4), (0, 2), colspan=2, rowspan=2)
        ax_raw = plt.subplot2grid((2, 4), (0, 0), colspan=lhs_colspan)
        ax_filt = plt.subplot2grid((2, 4), (1, 0), colspan=lhs_colspan)
        axes = [ax_raw, ax_filt]
    else:
        fig = None
        axes = None

    if show_plots:
        amp = np.abs(fft_pix_data)
        ax_raw.semilogy(w_vec[l_ind:r_ind], amp[l_ind:r_ind])
        ax_raw.semilogy(w_vec[l_ind:r_ind], (composite_filter[l_ind:r_ind] + np.min(amp))*(np.max(amp)-np.min(amp)))
        if noise_floor is not None:
            ax_raw.semilogy(w_vec[l_ind:r_ind], np.ones(r_ind-l_ind)*noise_floor)
        ax_raw.set_title('Raw Signal')

    fft_pix_data *= composite_filter

    if noise_floor is not None:
        fft_pix_data[np.abs(fft_pix_data) < noise_floor] = 1E-16  # DON'T use 0 here. ipython kernel dies

    if show_plots:
        ax_filt.semilogy(w_vec[l_ind:r_ind], np.abs(fft_pix_data[l_ind:r_ind]))
        ax_filt.set_title('Filtered Signal')
        ax_filt.set_xlabel('Frequency(kHz)')
        if noise_floor is not None:
            ax_filt.set_ylim(bottom=noise_floor)  # prevents the noise threshold from messing up plots

    filt_data = np.real(np.fft.ifft(np.fft.ifftshift(fft_pix_data)))

    if show_loops:
        if use_rainbow_plots:
            rainbow_plot(ax_loops, excit_wfm[l_resp_ind:r_resp_ind], filt_data[l_resp_ind:r_resp_ind] * 1E+3)
        else:
            ax_loops.plot(excit_wfm[l_resp_ind:r_resp_ind], filt_data[l_resp_ind:r_resp_ind]*1E+3)
        ax_loops.set_title('AI vs AO') 
        ax_loops.set_xlabel('Input Bias (V)')
        ax_loops.set_ylabel('Deflection (mV)')
        axes.append(ax_loops)
        fig.tight_layout()
    return filt_data, fig, axes

# ##############################################################################


def fft_filter_dataset(h5_main, filter_parms, write_filtered=True, write_condensed=False, num_cores=None):
    # TODO: Can simplify this function substantially. Collapse / absorb the serial and parallel functions...
    """
    Filters G-mode data using specified filter parameters and writes results to file.
        
    Parameters
    ----------
    h5_main : HDF5 dataset object
        Dataset containing the raw data
    filter_parms : dictionary
        Dictionary that contains all the filtering parameters
    write_filtered : (optional) Boolean - default True
        Whether or not to write filtered data to file
    write_condensed : (optional) Boolean - default False
        Whether or not to write condensed filtered data to file
    num_cores : unsigned int
        Number of cores to use for processing data in parallel
        
    Returns
    -------
    HDF5 group reference containing filtered dataset
    """ 
    
    def __max_pixel_read(h5_raw, num_cores, store_filt=True, hot_bins=None, bytes_per_bin=2, max_RAM_gb=16):
        """
        Returns the maximum number of pixels that can be stored in memory considering the output data and the
        number of cores
        
        Parameters
        ----------
        h5_raw : HDF5 Dataset reference
            Dataset containing the raw data
        num_cores : unsigned int
            Number of cores to use for processing data in parallel
        store_filt : boolean (optional)
            Write filtered data back to h5
        hot_bins : 1D numpy array (optional)
            Bins in the frequency domain to be saved to h5
        bytes_per_bin : unsigned int (optional)
            bytes per unit in the raw data - typically 2 bytes
        max_RAM_gb : unsigned int (optional)
            Maximum system memory that can be used. This is NOT respected well. Please fix this.            
        """
        
        # double the memory requirement if storing filtered data
        if store_filt:
            bytes_per_bin *= 2   
        bytes_per_pix = h5_raw.shape[1] * bytes_per_bin
        # account for the hot bins separately
        if hot_bins is not None:
            bytes_per_pix += len(hot_bins) * 8  # complex64
        
        # Multiply this memory requirement per pixel by the number of cores
        bytes_per_pix *= num_cores
        
        max_pix = int(np.rint(max_RAM_gb*1024**3 / bytes_per_pix))
        max_pix = max(1, min(h5_raw.shape[0], max_pix))
        print('Allowed to read', max_pix, 'of', h5_raw.shape[0], 'pixels')

        return np.uint(max_pix)
        
    def __filter_chunk(raw_mat, parm_dict, recom_cores):
        """
        This function delegates the actual fitting responsibility to the
        appropriate function. Decides whether or not serial / parallel processing
        is appropriate, number of cores, number of chunks, etc.
        
        Parameters
        ----------
        raw_mat : 2D numpy array
            Raw data arranged as [repetition, points per measurement]
        parm_dict : Dictionary
            Parameters necessary for filtering
        recom_cores : unsigned int
            Number of cores to use for processing data in parallel
        
        Returns
        -------
        (noise_floors, filt_data, cond_data)
        
        noise_floors : 1D numpy array
            Contains the noise floors per set of measurements
        filt_data : 2D numpy array or None
            filtered data arranged as [repetition, points per measurement]
        cond_data : 2D complex numpy array
            [set of measurements, frequency bins containing data]

        """
        max_cores = max(1, cpu_count()-2)
            
        recom_chunks = int(raw_mat.shape[0]/recom_cores)

        print('recom cores:', recom_cores, 'Total pixels:', raw_mat.shape[0], ', Recom chunks:', recom_chunks)

        if recom_cores > 1 and recom_chunks < 10:
            min_jobs = 20
            reduced_cores = int(raw_mat.shape[0]/min_jobs)
            # intelligently set the cores now. 
            recom_cores = min(max_cores, reduced_cores)
            print('Not enough jobs per core. Reducing cores to', recom_cores)

        if recom_cores > 1:
            return filter_chunk_parallel(raw_mat, parm_dict, recom_cores)
        else:
            return filter_chunk_serial(raw_mat, parm_dict)

    max_cores = max(1, cpu_count() - 2)
    if not num_cores:
        num_cores = max_cores
    
    if write_filtered is False and write_condensed is False:
        warn('You need to write the filtered and/or the condensed dataset to the file')
        return

    if 'num_pix' not in filter_parms:
        filter_parms['num_pix'] = 1

    num_effective_pix = h5_main.shape[0]*1.0/filter_parms['num_pix']
    if num_effective_pix % 1 > 0:
        warn('Number of pixels not divisible by the number of pixels to use for FFT filter')
        return
    num_effective_pix = int(num_effective_pix)
        
    num_pts = h5_main.shape[1]*filter_parms['num_pix']

    noise_band_filter = 1
    low_pass_filter = 1
    harmonic_filter = 1

    if 'band_filt_[Hz]' in filter_parms:
        if isinstance(filter_parms['band_filt_[Hz]'], Iterable):
            band_filt = filter_parms['band_filt_[Hz]']
            noise_band_filter = noiseBandFilter(num_pts, filter_parms['samp_rate_[Hz]'], band_filt[0], band_filt[1])
    if 'LPF_cutOff_[Hz]' in filter_parms:
        if filter_parms['LPF_cutOff_[Hz]'] > 0:
            low_pass_filter = makeLPF(num_pts, filter_parms['samp_rate_[Hz]'], filter_parms['LPF_cutOff_[Hz]'])

    if 'comb_[Hz]' in filter_parms:
        if isinstance(filter_parms['comb_[Hz]'], Iterable):
            harmonic_filter = harmonicsPassFilter(num_pts, filter_parms['samp_rate_[Hz]'], filter_parms['comb_[Hz]'][0],
                                                  filter_parms['comb_[Hz]'][1], filter_parms['comb_[Hz]'][2])

    composite_filter = noise_band_filter * low_pass_filter * harmonic_filter
    
    # ioHDF now handles automatic indexing
    grp_name = h5_main.name.split('/')[-1] + '-FFT_Filtering_'

    doing_noise_floor_filter = False

    if 'noise_threshold' in filter_parms:
        if 0 < filter_parms['noise_threshold'] < 1:
            ds_noise_floors = MicroDataset('Noise_Floors',
                                           data=np.zeros(shape=num_effective_pix, dtype=np.float32))
            doing_noise_floor_filter = True
        else:
            # Illegal inputs will be deleted from dictionary
            warn('Provided noise floor threshold: {} not within (0,1)'.format(filter_parms['noise_threshold']))
            del filter_parms['noise_threshold']

    if not doing_noise_floor_filter and not isinstance(composite_filter, np.ndarray):
        warn("No filtering being performed on this dataset. Exiting!")

    if isinstance(composite_filter, np.ndarray):
        ds_comp_filt = MicroDataset('Composite_Filter', np.float32(composite_filter))
                                        
    grp_filt = MicroDataGroup(grp_name, h5_main.parent.name)
    filter_parms['timestamp'] = getTimeStamp()
    filter_parms['algorithm'] = 'GmodeUtils-Parallel'
    grp_filt.attrs = filter_parms
    if isinstance(composite_filter, np.ndarray):
        grp_filt.addChildren([ds_comp_filt])
    if doing_noise_floor_filter:
        grp_filt.addChildren([ds_noise_floors])

    if write_filtered:
        ds_filt_data = MicroDataset('Filtered_Data', data=[], maxshape=h5_main.maxshape,
                                    dtype=np.float32, chunking=h5_main.chunks, compression='gzip')
        grp_filt.addChildren([ds_filt_data])
    
    hot_inds = None

    h5_pos_inds = getAuxData(h5_main, auxDataName=['Position_Indices'])[0]
    h5_pos_vals = getAuxData(h5_main, auxDataName=['Position_Values'])[0]

    if write_condensed:
        hot_inds = np.where(composite_filter > 0)[0]
        hot_inds = np.uint(hot_inds[int(0.5*len(hot_inds)):])  # only need to keep half the data
        ds_spec_inds, ds_spec_vals = build_ind_val_dsets([int(0.5*len(hot_inds))], is_spectral=True,
                                                         labels=['hot_frequencies'], units=[''], verbose=False)
        ds_spec_vals.data = np.atleast_2d(hot_inds)  # The data generated above varies linearly. Override.
        ds_cond_data = MicroDataset('Condensed_Data', data=[], maxshape=(num_effective_pix, len(hot_inds)),
                                    dtype=np.complex, chunking=(1, len(hot_inds)), compression='gzip')
        grp_filt.addChildren([ds_spec_inds, ds_spec_vals, ds_cond_data])
        if filter_parms['num_pix'] > 1:
            # need to make new position datasets by taking every n'th index / value:
            new_pos_vals = np.atleast_2d(h5_pos_vals[slice(0, None, filter_parms['num_pix']), :])
            ds_pos_inds, ds_pos_vals = build_ind_val_dsets([int(np.unique(h5_pos_inds[:, dim_ind]).size /
                                                                filter_parms['num_pix'])
                                                            for dim_ind in range(h5_pos_inds.shape[1])],
                                                           is_spectral=False,
                                                           labels=h5_pos_inds.attrs['labels'],
                                                           units=h5_pos_inds.attrs['units'], verbose=False)
            h5_pos_vals.data = np.atleast_2d(new_pos_vals)  # The data generated above varies linearly. Override.
            grp_filt.addChildren([ds_pos_inds, ds_pos_vals])

    hdf = ioHDF5(h5_main.file)
    h5_filt_refs = hdf.writeData(grp_filt)
    if isinstance(composite_filter, np.ndarray):
        h5_comp_filt = getH5DsetRefs(['Composite_Filter'], h5_filt_refs)[0]
    if doing_noise_floor_filter:
        h5_noise_floors = getH5DsetRefs(['Noise_Floors'], h5_filt_refs)[0]
    
    # Now need to link appropriately:
    if write_filtered:
        h5_filt_data = getH5DsetRefs(['Filtered_Data'], h5_filt_refs)[0]
        copyAttributes(h5_main, h5_filt_data, skip_refs=False)
        if isinstance(composite_filter, np.ndarray):
            linkRefs(h5_filt_data, [h5_comp_filt])
        if doing_noise_floor_filter:
            linkRefs(h5_filt_data, [h5_noise_floors])

        """link_as_main(h5_filt_data, h5_pos_inds, h5_pos_vals,
                     getAuxData(h5_main, auxDataName=['Spectroscopic_Indices'])[0],
                     getAuxData(h5_main, auxDataName=['Spectroscopic_Values'])[0])"""
      
    if write_condensed:
        h5_cond_data = getH5DsetRefs(['Condensed_Data'], h5_filt_refs)[0]
        if isinstance(composite_filter, np.ndarray):
            linkRefs(h5_cond_data, [h5_comp_filt])
        if doing_noise_floor_filter:
            linkRefs(h5_cond_data, [h5_noise_floors])

        if filter_parms['num_pix'] > 1:
            h5_pos_inds = getH5DsetRefs(['Position_Indices'], h5_filt_refs)[0]
            h5_pos_vals = getH5DsetRefs(['Position_Values'], h5_filt_refs)[0]

        link_as_main(h5_cond_data, h5_pos_inds, h5_pos_vals,
                     getH5DsetRefs(['Spectroscopic_Indices'], h5_filt_refs)[0],
                     getH5DsetRefs(['Spectroscopic_Values'], h5_filt_refs)[0])
        
    rot_pts = 0
    if 'phase_rot_[pts]' in filter_parms.keys():
        rot_pts = int(filter_parms['phase_rot_[pts]'])
                  
    print('Filtering data now. Be patient, this could take a few minutes') 

    max_pix = __max_pixel_read(h5_main, max(1, cpu_count() - 2), store_filt=write_filtered, hot_bins=hot_inds,
                               bytes_per_bin=2, max_RAM_gb=16)
    # Ensure that whole sets of pixels can be read.
    max_pix = np.uint(filter_parms['num_pix']*np.floor(max_pix / filter_parms['num_pix']))
    
    parm_dict = {'filter_parms': filter_parms, 'composite_filter': composite_filter,
                 'rot_pts': rot_pts, 'hot_inds': hot_inds}

    t_start = time()
    st_pix = 0
    line_count = 0
    while st_pix < h5_main.shape[0]:
        en_pix = int(min(h5_main.shape[0], st_pix + max_pix))
        print('Reading pixels:', st_pix, 'to', en_pix, 'of', h5_main.shape[0])
        raw_mat = h5_main[st_pix:en_pix, :]
        # reshape to (set of pix, data in each set of pix)
        # print 'raw mat originally of shape:', raw_mat.shape
        raw_mat = raw_mat.reshape(-1, filter_parms['num_pix'] * raw_mat.shape[1])
        num_lines = raw_mat.shape[0]
        # print 'After collapsing pixels, raw mat now of shape:', raw_mat.shape
        (nse_flrs, filt_data, cond_data) = __filter_chunk(raw_mat, parm_dict, num_cores)
        # Insert things into appropriate HDF datasets
        print('Writing filtered data to h5')
        # print 'Noise floors of shape:', nse_flrs.shape
        if doing_noise_floor_filter:
            h5_noise_floors[line_count: line_count + num_lines] = nse_flrs
        if write_condensed:
            # print('Condensed data of shape:', cond_data.shape)
            h5_cond_data[line_count: line_count + num_lines, :] = cond_data
        if write_filtered:
            # print('Filtered data of shape:', filt_data.shape)
            h5_filt_data[st_pix:en_pix, :] = filt_data
        hdf.flush()
        st_pix = en_pix

    print('FFT filtering took {} seconds'.format(time() - t_start))

    if isinstance(composite_filter, np.ndarray):
        return h5_comp_filt.parent
    if doing_noise_floor_filter:
        return h5_noise_floors.parent
              
# #############################################################################


def filter_chunk_parallel(raw_data, parm_dict, num_cores):
    # TODO: Need to check to ensure that all cores are indeed being utilized
    """
    Filters the provided dataset in parallel
    
    Parameters
    ----------
    raw_data : 2D numpy array
        Raw data arranged as [repetition, points per measurement]
    parm_dict : Dictionary
        Parameters necessary for filtering
    num_cores : unsigned int
        Number of cores to use for processing data in parallel
    
    Returns
    -------
    (noise_floors, filt_data, cond_data)
    
    noise_floors : 1D numpy array
        Contains the noise floors per set of measurements
    filt_data : 2D numpy array or None
        filtered data arranged as [repetition, points per measurement]
    cond_data : 2D complex numpy array or None
        [set of measurements, frequency bins containing data]

    """
    # Make place-holders to hold the data:
    pix_per_set = parm_dict['filter_parms']['num_pix']
    num_sets = raw_data.shape[0]
    pts_per_set = raw_data.shape[1]
    # print('sending unit filter data of size', pts_per_set)

    noise_floors = None
    noise_thresh = None
    if 'noise_threshold' in parm_dict['filter_parms']:
        noise_thresh = parm_dict['filter_parms']['noise_threshold']
        if 0 < noise_thresh < 1:
            noise_floors = np.zeros(shape=num_sets, dtype=np.float32)
        else:
            noise_thresh = None

    filt_data = None
    if not parm_dict['rot_pts']:
        filt_data = np.zeros(shape=(num_sets*pix_per_set, int(pts_per_set/pix_per_set)), dtype=raw_data.dtype)

    cond_data = None
    if parm_dict['hot_inds'] is not None:
        cond_data = np.zeros(shape=(num_sets, parm_dict['hot_inds'].size), dtype=np.complex64)
        
    # Set up single parameter:
    if sys.version_info.major == 3:
        zip_fun = zip
    else:
        zip_fun = itertools.izip
    sing_parm = zip_fun(raw_data, itertools.repeat(parm_dict))
    
    # Setup parallel processing:
    # num_cores = 10
    pool = Pool(processes=num_cores, maxtasksperchild=None)
    
    # Start parallel processing:
    num_chunks = int(np.ceil(raw_data.shape[0]/num_cores))
    parallel_results = pool.imap(unit_filter, sing_parm, chunksize=num_chunks)
    pool.close()
    pool.join()
    
    print('Done parallel computing. Now extracting data and populating matrices')
    
    # Extract data for each line...
    print_set = np.linspace(0, num_sets-1, 10, dtype=int)
    for set_ind, current_results in enumerate(parallel_results):
        if set_ind in print_set:
            print('Reading...', np.rint(100 * set_ind / num_sets), '% complete')

        temp_noise, filt_data_set, cond_data_set = current_results

        if noise_thresh is not None:
            noise_floors[set_ind] = temp_noise
        if parm_dict['hot_inds'] is not None:
            cond_data[set_ind, :] = cond_data_set
        if parm_dict['rot_pts'] is not None:
            filt_data[set_ind*pix_per_set:(set_ind+1)*pix_per_set, :] = filt_data_set
            
    return noise_floors, filt_data, cond_data
 

def filter_chunk_serial(raw_data, parm_dict):
    """
    Filters the provided dataset serially
    
    Parameters
    ----------
    raw_data : 2D numpy array
        Raw data arranged as [repetition, points per measurement]
    parm_dict : Dictionary
        Parameters necessary for filtering
    
    Returns
    -------
    (noise_floors, filt_data, cond_data)
    
    noise_floors : 1D numpy array
        Contains the noise floors per set of measurements
    filt_data : 2D numpy array or None
        filtered data arranged as [repetition, points per measurement]
    cond_data : 2D complex numpy array or None
        [set of measurements, frequency bins containing data]

    """
    # Make place-holders to hold the data:
    pix_per_set = parm_dict['filter_parms']['num_pix']
    num_sets = raw_data.shape[0]
    pts_per_set = raw_data.shape[1]
    # print ('sending unit filter data of size', pts_per_set)

    noise_floors = None
    noise_thresh = None
    if 'noise_threshold' in parm_dict['filter_parms']:
        noise_thresh = parm_dict['filter_parms']['noise_threshold']
        if 0 < noise_thresh < 1:
            noise_floors = np.zeros(shape=num_sets, dtype=np.float32)
        else:
            noise_thresh = None

    filt_data = None
    if parm_dict['rot_pts'] is not None:
        filt_data = np.zeros(shape=(num_sets*pix_per_set, int(pts_per_set/pix_per_set)), dtype=raw_data.dtype)

    cond_data = None
    if parm_dict['hot_inds'] is not None:
        cond_data = np.zeros(shape=(num_sets, parm_dict['hot_inds'].size), dtype=np.complex64)
    
    # Filter each line
    print_set = np.linspace(0, num_sets-1, 10, dtype=int)
    for set_ind in range(num_sets):
        if set_ind in print_set:
            print('Reading...', np.rint(100 * set_ind / num_sets), '% complete')

        # parm_dict['t_raw'] = raw_data[set_ind,:]
        (temp_noise, filt_data_set, cond_data_set) = unit_filter((raw_data[set_ind, :], parm_dict))
        if noise_thresh is not None:
            noise_floors[set_ind] = temp_noise
        if parm_dict['hot_inds'] is not None:
            cond_data[set_ind, :] = cond_data_set
        if parm_dict['rot_pts'] is not None:
            filt_data[set_ind*pix_per_set:(set_ind+1)*pix_per_set, :] = filt_data_set
            
    return noise_floors, filt_data, cond_data

     
def unit_filter(single_parm):
    """
    Filters a single instance of a signal. 
    This is the function that is called in parallel
    
    Parameters
    ----------
    single_parm : Tuple
        Parameters and data for filtering a single data instance. 
        Constructed as (raw_data_vec, parm_dict)
    
    Returns
    -------
    noise_floors : float
        Noise floor
    filt_data : 1D numpy array or None
        filtered data
    cond_data : 1D complex numpy array or None
        frequency bins containing data
    """
    # unpack all the variables from the sole input
    t_raw = single_parm[0]
    parm_dict = single_parm[1]
    # t_raw = parm_dict['t_raw']
    filter_parms = parm_dict['filter_parms']
    composite_filter = parm_dict['composite_filter']
    rot_pts = parm_dict['rot_pts']
    hot_inds = parm_dict['hot_inds']

    noise_floor = None

    t_raw = t_raw.reshape(-1)
    f_data = np.fft.fftshift(np.fft.fft(t_raw))

    if 'noise_threshold' in filter_parms:
        if 0 < filter_parms['noise_threshold'] < 1:
            noise_floor = getNoiseFloor(f_data, filter_parms['noise_threshold'])[0]
            f_data[np.abs(f_data) < noise_floor] = 1E-16  # DON'T use 0 here. ipython kernel dies

    f_data = f_data * composite_filter

    cond_data = None
    filt_data = None
    if hot_inds is not None:
        cond_data = f_data[hot_inds]
    if rot_pts is not None:
        t_clean = np.real(np.fft.ifft(np.fft.ifftshift(f_data)))
        filt_mat = t_clean.reshape(filter_parms['num_pix'], -1)
        if rot_pts > 0:
            filt_data = np.roll(filt_mat, rot_pts, axis=1)
        else:
            filt_data = filt_mat
            
    return noise_floor, filt_data, cond_data

###############################################################################


def decompress_response(f_condensed_mat, num_pts, hot_inds):
    """
    Returns the time domain representation of waveform(s) that are compressed in the frequency space
    
    Parameters
    ----------
    f_condensed_mat : 1D or 2D complex numpy arrays
        Frequency domain signals arranged as [position, frequency]. 
        Only the positive frequncy bins must be in the compressed dataset. 
        The dataset is assumed to have been FFT shifted (such that 0 Hz is at the center).
    num_pts : unsigned int
        Number of points in the time domain signal
    hot_inds : 1D unsigned int numpy array
        Indices of the frequency bins in the compressed data. 
        This index array will be necessary to reverse map the condensed 
        FFT into its original form
        
    Returns
    -------
    time_resp : 2D numpy array
        Time domain response arranged as [position, time]
        
    Notes
    -----
    Memory is given higher priority here, so this function loops over the position
    instead of doing the inverse FFT on the complete data.

    """
    f_condensed_mat = np.atleast_2d(f_condensed_mat)
    hot_inds_mirror = np.flipud(num_pts - hot_inds)
    time_resp = np.zeros(shape=(f_condensed_mat.shape[0], num_pts), dtype=np.float32)
    for pos in range(f_condensed_mat.shape[0]):
        f_complete = np.zeros(shape=num_pts, dtype=np.complex)
        f_complete[hot_inds] = f_condensed_mat[pos, :]
        # Now add the mirror (FFT in negative X axis that was removed)
        f_complete[hot_inds_mirror] = np.flipud(f_condensed_mat[pos, :])
        time_resp[pos, :] = np.real(np.fft.ifft(np.fft.ifftshift(f_complete)))
    
    return np.squeeze(time_resp)


def reshape_from_lines_to_pixels(h5_main, pts_per_cycle, scan_step_x_m=1):
    """
    Breaks up the provided raw G-mode dataset into lines and pixels (from just lines)

    Parameters
    ----------
    h5_main : h5py.Dataset object
        Reference to the main dataset that contains the raw data that is only broken up by lines
    pts_per_cycle : unsigned int
        Number of points in a single pixel
    scan_step_x_m : float
        Step in meters for pixels

    Returns
    -------
    h5_resh : h5py.Dataset object
        Reference to the main dataset that contains the reshaped data
    """
    if h5_main.shape[1] % pts_per_cycle != 0:
        warn('Error in reshaping the provided dataset to pixels. Check points per pixel')
        raise ValueError
        return
    num_cols = int(h5_main.shape[1] / pts_per_cycle)

    h5_spec_vals = getAuxData(h5_main, auxDataName=['Spectroscopic_Values'])[0]
    h5_pos_vals = getAuxData(h5_main, auxDataName=['Position_Values'])[0]
    single_AO = h5_spec_vals[:, :pts_per_cycle]

    ds_spec_inds, ds_spec_vals = build_ind_val_dsets([single_AO.size], is_spectral=True,
                                                     labels=h5_spec_vals.attrs['labels'],
                                                     units=h5_spec_vals.attrs['units'], verbose=False)
    ds_spec_vals.data = np.atleast_2d(single_AO)  # The data generated above varies linearly. Override.

    ds_pos_inds, ds_pos_vals = build_ind_val_dsets([num_cols, h5_main.shape[0]], is_spectral=False,
                                                   steps=[scan_step_x_m, h5_pos_vals[1, 0]],
                                                   labels=['X', 'Y'], units=['m', 'm'], verbose=False)

    ds_reshaped_data = MicroDataset('Reshaped_Data', data=np.reshape(h5_main.value, (-1, pts_per_cycle)),
                                    compression='gzip', chunking=(10, pts_per_cycle))

    # write this to H5 as some form of filtered data.
    resh_grp = MicroDataGroup(h5_main.name.split('/')[-1] + '-Reshape_', parent=h5_main.parent.name)
    resh_grp.addChildren([ds_reshaped_data, ds_pos_inds, ds_pos_vals, ds_spec_inds, ds_spec_vals])

    hdf = ioHDF5(h5_main.file)
    print('Starting to reshape G-mode line data. Please be patient')
    h5_refs = hdf.writeData(resh_grp)

    h5_resh = getH5DsetRefs(['Reshaped_Data'], h5_refs)[0]
    # Link everything:
    linkRefs(h5_resh,
             getH5DsetRefs(['Position_Indices', 'Position_Values', 'Spectroscopic_Indices', 'Spectroscopic_Values'],
                           h5_refs))

    # Copy the two attributes that are really important but ignored:
    copy_main_attributes(h5_main, h5_resh)

    print('Finished reshaping G-mode line data to rows and columns')

    return h5_resh
