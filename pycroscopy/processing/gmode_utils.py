# -*- coding: utf-8 -*-
"""
Created on Thu May 05 13:29:12 2016

@author: Suhas Somnath
"""

import itertools
from multiprocessing import Pool, cpu_count
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np

from .fft import getNoiseFloor, noiseBandFilter, makeLPF, harmonicsPassFilter
from ..io.hdf_utils import getH5DsetRefs, getH5GroupRef, linkRefs
from ..io.io_utils import getTimeStamp
from ..io.microdata import MicroDataGroup, MicroDataset
from ..viz.plot_utils import rainbowPlot


###############################################################################
def testFilter(resp_wfm, filter_parms, samp_rate, show_plots=True, rainbow_plot=True,
               excit_wfm=None, central_resp_size=None):
    """
    Filters the provided response with the provided filters. Use this only to test filters.
    This function does not care about the file structure etc.
    
    Parameters
    ----------
    resp_wfm : 1D numpy float array
        Raw response waveform in the time domain
    filter_parms : dictionary
        Dictionary that contains all the filtering parameters
    samp_rate : unsigned int 
        Sampling rate in Hertz
    show_plots : (Optional) Boolean
        Whether or not to plot FFTs before and after filtering
    excit_wfm : (Optional) 1D numpy float array
        Excitation waveform in the time domain. This waveform is necessary for plotting loops. 
    central_resp_size (Optional) : unsigned int
        Number of responce sample points from the center of the waveform to show in plots. Useful for SPORC 
    
    Filter Parameters
    -----------------
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
    
    Returns
    -------
    filt_data : 1D numpy float array
        Filtered signal in the time domain
    """
    
    num_pts = len(resp_wfm)
    
    show_loops = type(excit_wfm) != type(None)
    show_plots = show_plots or show_loops
    
    noise_band_filter = 1
    low_pass_filter = 1
    harmonic_filter = 1
    
    if type(filter_parms['band_filt_[Hz]']) in [list, np.ndarray]:
        noise_band_filter = noiseBandFilter(num_pts, samp_rate, filter_parms['band_filt_[Hz]'][0], filter_parms['band_filt_[Hz]'][1])
    if filter_parms['LPF_cutOff_[Hz]'] > 0:
        low_pass_filter = makeLPF(num_pts,samp_rate, filter_parms['LPF_cutOff_[Hz]'])
    if type(filter_parms['comb_[Hz]']) in [list, np.ndarray]:
        harmonic_filter = harmonicsPassFilter(num_pts, samp_rate, filter_parms['comb_[Hz]'][0], filter_parms['comb_[Hz]'][1], filter_parms['comb_[Hz]'][2])
    composite_filter =  noise_band_filter * low_pass_filter * harmonic_filter
        
    F_pix_data = np.fft.fftshift(np.fft.fft(resp_wfm))
    
    if show_plots:       
        l_ind = int(0.5*num_pts)
        r_ind = max(np.where(composite_filter>0)[0])    
        w_vec = np.linspace(-0.5*samp_rate,0.5*samp_rate,num_pts)*1E-3
        if central_resp_size:
            sz = int(0.5*central_resp_size)
            l_resp_ind = -sz+l_ind
            r_resp_ind = l_ind+sz
        else:
            l_resp_ind = l_ind
            r_resp_ind = num_pts
        
        fig = plt.figure(figsize=(12, 8))
        lhs_colspan = 2
        if show_loops == False:
            lhs_colspan = 4
        else:
            ax_loops = plt.subplot2grid((2, 4), (0, 2), colspan=2, rowspan=2)
        ax_raw = plt.subplot2grid((2, 4), (0, 0), colspan=lhs_colspan)
        ax_filt = plt.subplot2grid((2, 4), (1, 0), colspan=lhs_colspan)
        axes = [ax_raw, ax_filt]
                
    noise_floor = getNoiseFloor(F_pix_data, filter_parms['noise_threshold'])[0]
    if show_plots:
        amp = np.abs(F_pix_data)
        ax_raw.plot(w_vec[l_ind:r_ind],np.log(amp[l_ind:r_ind]))
        ax_raw.plot(w_vec[l_ind:r_ind],np.log((composite_filter[l_ind:r_ind] + min(amp))*(max(amp)-min(amp))))
        ax_raw.plot(w_vec[l_ind:r_ind],np.log(np.ones(r_ind-l_ind)*noise_floor))
        ax_raw.set_title('Raw Signal')
    F_pix_data = F_pix_data * composite_filter
    F_pix_data[np.abs(F_pix_data) < noise_floor] = 0
    if show_plots:
        ax_filt.plot(w_vec[l_ind:r_ind],np.log(np.abs(F_pix_data[l_ind:r_ind])))
        ax_filt.set_title('Filtered Signal')
        ax_filt.set_xlabel('Frequency(kHz)')
    filt_data = np.real(np.fft.ifft(np.fft.ifftshift(F_pix_data)))
    if show_loops:
        if rainbow_plot:
            rainbowPlot(ax_loops, excit_wfm[l_resp_ind:r_resp_ind], filt_data[l_resp_ind:r_resp_ind]*1E+3)
        else:
            ax_loops.plot(excit_wfm[l_resp_ind:r_resp_ind], filt_data[l_resp_ind:r_resp_ind]*1E+3)              
        ax_loops.set_title('AI vs AO') 
        ax_loops.set_xlabel('Input Bias (V)')
        ax_loops.set_ylabel('Deflection (mV)')
        axes.append(ax_loops)
    fig.tight_layout()
    return filt_data, fig, axes
        
  
###############################################################################        

def fftFilterRawData(hdf, h5_main, filter_parms, write_filtered=True, 
                     write_condensed=False, num_cores=None):
    """
    Filters G-mode data using specified filter parameters and writes results to file.
        
    Parameters
    ----------
    hdf : Active ioHDF object
        Object that will be used for writing back to the data file
    h5_main : HDF5 dataset object
        Dataset containing the raw data
    filter_parms : dictionary
        Dictionary that contains all the filtering parameters
    write_filtered (optional) : Boolean - default True
        Whether or not to write filtered data to file
    write_condensed (optional) : Boolean - default False
        Whether or not to write condensed filtered data to file
    num_cores : unsigned int
        Number of cores to use for processing data in parallel
        
    Returns
    -------
    HDF5 group reference containing filtered dataset
    """ 
    
    def __maxPixelRead__(h5_raw, num_cores, store_filt=True, hot_bins=None, bytes_per_bin=2, max_RAM_gb=16):
        """
        Returns the maximum number of pixels that can be stored in memory considering the output data and the number of cores
        
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
        if type(hot_bins) != type(None):
            bytes_per_pix += len(hot_bins) * 8 # complex64
        
        # Multiply this memory requirement per pixel by the number of cores
        bytes_per_pix *= num_cores
        
        max_pix = np.rint(max_RAM_gb*1024**3 / bytes_per_pix)
        max_pix = max(1,min(h5_raw.shape[0], max_pix))
        print 'Allowed to read', max_pix, 'of', h5_raw.shape[0], 'pixels'        
        return np.uint(max_pix)
        
    def __filterChunk__(raw_mat,parm_dict,recom_cores):
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
        max_cores = max(1,cpu_count()-2)
            
        recom_chunks = int(raw_mat.shape[0]/recom_cores)
        
        print 'recom cores:',recom_cores,'Total pixels:',raw_mat.shape[0],', Recom chunks:',recom_chunks
        
        if recom_cores > 1 and recom_chunks < 10:
            min_jobs = 20
            reduced_cores = int(raw_mat.shape[0]/min_jobs)
            # intelligently set the cores now. 
            recom_cores= min(max_cores,reduced_cores)
            print 'Not enough jobs per core. Reducing cores to', recom_cores
            
        if recom_cores > 1:
            return filterChunkParallel(raw_mat, parm_dict, recom_cores)
        else:
            return filterChunkSerial(raw_mat, parm_dict)
            
  
    max_cores = max(1,cpu_count()-2)
    if not num_cores:
        num_cores = max_cores
    
    if write_filtered == False and write_condensed == False:
       warn('You need to write the filtered and/or the condensed dataset to the file')
       return
    
    num_effective_pix = h5_main.shape[0]*1.0/filter_parms['num_pix']
    if num_effective_pix % 1 > 0:
        warn('Number of pixels not divisible by the number of pixels to use for FFT filter')
        return
    num_effective_pix = int(num_effective_pix)
        
    num_pts = h5_main.shape[1]*filter_parms['num_pix']
    band_filt = filter_parms['band_filt_[Hz]']
    noise_band_filter = noiseBandFilter(num_pts, filter_parms['samp_rate_[Hz]'], band_filt[0], band_filt[1])
    low_pass_filter = makeLPF(num_pts, filter_parms['samp_rate_[Hz]'], filter_parms['LPF_cutOff_[Hz]'])
    composite_filter = noise_band_filter * low_pass_filter    
    
    # ioHDF now handles automatic indexing
    grp_name = h5_main.name.split('/')[-1] + '-FFT_Filtering_' 
        
    ds_comp_filt = MicroDataset('Composite_Filter',np.float32(composite_filter))
    ds_noise_floors = MicroDataset('Noise_Floors', 
                      data=np.zeros(shape=(num_effective_pix), dtype=np.float32))
                                        
    grp_filt = MicroDataGroup(grp_name,h5_main.parent.name)
    filter_parms['timestamp'] = getTimeStamp()
    filter_parms['algorithm'] = 'GmodeUtils-Parallel'
    grp_filt.attrs = filter_parms
    grp_filt.addChildren([ds_comp_filt,ds_noise_floors])
     
    if write_filtered:
        ds_filt_data = MicroDataset('Filtered_Data', data=[],maxshape=h5_main.maxshape, 
                                    dtype=np.float32, chunking=h5_main.chunks, compression='gzip')
        grp_filt.addChildren([ds_filt_data])
    
    hot_inds = None        
    if write_condensed:
        hot_inds = np.where(composite_filter > 0)[0]
        hot_inds = np.uint(hot_inds[int(0.5*len(hot_inds)):]) # only need to keep half the data
        ds_cond_bins = MicroDataset('Condensed_Bins',hot_inds)    
        ds_cond_data = MicroDataset('Condensed_Data', data=[],maxshape=(num_effective_pix,len(hot_inds)), 
                                dtype=np.complex, chunking=(1,len(hot_inds)), compression='gzip')
        grp_filt.addChildren([ds_cond_bins, ds_cond_data])
                
    #grp_filt.showTree()
    h5_filt_refs = hdf.writeData(grp_filt)
    
    h5_filtr_grp = getH5GroupRef(grp_name, h5_filt_refs)
    
    h5_comp_filt = getH5DsetRefs(['Composite_Filter'], h5_filt_refs)[0]
    h5_noise_floors = getH5DsetRefs(['Noise_Floors'], h5_filt_refs)[0]
    
    # Now need to link appropriately:
    if write_filtered:
        h5_filt_data = getH5DsetRefs(['Filtered_Data'], h5_filt_refs)[0]
        linkRefs(h5_filt_data, [h5_comp_filt, h5_main, h5_noise_floors])
      
    if write_condensed:
        h5_cond_bins = getH5DsetRefs(['Condensed_Bins'], h5_filt_refs)[0]
        h5_cond_data = getH5DsetRefs(['Condensed_Data'], h5_filt_refs)[0]
        linkRefs(h5_cond_data, [h5_cond_bins, h5_comp_filt, h5_main, h5_noise_floors])
        
    rot_pts = 0
    if 'phase_rot_[pts]' in filter_parms.keys():
        rot_pts = int(filter_parms['phase_rot_[pts]'])
                  
    print('Filtering data now. Be patient, this could take a few minutes') 

    max_pix = __maxPixelRead__(h5_main, 10, store_filt=write_filtered, hot_bins=hot_inds, bytes_per_bin=2, max_RAM_gb=16)
    # Ensure that whole sets of pixels can be read.
    max_pix = np.uint(filter_parms['num_pix']*np.floor(max_pix/filter_parms['num_pix']))
    
    parm_dict = {'filter_parms':filter_parms, 'composite_filter': composite_filter,
                 'rot_pts': rot_pts, 'hot_inds': hot_inds}
    
    st_pix = 0
    line_count = 0
    while st_pix < h5_main.shape[0]:
        en_pix = int(min(h5_main.shape[0],st_pix + max_pix))
        print 'Reading pixels:', st_pix, 'to',en_pix, 'of', h5_main.shape[0]
        raw_mat = h5_main[st_pix:en_pix,:]
        # reshape to (set of pix, data in each set of pix)
        # print 'raw mat originally of shape:', raw_mat.shape
        raw_mat = raw_mat.reshape(-1,filter_parms['num_pix']*raw_mat.shape[1])
        num_lines = raw_mat.shape[0]
        # print 'After collapsing pixels, raw mat now of shape:', raw_mat.shape
        (nse_flrs, filt_data, cond_data) = __filterChunk__(raw_mat,parm_dict,num_cores)        
        # Insert things into appropriate HDF datasets
        print 'Writing filtered data to h5'
        # print 'Noise floors of shape:', nse_flrs.shape
        h5_noise_floors[line_count:line_count+num_lines] = nse_flrs
        if write_condensed:
            #print 'Condensed data of shape:', cond_data.shape
            h5_cond_data[line_count:line_count+num_lines,:] = cond_data
        if write_filtered:
            #print 'Filtered data of shape:', filt_data.shape
            h5_filt_data[st_pix:en_pix,:] = filt_data
        hdf.flush()
        st_pix = en_pix
    
    return h5_filtr_grp
              
###############################################################################  
              
def filterChunkParallel(raw_data, parm_dict, num_cores):
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
    #print 'sending unit filter data of size', pts_per_set
    noise_floors = np.zeros(shape=(num_sets), dtype=np.float32)
    filt_data = None
    if not parm_dict['rot_pts']:
        filt_data = np.zeros(shape=(num_sets*pix_per_set,pts_per_set/pix_per_set), dtype=raw_data.dtype)
    cond_data = None
    if type(parm_dict['hot_inds']) != type(None):
        cond_data = np.zeros(shape=(num_sets,parm_dict['hot_inds'].size), dtype=np.complex64)
        
    # Set up single parameter:
    sing_parm = itertools.izip(raw_data,itertools.repeat(parm_dict))
    
    # Setup parallel processing:
    #num_cores = 10
    pool=Pool(processes=num_cores, maxtasksperchild=None)
    
    # Start parallel processing:
    num_chunks = int(np.ceil(raw_data.shape[0]/num_cores))
    parallel_results=pool.imap(unitFilter, sing_parm, chunksize=num_chunks)
    pool.close()
    pool.join()
    
    print('Done parallel computing. Now extracting data and populating matrices')
    
    # Extract data for each line...
    print_set = np.linspace(0,num_sets-1,10, dtype=int) 
    for set_ind in xrange(num_sets):
        if set_ind in print_set:
            print 'Reading...', np.rint(100*set_ind/num_sets) , '% complete'
        
        (noise_floors[set_ind], filt_data_set, cond_data_set) = parallel_results.next()
        if type(parm_dict['hot_inds']) != type(None):
            cond_data[set_ind,:] = cond_data_set
        if parm_dict['rot_pts'] != None:
            filt_data[set_ind*pix_per_set:(set_ind+1)*pix_per_set,:] = filt_data_set      
            
    return (noise_floors, filt_data, cond_data)      
 
 
 
def filterChunkSerial(raw_data, parm_dict):        
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
    #print 'sending unit filter data of size', pts_per_set
    noise_floors = np.zeros(shape=(num_sets), dtype=np.float32)
    filt_data = None
    if parm_dict['rot_pts'] != None:
        filt_data = np.zeros(shape=(num_sets*pix_per_set,pts_per_set/pix_per_set), dtype=raw_data.dtype)
    cond_data = None
    if parm_dict['hot_inds'] != None:
        cond_data = np.zeros(shape=(num_sets,parm_dict['hot_inds'].size), dtype=np.complex64)
    
    # Filter each line
    print_set = np.linspace(0,num_sets-1,10, dtype=int) 
    for set_ind in xrange(num_sets):
        if set_ind in print_set:
            print 'Reading...', np.rint(100*set_ind/num_sets) , '% complete'
        
        #parm_dict['t_raw'] = raw_data[set_ind,:]
        (noise_floors[set_ind], filt_data_set, cond_data_set) = unitFilter((raw_data[set_ind,:],parm_dict))
        if parm_dict['hot_inds'] != None:
            cond_data[set_ind,:] = cond_data_set
        if parm_dict['rot_pts'] != None:
            filt_data[set_ind*pix_per_set:(set_ind+1)*pix_per_set,:] = filt_data_set      
            
    return (noise_floors, filt_data, cond_data)

     
def unitFilter(single_parm):
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
    #t_raw = parm_dict['t_raw']
    filter_parms = parm_dict['filter_parms']
    composite_filter = parm_dict['composite_filter']
    rot_pts = parm_dict['rot_pts']
    hot_inds = parm_dict['hot_inds']

    t_raw = t_raw.reshape(-1)
    F_data = np.fft.fftshift(np.fft.fft(t_raw))
    noise_floor = getNoiseFloor(F_data, filter_parms['noise_threshold'])[0]
    F_data = F_data * composite_filter
    F_data[np.abs(F_data) < noise_floor] = 0
    cond_data = None
    filt_data = None
    if hot_inds != None:
        cond_data = F_data[hot_inds]
    if rot_pts != None:
        t_clean = np.real(np.fft.ifft(np.fft.ifftshift(F_data)))
        filt_mat = t_clean.reshape(filter_parms['num_pix'],-1) 
        if rot_pts > 0:
            filt_data = np.roll(filt_mat, rot_pts, axis=1)
        else:
            filt_data = filt_mat
            
    return (noise_floor, filt_data, cond_data)

###############################################################################

def deCompressResponse(F_condensed_mat, num_pts, hot_inds):
    """
    Returns the time domain representation of waveform(s) that are compressed in the frequency space
    
    Parameters
    ----------
    F_condensed_mat : 1D or 2D complex numpy arrays
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
        
    Implemntation Note
    ------------------
    Memory is given higher priority here, so this function loops over the position
    instead of doing the inverse FFT on the complete data.
    """
    F_condensed_mat = np.atleast_2d(F_condensed_mat)
    hot_inds_mirror = np.flipud(num_pts - hot_inds)
    time_resp = np.zeros(shape=(F_condensed_mat.shape[0],num_pts), dtype=np.float32)    
    for pos in xrange(F_condensed_mat.shape[0]):
        F_complete = np.zeros(shape=(num_pts), dtype=np.complex)
        F_complete[hot_inds] = F_condensed_mat[pos,:]
        # Now add the mirror (FFT in negative X axis that was removed)
        F_complete[hot_inds_mirror] = np.flipud(F_condensed_mat[pos,:])    
        time_resp[pos,:] = np.real(np.fft.ifft(np.fft.ifftshift(F_complete)))
    
    return np.squeeze(time_resp)

              
