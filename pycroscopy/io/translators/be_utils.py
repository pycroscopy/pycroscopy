# -*- coding: utf-8 -*-
"""
Created on Tue Jan 05 07:55:56 2016

@author: Suhas Somnath
"""

from os import path
from warnings import warn

import h5py
import matplotlib.pyplot as plt
import numpy as np

from ..be_hdf_utils import getActiveUDVSsteps,maxReadPixels
from ..hdf_utils import getAuxData, getDataSet, getH5DsetRefs, linkRefs
from ..io_hdf5 import ioHDF5
from ..io_utils import getAvailableMem
from ..microdata import MicroDataset,MicroDataGroup
from ...processing.proc_utils import buildHistogram
from ...viz.plot_utils import plot1DSpectrum, plot2DSpectrogram, plotHistgrams


#%%############################################################################

def parmsToDict(filepath,parms_to_remove = []): 
    """
    Translates the parameters in the text file into a dictionary. 
    Also indentifies whether this is a BEPS or BELine dataset.
    
    Parameters
    -----------
    filepath : String / Unicode
        Absolute path of the parameters text file.
    parms_to_remove : List of string (Optional)
        keys that this function should attempt to remove from the dictionary
    
    Returns
    ----------
    isBEPS : Boolean
        whether this dataset is BEPS or BE Line.
    parm_dict : Dictionary
        experimental parameters
    """      
    isBEPS = False;
    f = open(filepath,'r')
    parm_dict = {};
    lines = f.readlines()
    f.close()
     
    prefix = 'File_'
    for line in lines:
        line = line.rstrip();
        fields = line.split(" : ");
     
        # Ignore the parameters describing the GUI choices
        if prefix == 'Multi_':
            continue
     
        """
        Check if the line is a group header or parameter/value pair
        """
        if len(fields) == 2:
            """
            Get the current name/value pair, and clean up the name
            """
            name = fields[0].strip().replace('# ','num_').replace('#','num_').replace(' ','_')
            value = fields[1] 
            
            """
            Rename specific parameters
            """
            if name == '1_mode':
                name = 'mode'
            if name == 'IO_rate':
                name = 'IO_rate_[Hz]'
                value = int(value.split()[0])*1E6
            if name == 'AO_range':
                name = 'AO_range_[V]'
                value = ' '.join(value.split()[:-1])
            if name == 'AO_amplifier':
                value = value.split()[0]
            if name == 'cycle_time[s]':
                name = 'cycle_time_[s]'
            if name == 'FORC_V_high1[V]':
                name = 'FORC_V_high1_[V]'
            if name == 'FORC_V_high2[V]':
                name = 'FORC_V_high2_[V]'
            if name == 'FORC_V_low1[V]':
                name = 'FORC_V_low1_[V]'
            if name == 'FORC_V_low2[V]':
                name = 'FORC_V_low2_[V]'
            if name == 'amplitude[V]':
                name = 'amplitude_[V]'
            if name == 'offset[V]':
                name = 'offset_[V]'
            if name == 'set_pulst_amplitude[V]':
                name = 'set_pulst_amplitude_[V]'
            if name == 'set_pulst_duration[s]':
                name = 'set_pulst_duration_[s]'
            if name == 'step_edge_smoothing[s]':
                name = 'step_edge_smoothing_[s]'
            
                
            """
            Append the prefix to the name
            """
            name = prefix+name.lstrip(prefix)
            
            """
            Write parameter to parm_dict    
            """
            try:
                num = float(value)
                parm_dict[name] = num
                if num == int(num):
                    parm_dict[name] = int(num)
            except ValueError:
                parm_dict[name] = value
            except:
                raise
        elif len(fields) == 1:
            """
            Change the parameter prefix to the new one from the group header
            """
            prefix = fields[0].strip('<').strip('>')
            prefix = prefix.split()[0]+'_'
        
            """
            Check if there are VS Parameters.  Set isBEPS to true if so.
            """
            isBEPS = isBEPS or prefix == 'VS_'
    
    if isBEPS:
        useless_parms = \
            ['Multi_1_BE_response_spectra',
             'Multi_2_BE_amplitude_spectra',
             'Multi_3_BE_phase_spectra',
             'Multi_4_BE_response_spectrogram',
             'Multi_5_BE_amplitude_spectrogram',
             'Multi_6_BE_phase_spectrogram',
             'Multi_7_VS_response_loops',
             'Multi_8_VS_amplitude_loops',
             'Multi_9_VS_phase_loops',
             'Multi_10_topography',
             'Multi_11_mean_channel_2'];
    else:
        useless_parms = \
            ['Multi_1_BE_response_spectra',
             'Multi_2_BE_amplitude_spectra',
             'Multi_3_BE_phase_spectra',
             'Multi_4_BE_response_spectrogram',
             'Multi_5_BE_amplitude_spectrogram',
             'Multi_6_BE_phase_spectrogram',
             'Multi_7_amplitude_map',
             'Multi_8_resonance_map',
             'Multi_9_Q_map',
             'Multi_10_phase_map',
             'Multi_11_topography',
             'Multi_12_AI_time_domain',
             'Multi_13_AI_Fourier_amplitude'];
        
    useless_parms.extend(parms_to_remove);
    
    # Now remove the list of useless parameters:
    for uparm in useless_parms:
        try:
            del parm_dict[uparm];
        except KeyError:
            #warn('Parameter to be deleted does not exist')
            pass;
        except:
            raise
    del uparm, useless_parms;
    
    if isBEPS:
        # fix the DC type in the parms:
        if parm_dict['VS_measure_in_field_loops'] == 'out-of-field only': 
            parm_dict['VS_measure_in_field_loops'] = 'out-of-field'
        elif parm_dict['VS_measure_in_field_loops'] == 'in-field only': 
            parm_dict['VS_measure_in_field_loops'] = 'in-field'
        
    return (isBEPS,parm_dict);
    
###############################################################################
    
def analyzePhaseCondition():
    """
    phase_cond = sign(sum(Amp(n)*gradient(unwrap(phase(n))))/N) 

    where n is the bin index going from 1 to N for a single resonance curve.
    You can probably determine whether to change the sign of the imaginary part based on the sign of phase_cond.
    
    I think that if phase_cond = -1 then don’t do anything, phase_cond = 1 then take the complex conjugate.
    
    You should be able to find a single value for the entire data set.
    """
    pass
    
###############################################################################

def getSpectroscopicParmLabel(expt_type):
    """
    Returns the label for the spectroscopic parameter in the plot group. 
    
    Parameters
    ----------
    expt_type : str
        Type of the experiment - found in the parms.txt file
    
    Returns
    -------
    str
        label for the spectroscopic parameter axis in the plot
    """
    
    if expt_type in ['DC modulation mode','current mode']:
        return 'DC Bias'
    elif expt_type == 'AC modulation mode with time reversal':
        return 'AC amplitude'
    return 'User Defined'

###############################################################################

def normalizeBEresponse(spectrogram_mat, FFT_BE_wave, harmonic):
    """
    This function normalizes the BE waveform to correct the phase by diving by the excitation

    Parameters
    ------------
    spectrogram_mat : 2D complex numpy array
        BE response arranged as [bins, steps]
    FFT_BE_wave : 1D complex numpy array
        FFT of the BE waveform at the appropriate bins. Number of bins must match with spectrogram_mat
    harmonic : unsigned int
        nth harmonic of the excitation waveform

    Returns
    ----------
    spectrogram_mat : 2D complex numpy array
        Normalized BE response spectrogram

    """

    BE_wave = np.fft.ifftshift(np.fft.ifft(FFT_BE_wave))
    scaling_factor = 1

    if harmonic == 2:
        scaling_factor = np.fft.fftshift(np.fft.fft(BE_wave ** 2)) / (2 * np.exp(1j * 3 * np.pi * 0.5))
    elif harmonic == 3:
        scaling_factor = np.fft.fftshift(np.fft.fft(BE_wave ** 3)) / (4 * np.exp(1j * np.pi))
    elif harmonic >= 4:
        print "Warning these high harmonics are not supported in translator."

    # Generate transfer functions
    F_AO_spectrogram = np.transpose(np.tile(FFT_BE_wave / scaling_factor, [spectrogram_mat.shape[1], 1]))
    # Divide by transfer function
    spectrogram_mat = spectrogram_mat / (F_AO_spectrogram)

    return spectrogram_mat
    
###############################################################################
    
def generatePlotGroups(h5_main, hdf, mean_resp, folder_path, basename, max_resp=[], min_resp=[], 
                       max_mem_mb=1024, spec_label='None', ignore_plot_groups=[], 
                        show_plots=True, save_plots=True, do_histogram=False):
    """
    Generates the spatially averaged datasets for the given raw dataset. 
    The averaged datasets are necessary for quick visualization of the quality of data. 
    
    Parameters
    ----------
    h5_main : H5 reference
        to the main dataset
    hdf : active ioHDF instance 
        for writing to same H5 file
    mean_resp : 1D numpy array
        spatially averaged amplitude
    folder_path : String
        Absolute path of the data folder
    basename : String
        base name of the dataset
    max_resp : 1D numpy array
        Maximum amplitude for all pixels
    min_resp : 1D numpy array
        Minimum amplitude for all pixels
    max_mem_mb : Unisigned integer
        Maximum memory that can be used for generating histograms
    spec_label : String
        Parameter that is varying
    ignore_plot_groups : (optional) List of strings
        Names of the plot groups (UDVS columns) that should be ignored
    show_plots : (optional) Boolean
        Whether or not to show plots
    save_plots : (optional) Boolean
        Whether or not to save generated plots
    do_histogram : Boolean (Optional. Default = False)
        Whether or not to generate hisograms. 
        Caution - Histograms can take a fair amount of time to compute.
                
    Returns: 
    ---------
    None
    """

    grp = h5_main.parent
    h5_freq = grp['Bin_Frequencies']    
    UDVS = grp['UDVS']
    spec_inds = grp['Spectroscopic_Indices']
    UDVS_inds = grp['UDVS_Indices']
    spec_vals = grp['Spectroscopic_Values']
                    
#     std_cols = ['wave_type','Frequency','DC_Offset','wave_mod','AC_Amplitude','dc_offset','ac_amplitude']
    
    col_names = UDVS.attrs['labels']
    
    if len(col_names) <= 5:
        """
        No plot groups are defined in the UDVS table.
        All plot group datasets will be written to the 
        Channel group
        """
        
        return
    
    # Removing the standard columns
    col_names = UDVS.attrs['labels'][5:]

#     col_names = [col for col in col_names if col not in std_cols + ignore_plot_groups]
    
    freq_inds = spec_inds[spec_inds.attrs['Frequency']].flatten()
    
    for col_name in col_names:
        ref = UDVS.attrs[col_name]
#         Make sure we're actually dealing with a reference of some type
        if not isinstance(ref,h5py.RegionReference):
            continue
        #4. Access that column of the data through region reference
        steps = np.where(np.isfinite(UDVS[ref]))[0]
        step_inds = np.array([np.where(UDVS_inds.value == step)[0] for step in steps]).flatten()
        """selected_UDVS_steps = UDVS[ref]
        selected_UDVS_steps = selected_UDVS_steps[np.isfinite(selected_UDVS_steps)]"""
        
        
        (step_averaged_vec, mean_spec) = reshapeMeanData(spec_inds, step_inds, mean_resp)
            
        """ 
        Need to account for cases with multiple excitation waveforms
        This will affect the frequency indices / values
        We are assuming that there is only one excitation waveform per plot group
        """
        freq_slice = np.unique(freq_inds[step_inds])
        freq_vec = h5_freq.value[freq_slice]
        
        num_bins = len(freq_slice) # int(len(freq_inds)/len(UDVS[ref]))
        pg_data = np.repeat(UDVS[ref],num_bins) 
            
        ds_mean_spec = MicroDataset('Mean_Spectrogram',mean_spec, dtype=np.complex64)
        ds_step_avg = MicroDataset('Step_Averaged_Response',step_averaged_vec, dtype=np.complex64)
        ds_spec_parm = MicroDataset('Spectroscopic_Parameter',np.squeeze(pg_data[step_inds]))# cannot assume that this is DC offset, could be AC amplitude....
        ds_spec_parm.attrs = {'name': spec_label}
        ds_freq = MicroDataset('Bin_Frequencies',freq_vec)
        
        plot_grp = MicroDataGroup('{:s}'.format('Spatially_Averaged_Plot_Group_'),grp.name[1:])
        plot_grp.attrs['Name'] = col_name
        plot_grp.addChildren([ds_mean_spec, ds_step_avg, ds_spec_parm, ds_freq])
        
        h5_plt_grp_refs = hdf.writeData(plot_grp)
        
        h5_mean_spec = getH5DsetRefs(['Mean_Spectrogram'], h5_plt_grp_refs)[0]
        h5_step_avg = getH5DsetRefs(['Step_Averaged_Response'], h5_plt_grp_refs)[0]
        h5_spec_parm = getH5DsetRefs(['Spectroscopic_Parameter'], h5_plt_grp_refs)[0]
        h5_freq_vec = getH5DsetRefs(['Bin_Frequencies'], h5_plt_grp_refs)[0]
        
        # Linking the datasets with the frequency and the spectroscopic variable:
        linkRefs(h5_mean_spec, [h5_spec_parm, h5_freq_vec])
        linkRefs(h5_step_avg, [h5_freq_vec])
        
        """
        Create Region Reference for the plot group in the Raw_Data, Spectroscopic_Indices 
        and Spectroscopic_Values Datasets
        """
        raw_ref = h5_main.regionref[:,step_inds]
        spec_inds_ref = spec_inds.regionref[:,step_inds]
        spec_vals_ref = spec_vals.regionref[:,step_inds]
        
        ref_name = col_name.replace(' ','_').replace('-','_')+'_Plot_Group'
        h5_main.attrs[ref_name] = raw_ref
        spec_inds.attrs[ref_name] = spec_inds_ref
        spec_vals.attrs[ref_name] = spec_vals_ref
        
        hdf.flush()
        
        if do_histogram:
            """
            Build the histograms for the current plot group
            """
            hist = BEHistogram()
            hist_mat, hist_labels, hist_indices, hist_indices_labels = hist.buildPlotGroupHist(h5_main, step_inds, max_response=max_resp, min_response=min_resp, max_mem_mb=max_mem_mb)
            ds_hist = MicroDataset('Histograms',hist_mat, dtype=np.int32, chunking=(1,hist_mat.shape[1]),compression='gzip')
            hist_slice_dict = dict()
            for hist_ind, hist_dim in enumerate(hist_labels):
                hist_slice_dict[hist_dim] = (slice(hist_ind,hist_ind+1), slice(None))
            ds_hist.attrs['labels'] = hist_slice_dict
            ds_hist.attrs['units'] = ['V','','V','V']
            ds_hist_indices = MicroDataset('Indices',hist_indices,dtype=np.uint)
            ds_hist_values = MicroDataset('Values',hist_indices,dtype=np.float32)
            hist_ind_dict = dict()
            for hist_ind_ind, hist_ind_dim in enumerate(hist_indices_labels):
                hist_ind_dict[hist_ind_dim] = (slice(hist_ind_ind, hist_ind_ind+1),slice(None))
            ds_hist_indices.attrs['labels'] = hist_ind_dict
            ds_hist_values.attrs['labels'] = hist_ind_dict

            hist_grp = MicroDataGroup('Histogram',h5_mean_spec.parent.name[1:])

            hist_grp.addChildren([ds_hist, ds_hist_indices, ds_hist_values])
            
            h5_hist_grp_refs = hdf.writeData(hist_grp)
            
            h5_hist = getH5DsetRefs(['Histograms'], h5_hist_grp_refs)[0]
            h5_hist_inds = getH5DsetRefs(['Indices'], h5_hist_grp_refs)[0]
            h5_hist_vals = getH5DsetRefs(['Values'], h5_hist_grp_refs)[0]
            
            linkRefs(h5_hist, 
                         getH5DsetRefs(['Indices','Values'], 
                                       h5_hist_grp_refs))
            
            h5_hist.attrs['Spectroscopic_Indices'] = h5_hist_inds.ref
            h5_hist.attrs['Spectroscopic_Values'] = h5_hist_vals.ref
                
        else:
            """
            Write the min and max response vectors so that histograms can be generated later.
            """
            ds_max_resp = MicroDataset('Max_Response', max_resp)
            ds_min_resp = MicroDataset('Min_Response', min_resp)
            plot_grp.addChildren([ds_max_resp, ds_min_resp])
        
        if save_plots or show_plots:
            fig_title = '_'.join(grp.name[1:].split('/')+[col_name])
            path_1d = None
            path_2d = None
            path_hist = None
            if save_plots:
                path_1d = path.join(folder_path,basename + '_Step_Avg_' + fig_title + '.png')
                path_2d = path.join(folder_path,basename + '_Mean_Spec_' + fig_title + '.png')
                path_hist = path.join(folder_path,basename + '_Histograms_' + fig_title + '.png')
            plot1DSpectrum(step_averaged_vec, freq_vec, fig_title, figure_path=path_1d)
            plot2DSpectrogram(mean_spec, freq_vec, fig_title, figure_path=path_2d)
            if do_histogram:
                plotHistgrams(hist_mat, hist_indices, grp.name, figure_path=path_hist)
            
            if show_plots:
                plt.show()
            plt.close('all')
        # print('Generated spatially average data for group: %s' %(col_name))
    print('Completed generating spatially averaged plot groups')

###############################################################################
    
def reshapeMeanData(spec_inds, step_inds, mean_resp):
    """
    Takes in the mean data vector and rearranges that data according to 
    plot group as [step number,bins]
    
    Parameters
    -----------
    spec_inds : 2D numpy array
        UDVS_Indices as a 2D mat
    step_inds : 1D numpy array or list
        UDVS step indices corresponding to this plot group
    mean_resp : 1D numpy complex array
        position averaged BE data
            
    Returns
    ----------
    step_averaged_vec : 1D complex numpy array
        Mean (position averaged) spectrogram averaged over the UDVS steps as well
    mean_spectrogram : 2D complex numpy array
        Position averaged data arranged as [step number,bins]
    """
    num_bins = len(np.unique(spec_inds[0,step_inds]))
    # Stephen says that we can assume that the number of bins will NOT change in a plot group
    mean_spectrogram = mean_resp[step_inds].reshape(-1,num_bins)
        
    step_averaged_vec = np.mean(mean_spectrogram,axis=0)
    return (step_averaged_vec, mean_spectrogram)
    
###############################################################################

def visualizePlotGroups(h5_filepath):
    """
    Visualizes the plot groups present in the provided BE data file
    
    Parameters
    -------------
    h5_filepath : String / Uniciode
        Absolute path of the h5 file
    
    Outputs:
    --------------
    None
    """
    with h5py.File(h5_filepath,mode='r') as h5f:
        expt_type = h5f.attrs.get('data_type')
        if expt_type not in ['BEPSData','BELineData']:
            warn('Invalid data format')
            return         
        for grp_name in h5f.keys():            
            grp = h5f[grp_name]['Channel_000']            
            for plt_grp_name in grp.keys():                
                if plt_grp_name.startswith('Spatially_Averaged_Plot_Group_'): 
                    plt_grp = grp[plt_grp_name]
                    if expt_type == 'BEPSData':
                        spect_data = plt_grp['Mean_Spectrogram'].value
                        plot2DSpectrogram(spect_data,plt_grp['Bin_Frequencies'].value,plt_grp.attrs['Name'])
                    step_avg_data = plt_grp['Step_Averaged_Response']
                    plot1DSpectrum(step_avg_data,plt_grp['Bin_Frequencies'].value,plt_grp.attrs['Name'])
                    try:
                        hist_data = plt_grp['Histograms']
                        hist_bins = plt_grp['Histograms_Indicies']
                        plotHistgrams(hist_data, hist_bins, plt_grp.attrs['Name'])
                    except:
                        pass
                    
    plt.show()
    plt.close('all')
    
###############################################################################

def trimUDVS(udvs_mat, udvs_labs, udvs_units, target_col_names):
    """
    Removes unused (typically default) plot groups
    
    Parameters
    ----------
    udvs_mat : 2D numpy array
        UDVS table arranged as [steps, col]
    udvs_labs : list of strings
        Column names of the UDVS table
    udvs_units : list of strings
        Units for the columns of the UDVS table
    target_col_names : list of strings
        Column names that need to be removed
        
    Returns
    -----------
    udvs_mat : 2D numpy array
        Truncated UDVS table
    udvs_labs : list of strings 
        Truncated list of UDVS column names
    udvs_units : list of strings
        Truncated list of UDVS column units
    """
    
    if len(target_col_names) == 0:
        return (udvs_mat, udvs_labs, udvs_units)
        
    if len(udvs_labs) != udvs_mat.shape[1]:
        warn('Error: Incompatible UDVS matrix and labels. Not truncating!')
        return (udvs_mat, udvs_labs, udvs_units)
    
    # First figure out the column indices
    col_inds = []
    found_cols = []
    for target in target_col_names:
        for ind, udvs_col in enumerate(udvs_labs):
            if udvs_col == target:
                col_inds.append(ind)
                found_cols.append(udvs_col)
                break
    col_inds = np.sort(np.unique(col_inds))
    
    # Now remove from the labels and the matrix
    udvs_mat = np.delete(udvs_mat, col_inds, axis=1)
#     col_inds.sort(reverse=True)
    [udvs_units.pop(ind) for ind in xrange(len(col_inds),0,-1)]
    udvs_labs = [col for col in udvs_labs if col not in found_cols]
    
    return (udvs_mat, udvs_labs, udvs_units)

###############################################################################

def createSpecVals(udvs_mat, spec_inds, bin_freqs, bin_wfm_type, parm_dict, 
                   udvs_labs, udvs_units):
    """
    This function will determine the proper Spectroscopic Value array for the 
    dataset
    
    Chris Smith -- csmith55@utk.edu
    
    Parameters
    ----------
    udvs_mat : numpy array 
        UDVS table from dataset
    spec_inds : numpy array 
        Spectroscopic Indices table from dataset
    bin_freqs : numpy array 
        Bin frequencies
    bin_wfm_type : numpy array 
        waveform type for each frequency index
    parm_dict : dictionary
        parameters for dataset
    udvs_labs : list of strings
        labels for the columns of the UDVS matrix
    udvs_units : list of strings
        units for the columns of the UDVS matrix
    
    Returns
    -----------
    ds_spec_val_mat : numpy array 
        Spectroscopic Values table
    ds_spec_val_labs : list of strings
        names of the columns of the Spectroscopic Values table
    ds_spec_val_units : list of strings
        units of the columns of the Spectroscopic Values table
    ds_spec_val_labs_names : list of Strings
        labels with the names of their parameters
    """
    def __FindSpecValIndices(udvs_mat, spec_inds, usr_defined=False):
        """
        This function finds the Spectroscopic Values associated with the dataset that
        have more than one unique value
        
        Parameters
        ----------
        udvs_mat : numpy array containing the UDVS table
        spec_inds : numpy array contain Spectroscopic indices table
    
        Returns
        -------
        iSpec_var : integer array holding column indices in UDVS that change
        ds_spec_val_mat : array holding all spectral values for columns in iSpec_var
        """
#         Copy even step values of DC_offset into odd steps 
        UDVS = np.copy(udvs_mat)

        if not usr_defined:
            DC = UDVS[:,1]
            for step in xrange(0,DC.size,2):
                DC[step+1]=DC[step]
            UDVS[:,1] = DC
        
        """
        icheck is an array containing all UDVS steps which should be checked.
        """
        icheck = np.unique(spec_inds[1])
        """
        Keep only the UDVS values for steps which we care about and the 
        first 5 columns
        """
        UDVS = UDVS[(icheck),:5]
#         UDVS = np.array([UDVS[i] for i in icheck])
        
        """
        Transpose UDVS for ease of looping later on and store the number of steps
        as num_cols
        """
        num_cols = np.size(UDVS,1)
        """
        Initialize the iSpec_var as an empty array.  It will store the index of the 
        UDVS label for any column which has more than one unique value
        """    
        iSpec_var = []
        
        """
        Loop over all columns in udvs_mat
        """
        for i in xrange(1,num_cols):
            """
            Find all unique values in the current column
            """
            toosmall = np.where(abs(UDVS[:,i]) < 1E-5)[0]
            UDVS[toosmall,i] = 0
            uvals = np.unique(UDVS[:,i])
            """
            np.unique considers all NaNs to be unique values
            These two lines find the indices of all NaNs in the unique value array 
            and removes all but the first
            """
            nanvals = np.where(np.isnan(uvals))[0]
            uvals = np.delete(uvals,nanvals[1:])
            """
            Check if more that one unique value
            Append column number to iSpec_var if true
            """
            if (uvals.size > 1): 
                iSpec_var = np.append(iSpec_var, int(i))
        
        iSpec_var = np.asarray(iSpec_var,np.int)
        ds_spec_val_mat = UDVS[:,iSpec_var]
        
        return iSpec_var, ds_spec_val_mat
                    
    def __BEPSVals(udvs_mat, spec_inds, bin_freqs, bin_wfm_type, parm_dict, udvs_labs, udvs_units):
        """
        Returns the Spectroscopic Value array for a BEPS dataset
    
        Parameters
        ----------
        udvs_mat : hdf5 dataset reference to UDVS dataset
        spec_inds : numpy array containing Spectroscopic indices table 
        bin_freqs : 1D numpy array of frequencies
        bin_wfm_type : numpy array containing the waveform type for each frequency index
        parm_dict : parameter dictinary for dataset
        udvs_labs : list of labels for the columns of the UDVS matrix
        udvs_units : list of units for the columns of the UDVS matrix
        
        Returns
        -------
        ds_spec_val_mat : list holding final Spectroscopic Value table 
        """
        
        """
        Check the mode to determine if DC, AC, or something else
        """
        mode = parm_dict['VS_mode']
        
        if mode == 'DC modulation mode' or mode == 'current mode':
            """ 
            First we call the FindSpecVals function to get the columns in UDVS of  
            interest and return a first pass on the spectral value array 
            """ 
            iSpecVals, inSpecVals = __FindSpecValIndices(udvs_mat, spec_inds)   
            
            return __BEPSDC(udvs_mat,inSpecVals, bin_freqs, bin_wfm_type, parm_dict)
        elif mode == 'AC modulation mode with time reversal':
            """ 
            First we call the FindSpecVals function to get the columns in UDVS of  
            interest and return a first pass on the spectral value array 
            """ 
            iSpecVals, inSpecVals = __FindSpecValIndices(udvs_mat, spec_inds)   
                       
            return __BEPSAC(udvs_mat,inSpecVals, bin_freqs, bin_wfm_type, parm_dict)
        else:
            """ 
            First we call the FindSpecVals function to get the columns in UDVS of  
            interest and return a first pass on the spectral value array 
            """ 
            iSpecVals, inSpecVals = __FindSpecValIndices(udvs_mat, spec_inds, usr_defined=True)   
                                   
            return __BEPSgen(udvs_mat,inSpecVals, bin_freqs, bin_wfm_type, parm_dict, udvs_labs, iSpecVals, udvs_units)
            
    
    def __BEPSDC(udvs_mat,inSpecVals, bin_freqs, bin_wfm_type, parm_dict):
        """
        Calculates Spectroscopic Values for BEPS data in DC modulation mode
        
        Parameters
        ----------
        udvs_mat : hdf5 dataset reference to UDVS dataset
        inSpecVals : list holding initial guess at spectral values 
        bin_freqs : 1D numpy array of frequencies
        bin_wfm_type : numpy array containing the waveform type for each frequency index
        parm_dict : parameter dictinary for dataset
                        
        Returns
        -------
        ds_spec_val_mat : list holding final Spectroscopic Value table 
        SpecValsLabels : list holding labels of column names of ds_spec_val_mat
        """
        hascycles = False
        hasFORCS = False
        
        print('inshape',np.shape(inSpecVals))
        """
        All DC datasets will need Spectroscopic Value fields for Bin, DC, and Field
        
        Bin     - Bin number
        DC      - dc offset from UDVS
        Field   - 0 if in-field, 1 if out-of-field 
        """
        nrow = 2
        ds_spec_val_labs = ['Frequency','DC_Offset']
        ds_spec_val_units = ['Hz', 'V']
        
        """
        Get the number of and steps in the current dataset
        """
        numsteps = udvs_mat.shape[0]
        
        """
        Get the wave form for each step from udvs_mat
        """
        wave_form = udvs_mat[:,3]        
        
        """
        Define list of attribute names needed from the group metadata
        """
        field_type = parm_dict['VS_measure_in_field_loops']
        numcycles = parm_dict['VS_number_of_cycles']
        numFORCs = parm_dict['FORC_num_of_FORC_cycles']
        numcyclesteps = parm_dict['VS_steps_per_full_cycle']
        cycle_fraction = parm_dict['VS_cycle_fraction']
        
        if field_type == 'in and out-of-field':
            ds_spec_val_labs.append('Field')
            ds_spec_val_units.append('')
            nrow += 1
        
        frac = {'full':1.0,'1/2':0.5,'1/4':0.25,'3/4':0.75}
        
        numcyclesteps = frac[cycle_fraction]*numcyclesteps
        
        """
        Check the number of cycles and FORCs
        Add to Spectroscopic Values Labels as needed 
        """
        if numcycles > 1:
            hascycles = True
            nrow += 1
            ds_spec_val_labs.append('Cycle')
            ds_spec_val_units.append('')
            
        if numFORCs > 1:
            hasFORCS = True
            """
        It's possible to have 1 cycle with multiple FORCs so force cycle tracking 
        if FROCs exist
            """
            nrow += 1
            ds_spec_val_labs.append('FORC')
            ds_spec_val_units.append('')
            numFORCsteps = numcycles*numcyclesteps*2
        
        """
        Check the field type
        For in-field and out-of-field we know all values of field ahead of time
        For in and out-of-field we must check at each step
        If something else is in field_type, we default to in and out-of-field and print message
        """
        if field_type == 'out-of-field':
            field = 1
            numsteps = numsteps/2
            numcyclesteps = numcyclesteps/2
            swapfield = [1,1]
            field_names = ['out-of-field']
        elif field_type == 'in-field':
            field = 0
            numsteps = numsteps/2
            numcyclesteps = numcyclesteps/2
            swapfield = [0,0]
            field_names = ['in-field']
        elif field_type == 'in and out-of-field':
            field = 0
            swapfield = [1,0]
            field_names = ['out-of-field','in-field']
        else:
            warn('{} is not a known field type'.format(field_type))
            field = 0
            swapfield = [1,0]
            
        """
        Initialize ds_spec_val_mat so that we can append to it in loop
        """
        ds_spec_val_mat = np.empty([nrow,1])
    
        """
        Main loop over all steps
        """
        FORC=-1
        cycle=-1
        for step in xrange(numsteps):
            """
            Calculate the cycle number if needed
            """
            if hasFORCS:
                FORC = np.floor(step/numFORCsteps)
                stepinFORC = step-FORC*numFORCsteps
                cycle = np.floor(stepinFORC/numcyclesteps/2)
            elif hascycles:
                cycle = np.floor(step/numcyclesteps/2)
            
            """
            Change field if needed
            """
            field = swapfield[field]
            """
            Get bins for current step based on waveform
            """
            this_wave = np.where(bin_wfm_type==wave_form[step])[0]
            
            """
            Loop over bins
            """
            for thisbin in this_wave:
                colVal = np.array([[bin_freqs[thisbin]],[inSpecVals[step][0]]])
                
                if field_type == 'in and out-of-field':
                    colVal = np.append(colVal, [[field]], axis=0)
                """
                Add entries to cycle and/or FORC as needed
                """
                if hascycles:
                    colVal = np.append(colVal, [[cycle]], axis=0)
                if hasFORCS:
                    colVal = np.append(colVal, [[FORC]], axis=0)
                
                ds_spec_val_mat = np.append(ds_spec_val_mat, colVal, axis=1)
    
        return ds_spec_val_mat[:,1:], ds_spec_val_labs, ds_spec_val_units, [['Field',field_names]]
        
    def __BEPSAC(udvs_mat, inSpecVals, bin_freqs, bin_wfm_type, parm_dict):
        """
        Calculates Spectroscopic Values for BEPS data in AC modulation mode with time 
                reversal
        
        Parameters
        ----------
        udvs_mat : hdf5 dataset reference to UDVS dataset
        inSpecVals : list holding initial guess at spectral values 
        bin_freqs : 1D numpy array of frequencies
        bin_wfm_type : numpy array containing the waveform type for each frequency index
        parm_dict : parameter dictinary for dataset            
            
        Returns
        -------
        ds_spec_val_mat : list holding final Spectroscopic Value table 
        SpecValsLabels : list holding labels of column names of ds_spec_val_mat
        """
        
        hascycles = False
        hasFORCS = False
    
        """
        All AC datasets will need Spectroscopic Value fields for Bin, AC, and Direction
        
        Bin     - Bin number
        AC      - AC amplitude from UDVS
        forrev   - 1 if forward, -1 if reverse 
        """
        nrow = 3
        ds_spec_val_labs = ['Frequency','AC_Amplitude','Direction']
        ds_spec_val_units = ['Hz', 'A', '']
        
        """
        Get the number of bins and steps in the current dataset
        """
        numsteps = np.shape(udvs_mat)[0]

        """
        Get the wave form for each step from udvs_mat
        """
        wave_form = udvs_mat[:,3]        
        
        """
        Define list of attribute names needed from the group metadata
        """
        numcycles = parm_dict['VS_number_of_cycles']
        numFORCs = parm_dict['FORC_num_of_FORC_cycles']
        numcyclesteps = parm_dict['VS_steps_per_full_cycle']
        cycle_fraction = parm_dict['VS_cycle_fraction']
        
        frac = {'full':1.0,'1/2':0.5,'1/4':0.25,'3/4':0.75}
        
        numcyclesteps = frac[cycle_fraction]*numcyclesteps
        
        """
        Check the number of cycles and FORCs
        Add to Spectroscopic Values Labels as needed 
        """
        if numcycles > 1:
            hascycles = True
            nrow += 1
            ds_spec_val_labs.append('Cycle')
            ds_spec_val_units.append('')
            
        if numFORCs > 1:
            hasFORCS = True
            nrow += 1
            ds_spec_val_labs.append('FORC')
            ds_spec_val_units.append('')
            numFORCsteps = numcycles*numcyclesteps

        """
        Initialize ds_spec_val_mat so that we can append to it in loop
        """
        ds_spec_val_mat = np.empty([nrow,1])

        """
        Main loop over all steps
        """
        FORC=-1
        cycle=-1
        for step in xrange(numsteps):
            """
            Calculate the cycle number if needed
            """
            if hasFORCS:
                FORC = np.floor(step/numFORCsteps)
                stepinFORC = step-FORC*numFORCsteps
                cycle = np.floor(stepinFORC/numcyclesteps)
            elif hascycles:
                cycle = np.floor(step/numcyclesteps)
            
            """
            Check the wave_mod
            """
            wmod = inSpecVals[step][1]
            forrev = np.sign(wmod)
            wmod = abs(wmod)
            
            """
            Get bins for current step based on waveform
            """
            this_wave = np.where(bin_wfm_type==wave_form[step])[0]
            
            """
            Loop over bins
            """
            for thisbin in this_wave:
                colVal = np.array([[bin_freqs[thisbin]],[inSpecVals[step][0]],[forrev]])
                """
                Add entries to cycle and/or FORC as needed
                """
                if hascycles:
                    colVal = np.append(colVal, [[cycle]], axis=0)
                if hasFORCS:
                    colVal = np.append(colVal, [[FORC]], axis=0)
                
                ds_spec_val_mat = np.append(ds_spec_val_mat, colVal, axis=1)
    
        return ds_spec_val_mat[:,1:], ds_spec_val_labs, ds_spec_val_units, [['Direction',['reverse','forward']]]
    
    
    def __BEPSgen(udvs_mat,inSpecVals, bin_freqs, bin_wfm_type, parm_dict, udvs_labs, iSpecVals, udvs_units):
        """
        Calculates Spectroscopic Values for BEPS data in generic mode
        
        Parameters
        ----------
        udvs_mat : hdf5 dataset reference to UDVS dataset
        inSpecVals : list holding initial guess at spectral values 
        bin_freqs : 1D numpy array of frequencies
        bin_wfm_type : numpy array containing the waveform type for each frequency index
        parm_dict : parameter dictinary for dataset            
        udvs_labs : list of labels for the columns of the UDVS matrix
            
        Returns
        -------
        ds_spec_val_mat -- list holding final Spectroscopic Value table 
        SpecValsLabels -- list holding labels of column names of ds_spec_val_mat
        """
        

        """
        Get the number of bins and steps in the current dataset
        """
        numsteps = udvs_mat.shape[0]
        
        """
        Get the wave form for each step from udvs_mat
        """
        wave_form = udvs_mat[:,3]        
        
        """
        All datasets will need Spectroscopic Value fields for Bin,
        everything else must be defined
        
        Bin     - Bin number
        """
        ds_spec_val_labs = ['Frequency']
        ds_spec_val_units = ['Hz']
        
        ds_spec_val_labs.extend(udvs_labs[(iSpecVals[:])])
        ds_spec_val_units.extend([udvs_units[i] for i in iSpecVals])
        nrow = len(ds_spec_val_labs)
        
        """
        Initialize ds_spec_val_mat so that we can append to it in loop
        """
        ds_spec_val_mat = np.empty([nrow,1])
    
        """
        Main loop over all steps
        """
        for step in xrange(numsteps):
            """
            Get the wave form for each step from udvs_mat
            """
            this_wave = np.where(bin_wfm_type==wave_form[step])[0]
            
            for thisbin in this_wave:
                colVal = np.array([[bin_freqs[thisbin]]])
                colVal = np.append(colVal, [[row] for row in inSpecVals[step,:]], axis=0)
                
                ds_spec_val_mat = np.append(ds_spec_val_mat, colVal, axis=1)
    
        return ds_spec_val_mat[:,1:], ds_spec_val_labs, ds_spec_val_units, []

    def __BEPSSpecInds(ds_spec_val_mat):
        """
        Create new Spectroscopic Indices table from the changes in the 
        Spectroscopic Values
        
        Parameters
        ----------
        ds_spec_val_mat : numpy array of floats, 
            Holds the spectroscopic values to be indexed
        
        Returns
        -------
        ds_spec_inds_mat : numpy array of uints the same shape as ds_spec_val_mat
            Indices corresponding to the values in ds_spec_val_mat
        """
        ds_spec_inds_mat = np.zeros_like(ds_spec_val_mat, dtype = np.int32)
        
        """
        Find how quickly the spectroscopic values are changing in each row 
        and the order of row from fastest changing to slowest.
        """
        change_count = [len(np.where([row[i] != row[i-1] for i in xrange(len(row))])[0]) for row in ds_spec_val_mat]
        change_sort = np.argsort(change_count)[::-1]
        
        """
        Determine everywhere the spectroscopic values change and build 
        index table based on those changed
        """
        indices = np.zeros(ds_spec_val_mat.shape[0])
        for jcol in xrange(1,ds_spec_val_mat.shape[1]):
            this_col = ds_spec_val_mat[change_sort,jcol]
            last_col = ds_spec_val_mat[change_sort,jcol-1]

            """
            Check if current column values are different than those 
            in last column.
            """
            changed = np.where(this_col != last_col)[0]
            
            """
            If only one row changed, increment the index for that 
            column
            If more than one row has changed, increment the index for 
            the last row that changed and set all others to zero
            """
            if len(changed) == 1:
                indices[changed]+=1
            elif len(changed > 1):
                for change in changed[:-1]: 
                    indices[change]=0
                indices[changed[-1]]+=1
                
            """
            Store the indices for the current column in the dataset
            """
            ds_spec_inds_mat[change_sort,jcol] = indices
        
        return ds_spec_inds_mat
    
    """
********************************************************************************************
    END OF INTERNAL FUNCTION LIST
    """
    dtype = parm_dict['data_type']
    if dtype == 'BELineData':
        ds_spec_val_mat = bin_freqs.reshape([1,-1])
        ds_spec_inds_mat = np.zeros_like(ds_spec_val_mat, dtype = np.int32)
        ds_spec_val_labs = ['Frequency']
        ds_spec_val_units = ['Hz']
        spec_vals_labs_names = []
        ds_spec_inds_mat[0,:] = np.arange(ds_spec_val_mat.shape[1])
        
    elif dtype == 'BEPSData':
        # Call __BEPSVals to finish the refining of the Spectroscopic Value array        
        ds_spec_val_mat, ds_spec_val_labs, ds_spec_val_units, spec_vals_labs_names = __BEPSVals(udvs_mat, spec_inds, bin_freqs, bin_wfm_type, parm_dict, udvs_labs, udvs_units)
        mode = parm_dict['VS_mode']
        ds_spec_inds_mat = __BEPSSpecInds(ds_spec_val_mat)
    
    else:
        warn('Unknown format! Cannot generate Spectroscopic Values!')
        ds_spec_val_mat = []
        ds_spec_inds_mat = []
        ds_spec_val_labs = []
        ds_spec_val_units = []
        spec_vals_labs_names = []
    
    return ds_spec_val_mat, ds_spec_inds_mat, ds_spec_val_labs, ds_spec_val_units, spec_vals_labs_names


"""
BEHistogram Class and Functions
"""
class BEHistogram():
    """
    Class just functions as a container so we can have shared objects
    Chris Smith -- csmith55@utk.edu
    """
    def addBEHist(self,h5_path, max_mem_mb=1024, show_plot=True, save_plot=True):
        """
        This function adds Histgrams from the Main Data to the Plot Groups for
        an existing hdf5 BEPS datafile.

        Parameters
        ----------
        h5_path : string
            the path to the hdf5 datafile
        max_mem_mb : unsigned integer
            the maximum amount of memory to use during the binning
        show_plot : Boolean
            Should plot of the histograms be drawn after they are
            created
        save_plot : Boolean
            Should plots of the histograms be saved

        Returns
        -------
        None
        """
        hdf = ioHDF5(h5_path)
        h5_file = hdf.file

        print('Adding Histograms to file {}'.format(h5_file.name))
        print('Path to HDF5 file is {}'.format(hdf.path))

        max_mem = min(max_mem_mb*1024**2,0.75*getAvailableMem())

        h5_main = getDataSet(h5_file, 'Raw_Data')
        h5_udvs = getDataSet(h5_file,'UDVS')

        m_groups = [data.parent for data in h5_main]
        print('{} Measurement groups found.'.format(len(m_groups)))

        
        for im, group in enumerate(m_groups):

            p_groups = []

            mspecs = getDataSet(group,'Mean_Spectrogram')
            p_groups.extend([mspec.parent for mspec in mspecs])

            print('{} Plot groups in {}'.format(len(p_groups),group.name))

            for ip, p_group in enumerate(p_groups):
                try:
                    max_resp = getDataSet(group,'Max_Response')
                    min_resp = getDataSet(group,'Min_Response')
                except:
                    warn('Maximum and Minimum Response vectors not found for {}.'.format(p_group.name))
                    max_resp = []
                    min_resp = []

                print 'Creating BEHistogram for Plot Group {}'.format(p_group.name)
                udvs_lab = p_group.attrs['Name']
                udvs_col = h5_udvs[im][h5_udvs[im].attrs[udvs_lab]]
                actual_udvs_steps = np.where(np.isnan(udvs_col)==False)[0]

                """
                Add the BEHistogram for the current plot group
                """
                plot_grp = MicroDataGroup(p_group.name.split('/')[-1], group.name[1:])
                plot_grp.attrs['Name'] = udvs_lab
                hist = BEHistogram()
                hist_mat, hist_labels, hist_indices, hist_indices_labels = hist.buildPlotGroupHist(h5_main[im], actual_udvs_steps, max_response=max_resp, min_response=min_resp, max_mem_mb=max_mem)
                ds_hist = MicroDataset('Histograms',hist_mat, dtype=np.int32, chunking=(1,hist_mat.shape[1]),compression='gzip')
                hist_slice_dict = dict()
                for hist_ind, hist_dim in enumerate(hist_labels):
                    hist_slice_dict[hist_dim] = (slice(hist_ind,hist_ind+1), slice(None))
                ds_hist.attrs['labels'] = hist_slice_dict
                ds_hist_indices = MicroDataset('Histograms_Indices',hist_indices,dtype=np.int32)
                hist_ind_dict = dict()
                for hist_ind_ind, hist_ind_dim in enumerate(hist_indices_labels):
                    hist_ind_dict[hist_ind_dim] = (slice(hist_ind_ind, hist_ind_ind+1),slice(None))
                ds_hist_indices.attrs['labels'] = hist_ind_dict
                ds_hist_labels = MicroDataset('Histograms_Labels',np.array(hist_labels))
                plot_grp.addChildren([ds_hist, ds_hist_indices, ds_hist_labels])
                hdf.writeData(plot_grp)

                if show_plot or save_plot:
                    if save_plot:
                        basename,junk = path.splitext(h5_path)
                        plotfile = '{}_MG{}_PG{}_Histograms.png'.format(basename,im,ip)
                    plotHistgrams(hist_mat, hist_indices, p_group, plotfile)
                    if show_plot:
                        plt.show()

        hdf.close()


    def buildBEHist(self,h5_main, max_response=[], min_response=[],max_mem_mb=1024, max_bins=256, debug=False):
        """
        Creates Histograms from dataset

        Parameters
        ----------
            h5_path : hdf5 reference to Main_Dataset

        Outputs:

        """

        free_mem = getAvailableMem()
        if debug: print 'We have {} bytes of memory available'.format(free_mem)
        self.max_mem = min(max_mem_mb*1024**2,0.75*free_mem)

        """
        Check that max_response and min_response have been defined.
        Call __getminmaxresponse__ is not
        """
        if max_response == [] or min_response == []:
            max_response = np.max(np.abs(h5_main),axis=1)
            min_response = np.min(np.abs(h5_main),axis=1)


        self.max_response = np.mean(max_response)+3*np.std(max_response)
        self.min_response = np.max([0,np.mean(min_response)-3*np.std(min_response)])

        """
        Loop over all datasets
        """
        active_udvs_steps = getActiveUDVSsteps(h5_main) # technically needs to be done only once
        self.num_udvs_steps = len(active_udvs_steps)

        """
        Load auxilary datasets and extract needed parameters
        """
        spec_ind_mat = getAuxData(h5_main,auxDataName=['Spectroscopic_Indices'])[0]
        self.N_spectral_steps = np.shape(spec_ind_mat)[0]

        """
        Set up frequency axis of histogram, same for all histograms in a single dataset
        """
        freqs_mat = getAuxData(h5_main,auxDataName=['Bin_Frequencies'])[0]
        x_hist = np.array(spec_ind_mat)

        self.N_bins = np.size(freqs_mat)
        self.N_freqs = np.size(np.unique(freqs_mat))
        # print 'There are {} total frequencies in this dataset'.format(self.N_bins)
        del freqs_mat, spec_ind_mat

        self.N_pixels = np.shape(h5_main)[1]
        # print 'There are {} pixels in this dataset'.format(self.N_pixels)

        self.N_y_bins = np.int(np.min( (max_bins, np.rint(np.sqrt(self.N_pixels*self.N_spectral_steps)))))
#         self.N_y_bins = np.min( (max_bins, np.rint(2*(self.N_pixels*self.N_spectral_steps)**(1.0/3.0))))
        # print '{} bins will be used'.format(self.N_y_bins)

        ds_hist = self.__datasetHist(h5_main, active_udvs_steps, x_hist,debug)

        return ds_hist

    def buildPlotGroupHist(self, h5_main, active_spec_steps, max_response=[],
                           min_response=[], max_mem_mb=1024, max_bins=256,
                           std_mult=3):
        """
        Creates Histograms for a given plot group

        Parameters
        ----------
        h5_main : HDF5 Dataset object
            Dataset to be historammed
        activ_spec_steps : numpy array
            active spectral steps in the current plot group
        max_response : numpy array
            maximum amplitude at each pixel
        min_response : numpy array
            minimum amplitude at each pixel
        max_mem : Unsigned integer
            maximum number of Mb allowed for use.  Used to calculate the
            number of pixels to load in a chunk
        max_bins : integer
            maximum number of spectroscopic bins
        std_mult : integer
            number of standard deviations from the mean of
            max_response and min_response to include in
            binning

        Returns
        -------
        hist_mat : 2d numpy array
            4 histograms as 1d arrays
        hist_labels : list of strings
            names for the 4 rows in hist_mat
        hist_indices : 2d numpy array
            the frequency and spectroscopic bins of each column in hist_mat
        hist_index_labels : list of strings
            labels for the hist_indices array

        """
        debug=False

        free_mem = getAvailableMem()
        if debug: print('We have {} bytes of memory available'.format(free_mem))
        self.max_mem = min(max_mem_mb,0.75*free_mem)

        """
        Check that max_response and min_response have been defined.
        Call __getminmaxresponse__ is not
        """
        if max_response == [] or min_response == []:
            max_response = np.amax(np.abs(h5_main),axis=0)
            min_response = np.amin(np.abs(h5_main),axis=0)

        self.max_response = np.mean(max_response)+std_mult*np.std(max_response)
        self.min_response = np.mean(min_response)-std_mult*np.std(min_response)
        del max_response,min_response

        """
        Load auxilary datasets and extract needed parameters
        """
        step_ind_mat = getAuxData(h5_main,auxDataName=['UDVS_Indices'])[0].value
        spec_ind_mat = getAuxData(h5_main,auxDataName=['Spectroscopic_Indices'])[0].value
        self.N_spectral_steps = np.size(step_ind_mat)

        active_udvs_steps = np.unique(step_ind_mat[active_spec_steps])
        self.num_udvs_steps = len(active_udvs_steps)

        """
        Set up frequency axis of histogram, same for all histograms in a single dataset
        """
        freqs_mat = getAuxData(h5_main,auxDataName=['Bin_Frequencies'])[0]
        x_hist = np.array([spec_ind_mat[0],step_ind_mat], dtype=np.int32)

        self.N_bins = np.size(freqs_mat)
        self.N_freqs = np.size(np.unique(freqs_mat))

        del freqs_mat, step_ind_mat, spec_ind_mat

        self.N_pixels = np.shape(h5_main)[0]

#         self.N_y_bins = np.int(np.min( (max_bins, np.rint(np.sqrt(self.N_pixels*self.N_spectral_steps)))))
        self.N_y_bins = np.int(np.min( (max_bins, np.rint(2*(self.N_pixels*self.N_spectral_steps)**(1.0/3.0)))))


        ds_hist = self.__datasetHist(h5_main, active_udvs_steps, x_hist, debug)
        if debug: print(np.shape(ds_hist))
        if debug: print('ds_hist max',np.max(ds_hist),
                        'ds_hist min',np.min(ds_hist))

        hist_mat, hist_labels, hist_indices, hist_index_labels = self.__reshapeHist(ds_hist)

        return hist_mat, hist_labels, hist_indices, hist_index_labels

    def __reshapeHist(self,ds_hist):
        """
        Reshape the histogram matrix into table, and build the associated index table

        Parameters
        ----------
        ds_hist : numpy array
            the 4 histogram matrices

        Returns
        -------
        hist_mat : 2d numpy array
            the 4 histograms as 1d arrays
        hist_labels : list of strings
            names for the 4 rows in hist_mat
        hist_indices : 2d numpy array
            the frequency and spectroscopic bins of each column in hist_mat
        hist_index_labels : list of strings
            labels for the hist_indices array
        """
        hist_shape = ds_hist.shape

        hist_mat = np.reshape(ds_hist, (hist_shape[0],hist_shape[1]*hist_shape[2]))

        hist_labels = ['Amplitude','Phase','Real Part','Imaginary Part']

        hist_indices = np.zeros((2,hist_mat.shape[1]), dtype=np.int32)

        hist_index_labels = ['Frequency Bin','Spectroscopic Bin']

        for isbin in xrange(hist_shape[1]):
            for ifbin in xrange(hist_shape[2]):
                ihbin = ifbin+isbin*hist_shape[2]
                hist_indices[0,ihbin] = ifbin
                hist_indices[1,ihbin] = isbin

        return hist_mat, hist_labels, hist_indices, hist_index_labels

    def __datasetHist(self, h5_main, active_udvs_steps, x_hist, debug=False):
        """
        Create the histogram for a single dataset

        Parmeters
        ---------
        h5_main : HDF5 Dataset
            Main_Dataset to be histogramed
        activ_udvs_steps : numpy array
            the active udvs steps in the current plot group
        x_hist : 1d numpy array
            the spectroscopic indices matrix, used to find the
            spectroscopic indices of each udvs step

        Returns
        -------
        ds_hist : numpy array
            the 4 histogram matrices
"""


        """
        Estimate maximum number of pixels to read at once
        """
        max_pixels = maxReadPixels(self.max_mem,self.N_pixels, self.num_udvs_steps, bytes_per_bin=h5_main.dtype.itemsize*self.N_y_bins*self.N_freqs)

        """
        Divide the pixels into chunks that will fit in memory
        """
        pix_chunks = np.append(np.arange(0,self.N_pixels,max_pixels,dtype=np.int),self.N_pixels)

        """
        Initialize the histograms
        """
        ds_hist = np.zeros((4,self.N_freqs,self.N_y_bins),dtype=np.int32)

        """
        loop over pixels
        """
        for ichunk in xrange(len(pix_chunks)-1):
            if debug: print 'pixel chunk',ichunk

            chunk = xrange(pix_chunks[ichunk],pix_chunks[ichunk+1])

            """
        Loop over active UDVS steps
            """
            for iudvs in xrange(self.num_udvs_steps):
                selected = (iudvs+chunk[0]*self.num_udvs_steps)%np.rint(self.num_udvs_steps*self.N_pixels/10) == 0
                if selected:
                    per_done = np.rint(100*(iudvs+chunk[0]*self.num_udvs_steps)/(self.num_udvs_steps*self.N_pixels))
                    print('Binning BEHistogram...{}% --pixels {}-{}, step # {}'.format(per_done,chunk[0],chunk[-1],iudvs))
                udvs_step = active_udvs_steps[iudvs]
                if debug: print('udvs step',udvs_step)

                """
        Get the correct Spectroscopic bins for the current UDVS step
        Read desired pixel chunk from these bins for Main_Data into data_mat
                """
                udvs_bins = np.where(x_hist[1] == udvs_step)[0]
                if debug:
                    print(np.shape(x_hist))
                data_mat = h5_main[pix_chunks[ichunk]:pix_chunks[ichunk+1],(udvs_bins)]

                """
        Get the frequecies that correspond to the current UDVS bins from the total x_hist
                """
                this_x_hist = np.take(x_hist[0], udvs_bins)
                this_x_hist = this_x_hist-this_x_hist[0]
                this_x_hist = np.transpose(np.tile(this_x_hist,(1,pix_chunks[ichunk+1]-pix_chunks[ichunk])))
                this_x_hist = np.squeeze(this_x_hist)

                N_x_bins = np.shape(this_x_hist)[0]
                if debug:
                    print('N_x_bins',N_x_bins)
                    print(this_x_hist)
                    print(np.shape(this_x_hist))
                """
        Create weighting vector.  If setting all to one value, can be a scalar.
                """
                weighting_vec = 1

                if debug: print(np.shape(data_mat))

                """
        Set up the list of functions to call and their corresponding maxima and minima
                """
                func_list = [np.abs,np.angle,np.real,np.imag]
                max_list = [self.max_response,np.pi,self.max_response,self.max_response]
                min_list = [self.min_response,-np.pi,self.min_response,self.min_response]
                """
        Get the Histograms and store in correct place in ds_hist
                """
                for ifunc,func in enumerate(func_list):
                    chunk_hist = buildHistogram(this_x_hist,
                                                data_mat,
                                                N_x_bins,
                                                self.N_y_bins,
                                                weighting_vec,
                                                min_list[ifunc],
                                                max_list[ifunc],
                                                func,
                                                debug)
                    if debug:
                        print('chunkhist-amp',np.shape(chunk_hist))
                        print(chunk_hist.dtype)

                    for (i,ifreq) in enumerate(udvs_bins):
                        ids_freq =this_x_hist[i]
                        if debug:
                            print(i,ifreq)
                            print(ids_freq)
                        ds_hist[ifunc,ids_freq,:] = np.add(ds_hist[ifunc,ids_freq,:],chunk_hist[i,:])

        return ds_hist

    