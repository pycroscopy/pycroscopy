# -*- coding: utf-8 -*-
"""
Created on Tue Jan 05 07:55:56 2016

@author: Suhas Somnath, Chris Smith, Rama K. Vasudevan
"""

from __future__ import division, print_function, absolute_import, unicode_literals

from os import path
from warnings import warn

import h5py
import matplotlib.pyplot as plt
import numpy as np
import xlrd as xlreader

from pyUSID.io.hdf_utils import get_auxiliary_datasets, find_dataset, get_h5_obj_refs, link_h5_objects_as_attrs, \
    get_attr, create_indexed_group, write_simple_attrs, write_main_dataset, Dimension
from pyUSID.io.write_utils import create_spec_inds_from_vals
from pyUSID.io.io_utils import get_available_memory, recommend_cpu_cores
from ....analysis.optimize import Optimize
from ....processing.histogram import build_histogram
from ....viz.be_viz_utils import plot_1d_spectrum, plot_2d_spectrogram, plot_histograms
from ...hdf_writer import HDFwriter
from ...virtual_data import VirtualDataset, VirtualGroup

nf32 = np.dtype({'names': ['super_band', 'inter_bin_band', 'sub_band'],
                 'formats': [np.float32, np.float32, np.float32]})


def parmsToDict(filepath, parms_to_remove=[]):
    """
    Translates the parameters in the text file into a dictionary. 
    Also indentifies whether this is a BEPS or BELine dataset.
    
    Parameters
    -----------
    filepath : String / Unicode
        Absolute path of the parameters text or spreadsheet file.
    parms_to_remove : List of string (Optional)
        keys that this function should attempt to remove from the dictionary
    
    Returns
    ----------
    isBEPS : Boolean
        whether this dataset is BEPS or BE Line.
    parm_dict : Dictionary
        experimental parameters
    """
    verbose = False

    lines = list()

    if filepath.lower().endswith('.txt'):
        file_handle = open(filepath, 'r')
        raw_lines = file_handle.readlines()
        file_handle.close()
        for line in raw_lines:
            line = line.rstrip()
            if len(line) > 0:
                lines.append(line.split(" : "))

    elif filepath.lower().endswith('.xls') or filepath.lower().endswith('.xlsx'):
        workbook = xlreader.open_workbook(filepath)
        worksheet = workbook.sheet_by_index(0)
        for row in range(worksheet.nrows):
            temp = list()
            for col in range(worksheet.ncols):
                try:
                    val = str(worksheet.cell(row, col).value).strip()
                    if len(val) > 0:
                        temp.append(val)
                except ValueError:
                    pass
                except:
                    raise
            lines.append(temp)
    else:
        warn('Parameter file not of expected format: text or spreadsheet')
        return None

    if verbose:
        print("Finished reading the file")

    is_beps = False
    parm_dict = dict()

    prefix = 'File_'
    for fields in lines:
        # Ignore the parameters describing the GUI choices
        if prefix == 'Multi_':
            continue

        # Check if the line is a group header or parameter/value pair
        if len(fields) == 2:
            # Get the current name/value pair, and clean up the name
            name = fields[0].strip().replace('# ', 'num_').replace('#', 'num_').replace(' ', '_')
            value = fields[1]

            # Rename specific parameters
            if name == '1_mode':
                name = 'mode'
            if name == 'IO_rate':
                name = 'IO_rate_[Hz]'
                value = int(value.split()[0]) * 1E6
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

            # Append the prefix to the name
            name = prefix + name.lstrip(prefix)

            # Write parameter to parm_dict
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
            # Change the parameter prefix to the new one from the group header
            prefix = fields[0].strip('<').strip('>')
            prefix = prefix.split()[0] + '_'

            # Check if there are VS Parameters.  Set isBEPS to true if so.
            is_beps = is_beps or prefix == 'VS_'

    if is_beps:
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
             'Multi_11_mean_channel_2']
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
             'Multi_13_AI_Fourier_amplitude']

    if verbose:
        print("Finished parsing the text pairs. isBEPS = {}".format(is_beps))

    useless_parms.extend(parms_to_remove)

    # Now remove the list of useless parameters:
    for uparm in useless_parms:
        try:
            del parm_dict[uparm]
        except KeyError:
            # warn('Parameter to be deleted does not exist')
            pass
        except:
            raise
    del uparm, useless_parms

    if verbose:
        print("Finished removing useless parameters")

    if is_beps:
        # fix the DC type in the parms:
        if parm_dict['VS_measure_in_field_loops'] == 'out-of-field only':
            parm_dict['VS_measure_in_field_loops'] = 'out-of-field'
        elif parm_dict['VS_measure_in_field_loops'] == 'in-field only':
            parm_dict['VS_measure_in_field_loops'] = 'in-field'

    return is_beps, parm_dict


###############################################################################


def requires_conjugate(chosen_spectra, default_q=10):
    """
    Determines whether or not the conjugate of the data needs to be taken based on the quality factor

    Parameters
    ----------
    chosen_spectra : 2D complex numpy array
        N random spectra arranged as [instance, frequency]
    default_q : unsigned int, Optional
        Default value of Q factor that the SHO guess function results in for poor guesses

    Returns
    -------
    do_conjugate : Boolean
        Whether or not to take the conjugate of the data
    """
    # Do the SHO Guess for each of these
    opt = Optimize(data=chosen_spectra)

    fitguess_results = opt.computeGuess(strategy='complex_gaussian',
                                        processors=recommend_cpu_cores(chosen_spectra.shape[0]),
                                        options={'frequencies': np.arange(chosen_spectra.shape[1])})

    q_results = np.array(fitguess_results)[:, 2]
    good_q = q_results[np.where(q_results != default_q)]

    if np.mean(good_q) < 0:
        return True
    else:
        return False


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

    if expt_type in ['DC modulation mode', 'current mode']:
        return 'DC Bias'
    elif expt_type == 'AC modulation mode with time reversal':
        return 'AC amplitude'
    return 'User Defined'


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
        print("Warning these high harmonics are not supported in translator.")

    # Generate transfer functions
    F_AO_spectrogram = np.transpose(np.tile(FFT_BE_wave / scaling_factor, [spectrogram_mat.shape[1], 1]))
    # Divide by transfer function
    spectrogram_mat = spectrogram_mat / F_AO_spectrogram

    return spectrogram_mat


def generatePlotGroups(h5_main, mean_resp, folder_path, basename, max_resp=[], min_resp=[],
                       max_mem_mb=1024, spec_label='None',
                       show_plots=True, save_plots=True, do_histogram=False,
                       debug=False):
    """
    Generates the spatially averaged datasets for the given raw dataset. 
    The averaged datasets are necessary for quick visualization of the quality of data. 
    
    Parameters
    ----------
    h5_main : H5 reference
        to the main dataset
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
    show_plots : (optional) Boolean
        Whether or not to show plots
    save_plots : (optional) Boolean
        Whether or not to save generated plots
    do_histogram : Boolean (Optional. Default = False)
        Whether or not to generate hisograms. 
        Caution - Histograms can take a fair amount of time to compute.
    debug : Boolean, Optional
        If True, then extra debug statements are printed.
        Default False
    """
    # Too
    assert isinstance(h5_main, h5py.Dataset)
    h5_f = h5_main.file

    grp = h5_main.parent
    h5_freq = grp['Bin_Frequencies']
    UDVS = grp['UDVS']
    spec_inds = h5_main.h5_spec_inds
    UDVS_inds = grp['UDVS_Indices']
    spec_vals = h5_main.h5_spec_vals

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
    col_names = get_attr(UDVS, 'labels')[5:]

    #     col_names = [col for col in col_names if col not in std_cols + ignore_plot_groups]

    freq_inds = spec_inds[spec_inds.attrs['Frequency']].flatten()

    for col_name in col_names:

        # Make sure the column name is a regular Python string
        col_name = str(col_name)

        ref = UDVS.attrs[col_name]
        #         Make sure we're actually dealing with a reference of some type
        if not isinstance(ref, h5py.RegionReference):
            continue
        # 4. Access that column of the data through region reference
        steps = np.where(np.isfinite(UDVS[ref]))[0]
        step_inds = np.array([np.where(UDVS_inds.value == step)[0] for step in steps]).flatten()
        """selected_UDVS_steps = UDVS[ref]
        selected_UDVS_steps = selected_UDVS_steps[np.isfinite(selected_UDVS_steps)]"""

        (step_averaged_vec, mean_spec) = reshape_mean_data(spec_inds, step_inds, mean_resp)

        """ 
        Need to account for cases with multiple excitation waveforms
        This will affect the frequency indices / values
        We are assuming that there is only one excitation waveform per plot group
        """
        freq_slice = np.unique(freq_inds[step_inds])
        freq_vec = h5_freq.value[freq_slice]

        num_bins = len(freq_slice)  # int(len(freq_inds)/len(UDVS[ref]))
        pg_data = np.repeat(UDVS[ref], num_bins)

        plot_grp = create_indexed_group(grp, 'Spatially_Averaged_Plot_Group')
        write_simple_attrs(plot_grp, {'Name': col_name})

        h5_mean_spec = plot_grp.create_dataset('Mean_Spectrogram',
                                               data=mean_spec,
                                               dtype=np.complex64)
        h5_step_avg = plot_grp.create_dataset('Step_Averaged_Response',
                                              data=step_averaged_vec,
                                              dtype=np.complex64)
        # cannot assume that this is DC offset, could be AC amplitude....
        h5_spec_parm = plot_grp.create_dataset('Spectroscopic_Parameter',
                                               data=np.squeeze(pg_data[step_inds]),
                                               dtype=np.uint32)
        write_simple_attrs(h5_spec_parm, {'name': spec_label})
        h5_freq_vec = plot_grp.create_dataset('Bin_Frequencies',
                                          data=freq_vec,
                                          dtype=h5_freq.dtype)

        # Linking the datasets with the frequency and the spectroscopic variable:
        link_h5_objects_as_attrs(h5_mean_spec, [h5_spec_parm, h5_freq_vec])
        link_h5_objects_as_attrs(h5_step_avg, [h5_freq_vec])

        """
        Create Region Reference for the plot group in the Raw_Data, Spectroscopic_Indices 
        and Spectroscopic_Values Datasets
        """
        raw_ref = h5_main.regionref[:, step_inds]
        spec_inds_ref = spec_inds.regionref[:, step_inds]
        spec_vals_ref = spec_vals.regionref[:, step_inds]

        ref_name = col_name.replace(' ', '_').replace('-', '_') + '_Plot_Group'
        h5_main.attrs[ref_name] = raw_ref
        spec_inds.attrs[ref_name] = spec_inds_ref
        spec_vals.attrs[ref_name] = spec_vals_ref

        h5_f.flush()

        if do_histogram:
            """
            Build the histograms for the current plot group
            """
            hist = BEHistogram()
            hist_mat, hist_labels, hist_indices, hist_indices_labels = \
                hist.buildPlotGroupHist(h5_main, step_inds, max_response=max_resp,
                                        min_response=min_resp, max_mem_mb=max_mem_mb, debug=debug)

            hist_grp = create_indexed_group(plot_grp, 'Histogram')

            hist_spec_dims = list()
            hist_units = ['V', '', 'V', 'V']
            for hist_ind, hist_dim in enumerate(hist_labels):
                hist_spec_dims.append(Dimension(hist_dim,
                                                hist_units[hist_ind],
                                                hist_indices[hist_ind]))

            h5_hist = write_main_dataset(hist_grp, hist_mat, 'Histograms',
                                         'Counts', 'a.u.',
                                         None, hist_spec_dims,
                                         h5_pos_inds=h5_main.h5_pos_inds, h5_pos_vals=h5_main.h5_pos_vals,
                                         dtype=np.int32,
                                         chunking=(1, hist_mat.shape[1]),
                                         compression='gzip')

        else:
            """
            Write the min and max response vectors so that histograms can be generated later.
            """
            h5_max_resp = plot_grp.create_dataset('Max_Response',
                                                  data=max_resp)
            h5_min_resp = plot_grp.create_dataset('Min_Response',
                                                  data=min_resp)

        if save_plots or show_plots:
            fig_title = '_'.join(grp.name[1:].split('/') + [col_name])

            fig_1d, axes_1d = plot_1d_spectrum(step_averaged_vec, freq_vec, fig_title)
            if save_plots:
                path_1d = path.join(folder_path, basename + '_Step_Avg_' + fig_title + '.png')
                path_2d = path.join(folder_path, basename + '_Mean_Spec_' + fig_title + '.png')
                path_hist = path.join(folder_path, basename + '_Histograms_' + fig_title + '.png')

                fig_1d.savefig(path_1d, format='png', dpi=300)
            if mean_spec.shape[0] > 1:
                fig_2d, axes_2d = plot_2d_spectrogram(mean_spec, freq_vec, title=fig_title)
                if save_plots:
                    fig_2d.savefig(path_2d, format='png', dpi=300)

            if do_histogram:
                plot_histograms(hist_mat, hist_indices, grp.name, figure_path=path_hist)

            if show_plots:
                plt.show()
            else:
                plt.close('all')

            # print('Generated spatially average data for group: %s' %(col_name))
    print('Completed generating spatially averaged plot groups')


###############################################################################


def reshape_mean_data(spec_inds, step_inds, mean_resp):
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
    num_bins = len(np.unique(spec_inds[0, step_inds]))
    # Stephen says that we can assume that the number of bins will NOT change in a plot group
    mean_spectrogram = mean_resp[step_inds].reshape(-1, num_bins)

    step_averaged_vec = np.mean(mean_spectrogram, axis=0)
    return step_averaged_vec, mean_spectrogram


###############################################################################


def visualize_plot_groups(h5_filepath):
    """
    Visualizes the plot groups present in the provided BE data file
    
    Parameters
    ----------
    h5_filepath : String / Uniciode
        Absolute path of the h5 file
    """
    with h5py.File(h5_filepath, mode='r') as h5f:
        expt_type = h5f.attrs.get('data_type')
        if expt_type not in ['BEPSData', 'BELineData']:
            warn('Invalid data format')
            return
        for grp_name in h5f.keys():
            grp = h5f[grp_name]['Channel_000']
            for plt_grp_name in grp.keys():
                if plt_grp_name.startswith('Spatially_Averaged_Plot_Group_'):
                    plt_grp = grp[plt_grp_name]
                    if expt_type == 'BEPSData':
                        spect_data = plt_grp['Mean_Spectrogram'].value
                        _ = plot_2d_spectrogram(spect_data, plt_grp['Bin_Frequencies'].value,
                                                title=plt_grp.attrs['Name'])
                    step_avg_data = plt_grp['Step_Averaged_Response']
                    _ = plot_1d_spectrum(step_avg_data, plt_grp['Bin_Frequencies'].value, plt_grp.attrs['Name'])
                    try:
                        hist_data = plt_grp['Histograms']
                        hist_bins = plt_grp['Histograms_Indicies']
                        plot_histograms(hist_data, hist_bins, plt_grp.attrs['Name'])
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
        return udvs_mat, udvs_labs, udvs_units

    if len(udvs_labs) != udvs_mat.shape[1]:
        warn('Error: Incompatible UDVS matrix and labels. Not truncating!')
        return udvs_mat, udvs_labs, udvs_units

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
    # col_inds.sort(reverse=True)
    [udvs_units.pop(ind) for ind in range(len(col_inds), 0, -1)]
    udvs_labs = [col for col in udvs_labs if col not in found_cols]

    return udvs_mat, udvs_labs, udvs_units


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
            DC = UDVS[:, 1]
            for step in range(0, DC.size, 2):
                DC[step + 1] = DC[step]
            UDVS[:, 1] = DC

        """
        icheck is an array containing all UDVS steps which should be checked.
        """
        icheck = np.unique(spec_inds[1])
        """
        Keep only the UDVS values for steps which we care about and the 
        first 5 columns
        """
        UDVS = UDVS[icheck, :5]
        #         UDVS = np.array([UDVS[i] for i in icheck])

        """
        Transpose UDVS for ease of looping later on and store the number of steps
        as num_cols
        """
        num_cols = np.size(UDVS, 1)
        """
        Initialize the iSpec_var as an empty array.  It will store the index of the 
        UDVS label for any column which has more than one unique value
        """
        iSpec_var = []

        """
        Loop over all columns in udvs_mat
        """
        for i in range(1, num_cols):
            """
            Find all unique values in the current column
            """
            toosmall = np.where(abs(UDVS[:, i]) < 1E-5)[0]
            UDVS[toosmall, i] = 0
            uvals = np.unique(UDVS[:, i])
            """
            np.unique considers all NaNs to be unique values
            These two lines find the indices of all NaNs in the unique value array 
            and removes all but the first
            """
            nanvals = np.where(np.isnan(uvals))[0]
            uvals = np.delete(uvals, nanvals[1:])
            """
            Check if more that one unique value
            Append column number to iSpec_var if true
            """
            if uvals.size > 1:
                iSpec_var = np.append(iSpec_var, int(i))

        iSpec_var = np.asarray(iSpec_var, np.int)
        ds_spec_val_mat = UDVS[:, iSpec_var]

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

            return __BEPSDC(udvs_mat, inSpecVals, bin_freqs, bin_wfm_type, parm_dict)

        elif mode == 'AC modulation mode with time reversal':
            """ 
            First we call the FindSpecVals function to get the columns in UDVS of  
            interest and return a first pass on the spectral value array 
            """
            iSpecVals, inSpecVals = __FindSpecValIndices(udvs_mat, spec_inds)

            return __BEPSAC(udvs_mat, inSpecVals, bin_freqs, bin_wfm_type, parm_dict)
        else:
            """ 
            First we call the FindSpecVals function to get the columns in UDVS of  
            interest and return a first pass on the spectral value array 
            """
            iSpecVals, inSpecVals = __FindSpecValIndices(udvs_mat, spec_inds, usr_defined=True)

            return __BEPSgen(udvs_mat, inSpecVals, bin_freqs, bin_wfm_type,
                             parm_dict, udvs_labs, iSpecVals, udvs_units)

    def __BEPSDC(udvs_mat, inSpecVals, bin_freqs, bin_wfm_type, parm_dict):
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

        # print('in shape:',np.shape(inSpecVals))
        """
        All DC datasets will need Spectroscopic Value fields for Bin, DC, and Field
        
        Bin     - Bin number
        DC      - dc offset from UDVS
        Field   - 0 if in-field, 1 if out-of-field 
        """
        nrow = 2
        ds_spec_val_labs = ['Frequency', 'DC_Offset']
        ds_spec_val_units = ['Hz', 'V']

        """
        Get the number of and steps in the current dataset
        """
        numsteps = udvs_mat.shape[0]

        """
        Get the wave form for each step from udvs_mat
        """
        wave_form = udvs_mat[:, 3]

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

        frac = {'full': 1.0, '1/2': 0.5, '1/4': 0.25, '3/4': 0.75}

        numcyclesteps = frac[cycle_fraction] * numcyclesteps

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
            numFORCsteps = numcycles * numcyclesteps * 2

        """
        Check the field type
        For in-field and out-of-field we know all values of field ahead of time
        For in and out-of-field we must check at each step
        If something else is in field_type, we default to in and out-of-field and print message
        """
        if field_type == 'out-of-field':
            field = 1
            numsteps = int(numsteps / 2)
            numcyclesteps = int(numcyclesteps / 2)
            swapfield = [1, 1]
            field_names = ['out-of-field']
        elif field_type == 'in-field':
            field = 0
            numsteps = int(numsteps / 2)
            numcyclesteps = int(numcyclesteps / 2)
            swapfield = [0, 0]
            field_names = ['in-field']
        elif field_type == 'in and out-of-field':
            field = 0
            swapfield = [1, 0]
            field_names = ['out-of-field', 'in-field']
        else:
            warn('{} is not a known field type'.format(field_type))
            field = 0
            swapfield = [1, 0]

        """
        Initialize ds_spec_val_mat so that we can append to it in loop
        """
        ds_spec_val_mat = np.empty([nrow, 1])

        """
        Main loop over all steps
        """
        FORC = -1
        cycle = -1
        for step in range(numsteps):
            """
            Calculate the cycle number if needed
            """
            if hasFORCS:
                FORC = np.floor(step / numFORCsteps)
                stepinFORC = step - FORC * numFORCsteps
                cycle = np.floor(stepinFORC / numcyclesteps / 2)
            elif hascycles:
                cycle = np.floor(step / numcyclesteps / 2)

            """
            Change field if needed
            """
            field = swapfield[field]
            """
            Get bins for current step based on waveform
            """
            this_wave = np.where(bin_wfm_type == wave_form[step])[0]

            """
            Loop over bins
            """
            for thisbin in this_wave:
                colVal = np.array([[bin_freqs[thisbin]], [inSpecVals[step][0]]])

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

        return ds_spec_val_mat[:, 1:], ds_spec_val_labs, ds_spec_val_units, [['Field', field_names]]

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
        ds_spec_val_labs = ['Frequency', 'AC_Amplitude', 'Direction']
        ds_spec_val_units = ['Hz', 'A', '']

        """
        Get the number of bins and steps in the current dataset
        """
        numsteps = np.shape(udvs_mat)[0]

        """
        Get the wave form for each step from udvs_mat
        """
        wave_form = udvs_mat[:, 3]

        """
        Define list of attribute names needed from the group metadata
        """
        numcycles = parm_dict['VS_number_of_cycles']
        numFORCs = parm_dict['FORC_num_of_FORC_cycles']
        numcyclesteps = parm_dict['VS_steps_per_full_cycle']
        cycle_fraction = parm_dict['VS_cycle_fraction']

        frac = {'full': 1.0, '1/2': 0.5, '1/4': 0.25, '3/4': 0.75}

        numcyclesteps = frac[cycle_fraction] * numcyclesteps

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
            numFORCsteps = numcycles * numcyclesteps

        """
        Initialize ds_spec_val_mat so that we can append to it in loop
        """
        ds_spec_val_mat = np.empty([nrow, 1])

        """
        Main loop over all steps
        """
        FORC = -1
        cycle = -1
        for step in range(numsteps):
            """
            Calculate the cycle number if needed
            """
            if hasFORCS:
                FORC = np.floor(step / numFORCsteps)
                stepinFORC = step - FORC * numFORCsteps
                cycle = np.floor(stepinFORC / numcyclesteps)
            elif hascycles:
                cycle = np.floor(step / numcyclesteps)

            """
            Check the wave_mod
            """
            wmod = inSpecVals[step][1]
            forrev = np.sign(wmod)

            """
            Get bins for current step based on waveform
            """
            this_wave = np.where(bin_wfm_type == wave_form[step])[0]

            """
            Loop over bins
            """
            for thisbin in this_wave:
                colVal = np.array([[bin_freqs[thisbin]], [inSpecVals[step][0]], [forrev]])
                """
                Add entries to cycle and/or FORC as needed
                """
                if hascycles:
                    colVal = np.append(colVal, [[cycle]], axis=0)
                if hasFORCS:
                    colVal = np.append(colVal, [[FORC]], axis=0)

                ds_spec_val_mat = np.append(ds_spec_val_mat, colVal, axis=1)

        return ds_spec_val_mat[:, 1:], ds_spec_val_labs, ds_spec_val_units, [['Direction', ['reverse', 'forward']]]

    def __BEPSgen(udvs_mat, inSpecVals, bin_freqs, bin_wfm_type, udvs_labs, iSpecVals, udvs_units):
        """
        Calculates Spectroscopic Values for BEPS data in generic mode
        
        Parameters
        ----------
        udvs_mat : hdf5 dataset reference to UDVS dataset
        inSpecVals : list holding initial guess at spectral values 
        bin_freqs : 1D numpy array of frequencies
        bin_wfm_type : numpy array containing the waveform type for each frequency index
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
        wave_form = udvs_mat[:, 3]

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
        ds_spec_val_mat = np.empty([nrow, 1])

        """
        Main loop over all steps
        """
        for step in range(numsteps):
            """
            Get the wave form for each step from udvs_mat
            """
            this_wave = np.where(bin_wfm_type == wave_form[step])[0]

            for thisbin in this_wave:
                colVal = np.array([[bin_freqs[thisbin]]])
                colVal = np.append(colVal, [[row] for row in inSpecVals[step, :]], axis=0)

                ds_spec_val_mat = np.append(ds_spec_val_mat, colVal, axis=1)

        return ds_spec_val_mat[:, 1:], ds_spec_val_labs, ds_spec_val_units, []

    """
********************************************************************************************
    END OF INTERNAL FUNCTION LIST
    """
    dtype = parm_dict['data_type']
    if dtype == 'BELineData':
        ds_spec_val_mat = bin_freqs.reshape([1, -1])
        ds_spec_inds_mat = np.zeros_like(ds_spec_val_mat, dtype=np.int32)
        ds_spec_val_labs = ['Frequency']
        ds_spec_val_units = ['Hz']
        spec_vals_labs_names = []
        ds_spec_inds_mat[0, :] = np.arange(ds_spec_val_mat.shape[1])

    elif dtype == 'BEPSData':
        # Call __BEPSVals to finish the refining of the Spectroscopic Value array        
        ds_spec_val_mat, ds_spec_val_labs, ds_spec_val_units, spec_vals_labs_names = __BEPSVals(udvs_mat, spec_inds,
                                                                                                bin_freqs, bin_wfm_type,
                                                                                                parm_dict, udvs_labs,
                                                                                                udvs_units)
        mode = parm_dict['VS_mode']
        ds_spec_inds_mat = create_spec_inds_from_vals(ds_spec_val_mat)

        # Make sure that the frequencies reset properly for user defined case
        spec_start = 0
        if mode == 'load user defined VS Wave from file':
            wave_form = udvs_mat[:, 3]
            for wave in wave_form:
                wave_freqs = bin_freqs[np.argwhere(bin_wfm_type == wave)].squeeze()
                num_bins = wave_freqs.size
                ds_spec_inds_mat[0, spec_start:spec_start + num_bins] = range(num_bins)
                spec_start += num_bins

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


class BEHistogram:
    # TODO: Make into Process class
    """
    Class just functions as a container so we can have shared objects
    Chris Smith -- csmith55@utk.edu
    """

    def __init__(self):
        self.max_mem = None
        self.max_response = None
        self.min_response = None
        self.num_udvs_steps = 1
        self.N_spectral_steps = 1
        self.N_bins = 1
        self.N_freqs = 1
        self.N_pixels = 1
        self.N_y_bins = 1

    def addBEHist(self, h5_path, max_mem_mb=1024, show_plot=True, save_plot=True):
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
        hdf = HDFwriter(h5_path)
        h5_file = hdf.file

        print('Adding Histograms to file {}'.format(h5_file.name))
        print('Path to HDF5 file is {}'.format(hdf.path))

        max_mem = min(max_mem_mb * 1024 ** 2, 0.75 * get_available_memory())

        h5_main = find_dataset(h5_file, 'Raw_Data')
        h5_udvs = find_dataset(h5_file, 'UDVS')

        m_groups = [data.parent for data in h5_main]
        print('{} Measurement groups found.'.format(len(m_groups)))

        for im, group in enumerate(m_groups):

            p_groups = []

            mspecs = find_dataset(group, 'Mean_Spectrogram')
            p_groups.extend([mspec.parent for mspec in mspecs])

            print('{} Plot groups in {}'.format(len(p_groups), group.name))

            for ip, p_group in enumerate(p_groups):
                try:
                    max_resp = find_dataset(group, 'Max_Response')
                    min_resp = find_dataset(group, 'Min_Response')
                except:
                    warn('Maximum and Minimum Response vectors not found for {}.'.format(p_group.name))
                    max_resp = []
                    min_resp = []

                print('Creating BEHistogram for Plot Group {}'.format(p_group.name))
                udvs_lab = p_group.attrs['Name']
                udvs_col = h5_udvs[im][h5_udvs[im].attrs[udvs_lab]]
                actual_udvs_steps = np.where(np.isnan(udvs_col) is False)[0]

                """
                Add the BEHistogram for the current plot group
                """
                plot_grp = VirtualGroup(p_group.name.split('/')[-1], group.name[1:])
                plot_grp.attrs['Name'] = udvs_lab
                hist = BEHistogram()
                hist_mat, hist_labels, hist_indices, hist_indices_labels = \
                    hist.buildPlotGroupHist(h5_main[im],
                                            actual_udvs_steps,
                                            max_response=max_resp,
                                            min_response=min_resp,
                                            max_mem_mb=max_mem)

                ds_hist = VirtualDataset('Histograms', hist_mat, dtype=np.int32,
                                         chunking=(1, hist_mat.shape[1]), compression='gzip')

                hist_slice_dict = dict()
                for hist_ind, hist_dim in enumerate(hist_labels):
                    hist_slice_dict[hist_dim] = (slice(hist_ind, hist_ind + 1), slice(None))
                ds_hist.attrs['labels'] = hist_slice_dict
                ds_hist_indices = VirtualDataset('Histograms_Indices', hist_indices, dtype=np.int32)
                hist_ind_dict = dict()
                for hist_ind_ind, hist_ind_dim in enumerate(hist_indices_labels):
                    hist_ind_dict[hist_ind_dim] = (slice(hist_ind_ind, hist_ind_ind + 1), slice(None))
                ds_hist_indices.attrs['labels'] = hist_ind_dict
                ds_hist_labels = VirtualDataset('Histograms_Labels', np.array(hist_labels))
                plot_grp.add_children([ds_hist, ds_hist_indices, ds_hist_labels])
                hdf.write(plot_grp)

                if show_plot or save_plot:
                    if save_plot:
                        basename, junk = path.splitext(h5_path)
                        plotfile = '{}_MG{}_PG{}_Histograms.png'.format(basename, im, ip)
                    plot_histograms(hist_mat, hist_indices, p_group, plotfile)
                    if show_plot:
                        plt.show()

        hdf.close()

    def buildBEHist(self, h5_main, max_response=[], min_response=[], max_mem_mb=1024, max_bins=256, debug=False):
        """
        Creates Histograms from dataset

        Parameters
        ----------
        h5_main : hdf5.Dataset
        max_response : list
        min_response : list
        max_mem_mb : int
        max_bins : int
        debug : bool

        Returns
        -------

        """

        free_mem = get_available_memory()
        if debug:
            print('We have {} bytes of memory available'.format(free_mem))
        self.max_mem = min(max_mem_mb * 1024 ** 2, 0.75 * free_mem)

        """
        Check that max_response and min_response have been defined.
        Call __getminmaxresponse__ is not
        """
        if max_response == [] or min_response == []:
            max_response = np.max(np.abs(h5_main), axis=1)
            min_response = np.min(np.abs(h5_main), axis=1)

        self.max_response = np.mean(max_response) + 3 * np.std(max_response)
        self.min_response = np.max([0, np.mean(min_response) - 3 * np.std(min_response)])

        """
        Loop over all datasets
        """
        active_udvs_steps = getActiveUDVSsteps(h5_main)  # technically needs to be done only once
        self.num_udvs_steps = len(active_udvs_steps)

        """
        Load auxilary datasets and extract needed parameters
        """
        spec_ind_mat = get_auxiliary_datasets(h5_main, aux_dset_name=['Spectroscopic_Indices'])[0]
        self.N_spectral_steps = np.shape(spec_ind_mat)[0]

        """
        Set up frequency axis of histogram, same for all histograms in a single dataset
        """
        freqs_mat = get_auxiliary_datasets(h5_main, aux_dset_name=['Bin_Frequencies'])[0]
        x_hist = np.array(spec_ind_mat)

        self.N_bins = np.size(freqs_mat)
        self.N_freqs = np.size(np.unique(freqs_mat))
        # print('There are {} total frequencies in this dataset'.format(self.N_bins))
        del freqs_mat, spec_ind_mat

        self.N_pixels = np.shape(h5_main)[1]
        # print('There are {} pixels in this dataset'.format(self.N_pixels))

        self.N_y_bins = np.int(np.min((max_bins, np.rint(np.sqrt(self.N_pixels * self.N_spectral_steps)))))
        #         self.N_y_bins = np.min( (max_bins, np.rint(2*(self.N_pixels*self.N_spectral_steps)**(1.0/3.0))))
        # print('{} bins will be used'.format(self.N_y_bins))

        ds_hist = self.__datasetHist(h5_main, active_udvs_steps, x_hist, debug)

        return ds_hist

    def buildPlotGroupHist(self, h5_main, active_spec_steps, max_response=[],
                           min_response=[], max_mem_mb=1024, max_bins=256,
                           std_mult=3, debug=False):
        """
        Creates Histograms for a given plot group

        Parameters
        ----------
        h5_main : HDF5 Dataset object
            Dataset to be historammed
        active_spec_steps : numpy array
            active spectral steps in the current plot group
        max_response : numpy array
            maximum amplitude at each pixel
        min_response : numpy array
            minimum amplitude at each pixel
        max_mem_mb : Unsigned integer
            maximum number of Mb allowed for use.  Used to calculate the
            number of pixels to load in a chunk
        max_bins : integer
            maximum number of spectroscopic bins
        std_mult : integer
            number of standard deviations from the mean of
            max_response and min_response to include in
            binning
        debug : boolean
            Turns on debug printing statements if true.  Default False.

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
        free_mem = get_available_memory()
        if debug:
            print('We have {} bytes of memory available'.format(free_mem))
        self.max_mem = min(max_mem_mb, 0.75 * free_mem)

        """
        Check that max_response and min_response have been defined.
        Call __getminmaxresponse__ is not
        """
        if max_response == [] or min_response == []:
            max_response = np.amax(np.abs(h5_main), axis=0)
            min_response = np.amin(np.abs(h5_main), axis=0)

        self.max_response = np.mean(max_response) + std_mult * np.std(max_response)
        self.min_response = np.mean(min_response) - std_mult * np.std(min_response)
        del max_response, min_response

        """
        Load auxilary datasets and extract needed parameters
        """
        step_ind_mat = get_auxiliary_datasets(h5_main, aux_dset_name=['UDVS_Indices'])[0].value
        spec_ind_mat = get_auxiliary_datasets(h5_main, aux_dset_name=['Spectroscopic_Indices'])[0].value
        self.N_spectral_steps = np.size(step_ind_mat)

        active_udvs_steps = np.unique(step_ind_mat[active_spec_steps])
        self.num_udvs_steps = len(active_udvs_steps)

        """
        Set up frequency axis of histogram, same for all histograms in a single dataset
        """
        freqs_mat = get_auxiliary_datasets(h5_main, aux_dset_name=['Bin_Frequencies'])[0]
        x_hist = np.array([spec_ind_mat[0], step_ind_mat], dtype=np.int32)

        self.N_bins = np.size(freqs_mat)
        self.N_freqs = np.size(np.unique(freqs_mat))

        del freqs_mat, step_ind_mat, spec_ind_mat

        self.N_pixels = np.shape(h5_main)[0]

        #         self.N_y_bins = np.int(np.min( (max_bins, np.rint(np.sqrt(self.N_pixels*self.N_spectral_steps)))))
        self.N_y_bins = np.int(np.min((max_bins, np.rint(2 * (self.N_pixels * self.N_spectral_steps) ** (1.0 / 3.0)))))

        ds_hist = self.__datasetHist(h5_main, active_udvs_steps, x_hist, debug)
        if debug:
            print(np.shape(ds_hist))
        if debug:
            print('ds_hist max', np.max(ds_hist), 'ds_hist min', np.min(ds_hist))

        hist_mat, hist_labels, hist_indices, hist_index_labels = self.__reshapeHist(ds_hist)

        return hist_mat, hist_labels, hist_indices, hist_index_labels

    @staticmethod
    def __reshapeHist(ds_hist):
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

        hist_mat = np.reshape(ds_hist, (hist_shape[0], hist_shape[1] * hist_shape[2]))

        hist_labels = ['Amplitude', 'Phase', 'Real Part', 'Imaginary Part']

        hist_indices = np.zeros((2, hist_mat.shape[1]), dtype=np.int32)

        hist_index_labels = ['Frequency Bin', 'Spectroscopic Bin']

        for isbin in range(hist_shape[1]):
            for ifbin in range(hist_shape[2]):
                ihbin = ifbin + isbin * hist_shape[2]
                hist_indices[0, ihbin] = ifbin
                hist_indices[1, ihbin] = isbin

        return hist_mat, hist_labels, hist_indices, hist_index_labels

    def __datasetHist(self, h5_main, active_udvs_steps, x_hist, debug=False):
        """
        Create the histogram for a single dataset

        Parameters
        ----------
        h5_main : HDF5 Dataset
            Main_Dataset to be histogramed
        active_udvs_steps : numpy array
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
        max_pixels = maxReadPixels(self.max_mem, self.N_pixels, self.num_udvs_steps,
                                   bytes_per_bin=h5_main.dtype.itemsize * self.N_y_bins * self.N_freqs)

        """
        Divide the pixels into chunks that will fit in memory
        """
        pix_chunks = np.append(np.arange(0, self.N_pixels, max_pixels, dtype=np.int), self.N_pixels)

        """
        Initialize the histograms
        """
        ds_hist = np.zeros((4, self.N_freqs, self.N_y_bins), dtype=np.int32)

        """
        loop over pixels
        """
        for ichunk in range(len(pix_chunks) - 1):
            if debug:
                print('pixel chunk', ichunk)

            chunk = range(pix_chunks[ichunk], pix_chunks[ichunk + 1])

            """
        Loop over active UDVS steps
            """
            for iudvs in range(self.num_udvs_steps):
                selected = (iudvs + chunk[0] * self.num_udvs_steps) % np.rint(
                    self.num_udvs_steps * self.N_pixels / 10) == 0
                if selected:
                    per_done = np.rint(
                        100 * (iudvs + chunk[0] * self.num_udvs_steps) / (self.num_udvs_steps * self.N_pixels))
                    print('Binning BEHistogram...{}% --pixels {}-{}, step # {}'.format(per_done, chunk[0],
                                                                                       chunk[-1], iudvs))
                udvs_step = active_udvs_steps[iudvs]
                if debug:
                    print('udvs step', udvs_step)

                """
        Get the correct Spectroscopic bins for the current UDVS step
        Read desired pixel chunk from these bins for Main_Data into data_mat
                """
                udvs_bins = np.where(x_hist[1] == udvs_step)[0]
                if debug:
                    print(np.shape(x_hist))
                data_mat = h5_main[pix_chunks[ichunk]:pix_chunks[ichunk + 1], udvs_bins]

                """
        Get the frequecies that correspond to the current UDVS bins from the total x_hist
                """
                this_x_hist = np.take(x_hist[0], udvs_bins)
                this_x_hist = this_x_hist - this_x_hist[0]
                this_x_hist = np.transpose(np.tile(this_x_hist, (1, pix_chunks[ichunk + 1] - pix_chunks[ichunk])))
                this_x_hist = np.squeeze(this_x_hist)

                N_x_bins = np.shape(this_x_hist)[0]
                if debug:
                    print('N_x_bins', N_x_bins)
                    print(this_x_hist)
                    print(np.shape(this_x_hist))
                """
        Create weighting vector.  If setting all to one value, can be a scalar.
                """
                weighting_vec = 1

                if debug:
                    print(np.shape(data_mat))

                """
        Set up the list of functions to call and their corresponding maxima and minima
                """
                func_list = [np.abs, np.angle, np.real, np.imag]
                max_list = [self.max_response, np.pi, self.max_response, self.max_response]
                min_list = [self.min_response, -np.pi, self.min_response, self.min_response]
                """
        Get the Histograms and store in correct place in ds_hist
                """
                for ifunc, func in enumerate(func_list):
                    chunk_hist = build_histogram(this_x_hist,
                                                 data_mat,
                                                 N_x_bins,
                                                 self.N_y_bins,
                                                 weighting_vec,
                                                 min_list[ifunc],
                                                 max_list[ifunc],
                                                 func,
                                                 debug)
                    if debug:
                        print('chunkhist-{}'.format(func.__name__), np.shape(chunk_hist))
                        print(chunk_hist.dtype)

                    for (i, ifreq) in enumerate(udvs_bins):
                        ids_freq = this_x_hist[i]
                        if debug:
                            print(i, ifreq)
                            print(ids_freq)
                        ds_hist[ifunc, ids_freq, :] = np.add(ds_hist[ifunc, ids_freq, :], chunk_hist[i, :])

        return ds_hist


def maxReadPixels(max_memory, tot_pix, bins_per_step, bytes_per_bin=4):
    """
    Calculates the maximum number of pixels that can be loaded into the
    specified memory size. This is particularly useful when applying a
    (typically parallel) operation / processing on each pixel.
    Example - Fitting response to a model.

    Parameters
    -------------
    max_memory : unsigned int
        Maximum memory (in bytes) that can be used.
        For example 4 GB would be = 4*((2**10)**3) bytes
    tot_pix : unsigned int
        Total number of pixels in dataset
    bins_per_step : unsigned int
        Number of bins that will be read (can be portion of each pixel)
    bytes_per_bin : (Optional) unsigned int
        size of each bin - set to 4 bytes

    Returns
    -------
    max_pix : unsigned int
        Maximum number of pixels that will be loaded
    """
    # alternatively try .nbytes
    bytes_per_step = bins_per_step * bytes_per_bin
    max_pix = np.rint(max_memory / bytes_per_step)
    # print('Allowed to read {} of {} pixels'.format(max_pix,tot_pix))
    max_pix = max(1, min(tot_pix, max_pix))
    return np.uint(max_pix)


def getActiveUDVSsteps(h5_raw):
    """
    Returns all the active UDVS steps in the data

    Parameters
    ----------
    h5_raw : HDF5 dataset reference
        Reference to the raw data

    Returns
    -----------
    steps : 1D numpy array
        Active UDVS steps
    """
    udvs_step_vec = get_auxiliary_datasets(h5_raw, aux_dset_name=['UDVS_Indices'])[0].value
    return np.unique(udvs_step_vec)


def getSliceForExcWfm(h5_bin_wfm, excit_wfm):
    """
    Returns the indices that correspond to the given excitation waveform
    that can be used to slice the bin datasets
    * Developer note - Replace the first parameter with the Raw_Data dataset

    Parameters
    ----------------
    h5_bin_wfm : Reference to HDF5 dataset
        Bin Waveform Indices
    excit_wfm : integer
        excitation waveform / wave type

    Returns
    --------------
    slc : slice object
        Slice with the start and end indices
    """
    temp = np.where(h5_bin_wfm.value == excit_wfm)[0]
    return slice(temp[0], temp[-1] + 1)  # Need to add one additional index otherwise, the last index will be lost


def getDataIndicesForUDVSstep(h5_udvs_inds, udvs_step_index):
    """
    Returns the spectroscopic indices that correspond to the given udvs_step_index
    that can be used to slice the main data matrix.
    * Developer note - Replace the first parameter with the Raw_Data dataset

    Parameters
    -------------
    h5_udvs_inds : Reference to HDF5 dataset
        UDVS_Indices dataset
    udvs_step_index : usigned int
        UDVS step index (base 0)

    Returns
    --------------
    ans : 1D numpy array
        Spectroscopic indices
    """
    spec_ind_udvs_step_col = h5_udvs_inds[h5_udvs_inds.attrs.get('UDVS_Step')]
    return np.where(spec_ind_udvs_step_col == udvs_step_index)[0]


def getSpecSliceForUDVSstep(h5_udvs_inds, udvs_step_index):
    """
    Returns the spectroscopic indices that correspond to the given udvs_step_index
    that can be used to slice the main data matrix
    * Developer note - Replace the first parameter with the Raw_Data dataset

    Parameters
    -------------
    h5_udvs_inds : Reference to HDF5 dataset
        UDVS_Indices dataset
    udvs_step_index : unsigned int
        UDVS step index (base 0)

    Returns
    ----------
    slc : slice object
        Object containing the start and end indices
    """
    temp = np.where(h5_udvs_inds.value == udvs_step_index)[0]
    return slice(temp[0], temp[-1] + 1)  # Need to add one additional index otherwise, the last index will be lost


def getForExcitWfm(h5_main, h5_other, wave_type):
    """
    Slices the provided H5 dataset by the provided wave type.
    Note that this is applicable to only certain H5 datasets such as the bin frequences, bin FFT etc.

    Parameters
    ----------
    h5_main : Reference to HDF5 dataset
        Raw_Data dataset
    h5_other :Reference to HDF5 dataset
        The dataset that needs to be sliced such as bin frequencies
    wave_type : unsigned int
        Excitation waveform type

    Returns
    ---------
    freq_vec : 1D numpy array
        data specific to specified excitation waveform
    """
    h5_bin_wfm_type = get_auxiliary_datasets(h5_main, aux_dset_name=['Bin_Wfm_Type'])[0]
    inds = np.where(h5_bin_wfm_type.value == wave_type)[0]
    return h5_other[slice(inds[0], inds[-1] + 1)]


def getIndicesforPlotGroup(h5_udvs_inds, ds_udvs, plt_grp_name):
    """
    For a provided plot group name in the udvs table, this function
    returns the corresponding spectroscopic indices that can be used to index / slice the main data set
    and the data within the udvs table for the requested plot group
    * Developer note - Replace the first parameter with the Raw_Data dataset

    Parameters
    ------------
    h5_udvs_inds : Reference to HDF5 dataset
        containing the UDVS indices
    ds_udvs : Reference to HDF5 dataset
        containing the UDVS table
    plt_grp_name : string
        name of the plot group in the UDVS table

    Returns
    -----------
    step_bin_indices : 2D numpy array
        Indices arranged as [step, bin] in the spectroscopic_indices table
        This is useful for knowing the number of bins and steps in this plot group.
        We are allowed to assume that the number of bins does NOT change within the plot group
    oneD_indices : 1D numpy array
        spectroscopic indices corresponding to the requested plot group
    udvs_plt_grp_col : 1D numpy array
        data contained within the udvs table for the requested plot group
    """

    # working on the UDVS table first:
    # getting the numpy array corresponding the requested plot group
    udvs_col_data = np.squeeze(ds_udvs[ds_udvs.attrs.get(plt_grp_name)])
    # All UDVS steps that are NOT part of the plot grop are empty cells in the table
    # and hence assume a nan value.
    # getting the udvs step indices that belong to this plot group:
    step_inds = np.where(np.isnan(udvs_col_data) is False)[0]
    # Getting the values in that plot group that were non NAN
    udvs_plt_grp_col = udvs_col_data[step_inds]

    # ---------------------------------

    # Now we use the udvs step indices calculated above to get
    # the indices in the spectroscopic indices table
    spec_ind_udvs_step_col = h5_udvs_inds[h5_udvs_inds.attrs.get('UDVS_Step')]
    num_bins = len(np.where(spec_ind_udvs_step_col == step_inds[0])[0])
    # Stepehen says that we can assume that the number of bins will NOT change in a plot group
    step_bin_indices = np.zeros(shape=(len(step_inds), num_bins), dtype=int)

    for indx, step in enumerate(step_inds):
        step_bin_indices[indx, :] = np.where(spec_ind_udvs_step_col == step)[0]

    oneD_indices = step_bin_indices.reshape((step_bin_indices.shape[0] * step_bin_indices.shape[1]))
    return step_bin_indices, oneD_indices, udvs_plt_grp_col


def generateTestSpectroscopicData(num_bins=7, num_steps=3, num_pos=4):
    """
    Generates a (preferably small) test data set using the given parameters.
    Data is filled with indices (with base 1 for simplicity).
    Use this for testing reshape operations etc.

    Parameters
    ----------
    num_bins : unsigned int (Optional. Default = 7)
        Number of bins
    num_steps : unsigned int (Optional. Default = 3)
        Number of spectroscopic steps
    num_pos : unsigned int (Optional. Default = 4)
        Number of fictional positions

    Returns
    --------------
    full_data : 2D numpy array
        Data organized as [steps x bins, positions]
    """
    full_data = np.zeros((num_steps * num_bins, num_pos))
    for pos in range(num_pos):
        bin_count = 0
        for step in range(num_steps):
            for bind in range(num_bins):
                full_data[bin_count, pos] = (pos + 1) * 100 + (step + 1) * 10 + (bind + 1)
                bin_count += 1
    return full_data


def isSimpleDataset(h5_main, isBEPS=True):
    """
    This function figures out if a single number defines the bins for all UDVS steps
    In such cases (udvs_steps x bins, pos) can be reshaped to (bins, positions x steps)
    for (theoretically) faster computation, especially for large datasets

    Actually, things are a lot simpler. Only need to check if number of bins for all excitation waveforms are equal

    Parameters
    -------------
    h5_main : Reference to HDF5 dataset
        Raw_Data dataset
    isBEPS : Boolean (default = True)
        Whether or not this dataset is BEPS

    Returns
    ----------
    data_type : Boolean
        Whether or not this dataset can be unraveled / flattened
    """

    if isBEPS:
        beps_modes = ['DC modulation mode', 'AC modulation mode with time reversal', 'current mode', 'Relaxation']
        if h5_main.parent.parent.attrs['VS_mode'] in beps_modes:
            # I am pretty sure that AC modulation also is simple
            return True
        else:
            # Could be user defined or some other kind I am not aware of
            # In many cases, some of these datasets could also potentially be simple datasets
            ds_udvs = get_auxiliary_datasets(h5_main, aux_dset_name=['UDVS'])[0]
            excit_wfms = ds_udvs[ds_udvs.attrs.get('wave_mod')]
            wfm_types = np.unique(excit_wfms)
            if len(wfm_types) == 1:
                # BEPS with single excitation waveform
                print('Single BEPS excitation waveform')
                return True
            else:
                # Multiple waveform types here
                harm_types = np.unique(np.abs(wfm_types))
                if len(harm_types) != 1:
                    # eg - excitaiton waveforms 1, 2, 3 NOT -1, +1
                    return False
                # In this case a single excitation waveform with forward and reverse was used.
                h5_bin_wfm_type = get_auxiliary_datasets(h5_main, aux_dset_name=['Bin_Wfm_Type'])[0]
                # Now for each wfm type, count number of bins.
                wfm_bin_count = []
                for wfm in wfm_types:
                    wfm_bin_count.append(np.where(h5_bin_wfm_type.value == wfm)[0])
                wfm_lengths = np.unique(np.array(wfm_bin_count))
                if len(wfm_lengths) == 1:
                    # BEPS with multiple excitation waveforms but each excitation waveform has same number of bins
                    print('All BEPS excitation waves have same number of bins')
                    return True
            return False
    else:
        # BE-Line
        return True
