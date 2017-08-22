# -*- coding: utf-8 -*-
"""
Created on Fri Mar 04 11:16:20 2016

@author: Suhas Somnath
"""

from __future__ import division, print_function, absolute_import, unicode_literals
from scipy.io.matlab import loadmat  # To load parameters stored in Matlab .mat file
import numpy as np


def readGmodeParms(parm_path):
    """
    Translates the parameters stored in a G-mode parms.mat file into a python dictionary
    
    Parameters
    ------------
    parm_path : String / unicode
        Absolute path of the parms.mat file
    
    Returns
    ----------
    parm_dict : Dictionary 
        dictionary of relevant parameters neatly formatted with units
    """
    parm_data = loadmat(parm_path, squeeze_me=True, struct_as_record=True)

    parm_dict = dict()
    IO_parms = parm_data['IOparms']
    parm_dict['IO_samp_rate_[Hz]'] = np.int32(IO_parms['sampRate'].item())
    parm_dict['IO_down_samp_rate_[Hz]'] = np.int32(IO_parms['downSampRate'].item())
    parm_dict['IO_AO0_amp'] = np.int32(IO_parms['AO0_amp'].item())
    parm_dict['IO_AO1_amp'] = np.int32(IO_parms['AO1_amp'].item())
    parm_dict['IO_AI_chans'] = np.int32(parm_data['aiChans'])

    env_parms = parm_data['envParms']
    parm_dict['envelope_mode'] = np.int32(env_parms['envMode'].item())
    parm_dict['envelope_type'] = np.int32(env_parms['envType'].item())
    parm_dict['envelope_smoothing'] = np.int32(env_parms['smoothing'].item())

    forc_parms = parm_data['forcParms']
    parm_dict['FORC_V_high_1_[V]'] = np.float(forc_parms['vHigh1'].item())
    parm_dict['FORC_V_high_2_[V]'] = np.float(forc_parms['vHigh2'].item())
    parm_dict['FORC_V_low_1_[V]'] = np.float(forc_parms['vLow1'].item())
    parm_dict['FORC_V_low_2_[V]'] = np.float(forc_parms['vLow2'].item())

    gen_sig = parm_data['genSig']
    parm_dict['wfm_f_fast_[Hz]'] = np.float(gen_sig['fFast'].item())
    parm_dict['wfm_d_fast_[s]'] = np.float(gen_sig['dFast'].item())
    parm_dict['wfm_p_slow_[s]'] = np.float(gen_sig['pSlow'].item())
    parm_dict['wfm_n_cycles'] = np.int32(gen_sig['nCycles'].item())
    parm_dict['wfm_swap_mode'] = np.int32(gen_sig['swapMode'].item())
    parm_dict['wfm_reps'] = np.int32(gen_sig['mReps'].item())

    scl_parms = parm_data['sclParms']
    parm_dict['wfm_amp_tip_fast_[V]'] = np.float(scl_parms['ampTipFast'].item())
    parm_dict['wfm_off_tip_fast_[V]'] = np.float(scl_parms['offTipFast'].item())
    parm_dict['wfm_amp_tip_slow_[V]'] = np.float(scl_parms['ampTipSlow'].item())
    parm_dict['wfm_off_tip_slow_[V]'] = np.float(scl_parms['offTipSlow'].item())
    parm_dict['wfm_amp_BD_fast_[V]'] = np.float(scl_parms['ampBDfast'].item())
    parm_dict['wfm_off_BD_fast_[V]'] = np.float(scl_parms['offBDfast'].item())

    parm_dict['grid_num_rows'] = parm_data['numrows']
    parm_dict['grid_num_cols'] = parm_data['numcols']

    return parm_dict
