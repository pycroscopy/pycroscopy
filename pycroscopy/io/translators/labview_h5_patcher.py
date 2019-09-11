# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 15:24:12 2015

@author: Chris R. Smith
"""

from __future__ import division, print_function, absolute_import, unicode_literals
from warnings import warn
import sys
import h5py
import os
import numpy as np
from pyUSID.io.translator import Translator
from pyUSID.io.hdf_utils import get_attr, link_as_main, check_and_link_ancillary, find_results_groups, find_dataset
from pyUSID.io.write_utils import create_spec_inds_from_vals

if sys.version_info.major == 3:
    unicode = str


class LabViewH5Patcher(Translator):
    """
    Patches the hdf5 files from the LabView V3 data aquisition software to meet the
    standards of the Pycroscopy data format.

    """

    def __init__(self):
        super(LabViewH5Patcher, self).__init__()

    def _parse_file_path(self, input_path):
        pass

    def _read_data(self):
        pass

    @staticmethod
    def is_valid_file(file_path):
        """
        Checks whether the provided file can be read by this translator

        Parameters
        ----------
        file_path : str
            Path to raw data file

        Returns
        -------
        obj : str
            Path to file that will be accepted by the translate() function if
            this translator is indeed capable of translating the provided file.
            Otherwise, None will be returned
        """
        if not isinstance(file_path, (str, unicode)):
            raise TypeError('file_path should be a string object')
        if not os.path.isfile(file_path):
            return None

        file_path = os.path.abspath(file_path)
        extension = os.path.splitext(file_path)[1][1:]
        if extension not in ['h5', 'hdf5']:
            return None
        try:
            h5_f = h5py.File(file_path, 'r+')
        except:
            return None

        # TODO: Make this check as lot stronger. Currently brittle
        if 'DAQ_software_version_name' not in h5_f.attrs.keys():
            return None

        if len(find_dataset(h5_f, 'Raw_Data')) < 1:
            return None
        return file_path

    def translate(self, h5_path, force_patch=False, **kwargs):
        """
        Add the needed references and attributes to the h5 file that are not created by the
        LabView data aquisition program.

        Parameters
        ----------
        h5_path : str
            path to the h5 file
        force_patch : bool, optional
            Should the check to see if the file has already been patched be ignored.
            Default False.

        Returns
        -------
        h5_file : str
            path to the patched dataset

        """
        # Open the file and check if a patch is needed
        h5_file = h5py.File(os.path.abspath(h5_path), 'r+')
        if h5_file.attrs.get('translator') is not None and not force_patch:
            print('File is already Pycroscopy ready.')
            h5_file.close()
            return h5_path

        '''
        Get the list of all Raw_Data Datasets
        Loop over the list and update the needed attributes
        '''
        raw_list = find_dataset(h5_file, 'Raw_Data')
        for h5_raw in raw_list:
            if 'quantity' not in h5_raw.attrs:
                h5_raw.attrs['quantity'] = 'quantity'
            if 'units' not in h5_raw.attrs:
                h5_raw.attrs['units'] = 'a.u.'

            # Grab the channel and measurement group of the data to check some needed attributes
            h5_chan = h5_raw.parent
            try:
                c_type = get_attr(h5_chan, 'channel_type')

            except KeyError:
                warn_str = "'channel_type' was not found as an attribute of {}.\n".format(h5_chan.name)
                warn_str += "If this is BEPS or BELine data from the LabView aquisition software, " + \
                            "please run the following piece of code.  Afterwards, run this function again.\n" + \
                            "CODE: " \
                            "hdf.file['{}'].attrs['channel_type'] = 'BE'".format(h5_chan.name)
                warn(warn_str)
                h5_file.close()
                return h5_path

            except:
                raise

            if c_type != 'BE':
                continue

            h5_meas = h5_chan.parent
            h5_meas.attrs['num_UDVS_steps'] = h5_meas.attrs['num_steps']

            # Get the object handles for the Indices and Values datasets
            h5_pos_inds = h5_chan['Position_Indices']
            h5_pos_vals = h5_chan['Position_Values']
            h5_spec_inds = h5_chan['Spectroscopic_Indices']
            h5_spec_vals = h5_chan['Spectroscopic_Values']

            # Make sure we have correct spectroscopic indices for the given values
            ds_spec_inds = create_spec_inds_from_vals(h5_spec_vals[()])
            if not np.allclose(ds_spec_inds, h5_spec_inds[()]):
                h5_spec_inds[:, :] = ds_spec_inds[:, :]
                h5_file.flush()

            # Get the labels and units for the Spectroscopic datasets
            h5_spec_labels = h5_spec_inds.attrs['labels']
            inds_and_vals = [h5_pos_inds, h5_pos_vals, h5_spec_inds, h5_spec_vals]
            for dset in inds_and_vals:
                spec_labels = dset.attrs['labels']
                try:
                    spec_units = dset.attrs['units']

                    if len(spec_units) != len(spec_labels):
                        raise KeyError

                except KeyError:
                    dset['units'] = ['' for _ in spec_labels]
                except:
                    raise

            for ilabel, label in enumerate(h5_spec_labels):
                label_slice = (slice(ilabel, ilabel + 1), slice(None))
                if label == '':
                    label = 'Step'
                h5_spec_inds.attrs[label] = h5_spec_inds.regionref[label_slice]
                h5_spec_vals.attrs[label] = h5_spec_vals.regionref[label_slice]

            # Link the references to the Indices and Values datasets to the Raw_Data
            link_as_main(h5_raw, h5_pos_inds, h5_pos_vals, h5_spec_inds, h5_spec_vals)

            # Also link the Bin_Frequencies and Bin_Wfm_Type datasets
            h5_freqs = h5_chan['Bin_Frequencies']
            aux_dset_names = ['Bin_Frequencies']
            aux_dset_refs = [h5_freqs.ref]
            check_and_link_ancillary(h5_raw, aux_dset_names, anc_refs=aux_dset_refs)

            '''
            Get all SHO_Fit groups for the Raw_Data and loop over them
            Get the Guess and Spectroscopic Datasets for each SHO_Fit group
            '''
            sho_list = find_results_groups(h5_raw, 'SHO_Fit')
            for h5_sho in sho_list:
                h5_sho_guess = h5_sho['Guess']
                h5_sho_spec_inds = h5_sho['Spectroscopic_Indices']
                h5_sho_spec_vals = h5_sho['Spectroscopic_Values']

                # Make sure we have correct spectroscopic indices for the given values
                ds_sho_spec_inds = create_spec_inds_from_vals(h5_sho_spec_inds[()])
                if not np.allclose(ds_sho_spec_inds, h5_sho_spec_inds[()]):
                    h5_sho_spec_inds[:, :] = ds_sho_spec_inds[:, :]

                # Get the labels and units for the Spectroscopic datasets
                h5_sho_spec_labels = get_attr(h5_sho_spec_inds, 'labels')
                link_as_main(h5_sho_guess, h5_pos_inds, h5_pos_vals, h5_sho_spec_inds, h5_sho_spec_vals)
                sho_inds_and_vals = [h5_sho_spec_inds, h5_sho_spec_vals]

                for dset in sho_inds_and_vals:
                    spec_labels = get_attr(dset, 'labels')
                    try:
                        spec_units = get_attr(dset, 'units')

                        if len(spec_units) != len(spec_labels):
                            raise KeyError

                    except KeyError:
                        spec_units = [''.encode('utf-8') for _ in spec_labels]
                        dset.attrs['units'] = spec_units

                    except:
                        raise

                # Make region references in the
                for ilabel, label in enumerate(h5_sho_spec_labels):
                    label_slice = (slice(ilabel, ilabel + 1), slice(None))
                    if label == '':
                        label = 'Step'.encode('utf-8')
                    h5_sho_spec_inds.attrs[label] = h5_sho_spec_inds.regionref[label_slice]
                    h5_sho_spec_vals.attrs[label] = h5_sho_spec_vals.regionref[label_slice]

            h5_file.flush()

        h5_file.attrs['translator'] = 'V3patcher'.encode('utf-8')

        h5_file.close()

        return h5_path
