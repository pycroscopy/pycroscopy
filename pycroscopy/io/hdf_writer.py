# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 12:29:33 2015

@author: Numan Laanait, Suhas Somnath, Chris Smith
"""

from __future__ import division, print_function, absolute_import, unicode_literals
import os
import subprocess
import sys
from time import time, sleep
from warnings import warn
import h5py

from pyUSID.io.hdf_utils import assign_group_index, write_simple_attrs, attempt_reg_ref_build, write_region_references
from .virtual_data import VirtualGroup, VirtualDataset, VirtualData
from ..__version__ import version

if sys.version_info.major == 3:
    unicode = str


class HDFwriter(object):
    def __init__(self, file_handle, force_new=False):
        """
        Main class that simplifies writing to pycroscopy hdf5 files.

        Parameters
        ----------
        file_handle : h5py.File object or str or unicode
            h5py.File - handle to an open file in 'w' or 'r+' mode
            str or unicode - Absolute path to an unopened the hdf5 file
        force_new : bool, optional
            If true, check if the file already exists and delete it if
            it does.  Ignored if the file_handle is a h5py.File object
        """
        warn('HDFWriter is available only for legacy purposes and will be REMOVED in a future release.\n'
             'Please consider using a combination of functions in hdf_utils such as write_main_dataset() instead',
             DeprecationWarning)

        if type(file_handle) in [str, unicode]:
            if force_new and os.path.exists(file_handle):
                os.remove(file_handle)

            try:
                self.file = h5py.File(file_handle, 'r+')
            except IOError:
                self.file = h5py.File(file_handle, 'w')

            self.path = file_handle
        elif type(file_handle) == h5py.File:
            # file handle is actually an open hdf file
            try:
                _ = file_handle.mode
            except ValueError:
                raise ValueError('A closed h5py.File was provided')

            if file_handle.mode == 'r':
                raise TypeError('HDFWriter cannot work with open HDF5 files in read mode. Change to r+ or w')
            self.file = file_handle.file
            self.path = file_handle.filename
        else:
            raise TypeError('Please provide a file path as a string or a valid h5py.File object')

    def clear(self):
        """
        Clear h5.file of all contents

        file.clear() only removes the contents, it does not free up previously allocated space.
        To do so, it's necessary to use the h5repack command after clearing.
        Because the file must be closed and reopened, it is best to call this
        function immediately after the creation of the HDFWriter object.
        """
        warn('This is unlikely to work on Windows due to driver issues')
        self.file.clear()
        self.repack()

    def repack(self):
        """
        Uses the h5repack command to recover cleared space in an hdf5 file.
        h5repack can also be used to change chunking and compression, but these options have
        not yet been implemented here.
        """
        warn('This is unlikely to work on Windows due to driver issues')
        self.close()
        tmpfile = self.path + '.tmp'

        '''
        Repack the opened hdf5 file into a temporary file
        '''
        try:
            repack_line = ' '.join(['h5repack', '"' + self.path + '"', '"' + tmpfile + '"'])
            subprocess.check_output(repack_line,
                                    stderr=subprocess.STDOUT,
                                    shell=True)
            # Check that the file is done being modified
            sleep(0.5)
            while time() - os.stat(tmpfile).st_mtime <= 1:
                sleep(0.5)
        except subprocess.CalledProcessError as err:
            print('Could not repack hdf5 file')
            raise Exception(err.output)
        except Exception:
            raise

        '''
        Delete the original file and move the temporary file to the originals path
        '''
        # TODO Find way to get the real OS error that works in and out of Spyder
        try:
            os.remove(self.path)
            os.rename(tmpfile, self.path)
        except Exception:
            print('Could not copy repacked file to original path.')
            print('The original file is located {}'.format(self.path))
            print('The repacked file is located {}'.format(tmpfile))
            raise

        '''
        Open the repacked file
        '''
        self.file = h5py.File(self.path, mode='r+')

    def close(self):
        """
        Close h5.file
        """
        self.file.close()

    def delete(self):
        """
        Delete's the existing file and creates an empty new file of the same path
        """
        self.close()
        os.remove(self.path)
        self.file = h5py.File(self.path, 'w')

    def flush(self):
        """
        Flush data from memory and commit to file.
        Use this after manually inserting data into the hdf dataset
        """
        self.file.flush()

    @staticmethod
    def __safe_abort(h5_file):
        """
        Performs the operations necessary to safely close oujt the file
        Call this function before raising exceptions so that the rest of the file is at least safe.

        Parameters
        ----------
        h5_file : h5py.File object
            File that needs to be closed

        Notes
        -----
        As of now, we feel that just flushing and closing is sufficient but this centralized function can be changed as
        newer revelations are revealed
        """
        if not isinstance(h5_file, h5py.File):
            raise TypeError('h5_obj should be a h5py File object but is instead of type '
                            '{}. UNABLE to safely abort'.format(type(h5_file)))
        h5_file.flush()
        h5_file.close()

    def write(self, data, print_log=False):
        """
        Writes data into the hdf5 file and assigns data attributes such as region references.
        The tree structure is inferred from the AFMData Object.

        Parameters
        ----------
        data : Instance of MicroData
            Tree structure describing the organization of the data
        print_log : Boolean (Optional)
            Whether or not to print all log statements - use for debugging purposes

        Returns
        -------
        ref_list : List of HDF5dataset or HDF5Datagroup references
            References to the objects written
        """

        h5_file = self.file

        h5_file.attrs['Pycroscopy version'] = version

        # Checking if the data is a VirtualGroup object
        if not isinstance(data, VirtualData):
            raise TypeError('Input expected to be of type MicroData but is of type: {} \n'.format(type(data)))

        if isinstance(data, VirtualDataset):
            # just want to write a single dataset:
            try:
                h5_parent = h5_file[data.parent]
            except KeyError:
                raise KeyError('Parent ({}) of provided VirtualDataset ({}) does not exist in the '
                               'file'.format(data.parent, data.name))
            h5_dset = HDFwriter._create_dataset(h5_parent, data, print_log=print_log)
            return [h5_dset]

        assert isinstance(data, VirtualGroup)  # just to avoid PEP8 warning

        # Populating the tree structure recursively
        ref_list = []

        # Figuring out if the first item in VirtualGroup tree is file or group
        if data.name == '' and data.parent == '/':
            # For file we just write the attributes
            write_simple_attrs(h5_file, data.attrs, obj_type='file', verbose=print_log)
            root = h5_file.name
            ref_list.append(h5_file)
        else:
            # For a group we write it and its attributes
            h5_grp = self._create_group(h5_file[data.parent], data, print_log=print_log)
            root = h5_grp.name
            ref_list.append(h5_grp)

        # Recursive function
        def __populate(child, parent):
            """
            Recursive function to build the tree from the top down.

            Parameters
            ----------
            child : VirtualGroup object
                tree to be written
            parent : h5py.Group or h5py.File object
                HDF5 object to build tree under

            Returns
            -------
            ref_list : list
                list of h5py.Dataset and h5py.Group objects created when populating the file
            """
            # Update the parent attribute with the true path
            child.parent = parent

            h5_parent_group = h5_file[parent]

            if isinstance(child, VirtualGroup):
                h5_obj = HDFwriter._create_group(h5_parent_group, child, print_log=print_log)
                # here we do the recursive function call
                for ch in child.children:
                    __populate(ch, parent + '/' + child.name)
            else:
                h5_obj = HDFwriter._create_dataset(h5_parent_group, child, print_log=print_log)

            ref_list.append(h5_obj)
            return ref_list

        # Recursive function is called at each stage beginning at the root

        for curr_child in data.children:
            __populate(curr_child, root)

        if print_log:
            print('Finished writing to h5 file.\n' +
                  'Right now you got yourself a fancy folder structure. \n' +
                  'Make sure you do some reference linking to take advantage of the full power of HDF5.')
        return ref_list

    @staticmethod
    def _create_group(h5_parent_group, micro_group, print_log=False):
        """
        Creates a h5py.Group object from the provided VirtualGroup object under h5_new_group and writes all attributes

        Parameters
        ----------
        h5_parent_group : h5py.Group object
            Parent group under which the new group object will be created
        micro_group : VirtualGroup object
            Definition for the new group
        print_log : bool, optional. Default=False
            Whether or not to print debugging statements

        Returns
        -------
        h5_new_group : h5py.Group
            The newly created group
        """
        if not isinstance(micro_group, VirtualGroup):
            HDFwriter.__safe_abort(h5_parent_group.file)
            raise TypeError('micro_group should be a VirtualGroup object but is instead of type '
                            '{}'.format(type(micro_group)))
        if not isinstance(h5_parent_group, h5py.Group):
            raise TypeError('h5_parent_group should be a h5py.Group object but is instead of type '
                            '{}'.format(type(h5_parent_group)))

        if micro_group.name == '':
            HDFwriter.__safe_abort(h5_parent_group.file)
            raise ValueError('VirtualGroup object with empty name will not be handled by this function')

        # First complete the name of the group by adding the index suffix
        if micro_group.indexed:
            micro_group.name = assign_group_index(h5_parent_group, micro_group.name, verbose=print_log)

        # Now, try to write the group
        try:
            h5_new_group = h5_parent_group.create_group(micro_group.name)
            if print_log:
                print('Created Group {}'.format(h5_new_group.name))
        except ValueError:
            h5_new_group = h5_parent_group[micro_group.name]
            if print_log:
                print('Found Group already exists {}'.format(h5_new_group.name))
        except Exception:
            HDFwriter.__safe_abort(h5_parent_group.file)
            raise

        # Write attributes
        write_simple_attrs(h5_new_group, micro_group.attrs, 'group', verbose=print_log)

        return h5_new_group

    @staticmethod
    def _create_simple_dset(h5_group, microdset):
        """
        Creates a simple h5py.Dataset object in the file. This is for those cases where the dataset contains
        small data matrices of known shape and value

        Parameters
        ----------
        h5_group : h5py.File or h5py.Group object
            Parent under which this dataset will be created
        microdset : VirtualDataset object
            Definition for the dataset

        Returns
        -------
        h5_dset : h5py.Dataset object
            Newly created datset object
        """
        if not isinstance(microdset, VirtualDataset):
            HDFwriter.__safe_abort(h5_group.file)
            raise TypeError('microdset should be a VirtualGroup object but is instead of type '
                            '{}'.format(type(microdset)))
        if not isinstance(h5_group, (h5py.Group, h5py.File)):
            raise TypeError('h5_group should be a h5py.Group or h5py.File object but is instead of type '
                            '{}. UNABLE to safely abort'.format(type(h5_group)))

        h5_dset = h5_group.create_dataset(microdset.name,
                                          data=microdset.data,
                                          compression=microdset.compression,
                                          dtype=microdset.dtype,
                                          chunks=microdset.chunking)
        return h5_dset

    @staticmethod
    def _create_empty_dset(h5_group, microdset):
        """
        Creates a h5py.Dataset object in the file. This is for those cases where the dataset is expected to be
        large and its contents cannot be held in memory. This function creates an empty dataset that can be filled in
        manually / incrementally

        Parameters
        ----------
        h5_group : h5py.File or h5py.Group object
            Parent under which this dataset will be created
        microdset : VirtualDataset object
            Definition for the dataset

        Returns
        -------
        h5_dset : h5py.Dataset object
            Newly created datset object
        """
        if not isinstance(microdset, VirtualDataset):
            HDFwriter.__safe_abort(h5_group.file)
            raise TypeError('microdset should be a VirtualGroup object but is instead of type '
                            '{}'.format(type(microdset)))
        if not isinstance(h5_group, (h5py.Group, h5py.File)):
            raise TypeError('h5_group should be a h5py.Group or h5py.File object but is instead of type '
                            '{}. UNABLE to safely abort'.format(type(h5_group)))

        h5_dset = h5_group.create_dataset(microdset.name, microdset.maxshape,
                                          compression=microdset.compression,
                                          dtype=microdset.dtype,
                                          chunks=microdset.chunking)
        return h5_dset

    @staticmethod
    def _create_resizeable_dset(h5_group, microdset):
        """
        Creates a simple h5py.Dataset object in the file. This is for those datasets whose dimensions in one or more
        dimensions are not known at the time of creation.

        Parameters
        ----------
        h5_group : h5py.File or h5py.Group object
            Parent under which this dataset will be created
        microdset : VirtualDataset object
            Definition for the dataset

        Returns
        -------
        h5_dset : h5py.Dataset object
            Newly created datset object
        """
        if not isinstance(microdset, VirtualDataset):
            HDFwriter.__safe_abort(h5_group.file)
            raise TypeError('microdset should be a VirtualGroup object but is instead of type '
                            '{}'.format(type(microdset)))
        if not isinstance(h5_group, (h5py.Group, h5py.File)):
            raise TypeError('h5_group should be a h5py.Group or h5py.File object but is instead of type '
                            '{}. UNABLE to safely abort'.format(type(h5_group)))

        # Allow user to specify maxshape to grow in specific dimensions only
        max_shape = microdset.maxshape
        if max_shape is None:
            max_shape = tuple([None for _ in range(len(microdset.data.shape))])

        if microdset.data is not None:
            shape = microdset.data.shape
        elif microdset.chunking is not None:
            warn('No data was given.  The base shape of of the dataset will be set to the chunk size.')
            shape = microdset.chunking
        else:
            raise ValueError("You must provide an initial data array or a chunk size to create a resizeable dataset.")

        h5_dset = h5_group.create_dataset(microdset.name,
                                          data=microdset.data,
                                          shape=shape,
                                          compression=microdset.compression,
                                          dtype=microdset.dtype,
                                          chunks=microdset.chunking,
                                          maxshape=max_shape)
        return h5_dset

    @staticmethod
    def _create_dataset(h5_group, microdset, print_log=False):
        """
        Creates a h5py.Dataset object in the file. This function handles all three kinds of dataset cases

        Parameters
        ----------
        h5_group : h5py.File or h5py.Group object
            Parent under which this dataset will be created
        microdset : VirtualDataset object
            Definition for the dataset
        print_log : bool, optional. Default=False
            Whether or not to print debugging statements

        Returns
        -------
        h5_dset : h5py.Dataset object
            Newly created datset object
        """
        if not isinstance(microdset, VirtualDataset):
            HDFwriter.__safe_abort(h5_group.file)
            raise TypeError('microdset should be a VirtualGroup object but is instead of type '
                            '{}'.format(type(microdset)))
        if not isinstance(h5_group, (h5py.Group, h5py.File)):
            raise TypeError('h5_group should be a h5py.Group or h5py.File object but is instead of type '
                            '{}. UNABLE to safely abort'.format(type(h5_group)))

        h5_file = h5_group.file

        if microdset.name in h5_group.keys():
            HDFwriter.__safe_abort(h5_file)
            raise ValueError('Dataset named {} already exists in group!'.format(h5_group[microdset.name].name))

        # A standardized procedure for safely creating any kind of dataset:
        def __create_dset(h5_parent_group, microdset_obj, build_func):
            try:
                h5_new_dset = build_func(h5_parent_group, microdset_obj)
            except Exception:
                HDFwriter.__safe_abort(h5_file)
                raise
            return h5_new_dset

        # Handle the different types of datasets
        if not microdset.resizable:
            if not bool(microdset.maxshape):
                # finite sized dataset and maxshape is not provided
                # Typically for small / ancillary datasets
                h5_dset = __create_dset(h5_group, microdset, HDFwriter._create_simple_dset)
            else:
                # In many cases, we DON'T need resizable datasets but we know the max-size
                # Here, we only allocate the space. The provided data is ignored
                h5_dset = __create_dset(h5_group, microdset, HDFwriter._create_empty_dset)
        else:
            # Resizable but the written files are significantly larger
            h5_dset = __create_dset(h5_group, microdset, HDFwriter._create_resizeable_dset)

        if print_log:
            print('Created Dataset {}'.format(h5_dset.name))

        HDFwriter._write_dset_attributes(h5_dset, microdset.attrs, print_log=print_log)

        return h5_dset

    @staticmethod
    def _write_dset_attributes(h5_dset, attrs, print_log=False):
        """
        Writes attributes to a h5py dataset

        Parameters
        ----------
        h5_dset : h5py.Dataset object
            h5py dataset to which the attributes will be written to.
            This function handles region references as well
        attrs : dict
            Dictionary containing the attributes as key-value pairs
        print_log : bool, optional. Default=False
            Whether or not to print debugging statements
        """
        if not isinstance(attrs, dict):
            HDFwriter.__safe_abort(h5_dset.file)
            raise TypeError('attrs should be a dictionary but is instead of type '
                            '{}'.format(type(attrs)))
        if not isinstance(h5_dset, h5py.Dataset):
            raise TypeError('h5_dset should be a h5py Dataset object but is instead of type '
                            '{}. UNABLE to safely abort'.format(type(h5_dset)))

        # First, set aside the complicated attribute(s)
        attr_dict = attrs.copy()
        labels_dict = attr_dict.pop('labels', None)

        # Next, write the simple ones using a centralized function
        write_simple_attrs(h5_dset, attr_dict, obj_type='dataset', verbose=print_log)

        if labels_dict is None:
            if print_log:
                print('Finished writing all attributes of dataset')
            return

        if isinstance(labels_dict, (tuple, list)):
            # What if the labels dictionary is just a list of names? make a dictionary using the names
            # This is the most that can be done.
            labels_dict = attempt_reg_ref_build(h5_dset, labels_dict, verbose=print_log)

        if len(labels_dict) == 0:
            if print_log:
                warn('No region references to write')
            return
        # Now, handle the region references attribute:
        write_region_references(h5_dset, labels_dict, verbose=print_log)
