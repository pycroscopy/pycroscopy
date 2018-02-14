# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 12:29:33 2015
Main Class in charge of writing/reading to/from hdf5 file.
@author: Numan Laanait, Suhas Somnath, Chris Smith
"""

# cannot import unicode_literals since it is not compatible with h5py just yet
from __future__ import division, print_function, absolute_import, unicode_literals
import os
import subprocess
import sys
from collections import Iterable
from time import time, sleep
from warnings import warn

import h5py
import numpy as np

from .microdata import MicroDataGroup
from ..__version__ import version

if sys.version_info.major == 3:
    unicode = str


class ioHDF5(object):
    def __init__(self, file_handle, cachemult=1):
        """
        Handles:
            + I/O operation from HDF5 file.
            + Utilities to get data and associated auxiliary.

        Parameters
        ----------
        file_handle : Object - String or Unicode or open hdf5 file
            Absolute path to the h5 file or an open hdf5 file
        cachemult : unsigned int (Optional. default = 1)
            Cache multiplier
        """
        if type(file_handle) in [str, unicode]:
            # file handle is actually a file path
            # propfaid = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
            # if cachemult != 1:
            #     settings = list(propfaid.get_cache())
            #     settings[2] *= cachemult
            #     propfaid.set_cache(*settings)
            # try:
            #     fid = h5py.h5f.open(file_handle, fapl=propfaid)
            #     self.file = h5py.File(fid, mode = 'r+')
            # except IOError:
            #     #print('Unable to open file %s. \n Making a new one! \n' %(filename))
            #     fid = h5py.h5f.create(file_handle, fapl=propfaid)
            #     self.file = h5py.File(fid, mode = 'w')
            # except:
            #     raise
            try:
                self.file = h5py.File(file_handle, 'r+')
            except IOError:
                self.file = h5py.File(file_handle, 'w')
            except:
                raise

            self.path = file_handle
        elif type(file_handle) == h5py.File:
            # file handle is actually an open hdf file
            if file_handle.mode == 'r':
                warn('ioHDF5 cannot work with open HDF5 files in read mode. Change to r+ or w')
                return
            self.file = file_handle.file
            self.path = file_handle.filename

    def clear(self):
        """
        Clear h5.file of all contents

        file.clear() only removes the contents, it does not free up previously allocated space.
        To do so, it's necessary to use the h5repack command after clearing.
        Because the file must be closed and reopened, it is best to call this
        function immediately after the creation of the ioHDF5 object.
        """
        self.file.clear()
        self.repack()

    def repack(self):
        """
        Uses the h5repack command to recover cleared space in an hdf5 file.
        h5repack can also be used to change chunking and compression, but these options have
        not yet been implemented here.
        """
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
        except:
            raise

        '''
        Delete the original file and move the temporary file to the originals path
        '''
        # TODO Find way to get the real OS error that works in and out of Spyder
        try:
            os.remove(self.path)
            os.rename(tmpfile, self.path)
        except:
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
        Delete h5.file
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

    def writeData(self, data, print_log=False):
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
        refList : List of HDF5dataset or HDF5Datagroup references
            References to the objects written
        """

        h5_file = self.file

        h5_file.attrs['Pycroscopy version'] = version

        # Checking if the data is an MicroDataGroup object
        if not isinstance(data, MicroDataGroup):
            warn('Input of type: {} \n'.format(type(data)))
            sys.exit("Input not of type MicroDataGroup.\n We're done here! \n")

        # Figuring out if the first item in AFMData tree is file or group
        if data.name == '' and data.parent == '/':
            # For file we just write the attributes

            for key, val in data.attrs.items():
                h5_file.attrs[key] = self.clean_string_att(val)
            if print_log:
                print('Wrote attributes of file {} \n'.format(h5_file.name))
            root = h5_file.name
        else:
            # For a group we write it and its attributes
            if data.indexed:
                ''' If the name of the requested group ends in a '_', the user expects
                the suffix index to be appended automatically. Here, we check to
                ensure that the chosen index is new.
                '''
                previous = list()
                for key in h5_file[data.parent].keys():
                    if data.name in key:
                        previous.append(key)
                if len(previous) == 0:
                    index = 0
                else:
                    # assuming that the last element of previous contains the highest index
                    last = h5_file[data.parent][previous[-1]].name
                    index = int(last.split('_')[-1]) + 1
                data.name += '{:03d}'.format(index)
            try:
                g = h5_file[data.parent].create_group(data.name)
                if print_log:
                    print('Created group {}'.format(g.name))
            except ValueError:
                g = h5_file[data.parent][data.name]
                if print_log:
                    print('Group already exists: {}'.format(g.name))
            except:
                h5_file.flush()
                h5_file.close()
                raise
            for key, val in data.attrs.items():
                if val is None:
                    continue
                if print_log:
                    print('Writing attribute: {} with value: {}'.format(key, val))
                g.attrs[key] = self.clean_string_att(val)
            if print_log:
                print('Wrote attributes to group: {} \n'.format(data.name))
            root = g.name

        # Populating the tree structure recursively
        ref_list = []

        # Recursive function

        def __populate(child, parent):
            """
            Recursive function to build the tree from the top down.

            Parameters
            ----------
            child
            parent

            Returns
            -------

            """
            # Update the parent attribute with the true path
            child.parent = parent

            if isinstance(child, MicroDataGroup):
                if child.indexed:
                    previous = np.where([child.name in key for key in h5_file[parent].keys()])[0]
                    if len(previous) == 0:
                        index = 0
                    else:
                        last = list(h5_file[parent].keys())[previous[-1]]
                        index = int(last.split('_')[-1]) + 1
                    child.name += '{:03d}'.format(index)
                try:
                    itm = h5_file[parent].create_group(child.name)
                    if print_log:
                        print('Created Group {}'.format(itm.name))
                except ValueError:
                    itm = h5_file[parent][child.name]
                    print('Found Group already exists {}'.format(itm.name))
                except:
                    h5_file.flush()
                    h5_file.close()
                    raise
                for key, val in child.attrs.items():
                    if val is None:
                        continue
                    if print_log:
                        print('Writing attribute: {} with value: {}'.format(key, val))
                    itm.attrs[key] = self.clean_string_att(val)
                if print_log:
                    print('Wrote attributes to group {}\n'.format(itm.name))
                # here we do the recursive function call
                for ch in child.children:
                    __populate(ch, parent + '/' + child.name)
            else:
                if not child.resizable:
                    if not bool(child.maxshape):
                        # finite sized dataset and maxshape is not provided
                        # Typically for small / ancilliary datasets
                        try:
                            itm = h5_file[parent].create_dataset(child.name,
                                                                 data=child.data,
                                                                 compression=child.compression,
                                                                 dtype=child.data.dtype,
                                                                 chunks=child.chunking)
                        except RuntimeError:
                            itm = h5_file[parent][child.name]
                            warn('Found Dataset already exists {}'.format(itm.name))
                        except:
                            h5_file.flush()
                            h5_file.close()
                            raise
                    else:
                        # In many cases, we DON'T need resizable datasets but we know the max-size
                        # Here, we only allocate the space. The provided data is ignored
                        # print child.name
                        try:
                            itm = h5_file[parent].create_dataset(child.name, child.maxshape,
                                                                 compression=child.compression,
                                                                 dtype=child.dtype,
                                                                 chunks=child.chunking)
                        except RuntimeError:
                            itm = h5_file[parent][child.name]
                            warn('Found Dataset already exists {}'.format(itm.name))
                        except:
                            h5_file.flush()
                            h5_file.close()
                            raise
                else:
                    # Resizable but the written files are significantly larger
                    max_shape = tuple([None for _ in range(len(child.data.shape))])
                    try:
                        itm = h5_file[parent].create_dataset(child.name,
                                                             data=child.data,
                                                             compression=child.compression,
                                                             dtype=child.data.dtype,
                                                             chunks=child.chunking,
                                                             maxshape=max_shape)
                    except RuntimeError:
                        itm = h5_file[parent][child.name]
                        warn('Found Dataset already exists {}'.format(itm.name))
                    except:
                        h5_file.flush()
                        h5_file.close()
                        raise

                if print_log:
                    print('Created Dataset {}'.format(itm.name))
                for key, val in child.attrs.items():
                    # print('Found some region references')
                    # writing region reference
                    if key == 'labels':
                        # print('Found some region references')
                        labels = child.attrs[key]  # labels here is a dictionary
                        self.write_region_references(itm, labels, print_log=print_log)
                        '''
                        Now make an attribute called 'labels' that is a list of strings 
                        First ascertain the dimension of the slicing:
                        '''
                        found_dim = False
                        for dimen, slobj in enumerate(list(labels.values())[0]):
                            # We make the assumption that checking the start is sufficient
                            if slobj.start is not None:
                                found_dim = True
                                break
                        if found_dim:
                            headers = [None] * len(labels)  # The list that will hold all the names
                            for col_name in labels.keys():
                                headers[labels[col_name][dimen].start] = col_name
                            if print_log:
                                print('Writing header attributes: {}'.format(key))
                            # Now write the list of col / row names as an attribute:
                            itm.attrs[key] = self.clean_string_att(headers)
                        else:
                            warn('Unable to write region labels for %s' % (itm.name.split('/')[-1]))

                        if print_log:
                            print('Wrote Region References of Dataset %s' % (itm.name.split('/')[-1]))
                    else:
                        if print_log:
                            print('Writing attribute: {} with value: {}'.format(key, val))
                        itm.attrs[key] = self.clean_string_att(child.attrs[key])
                        if print_log:
                            print('Wrote Attributes of Dataset %s \n' % (itm.name.split('/')[-1]))
                            # Make a dictionary of references
            ref_list.append(itm)
            return ref_list

        # Recursive function is called at each stage beginning at the root
        for child in data.children:
            __populate(child, root)

        if print_log:
            print('Finished writing to h5 file.\n' +
                  'Right now you got yourself a fancy folder structure. \n' +
                  'Make sure you do some reference linking to take advantage of the full power of HDF5.')
        return ref_list

    @staticmethod
    def clean_string_att(att_val):
        """
        Replaces any unicode objects within lists with their string counterparts to ensure compatibility with python 3.
        If the attribute is indeed a list of unicodes, the changes will be made in-place

        Parameters
        ----------
        att_val : object
            Attribute object

        Returns
        -------
        att_val : object
            Attribute object
        """
        try:
            if isinstance(att_val, Iterable):
                if type(att_val) in [unicode, str]:
                    return att_val
                elif np.any([type(x) in [str, unicode, bytes] for x in att_val]):
                    return np.array(att_val, dtype='S')
            if type(att_val) == np.str_:
                return str(att_val)
            return att_val
        except TypeError:
            warn('Failed to clean: {}'.format(att_val))
            raise

    @staticmethod
    def write_region_references(dataset, slices, print_log=False):
        """
        Creates attributes of a h5.Dataset that refer to regions in the arrays

        Parameters
        ----------
        dataset : h5.Dataset instance
            Dataset to which region references will be added as attributes
        slices : dictionary
            The slicing information must be formatted using tuples of slice objects.
            For example {'region_1':(slice(None, None), slice (0,1))}
        print_log : Boolean (Optional. Default = False)
            Whether or not to print status messages
        """
        if print_log:
            print('Starting to write Region References to Dataset', dataset.name, 'of shape:', dataset.shape)
        for sl in slices.keys():
            if print_log:
                print('About to write region reference:', sl, ':', slices[sl])
            if len(slices[sl]) == len(dataset.shape):
                dataset.attrs[sl] = dataset.regionref[slices[sl]]
                if print_log:
                    print('Wrote Region Reference:%s' % sl)
            else:
                warn('Region reference %s could not be written since the object size was not equal to the dimensions of'
                     ' the dataset' % sl)
                raise ValueError
