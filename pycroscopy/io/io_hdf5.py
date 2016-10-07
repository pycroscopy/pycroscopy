# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 12:29:33 2015
Main Class in charge of writing/reading to/from hdf5 file.
@author: Numan Laanait, Suhas Somnath, Chris Smith
"""

import os
import subprocess
import sys
from time import time, sleep
from warnings import warn

import h5py
import numpy as np

from .microdata import MicroDataGroup
from ..__version__ import version


class ioHDF5(object):

    def __init__(self, file_handle,cachemult=1):
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
            propfaid = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
            if cachemult != 1:
                settings = list(propfaid.get_cache())
                settings[2] *= cachemult
                propfaid.set_cache(*settings)
            try:
                fid = h5py.h5f.open(file_handle,fapl=propfaid)
                self.file = h5py.File(fid, mode = 'r+')
            except IOError:
                #print('Unable to open file %s. \n Making a new one! \n' %(filename))
                fid = h5py.h5f.create(file_handle,fapl=propfaid)
                self.file = h5py.File(fid, mode = 'w')
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
        tmpfile = self.path+'.tmp'

        '''
        Repack the opened hdf5 file into a temporary file
        '''
        try:
            repack_line = ' '.join(['h5repack',self.path,tmpfile])
            subprocess.check_output(repack_line,
                                    stderr=subprocess.STDOUT,
                                    shell=True)
            # Check that the file is done being modified
            while time()-os.stat(tmpfile).st_mtime <= 1:
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
        self.file = h5py.File(self.path, mode = 'r+')

    def close(self):
        '''Close h5.file'''
        self.file.close()

    def delete(self):
        ''' Delete h5.file'''
        self.close()
        os.remove(self.path)

    def flush(self):
        '''Flush data from memory and commit to file. 
        Use this after manually inserting data into the hdf dataset'''
        self.file.flush()

    def writeData(self, data, print_log=False):
        '''
        Writes data into the hdf5 file and assigns data attributes such as region references.
        The tree structure is inferred from the AFMData Object.
        
        Parameters
        ----------
        data : Instance of MicroData
            Tree structure describing the organization of the data
            
        Returns
        -------
        refList : List of HDF5dataset or HDF5Datagroup references
            References to the objects written
        '''

        f = self.file

        f.attrs['PySPM version']=version

        # Checking if the data is an MicroDataGroup object
        if not isinstance(data, MicroDataGroup):
            warn('Input of type: {} \n'.format(type(data)))
            sys.exit("Input not of type MicroDataGroup.\n We're done here! \n")

        # Figuring out if the first item in AFMData tree is file or group
        if data.name is '' and data.parent is '/':
            # For file we just write the attributes
            for key in data.attrs.iterkeys():
                f.attrs[key] = data.attrs[key]
            if print_log: print('Wrote attributes of file {} \n'.format(f.name))
            root = f.name
        else:
            # For a group we write it and its attributes
            if data.indexed:
                ''' If the name of the requested group ends in a '_', the user expects
                the suffix index to be appended automatically. Here, we check to
                ensure that the chosen index is new.
                '''
                previous = np.where([data.name in key for key in f[data.parent].keys()])[0]
                if len(previous)==0:
                    index = 0
                else:
                    # assuming that the last element of previous contains the highest index
                    last = f[data.parent].keys()[previous[-1]]
                    index = int(last.split('_')[-1])+1
                data.name+='{:03d}'.format(index)
            try:
                g = f[data.parent].create_group(data.name)
                if print_log: print('Created group {}'.format(g.name))
            except ValueError:
                g = f[data.parent][data.name]
                print('Group already exists: {}'.format(g.name))
            except:
                raise
            for key in data.attrs.iterkeys():
                if data.attrs[key] is None:
                    continue
                g.attrs[key] = data.attrs[key]
            if print_log: print('Wrote attributes to group: {} \n'.format(data.name))
            root = g.name

        # Populating the tree structure recursively
        refList = []
        # Recursive function
        def __populate(child, parent):

            if isinstance(child, MicroDataGroup):
                if child.indexed:
                    previous = np.where([child.name in key for key in f[parent].keys()])[0]
                    if len(previous)==0:
                        index = 0
                    else:
                        last = f[parent].keys()[previous[-1]]
                        index = int(last.split('_')[-1])+1
                    child.name+='{:03d}'.format(index)
                try:
                    itm = f[parent].create_group(child.name)
                    if print_log: print('Created Group {}'.format(itm.name))
                except ValueError:
                    itm = f[parent][child.name]
                    print('Found Group already exists {}'.format(itm.name))
                except:
                    raise
                for key in child.attrs.iterkeys():
                    itm.attrs[key] = child.attrs[key]
                if print_log: print('Wrote attributes to group {}\n'.format(itm.name))
                # here we do the recursive function call
                for ch in child.children:
                    __populate(ch, parent+'/'+child.name)
            else:
                if not child.resizable:
                    if not bool(child.maxshape):
                        # finite sized dataset and maxshape is not provided
                        # Typically for small / ancilliary datasets
                        try:
                            itm = f[parent].create_dataset(child.name,
                                                        data = child.data,
                                                        compression = child.compression,
                                                        dtype = child.data.dtype,
                                                        chunks= child.chunking)
                        except RuntimeError:
                            itm = f[parent][child.name]
                            warn('Found Dataset already exists {}'.format(itm.name))
                        except:
                            raise
                    else:
                        # In many cases, we DON'T need resizable datasets but we know the max-size
                        # Here, we only allocate the space. The provided data is ignored
                        # print child.name
                        try:
                            itm = f[parent].create_dataset(child.name, child.maxshape,
                                                        compression = child.compression,
                                                        dtype = child.dtype,
                                                        chunks= child.chunking)
                        except RuntimeError:
                            itm = f[parent][child.name]
                            warn('Found Dataset already exists {}'.format(itm.name))
                        except:
                            raise
                else:
                    # Resizable but the written files are significantly larger
                    max_shape = tuple([ None for i in range(len(child.data.shape))])
                    try:
                        itm = f[parent].create_dataset(child.name,
                                                    data = child.data,
                                                    compression = child.compression,
                                                    dtype = child.data.dtype,
                                                    chunks= child.chunking,
                                                    maxshape = max_shape)
                    except RuntimeError:
                        itm = f[parent][child.name]
                        warn('Found Dataset already exists {}'.format(itm.name))
                    except:
                        raise

                if print_log: print('Created Dataset {}'.format(itm.name))
                for key in child.attrs.iterkeys():
                    # print('Found some region references')
                    # writing region reference
                    if key is 'labels':
                        # print('Found some region references')
                        labels = child.attrs[key]# labels here is a dictionary
                        self.regionRefs(itm, labels, print_log=print_log)
                        '''
                        Now make an attribute called 'labels' that is a list of strings 
                        First ascertain the dimension of the slicing:
                        '''
                        found_dim = False
                        for dimen, slobj in enumerate(labels[labels.keys()[0]]):
                            # We make the assumption that checking the start is sufficient 
                            if slobj.start != None:
                                found_dim = True
                                break
                        if found_dim:
                            headers = [None]*len(labels) # The list that will hold all the names
                            for col_name in labels.keys():
                                headers[labels[col_name][dimen].start] = col_name
                            # Now write the list of col / row names as an attribute:
                            itm.attrs[key] = headers
                        else:
                            warn('Unable to write region labels for %s' %(itm.name.split('/')[-1]))

                        if print_log: print('Wrote Region References of Dataset %s' %(itm.name.split('/')[-1]))
                    else:
                        itm.attrs[key] = child.attrs[key]
                        if print_log: print('Wrote Attributes of Dataset %s \n' %(itm.name.split('/')[-1]))
                        # Make a dictionary of references
            refList.append(itm)
            return refList

        # Recursive function is called at each stage beginning at the root
        for child in data.children:
            __populate(child, root)

        if print_log:
            print('Finished writing to h5 file.\n'+
                  'Right now you got yourself a fancy folder structure. \n'+
                  'Make sure you do some reference linking to take advantage of the full power of HDF5.')
        return refList



    def regionRefs(self, dataset, slices, print_log=False):
        '''
        Creates attributes of a h5.Dataset that refer to regions in the arrays
        
        Parameters
        ----------
        dataset : h5.Dataset instance
            Dataset to which region references will be added as attributes
        slices : dictionary
            The slicing information must be formatted using tuples of slice objects. 
            For example {'region_1':(slice(None, None), slice (0,1))}
        '''
        for sl in slices.iterkeys():
            if print_log: print('Wrote Region Reference:%s to Dataset %s' %(sl, dataset.name))
            dataset.attrs[sl] = dataset.regionref[slices[sl]]

