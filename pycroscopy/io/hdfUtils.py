# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 21:14:25 2015

@author: Suhas Somnath, Chris Smith
"""
import h5py
from warnings import warn
import numpy as np

def getDataSet(h5Parent, dataName):
    '''
    Search for dataset objects in the hdf5 file with given name
    and returns a list of reference(s).

    Parameters
    -----------
    h5Parent : h5py.File reference.
        Reference to file, the file must be open and in read-mode.
    dataName : string.
        Name of Dataset object. If not unique, i.e. parent not specified,
        then references to all Dataset objects that contain this name are returned.

    Returns
    ---------
    list of h5py.Reference of the dataset.
    '''
    if isinstance(h5Parent, h5py.File) or isinstance(h5Parent, h5py.Group):
        dataList = []

        def findData(name, obj):
            if name.endswith(dataName) and isinstance(obj, h5py.Dataset):
                dataList.append(obj)

        h5Parent.visititems(findData)
        return dataList
    else:
        print('%s is not an hdf5 File or Group' % (h5Parent))


def getAuxData(parentData, **kwargs):
    '''
    Returns auxiliary dataset objects associated with some DataSet through its attributes.

    Parameters
    ------------
    parentData : h5py.Dataset
        Dataset object reference.
    auxDataName : list of strings, optional, default = all (DataSet.attrs).
        Name of auxiliary Dataset object to return.

    Returns
    -----------
    list of h5py.Reference of auxiliary dataset objects.
    '''
    auxDataName = kwargs.get('auxDataName', parentData.attrs.iterkeys())

    try:
        dataList = []
        f = parentData.file
        for auxName in auxDataName:
            ref = parentData.attrs[auxName]
            if isinstance(ref, h5py.Reference) and isinstance(f[ref], h5py.Dataset):
                dataList.append(f[ref])
    except KeyError:
        warn('%s is not an attribute of %s'
             % (str(auxName), parentData.name))
    except:
        raise

    return dataList


def getDataAttr(parentData, **kwargs):
    '''
    Returns attribute associated with some DataSet.

    Parameters
    -----------
    parentData : h5py.Dataset
        Dataset object reference.
    attrName : list of strings, optional, default = all (DataSet.attrs).
        Name of attribute object to return.

    Returns
    -------
    tuple containing (name,value) pairs of attributes
    '''
    attrName = kwargs.get('attrName', parentData.attrs.iterkeys())

    try:
        dataList = []
        for attr in attrName:
            ref = parentData.attrs[attr]
            dataList.append(ref)
    except KeyError:
        warn('%s is not an attribute of %s'
             % (str(attr), parentData.name))
    except:
        raise

    return dataList


def getH5DsetRefs(ds_names, h5_refs):
    '''
    Given a list of H5 dataset references and a list of dataset names,
    this method returns H5 Dataset objects corresponding to the names

    Parameters
    --------
    ds_names : List of strings
        names of target datasets
    h5_refs : List of H5 dataset references
        list containing the target reference

    Returns
    -------
    aux_dset : List of HDF5 dataset references
        Corresponding references
    '''
    aux_dset = []
    for dset in h5_refs:
        for ds_name in ds_names:
            if dset.name.split('/')[-1] == ds_name:
                aux_dset.append(dset)
    return aux_dset


def getH5GroupRef(group_name, h5_refs):
    '''
    Given a list of H5 references and a group name,
    this method returns H5 Datagroup object corresponding to the names.
    This function is especially useful when the suffix of the written group
    is unknown (due to the autoindexing in ioHDF5)

    Parameters
    -----------
    group_name : unicode / string
        Names of the datagroup
    h5_refs : List of H5 dataset references

    Returns:
    ----------
    h5_grp
    '''
    for dset in h5_refs:
        if dset.name.split('/')[-1].startswith(group_name):
            # assuming that this name will show up only once in the list
            return dset
    return None


def findH5group(h5_main, tool_name):
    '''
    Given a dataset and a tool name, return the list of all groups

    Parameters
    -----------
    h5_main : h5 dataset reference
        Reference to the target dataset to which the tool was applied
    tool_name : String / unicode
        Name of the tool applied to the target dataset

    Returns
    ---------
    groups : list of references to h5 group objects
        groups whose name contains the tool name and the dataset name
    '''
    dset_name = h5_main.name.split('/')[-1]
    parent_grp = h5_main.parent
    groups = []
    for key in parent_grp.keys():
        if dset_name in key and tool_name in key:
            groups.append(parent_grp[key])
    return groups


def getH5RegRefIndices(ref, h5_main, return_method='slices'):
    '''
    Given an hdf5 region reference and the dataset it refers to,
    return an array of indices within that dataset that
    correspond to the reference.

    Parameters
    ---------
        ref - HDF5 Region Reference
        h5_main - HDF5 object that the reference can be returned
                from
        return_method - String, what for should the reference indices be returned
            Options:
                slices - default, the reference is return as pairs of slices
                corners - the reference is returned as pairs of corners representing the
                        starting and ending indices of each block
                points - the reference is returns as a list of tuples of points

    Returns
    --------
        ref_inds - Array of indices in the source dataset that
                ref accesses
    '''

    if return_method == 'points':
        def __cornersToPointArray(start, stop):
            '''
            Convert a pair of tuples representing two opposite corners of an HDF5 region reference
            into a list of arrays for each dimension.

            Parameters:
            -----------
            start - Tuple holding the starting indices of the region
            stop - Tuple holding the final indices of the region

            Outputs:
            --------
            inds - Tuple of arrays containing the list of points in each dimension
            '''
            ranges = []
            for i in xrange(len(start)):
                if start[i] == stop[i]:
                    ranges.append([stop[i]])
                else:
                    ranges.append(np.arange(start[i], stop[i] + 1, dtype=np.uint))
            grid = np.meshgrid(*(ranges), indexing='ij')

            ref_inds = np.asarray(zip(*(x.flat for x in grid)))

            return ref_inds

        retfunc = __cornersToPointArray
    elif return_method == 'corners':
        def __cornersToCorners(start, stop):
            return start, stop

        retfunc = __cornersToCorners
    elif return_method == 'slices':
        def __cornersToSlices(start, stop):
            '''
            Convert a pair of tuples representing two opposite corners of an HDF5 region reference
            into a pair of slices.

            Parameters:
            -----------
            start - Tuple holding the starting indices of the region
            stop - Tuple holding the final indices of the region

            Outputs:
            --------
            slices - pair of slices representing the region
            '''
            slices = []
            for idim in xrange(len(start)):
                slices.append(slice(start[idim], stop[idim]))

            return slices

        retfunc = __cornersToSlices

    if isinstance(ref, h5py.RegionReference):
        region = h5py.h5r.get_region(ref, h5_main.id)
        reg_type = region.get_select_type()
        if reg_type == 2:
            '''
            Reference is hyperslabs
            '''
            ref_inds = []
            for start, end in region.get_select_hyper_blocklist():
                ref_inds.append(retfunc(start, end))
            ref_inds = np.array(ref_inds).reshape(-1, len(start))

        elif reg_type == 3:
            '''
            Reference is single block
            '''
            start, end = region.get_select_bounds()

            ref_inds = retfunc(start, end)
        else:
            warn('No method currently exists for converting this type of reference.')
    else:
        raise TypeError('Input ref must be an HDF5 Region Reference')

    return ref_inds


def checkAndLinkAncillary(hdf, h5_dset, anc_names, h5_main=None, anc_refs=None):
    '''
    This function will add references to auxilliary datasets as attributes
    of an input dataset.
    If the entries in anc_refs are valid references, they will be added
    as attributes with the name taken from the corresponding entry in
    anc_names.
    If an entry in anc_refs is not a valid reference, the function will
    attempt to get the attribute with the same name from the h5_main
    dataset

    @author: Suhas Somnath
    edited - Chris Smith

    Parameters:
    -----------
    hdf -- ioHDF object associated with the HDF5 file
    h5_dset -- HDF5 dataset to which the attributes will be written\
    anc_names -- list of strings containing the attribute names to be used
    h5_main -- Optional, HDF5 dataset from which attributes will be copied
            if anc_refs is None
    anc_refs -- Optional, HDF5 references that correspond to the strings in
            anc_names
    *Note: either h5_main or anc_ref MUST be provided and anc_ref has the
        higher priority if both are present.

    Outputs:
    --------
    None
    '''

    def __checkAndLinkSingle(h5_ref, ref_name):
        if isinstance(h5_ref, h5py.Reference):
            h5_dset.attrs[ref_name] = h5_ref
        elif isinstance(h5_ref, h5py.Dataset):
            h5_dset.attrs[ref_name] = h5_ref.ref
        elif h5_main is not None:
            h5_anc = getAuxData(h5_main, auxDataName=[ref_name])
            if len(h5_anc) == 1:
                hdf.linkRefAsAlias(h5_dset, h5_anc[0], ref_name)
        else:
            warnstring = '{} is not a valid h5py Reference and will be skipped.'.format(repr(h5_ref))
            warn(warnstring)

    if bool(np.iterable(anc_refs) and not isinstance(anc_refs, h5py.Dataset)):
        '''
        anc_refs can be iterated over
        '''
        for ref_name, h5_ref in zip(anc_names, anc_refs):
            __checkAndLinkSingle(h5_ref, ref_name)
    else:
        '''
        anc_refs is just a single value
        '''
        __checkAndLinkSingle(anc_refs, anc_names)

    hdf.flush()


def createRefFromIndices(h5_main, ref_inds):
    '''
    Create a region reference in the destination dataset using an iterable of pairs of indices
    representing the start and end points of a hyperslab block

    Parameters:
    -----------
    h5_main - HDF5 dataset which the region will be in
    ref_inds - Iterable of index pairs, [start indices, final indices] for each block in the
            hyperslab

    Outputs:
    --------
    new_ref - Region reference in h5_main for the blocks of points defined by ref_inds
    '''
    h5_space = h5_main.id.get_space()
    h5_space.select_none()

    for start, stop in ref_inds:
        block = stop - start + 1
        h5_space.select_hyperslab(tuple(start), (1, 1), block=tuple(block), op=1)

    if not h5_space.select_valid():
        warn('Could not create new region reference.')
        return None
    new_ref = h5py.h5r.create(h5_main.id, b'.', h5py.h5r.DATASET_REGION, space=h5_space)

    return new_ref


def reshape_to_Ndims(h5_main, h5_pos=None, h5_spec=None):
    '''
    Reshape the input 2D matrix to be N-dimensions based on the
    position and spectroscopic datasets.

    Inputs:
        h5_main : HDF5 Dataset, 2D data to be reshaped
        h5_pos : (Optional) HDF5 Dataset, Position indices corresponding to
                rows in ds_main
        h5_spec : (Optional) HDF5 Dataset, Spectroscopic indices corresponding
                to columns in ds_main

        If either h5_pos or h5_spec are not provided, the function will first
        attempt to find them as attributes of h5_main.  If that fails, it will
        generate dummy values for them.

    Outputs:
        ds_Nd : N-D numpy array
            N dimensional numpy array arranged as [positions slowest to fastest, spectroscopic slowest to fastest]
        success : boolean or string
            True if full reshape was successful
            "Positions" if it was only possible to reshape by
            the position dimensions
            False if no reshape was possible
    '''

    if h5_pos is None:
        '''
        Get the Position datasets from the references if possible
        '''
        if isinstance(h5_main, h5py.Dataset):
            try:
                ds_pos = h5_main.file[h5_main.attrs['Position_Indices']][()]
            except KeyError:
                print 'No position datasets found as attributes of {}'.format(h5_main.name)
                if len(h5_main.shape) > 1:
                    ds_pos = np.arange(h5_main.shape[0], dtype=np.uint8)
                else:
                    ds_pos = np.array(0, dtype=np.uint8)
            except:
                raise
        else:
            ds_pos = np.arange(h5_main.shape[0], dtype=np.uint8)
    elif isinstance(h5_pos, h5py.Dataset):
        '''
    Position Indices dataset was provided
        '''
        ds_pos = h5_pos[()]
    else:
        raise TypeError('Position Indices must be either h5py.Dataset or None')

    ##################################################

    if h5_spec is None:
        '''
        Get the Spectroscopic datasets from the references if possible
        '''
        if isinstance(h5_main, h5py.Dataset):
            try:
                ds_spec = h5_main.file[h5_main.attrs['Spectroscopic_Indices']][()]
            except KeyError:
                print 'No spectroscopic datasets found as attributes of {}'.format(h5_main.name)
                if len(h5_main.shape) > 1:
                    ds_spec = np.arange(h5_main.shape[1], dtype=np.uint8)
                else:
                    ds_spec = np.array(0, dtype=np.uint8)
            except:
                raise
        else:
            ds_spec = np.arange(h5_main.shape[1], dtype=np.uint8)

    elif isinstance(h5_spec, h5py.Dataset):
        '''
    Spectroscopic Indices dataset was provided
        '''
        ds_spec = h5_spec[()]
    else:
        raise TypeError('Spectroscopic Indices must be either h5py.Dataset or None')

    #######################################################

    '''
    Find how quickly the spectroscopic values are changing in each row
    and the order of rows from fastest changing to slowest.
    '''
    change_count = [len(np.where([row[i] != row[i - 1] for i in xrange(len(row))])[0]) for row in ds_spec]
    change_sort = np.argsort(change_count)[::-1]

    '''
    Get the number of unique values in the index arrays
    This gives us the size of each dimension
    Spectroscopic must go in the row order determined above
    '''
    pos_dims = [len(np.unique(col)) for col in np.array(np.transpose(ds_pos), ndmin=2)]
    spec_dims = [len(np.unique(row)) for row in np.array(ds_spec, ndmin=2)]

    if isinstance(h5_main, h5py.Dataset):
        ds_main = h5_main[()]
    elif isinstance(h5_main, np.ndarray):
        ds_main = h5_main
    spec_dims = [len(np.unique(row)) for row in np.array(ds_spec[change_sort], ndmin=2)]

    ds_main = h5_main[()]

    '''
    Now we reshape the dataset based on those dimensions
    We must use the spectroscopic dimensions in reverse order
    '''
    try:
        ds_Nd = np.reshape(ds_main, pos_dims + spec_dims[::-1])
    except ValueError:
        warn('Could not reshape dataset to full N-dimensional form.  Attempting reshape based on position only.')
        try:
            ds_Nd = np.reshape(ds_main, pos_dims[-1])
            return ds_Nd, 'Positions'
        except ValueError:
            warn('Reshape by position only also failed.  Will keep dataset in 2d form.')
            return ds_main, False
        except:
            raise
    except:
        raise

    '''
    Now we transpose the axes associated with the spectroscopic dimensions
    so that they are in the same order as in the index array
    '''
    swap_axes = np.append(np.arange(len(pos_dims)),
                          change_sort[::-1] + len(pos_dims))

    ds_Nd = np.transpose(ds_Nd, swap_axes)

    return ds_Nd, True


def copyAttributes(source, dest, skip_refs=True):
    '''
    Copy attributes from one h5object to another
    '''
    for attr, atval in source.attrs.iteritems():
        '''
        Don't copy references unless asked
        '''
        if isinstance(atval, h5py.Reference):
            if skip_refs:
                continue
            elif isinstance(atval, h5py.RegionReference):
                '''
                Dereference old reference, get the appropriate data
                slice and create new reference.
                '''
                try:
                    region = h5py.h5r.get_region(atval, source.id)

                    start, end = region.get_select_bounds()
                    ref_slice = []
                    for i in xrange(len(start)):
                        if start[i] == end[i]:
                            ref_slice.append(start[i])
                        else:
                            ref_slice.append(slice(start[i], end[i]))
                except:
                    print 'Could not create new region reference for {} in {}.'.format(attr, source.name)
                    continue

                dest.attrs[attr] = dest.regionref[tuple(ref_slice)]
                continue
            else:
                dest.attrs[attr] = source.file[atval].name
                continue
        dest.attrs[attr] = atval

    return dest


def checkIfMain(h5_main):
    '''
    Checks the input dataset to see if it has all the neccessary
    features to be considered a Main dataset.  This means it is
     2D and has the following attributes
        Position_Indices
        Position_Values
        Spectroscopic_Indices
        Spectroscopic_Values
    :param h5_main: HDF5 Dataset
    :return: success: Boolean, did all tests pass
    '''
    # Check that h5_main is a dataset
    success = isinstance(h5_main, h5py.Dataset)

    if not success:
        print '{} is not an HDF5 Dataset object.'.format(h5_main)
        return success

    h5_name = h5_main.name.split('/')[-1]

    # Check dimensionality
    success = np.all([success, len(h5_main.shape) == 2])

    if not success:
        print '{} is not 2D.'.format(h5_name)
        return success

    # Check for Datasets
    dset_names = ['Position_Indices', 'Position_Values',
                  'Spectroscopic_Indices', 'Spectroscopic_Values']

    for name in dset_names:
        try:
            ds = h5_main.file[h5_main.attrs[name]]
            success = np.all([success, isinstance(ds, h5py.Dataset)])
        except:
            print '{} not found as an attribute of {}.'.format(name, h5_name)
            success = False
            break

    return success
