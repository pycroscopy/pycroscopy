# -*- coding: utf-8 -*-
from __future__ import print_function

"""
Created on Tue Nov  3 21:14:25 2015

@author: Numan Laanait, Suhas Somnath, Chris Smith
"""
import h5py
from warnings import warn
import numpy as np
from .microdata import MicroDataset

__all__ = ['getDataSet', 'getH5DsetRefs', 'getH5RegRefIndices', 'get_dimensionality', 'get_sort_order',
           'getAuxData', 'getDataAttr', 'getH5GroupRef', 'checkIfMain', 'checkAndLinkAncillary',
           'createRefFromIndices', 'copyAttributes', 'reshape_to_Ndims', 'linkRefs', 'linkRefAsAlias',
           'findH5group']


def getDataSet(h5Parent, dataName):
    """
    Search for dataset objects in the hdf5 file with given name
    and returns a list of reference(s).

    Parameters
    ----------
    h5Parent : h5py.File reference.
        Reference to file, the file must be open and in read-mode.
    dataName : string.
        Name of Dataset object. If not unique, i.e. parent not specified,
        then references to all Dataset objects that contain this name are returned.

    Returns
    -------
    list of h5py.Reference of the dataset.
    """
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
    """
    Returns auxiliary dataset objects associated with some DataSet through its attributes.

    Parameters
    ----------
    parentData : h5py.Dataset
        Dataset object reference.
    auxDataName : list of strings, optional, default = all (DataSet.attrs).
        Name of auxiliary Dataset object to return.

    Returns
    -------
    list of h5py.Reference of auxiliary dataset objects.
    """
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
    """
    Returns attribute associated with some DataSet.

    Parameters
    ----------
    parentData : h5py.Dataset
        Dataset object reference.
    attrName : list of strings, optional, default = all (DataSet.attrs).
        Name of attribute object to return.

    Returns
    -------
    tuple containing (name,value) pairs of attributes
    """
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
    """
    Given a list of H5 dataset references and a list of dataset names,
    this method returns H5 Dataset objects corresponding to the names

    Parameters
    ----------
    ds_names : List of strings
        names of target datasets
    h5_refs : List of H5 dataset references
        list containing the target reference

    Returns
    -------
    aux_dset : List of HDF5 dataset references
        Corresponding references
    """
    aux_dset = []
    for ds_name in ds_names:
        for dset in h5_refs:
            if dset.name.split('/')[-1] == ds_name:
                aux_dset.append(dset)
    return aux_dset


def getH5GroupRef(group_name, h5_refs):
    """
    Given a list of H5 references and a group name,
    this method returns H5 Datagroup object corresponding to the names.
    This function is especially useful when the suffix of the written group
    is unknown (due to the autoindexing in ioHDF5)

    Parameters
    ----------
    group_name : unicode / string
        Names of the datagroup
    h5_refs : List of H5 dataset references

    Returns
    -------
    h5_grp : HDF5 Object Reference
        reference to group that matches the `group_name`
    """
    for dset in h5_refs:
        if dset.name.split('/')[-1].startswith(group_name):
            # assuming that this name will show up only once in the list
            return dset
    return None


def findH5group(h5_main, tool_name):
    """
    Given a dataset and a tool name, return the list of all groups

    Parameters
    ----------
    h5_main : h5 dataset reference
        Reference to the target dataset to which the tool was applied
    tool_name : String / unicode
        Name of the tool applied to the target dataset

    Returns
    -------
    groups : list of references to h5 group objects
        groups whose name contains the tool name and the dataset name
    """
    dset_name = h5_main.name.split('/')[-1]
    parent_grp = h5_main.parent
    groups = []
    for key in parent_grp.keys():
        if dset_name in key and tool_name in key:
            groups.append(parent_grp[key])
    return groups


def getH5RegRefIndices(ref, h5_main, return_method='slices'):
    """
    Given an hdf5 region reference and the dataset it refers to,
    return an array of indices within that dataset that
    correspond to the reference.

    Parameters
    ----------
    ref : HDF5 Region Reference
    h5_main : HDF5 Dataset
        dataset that the reference can be returned from
    return_method : {'slices', 'corners', 'points'}
        slices : the reference is return as pairs of slices

        corners : the reference is returned as pairs of corners representing
        the starting and ending indices of each block

        points : the reference is returns as a list of tuples of points

    Returns
    -------
    ref_inds : Numpy Array
        array of indices in the source dataset that ref accesses

    """

    if return_method == 'points':
        def __cornersToPointArray(start, stop):
            """
            Convert a pair of tuples representing two opposite corners of an HDF5 region reference
            into a list of arrays for each dimension.

            Parameters
            ----------
            start : Tuple
                the starting indices of the region
            stop : Tuple
                the final indices of the region

            Returns
            -------
            inds : Tuple of arrays
                the list of points in each dimension
            """
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
            """
            Convert a pair of tuples representing two opposite corners of an HDF5 region reference
            into a pair of slices.

            Parameters
            ----------
            start : Tuple
                the starting indices of the region
            stop : Tuple
                the final indices of the region

            Returns
            -------
            slices : list
                pair of slices representing the region
            """
            slices = []
            for idim in xrange(len(start)):
                slices.append(slice(start[idim], stop[idim]))

            return slices

        retfunc = __cornersToSlices

    if isinstance(ref, h5py.RegionReference):
        region = h5py.h5r.get_region(ref, h5_main.id)
        reg_type = region.get_select_type()
        if reg_type == 2:
            """
            Reference is hyperslabs
            """
            ref_inds = []
            for start, end in region.get_select_hyper_blocklist():
                ref_inds.append(retfunc(start, end))
            ref_inds = np.array(ref_inds).reshape(-1, len(start))

        elif reg_type == 3:
            """
            Reference is single block
            """
            start, end = region.get_select_bounds()

            ref_inds = retfunc(start, end)
        else:
            warn('No method currently exists for converting this type of reference.')
    else:
        raise TypeError('Input ref must be an HDF5 Region Reference')

    return ref_inds


def checkAndLinkAncillary(h5_dset, anc_names, h5_main=None, anc_refs=None):
    """
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

    Parameters
    ----------
    hdf : ioHDF5 object
        object associated with the HDF5 file
    h5_dset : HDF5 Dataset
        dataset to which the attributes will be written
    anc_names : list of str
        the attribute names to be used
    h5_main : HDF5 Dataset, optional
        dataset from which attributes will be copied if `anc_refs` is None
    anc_refs : list of HDF5 Object References, optional
        references that correspond to the strings in `anc_names`

    Returns
    -------
    None

    Notes
    -----
    Either `h5_main` or `anc_refs` MUST be provided and `anc_refs` has the
    higher priority if both are present.
    """

    def __checkAndLinkSingle(h5_ref, ref_name):
        if isinstance(h5_ref, h5py.Reference):
            h5_dset.attrs[ref_name] = h5_ref
        elif isinstance(h5_ref, h5py.Dataset):
            h5_dset.attrs[ref_name] = h5_ref.ref
        elif h5_main is not None:
            h5_anc = getAuxData(h5_main, auxDataName=[ref_name])
            if len(h5_anc) == 1:
                linkRefAsAlias(h5_dset, h5_anc[0], ref_name)
        else:
            warnstring = '{} is not a valid h5py Reference and will be skipped.'.format(repr(h5_ref))
            warn(warnstring)

    if bool(np.iterable(anc_refs) and not isinstance(anc_refs, h5py.Dataset)):
        """
        anc_refs can be iterated over
        """
        for ref_name, h5_ref in zip(anc_names, anc_refs):
            __checkAndLinkSingle(h5_ref, ref_name)
    elif anc_refs is not None:
        """
        anc_refs is just a single value
        """
        __checkAndLinkSingle(anc_refs, anc_names)
    elif isinstance(anc_names, str) or isinstance(anc_names, unicode):
        """
        Single name provided
        """
        __checkAndLinkSingle(None, anc_names)
    else:
        """
        Iterable of names provided
        """
        for name in anc_names:
            __checkAndLinkSingle(None, name)

    h5_dset.file.flush()


def createRefFromIndices(h5_main, ref_inds):
    """
    Create a region reference in the destination dataset using an iterable of pairs of indices
    representing the start and end points of a hyperslab block

    Parameters
    ----------
    h5_main : HDF5 dataset
        dataset the region will be created in
    ref_inds : Iterable
        index pairs, [start indices, final indices] for each block in the
        hyperslab

    Returns
    -------
    new_ref : HDF5 Region reference
        reference in `h5_main` for the blocks of points defined by `ref_inds`
    """
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
    """
    Reshape the input 2D matrix to be N-dimensions based on the
    position and spectroscopic datasets.

    Parameters
    ----------
        h5_main : HDF5 Dataset
            2D data to be reshaped
        h5_pos : HDF5 Dataset, optional
            Position indices corresponding to rows in `h5_main`
        h5_spec : HDF5 Dataset, optional
            Spectroscopic indices corresponding to columns in `h5_main`

    Returns
    -------
        ds_Nd : N-D numpy array
            N dimensional numpy array arranged as [positions slowest to fastest, spectroscopic slowest to fastest]
        success : boolean or string
            True if full reshape was successful

            "Positions" if it was only possible to reshape by
            the position dimensions

            False if no reshape was possible

    Notes
    -----
    If either `h5_pos` or `h5_spec` are not provided, the function will first
    attempt to find them as attributes of `h5_main`.  If that fails, it will
    generate dummy values for them.
    """

    if h5_pos is None:
        """
        Get the Position datasets from the references if possible
        """
        if isinstance(h5_main, h5py.Dataset):
            try:
                ds_pos = h5_main.file[h5_main.attrs['Position_Indices']][()]
            except KeyError:
                print('No position datasets found as attributes of {}'.format(h5_main.name))
                if len(h5_main.shape) > 1:
                    ds_pos = np.arange(h5_main.shape[0], dtype=np.uint8)
                else:
                    ds_pos = np.array(0, dtype=np.uint8)
            except:
                raise
        else:
            ds_pos = np.arange(h5_main.shape[0], dtype=np.uint8)
    elif isinstance(h5_pos, h5py.Dataset):
        """
    Position Indices dataset was provided
        """
        ds_pos = h5_pos[()]
    else:
        raise TypeError('Position Indices must be either h5py.Dataset or None')

    ##################################################

    if h5_spec is None:
        """
        Get the Spectroscopic datasets from the references if possible
        """
        if isinstance(h5_main, h5py.Dataset):
            try:
                ds_spec = h5_main.file[h5_main.attrs['Spectroscopic_Indices']][()]
            except KeyError:
                print ('No spectroscopic datasets found as attributes of {}'.format(h5_main.name))
                if len(h5_main.shape) > 1:
                    ds_spec = np.arange(h5_main.shape[1], dtype=np.uint8)
                else:
                    ds_spec = np.array(0, dtype=np.uint8)
            except:
                raise
        else:
            ds_spec = np.arange(h5_main.shape[1], dtype=np.uint8)

    elif isinstance(h5_spec, h5py.Dataset):
        """
    Spectroscopic Indices dataset was provided
        """
        ds_spec = h5_spec[()]
    else:
        raise TypeError('Spectroscopic Indices must be either h5py.Dataset or None')

    '''
    Sort the indices from fastest to slowest
    '''
    pos_sort = get_sort_order(np.transpose(ds_pos))
    spec_sort = get_sort_order(ds_spec)

    '''
    Get the size of each dimension in the sorted order
    '''
    pos_dims = get_dimensionality(np.transpose(ds_pos), pos_sort)
    spec_dims = get_dimensionality(ds_spec, spec_sort)

    ds_main = h5_main[()]

    """
    Now we reshape the dataset based on those dimensions
    We must use the spectroscopic dimensions in reverse order
    """
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

    """
    Now we transpose the axes associated with the spectroscopic dimensions
    so that they are in the same order as in the index array
    """
    swap_axes = np.append(np.argsort(pos_sort),
                          np.argsort(spec_sort)+len(pos_dims))

    ds_Nd = np.transpose(ds_Nd, swap_axes)

    return ds_Nd, True


def get_dimensionality(ds_index, index_sort=None):
    """
    Get the size of each index dimension in a specified sort order

    Parameters
    ----------
    ds_index : 2D HDF5 Dataset or numpy array
        Row matrix of indices
    index_sort : Iterable of unsigned integers (Optional)
        Order of rows sorted from fastest to slowest

    Returns
    -------
    sorted_dims : list of unsigned integers
        Dimensionality of each row in ds_index.  If index_sort is supplied, it will be in the sorted order
    """
    if index_sort is None:
        index_sort = np.arange(ds_index.shape[0])

    sorted_dims = [len(np.unique(col)) for col in np.array(ds_index[index_sort], ndmin=2)]

    return sorted_dims


def get_sort_order(ds_spec):
    """
    Find how quickly the spectroscopic values are changing in each row
    and the order of rows from fastest changing to slowest.

    Parameters
    ----------
    ds_spec : 2D HDF5 dataset or numpy array
        Rows of indices to be sorted from fastest changing to slowest

    Returns
    -------
    change_sort : List of unsigned integers
        Order of rows sorted from fastest changing to slowest
    """
    change_count = [len(np.where([row[i] != row[i - 1] for i in xrange(len(row))])[0]) for row in ds_spec]
    change_sort = np.argsort(change_count)[::-1]

    return change_sort


def copyAttributes(source, dest, skip_refs=True):
    """
    Copy attributes from one h5object to another
    """
    for attr, atval in source.attrs.iteritems():
        """
        Don't copy references unless asked
        """
        if isinstance(atval, h5py.Reference):
            if skip_refs:
                continue
            elif isinstance(atval, h5py.RegionReference):
                """
                Dereference old reference, get the appropriate data
                slice and create new reference.
                """
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
                    print('Could not create new region reference for {} in {}.'.format(attr, source.name))
                    continue

                dest.attrs[attr] = dest.regionref[tuple(ref_slice)]
                continue
            else:
                dest.attrs[attr] = source.file[atval].name
                continue
        dest.attrs[attr] = atval

    return dest


def checkIfMain(h5_main):
    """
    Checks the input dataset to see if it has all the neccessary
    features to be considered a Main dataset.  This means it is
    2D and has the following attributes
    Position_Indices
    Position_Values
    Spectroscopic_Indices
    Spectroscopic_Values

    Parameters
    ----------
    h5_main : HDF5 Dataset

    Returns
    -------
    success : Boolean
        True if all tests pass
    """
    # Check that h5_main is a dataset
    success = isinstance(h5_main, h5py.Dataset)

    if not success:
        print('{} is not an HDF5 Dataset object.'.format(h5_main))
        return success

    h5_name = h5_main.name.split('/')[-1]

    # Check dimensionality
    success = np.all([success, len(h5_main.shape) == 2])

    if not success:
        print('{} is not 2D.'.format(h5_name))
        return success

    # Check for Datasets
    dset_names = ['Position_Indices', 'Position_Values',
                  'Spectroscopic_Indices', 'Spectroscopic_Values']

    for name in dset_names:
        try:
            ds = h5_main.file[h5_main.attrs[name]]
            success = np.all([success, isinstance(ds, h5py.Dataset)])
        except:
            print('{} not found as an attribute of {}.'.format(name, h5_name))
            success = False
            break

    return success


def linkRefs(src, trg):
    """
    Creates Dataset attributes that contain references to other Dataset Objects.

    Parameters
    -----------
    src : Reference to h5.objects
        Reference to the the object to which attributes will be added
    trg : list of references to h5.objects
        objects whose references that can be accessed from src.attrs

    Returns
    --------
    None
    """
    for itm in trg:
        src.attrs[itm.name.split('/')[-1]] = itm.ref


def linkRefAsAlias(src, trg, trg_name):
    """
    Creates Dataset attributes that contain references to other Dataset Objects.
    This function is useful when the reference attribute must have a reserved name.
    Such as linking 'SHO_Indices' as 'Spectroscopic_Indices'

    Parameters
    ------------
    src : h5py.Dataset
        Reference to the the object to which attributes will be added
    trg : h5py.Dataset
        object whose reference that can be accessed from src.attrs
    trg_name : String
        Alias / alternate name for trg
    """
    src.attrs[trg_name] = trg.ref


def copyRegionRefs(h5_source, h5_target):
    """
    Check the input dataset for plot groups, copy them if they exist
    Also make references in the Spectroscopic Values and Indices tables

    Parameters
    ----------
    h5_source : HDF5 Dataset
            source dataset to copy references from
    h5_target : HDF5 Dataset
            target dataset the references from h5_source are copied to

    Returns
    -------
    None
    """
    '''
    Check both h5_source and h5_target to ensure that are Main
    '''
    if not all([checkIfMain(h5_source), checkIfMain(h5_target)]):
        raise TypeError('Inputs to copyRegionRefs must be HDF5 Datasets.')

    h5_source_inds = h5_source.file[h5_source.attrs['Spectroscopic_Indices']]

    h5_spec_inds = h5_target.file[h5_target.attrs['Spectroscopic_Indices']]
    h5_spec_vals = h5_target.file[h5_target.attrs['Spectroscopic_Values']]


    for key in h5_source.attrs.iterkeys():
        if '_Plot_Group' not in key:
            continue

        if h5_source_inds.shape[0] == h5_spec_inds.shape[0]:
            '''
            Spectroscopic dimensions are identical.
            Do direct copy.
            '''
            ref_inds = simpleRefCopy(h5_source, h5_target, key)

        else:
            '''
        Spectroscopic dimensions are different.
        Do the dimenion reducing copy.
            '''
            ref_inds = reducingRefCopy(h5_source, h5_target, h5_source_inds, h5_spec_inds, key)
        '''
        Create references for Spectroscopic Indices and Values
        Set the end-point of each hyperslab in the position dimension to the number of
        rows in the index array
        '''
        ref_inds[:, 1, 0][ref_inds[:, 1, 0] > h5_spec_inds.shape[0]] = h5_spec_inds.shape[0] - 1
        spec_inds_ref = createRefFromIndices(h5_spec_inds, ref_inds)
        h5_spec_inds.attrs[key] = spec_inds_ref
        spec_vals_ref = createRefFromIndices(h5_spec_vals, ref_inds)
        h5_spec_vals.attrs[key] = spec_vals_ref


def reducingRefCopy(h5_source, h5_target, h5_source_inds, h5_target_inds, key):
    """
    Copies a region reference from one dataset to another taking into account that a dimension
    has been lost from source to target

    Parameter
    ---------
    h5_source : HDF5 Dataset
            source dataset for region reference copy
    h5_target : HDF5 Dataset
            target dataset for region reference copy
    h5_source_inds : HDF5 Dataset
            indices of each dimension of the h5_source dataset
    h5_target_inds : HDF5 Dataset
            indices of each dimension of the h5_target dataset
    key : String
            Name of attribute in h5_source that contains
            the Region Reference to copy
    Return
    ------
    ref_inds : Nx2x2 array of unsigned integers
            Array containing pairs of points that define
            the corners of each hyperslab in the region
            reference
    """

    '''
    Determine which dimension is missing from the target
    '''
    lost_dim = []
    for dim in h5_source_inds.attrs['labels']:
        if dim not in h5_target_inds.attrs['labels']:
            lost_dim.append(np.where(h5_source_inds.attrs['labels'] == dim)[0])
    ref = h5_source.attrs[key]
    ref_inds = getH5RegRefIndices(ref, h5_source, return_method='corners')
    '''
    Convert to proper spectroscopic dimensions
    First is special case for a region reference that spans the entire dataset
    '''
    if len(ref_inds.shape) == 2 and all(ref_inds[0] == [0, 0]) and all(ref_inds[1] + 1 == h5_source.shape):
        ref_inds[1, 1] = h5_target.shape[1] - 1
        ref_inds = np.expand_dims(ref_inds, 0)
    else:
        '''
    More common case of reference made of hyperslabs
        '''
        spec_ind_zeroes = np.where(h5_source_inds[lost_dim] == 0)[1]

        ref_inds = ref_inds.reshape([-1, 2, 2])

        for start, stop in ref_inds[:-1]:
            start[1] = np.where(start[1] == spec_ind_zeroes)[0]
            stop[1] = np.where(stop[1] == spec_ind_zeroes - 1)[0] - 1

        ref_inds[-1, 0, 1] = np.where(ref_inds[-1, 0, 1] == spec_ind_zeroes)[0]
        stop = np.where(ref_inds[-1, 1, 1] == spec_ind_zeroes - 1)[0]
        if stop.size == 0:
            stop = len(spec_ind_zeroes)
        ref_inds[-1, 1, 1] = stop - 1
    '''
    Create the new reference from the indices
    '''
    h5_target.attrs[key] = createRefFromIndices(h5_target, ref_inds)

    return ref_inds


def simpleRefCopy(h5_source, h5_target, key):
    """
    Copies a region reference from one dataset to another
    without alteration

    Parameter
    ---------
    h5_source : HDF5 Dataset
            source dataset for region reference copy
    h5_target : HDF5 Dataset
            target dataset for region reference copy
    key : String
            Name of attribute in h5_source that contains
            the Region Reference to copy
    Return
    ------
    ref_inds : Nx2x2 array of unsigned integers
            Array containing pairs of points that define
            the corners of each hyperslab in the region
            reference
    """

    ref = h5_source.attrs[key]
    ref_inds = getH5RegRefIndices(ref, h5_source, return_method='corners')
    ref_inds = ref_inds.reshape([-1, 2, 2])
    ref_inds[:, 1, 1] = h5_target.shape[1] - 1
    target_ref = createRefFromIndices(h5_target, ref_inds)
    h5_target.attrs[key] = target_ref
    return ref_inds


def buildReducedSpec(h5_spec_inds, h5_spec_vals, keep_dim, step_starts, basename='Spectroscopic'):
    """
    Creates new Spectroscopic Indices and Values datasets from the input datasets
    and keeps the dimensions specified in not_freq

    Parameters
    ----------
    h5_spec_inds : HDF5 Dataset
            Spectroscopic indices dataset
    h5_spec_vals : HDF5 Dataset
            Spectroscopic values dataset
    keep_dim : Numpy Array, Boolean
            Array designating which rows of the input spectroscopic datasets to keep
    step_starts : Numpy Array, Unsigned Integers
            Array specifying the start of each step in the reduced datasets
    basename : String
            String to which '_Indices' and '_Values' will be appended to get the names
            of the new datasets

    Returns
    -------
    ds_inds : MicroDataset
            Reduced Spectroscopic indices dataset
    ds_vals : MicroDataset
            Reduces Spectroscopic values dataset
    """
    '''
    Extract all rows that we want to keep from input indices and values
    '''
    ind_mat = h5_spec_inds[keep_dim, :][:, step_starts]
    val_mat = h5_spec_vals[keep_dim, :][:, step_starts]
    '''
    Create new MicroDatasets to hold the data
    Name them based on basename
    '''
    ds_inds = MicroDataset(basename+'_Indices', ind_mat, dtype=h5_spec_inds.dtype)
    ds_vals = MicroDataset(basename+'_Values', val_mat, dtype=h5_spec_vals.dtype)
    # Extracting the labels from the original spectroscopic data sets
    sho_inds_labs = h5_spec_inds.attrs['labels'][keep_dim]
    # Creating the dimension slices for the new spectroscopic data sets
    inds_slices = dict()
    for row_ind, row_name in enumerate(sho_inds_labs):
        inds_slices[row_name] = (slice(row_ind, row_ind + 1), slice(None))

    # Adding the labels and units to the new spectroscopic data sets
    ds_inds.attrs['labels'] = inds_slices
    ds_inds.attrs['units'] = h5_spec_inds.attrs['units'][keep_dim]
    ds_vals.attrs['labels'] = inds_slices
    ds_vals.attrs['units'] = h5_spec_vals.attrs['units'][keep_dim]

    return ds_inds, ds_vals


def calc_chunks(dimensions, data_size, unit_chunks=None, max_chunk_mem=10240):
    """
    Calculate the chunk size for the HDF5 dataset based on the dimensions and the
    maximum chunk size in memory

    Parameters
    ----------
    dimensions : array_like of int
        Shape of the data to be chunked
    data_size : int
        Size of an entry in the data in bytes
    unit_chunks : array_like of int, optional
        Unit size of the chunking in each dimension.  Must be the same size as
        the shape of `ds_main`.  Default None, `unit_chunks` is set to 1 in all
        dimensions
    max_chunk_mem : int, optional
        Maximum size of the chunk in memory in bytes.  Default 10240b or 10kb

    Returns
    -------
    chunking : tuple of int
        Calculated maximum size of a chunk in each dimension that is as close to the
        requested `max_chunk_mem` as posible while having steps based on the input
        `unit_chunks`.
    """
    '''
    Ensure that dimensions is an array
    '''
    dimensions = np.asarray(dimensions, dtype=np.uint)
    '''
    Set the unit_chunks to all ones if not given.  Ensure it is an array if it is.
    '''
    if unit_chunks is None:
        unit_chunks = np.ones_like(dimensions)
    else:
        unit_chunks = np.asarray(unit_chunks, dtype=np.uint)

    if unit_chunks.shape != dimensions.shape:
        raise ValueError('Unit chunk size must have the same shape as the input dataset.')

    '''
    Save the original size of unit_chunks to use for incrementing the chunk size during
     loop
    '''
    base_chunks = unit_chunks

    '''
    Loop until chunk_size is greater than the maximum chunk_mem or the chunk_size is equal to
    that of dimensions
    '''
    while np.prod(unit_chunks)*data_size <= max_chunk_mem:
        '''
        Check if all chunk dimensions are greater or equal to the
        actual dimensions.  Exit the loop if true.
        '''
        if np.all(unit_chunks >= dimensions):
            break

        '''
        Find the index of the next chunk to be increased and increment it by the base_chunk
        size
        '''
        ichunk = np.argmax(dimensions/unit_chunks)
        unit_chunks[ichunk] += base_chunks[ichunk]

    '''
    Ensure that the size of the chunks is between one and the dimension size.
    '''
    unit_chunks = np.clip(unit_chunks, np.ones_like(unit_chunks), dimensions)

    chunking = tuple(unit_chunks)

    return chunking