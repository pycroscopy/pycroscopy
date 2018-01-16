# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 21:14:25 2015

@author: Suhas Somnath, Chris Smith, Numan Laanait
"""

from __future__ import division, print_function, absolute_import, unicode_literals
import sys
import h5py
import collections
from warnings import warn
import numpy as np
from .microdata import MicroDataset

__all__ = ['get_attr', 'getDataSet', 'getH5DsetRefs', 'getH5RegRefIndices', 'get_dimensionality', 'get_sort_order',
           'getAuxData', 'get_attributes', 'getH5GroupRefs', 'checkIfMain', 'checkAndLinkAncillary', 'copyRegionRefs',
           'createRefFromIndices', 'copyAttributes', 'reshape_to_Ndims', 'linkRefs', 'linkRefAsAlias',
           'findH5group', 'get_formatted_labels', 'reshape_from_Ndims', 'findDataset', 'print_tree', 'get_all_main',
           'copy_main_attributes', 'create_empty_dataset', 'calc_chunks', 'create_spec_inds_from_vals',
           'buildReducedSpec', 'check_for_old', 'get_source_dataset', 'get_unit_values', 'get_data_descriptor',
           'link_as_main', 'reducingRefCopy', 'simpleRefCopy']

if sys.version_info.major == 3:
    unicode = str


def print_tree(parent):
    """
    Simple function to recursively print the contents of an hdf5 group
    Parameters
    ----------
    parent : h5py.Group

    Returns
    -------
    None
    
    """

    def __print(name, obj):
        print(name)

    print(parent.name)
    parent.visititems(__print)


def get_all_main(parent, verbose=False):
    """
    Simple function to recursively print the contents of an hdf5 group

    Parameters
    ----------
    parent : h5py.Group
        HDF5 Group to search within
    verbose : bool
        If true, extra print statements are enabled

    Returns
    -------
    main_list : list of h5py.Dataset
        The datasets found in the file that meet the 'Main Data' criteria.

    """
    from .pycro_data import PycroDataset

    main_list = list()

    def __check(name, obj):
        if verbose:
            print(name, obj)
        if isinstance(obj, h5py.Dataset):
            if verbose:
                print(name, 'is an HDF5 Dataset.')
            ismain = checkIfMain(obj)
            if ismain:
                if verbose:
                    print(name, 'is a `Main` dataset.')
                main_list.append(PycroDataset(obj))

    if verbose:
        print('Checking the group {} for `Main` datasets.'.format(parent.name))
    parent.visititems(__check)

    return main_list


def getDataSet(h5_parent, data_name):
    """
    Search for dataset objects in the hdf5 file with given name
    and returns a list of reference(s).

    Parameters
    ----------
    h5_parent : h5py.File reference.
        Reference to file, the file must be open and in read-mode.
    data_name : string.
        Name of Dataset object. If not unique, i.e. parent not specified,
        then references to all Dataset objects that contain this name are returned.

    Returns
    -------
    list of h5py.Reference of the dataset.
    """
    from .pycro_data import PycroDataset

    if isinstance(h5_parent, h5py.File) or isinstance(h5_parent, h5py.Group):
        data_list = []

        def findData(name, obj):
            if name.endswith(data_name) and isinstance(obj, h5py.Dataset):
                try:
                    data_list.append(PycroDataset(obj))
                except TypeError:
                    data_list.append(obj)
                except:
                    raise

        h5_parent.visititems(findData)
        return data_list
    else:
        print('%s is not an hdf5 File or Group' % h5_parent)


def getAuxData(parent_data, auxDataName=None):
    """
    Returns auxiliary dataset objects associated with some DataSet through its attributes.

    Parameters
    ----------
    parent_data : h5py.Dataset
        Dataset object reference.
    auxDataName : list of strings, optional, default = all (DataSet.attrs).
        Name of auxiliary Dataset object to return.

    Returns
    -------
    list of h5py.Reference of auxiliary dataset objects.
    """
    if auxDataName is None:
        auxDataName = parent_data.attrs.keys()
    elif type(auxDataName) not in [list, tuple, set]:
        auxDataName = [auxDataName]  # typically a single string
    data_list = list()
    try:
        file_ref = parent_data.file
        for auxName in auxDataName:
            ref = parent_data.attrs[auxName]
            if isinstance(ref, h5py.Reference) and isinstance(file_ref[ref], h5py.Dataset):
                data_list.append(file_ref[ref])
    except KeyError:
        warn('%s is not an attribute of %s'
             % (str(auxName), parent_data.name))
    except:
        raise

    return data_list


def get_attr(h5_object, attr_name):
    """
    Returns the attribute from the h5py object

    Parameters
    ----------
    h5_object : h5py object
        dataset or datagroup object
    attr_name : str
        Name of the attribute of interest

    Returns
    -------
    att_val : object
        value of attribute, in certain cases (byte strings or list of byte strings) reformatted to readily usable forms
    """
    att_val = h5_object.attrs.get(attr_name)
    if att_val is None:
        raise KeyError("'{}' is not an attribute in '{}'".format(attr_name, h5_object.name))

    if isinstance(att_val, np.bytes_) or isinstance(att_val, bytes):
        att_val = att_val.decode('utf-8')

    elif type(att_val) == np.ndarray:
        if sys.version_info.major == 3: 
            if att_val.dtype.type in [np.bytes_, np.object_]:
                att_val = np.array([str(x, 'utf-8') for x in att_val])
            
    return att_val


def get_attributes(parent_data, attr_names=None):
    """
    Returns attribute associated with some DataSet.

    Parameters
    ----------
    parent_data : h5py.Dataset
        Dataset object reference.
    attr_names : string or list of strings, optional, default = all (DataSet.attrs).
        Name of attribute object to return.

    Returns
    -------
    Dictionary containing (name,value) pairs of attributes
    """
    if attr_names is None:
        attr_names = parent_data.attrs.keys()

    if type(attr_names) == str:
        attr_names = [attr_names]

    att_dict = {}

    for attr in attr_names:
        try:
            att_dict[attr] = get_attr(parent_data, attr)
        except KeyError:
            warn('%s is not an attribute of %s'
                 % (str(attr), parent_data.name))
        except:
            raise

    return att_dict


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
    from .pycro_data import PycroDataset
    aux_dset = []
    for ds_name in ds_names:
        for dset in h5_refs:
            if dset.name.split('/')[-1] == ds_name:
                try:
                    aux_dset.append(PycroDataset(dset))
                except TypeError:
                    aux_dset.append(dset)
                except:
                    raise
    return aux_dset


def getH5GroupRefs(group_name, h5_refs):
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
    group_list = list()
    for item in h5_refs:
        if item.name.split('/')[-1].startswith(group_name):
            group_list.append(item)
    return group_list


def findDataset(h5_group, ds_name):
    """
    Uses visit() to find all datasets with the desired name

    Parameters
    ----------
    h5_group : h5py.Group
        Group to search within for the Dataset
    ds_name : str
        Name of the dataset to search for

    Returns
    -------
    ds : list
        List of [Name, object] pairs corresponding to datasets that match `ds_name`.

    """
    from .pycro_data import PycroDataset

    # print 'Finding all instances of', ds_name
    ds = []

    def __find_name(name, obj):
        if ds_name in name.split('/')[-1] and isinstance(obj, h5py.Dataset):
            try:
                ds.append([name, PycroDataset(obj)])
            except TypeError:
                ds.append([name, obj])
            except:
                raise
        return

    h5_group.visititems(__find_name)

    return ds


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
        def __corners_to_point_array(start, stop):
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
            for i in range(len(start)):
                if start[i] == stop[i]:
                    ranges.append([stop[i]])
                else:
                    ranges.append(np.arange(start[i], stop[i] + 1, dtype=np.uint))
            grid = np.meshgrid(*ranges, indexing='ij')

            ref_inds = np.asarray(zip(*(x.flat for x in grid)))

            return ref_inds

        return_func = __corners_to_point_array
    elif return_method == 'corners':
        def __corners_to_corners(start, stop):
            return start, stop

        return_func = __corners_to_corners
    elif return_method == 'slices':
        def __corners_to_slices(start, stop):
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
            for idim in range(len(start)):
                slices.append(slice(start[idim], stop[idim]))

            return slices

        return_func = __corners_to_slices

    if isinstance(ref, h5py.RegionReference):
        region = h5py.h5r.get_region(ref, h5_main.id)
        reg_type = region.get_select_type()
        if reg_type == 2:
            """
            Reference is hyperslabs
            """
            ref_inds = []
            for start, end in region.get_select_hyper_blocklist():
                ref_inds.append(return_func(start, end))
            ref_inds = np.array(ref_inds).reshape(-1, len(start))

        elif reg_type == 3:
            """
            Reference is single block
            """
            start, end = region.get_select_bounds()

            ref_inds = return_func(start, end)
        else:
            warn('No method currently exists for converting this type of reference.')
            ref_inds = np.empty(0)
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

    def __check_and_link_single(h5_ref, ref_name):
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
            __check_and_link_single(h5_ref, ref_name)
    elif anc_refs is not None:
        """
        anc_refs is just a single value
        """
        __check_and_link_single(anc_refs, anc_names)
    elif isinstance(anc_names, str) or isinstance(anc_names, unicode):
        """
        Single name provided
        """
        __check_and_link_single(None, anc_names)
    else:
        """
        Iterable of names provided
        """
        for name in anc_names:
            __check_and_link_single(None, name)

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


def get_data_descriptor(h5_dset):
    """
    Returns a string of the form 'quantity (unit)'

    Parameters
    ----------
    h5_dset : h5py.Dataset object
        A 'main' dataset in pycroscopy

    Returns
    -------
    descriptor : String
        string of the form 'quantity (unit)'
    """
    try:
        quant = get_attr(h5_dset, 'quantity')
    except KeyError:
        quant = 'Unknown quantity'
    try:
        unit = get_attr(h5_dset, 'units')
    except KeyError:
        unit = 'unknown units'
    return '{} ({})'.format(quant, unit)


def get_formatted_labels(h5_dset):
    """
    Takes any dataset which has the labels and units attributes and returns a list of strings
    formatted as 'label k (unit k)'

    Parameters
    ----------
    h5_dset : h5py.Dataset object
        dataset which has labels and units attributes

    Returns
    -------
    labels : list
        list of strings formatted as 'label k (unit k)'
    """
    try:
        labs = get_attr(h5_dset, 'labels')
        try:
            units = get_attr(h5_dset, 'units')
        except KeyError:
            warn('units attribute was missing')
            units = ['' for _ in labs]

        if len(labs) != len(units):
            warn('Labels and units have different sizes!')
            return None
        labels = []
        for lab, unit in zip(labs, units):
            labels.append('{} ({})'.format(lab, unit))
        return labels
    except KeyError:
        warn('labels attribute was missing')
        return None


def reshape_to_Ndims(h5_main, h5_pos=None, h5_spec=None, get_labels=False, verbose=False, sort_dims=False):
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
    get_labels : bool, optional
        Whether or not to return the dimension labels.  Default False
    verbose : bool, optional
        Whether or not to print debugging statements
    sort_dims : bool
        If True, the data is sorted so that the dimensions are in order from fastest to slowest
        If False, the data is kept in the original order
        If `get_labels` is also True, the labels are sorted as well.

    Returns
    -------
    ds_Nd : N-D numpy array
        N dimensional numpy array arranged as [positions slowest to fastest, spectroscopic slowest to fastest]
    success : boolean or string
        True if full reshape was successful

        "Positions" if it was only possible to reshape by
        the position dimensions

        False if no reshape was possible
    ds_labels : list of str
        List of the labels of each dimension of `ds_Nd`

    Notes
    -----
    If either `h5_pos` or `h5_spec` are not provided, the function will first
    attempt to find them as attributes of `h5_main`.  If that fails, it will
    generate dummy values for them.

    """
    pos_labs = np.array(['Positions'])
    spec_labs = np.array(['Spectral_Step'])
    if h5_pos is None:
        """
        Get the Position datasets from the references if possible
        """
        if isinstance(h5_main, h5py.Dataset):
            try:
                h5_pos = h5_main.file[h5_main.attrs['Position_Indices']]
                ds_pos = h5_pos[()]
                pos_labs = get_attr(h5_pos, 'labels')
            except KeyError:
                print('No position datasets found as attributes of {}'.format(h5_main.name))
                if len(h5_main.shape) > 1:
                    ds_pos = np.arange(h5_main.shape[0], dtype=np.uint8).reshape(-1, 1)
                    pos_labs = np.array(['Position Dimension {}'.format(ipos) for ipos in range(ds_pos.shape[1])])
                else:
                    ds_pos = np.array(0, dtype=np.uint8).reshape(-1, 1)
            except:
                raise
        else:
            ds_pos = np.arange(h5_main.shape[0], dtype=np.uint32).reshape(-1, 1)
            pos_labs = np.array(['Position Dimension {}'.format(ipos) for ipos in range(ds_pos.shape[1])])
    elif isinstance(h5_pos, h5py.Dataset):
        """
    Position Indices dataset was provided
        """
        ds_pos = h5_pos[()]
        pos_labs = get_attr(h5_pos, 'labels')
    elif isinstance(h5_pos, np.ndarray):
        ds_pos = np.atleast_2d(h5_pos)
        pos_labs = np.array(['Position Dimension {}'.format(ipos) for ipos in range(ds_pos.shape[1])])
    else:
        raise TypeError('Position Indices must be either h5py.Dataset or None')

    ##################################################

    if h5_spec is None:
        """
        Get the Spectroscopic datasets from the references if possible
        """
        if isinstance(h5_main, h5py.Dataset):
            try:
                h5_spec = h5_main.file[h5_main.attrs['Spectroscopic_Indices']]
                ds_spec = h5_spec[()]
                spec_labs = get_attr(h5_spec, 'labels')
            except KeyError:
                print('No spectroscopic datasets found as attributes of {}'.format(h5_main.name))
                if len(h5_main.shape) > 1:
                    ds_spec = np.arange(h5_main.shape[1], dtype=np.uint8).reshape([1, -1])
                    spec_labs = np.array(['Spectral Dimension {}'.format(ispec) for ispec in range(ds_spec.shape[0])])
                else:
                    ds_spec = np.array(0, dtype=np.uint8).reshape([1, 1])
            except:
                raise
        else:
            ds_spec = np.arange(h5_main.shape[1], dtype=np.uint8).reshape([1, -1])
            spec_labs = np.array(['Spectral Dimension {}'.format(ispec) for ispec in range(ds_spec.shape[0])])

    elif isinstance(h5_spec, h5py.Dataset):
        """
    Spectroscopic Indices dataset was provided
        """
        ds_spec = h5_spec[()]
        spec_labs = get_attr(h5_spec, 'labels')
    elif isinstance(h5_spec, np.ndarray):
        ds_spec = h5_spec
        spec_labs = np.array(['Spectral Dimension {}'.format(ispec) for ispec in range(ds_spec.shape[0])])
    else:
        raise TypeError('Spectroscopic Indices must be either h5py.Dataset or None')

    '''
    Sort the indices from fastest to slowest
    '''
    pos_sort = get_sort_order(np.transpose(ds_pos))
    spec_sort = get_sort_order(ds_spec)

    if verbose:
        print('Position dimensions:', pos_labs)
        print('Position sort order:', pos_sort)
        print('Spectroscopic Dimensions:', spec_labs)
        print('Spectroscopic sort order:', spec_sort)

    '''
    Get the size of each dimension in the sorted order
    '''
    pos_dims = get_dimensionality(np.transpose(ds_pos), pos_sort)
    spec_dims = get_dimensionality(ds_spec, spec_sort)

    if verbose:
        print('\nPosition dimensions (sort applied):', pos_labs[pos_sort])
        print('Position dimensionality (sort applied):', pos_dims)
        print('Spectroscopic dimensions (sort applied):', spec_labs[spec_sort])
        print('Spectroscopic dimensionality (sort applied):', spec_dims)

    ds_main = h5_main[()]

    """
    Now we reshape the dataset based on those dimensions
    numpy reshapes correctly when the dimensions are arranged from slowest to fastest. 
    Since the sort orders we have are from fastest to slowest, we need to reverse the orders
    for both the position and spectroscopic dimensions
    """
    try:
        ds_Nd = np.reshape(ds_main, pos_dims[::-1] + spec_dims[::-1])

    except ValueError:
        warn('Could not reshape dataset to full N-dimensional form.  Attempting reshape based on position only.')
        try:
            ds_Nd = np.reshape(ds_main, pos_dims[::-1] + [-1])

        except ValueError:
            warn('Reshape by position only also failed.  Will keep dataset in 2d form.')
            if get_labels:
                return ds_main, False, ['Position', 'Spectral Step']
            else:
                return ds_main, False

        # No exception
        else:
            if get_labels:
                return ds_Nd, 'Positions', ['Position'] + spec_labs
            else:
                return ds_Nd, 'Positions'

    all_labels = np.hstack((pos_labs[pos_sort][::-1],
                            spec_labs[spec_sort][::-1]))

    if verbose:
        print('\nAfter first reshape, labels are', all_labels)
        print('Data shape is', ds_Nd.shape)

    """
    Now we transpose the axes for both the position and spectroscopic dimensions
    so that they are in the same order as in the index array
    """
    if not sort_dims:
        swap_axes = np.append(pos_sort.size - 1 - pos_sort,
                              spec_sort.size - spec_sort - 1 + len(pos_dims))
    else:
        swap_axes = np.append(pos_sort[::-1], spec_sort[::-1] + len(pos_dims))

    if verbose:
        print('\nAxes will permuted in this order:', swap_axes)
        print('New labels ordering:', all_labels[swap_axes])

    ds_Nd = np.transpose(ds_Nd, swap_axes)

    results = [ds_Nd, True]

    if verbose:
        print('Dataset now of shape:', ds_Nd.shape)

    if get_labels:
        '''
        Get the labels in the proper order
        '''
        if isinstance(h5_pos, h5py.Dataset):
            pos_labs = get_attr(h5_pos, 'labels')
        else:
            pos_labs = np.array(['' for _ in pos_dims])
        if isinstance(h5_spec, h5py.Dataset):
            spec_labs = get_attr(h5_spec, 'labels')
        else:
            spec_labs = np.array(['' for _ in spec_dims])

        ds_labels = np.hstack([pos_labs[pos_sort[::-1]], spec_labs[spec_sort[::-1]]])

        results.append(ds_labels[swap_axes])

    return results


def reshape_from_Ndims(ds_Nd, h5_pos=None, h5_spec=None):
    """
    Reshape the input 2D matrix to be N-dimensions based on the
    position and spectroscopic datasets.

    Parameters
    ----------
    ds_Nd : numpy.array
        N dimensional numpy array arranged as [positions slowest to fastest, spectroscopic slowest to fastest]
    h5_pos : HDF5 Dataset
        Position indices corresponding to rows in the final 2d array
    h5_spec : HDF5 Dataset
        Spectroscopic indices corresponding to columns in the final 2d array

    Returns
    -------
    ds_2d : numpy.array
        2 dimensional numpy array arranged as [positions, spectroscopic]
    success : boolean or string
        True if full reshape was successful

        "Positions" if it was only possible to reshape by
        the position dimensions

        False if no reshape was possible

    Notes
    -----
    If either `h5_pos` or `h5_spec` are not provided, the function will
    assume the first dimension is position and the remaining are spectroscopic already
    in order from fastest to slowest.

    """

    if h5_pos is None:
        '''
    Get the Position datasets from the references if possible
        '''
        ds_pos = np.arange(ds_Nd.shape[0], dtype=np.uint8).reshape(-1, 1)
    elif isinstance(h5_pos, h5py.Dataset):
        '''
    Position Indices dataset was provided
        '''
        ds_pos = h5_pos[()]

    elif isinstance(h5_pos, np.ndarray):
        ds_pos = h5_pos
    else:
        raise TypeError('Position Indices must be either h5py.Dataset or None')

    ##################################################

    if h5_spec is None:
        '''
    Get the Spectroscopic datasets from the references if possible
        '''
        ds_spec = np.atleast_2d(np.arange(ds_Nd.shape[1], dtype=np.uint8))

    elif isinstance(h5_spec, h5py.Dataset):
        '''
    Spectroscopic Indices dataset was provided
        '''
        ds_spec = h5_spec[()]

    elif isinstance(h5_spec, np.ndarray):
        ds_spec = h5_spec
    else:
        raise TypeError('Spectroscopic Indices must be either h5py.Dataset or None')

    '''
    Sort the indices from fastest to slowest
    '''
    pos_sort = get_sort_order(np.transpose(ds_pos))
    spec_sort = get_sort_order(ds_spec)

    '''
    Now we transpose the axes associated with the spectroscopic dimensions
    so that they are in the same order as in the index array
    '''
    swap_axes = np.append(np.argsort(pos_sort),
                          spec_sort[::-1] + len(pos_sort))

    ds_Nd = np.transpose(ds_Nd, swap_axes)

    '''
    Now we reshape the dataset based on those dimensions
    We must use the spectroscopic dimensions in reverse order
    '''
    try:
        ds_2d = np.reshape(ds_Nd, [ds_pos.shape[0], ds_spec.shape[1]])
    except ValueError:
        warn('Could not reshape dataset to full N-dimensional form.  Attempting reshape based on position only.')
        raise
    except:
        raise

    return ds_2d, True


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

    sorted_dims = [len(np.unique(col)) for col in np.array(ds_index, ndmin=2)[index_sort]]

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
    change_count = [len(np.where([row[i] != row[i - 1] for i in range(len(row))])[0]) for row in ds_spec]
    change_sort = np.argsort(change_count)[::-1]

    return change_sort


def create_empty_dataset(source_dset, dtype, dset_name, new_attrs=dict(), skip_refs=False):
    """
    Creates an empty dataset in the h5 file based in the same group as the provided dataset

    Parameters
    ----------
    source_dset : h5py.Dataset object
        Source object that provides information on the group and shape of the dataset
    dtype : dtype
        Data type of the fit / guess datasets
    dset_name : String / Unicode
        Name of the dataset
    new_attrs : dictionary (Optional)
        Any new attributes that need to be written to the dataset
    skip_refs : boolean, optional
        Should ObjectReferences and RegionReferences be skipped when copying attributes from the
        `source_dset`

    Returns
    -------
    h5_new_dset : h5py.Dataset object
        Newly created dataset
    """
    h5_group = source_dset.parent
    try:
        # Check if the dataset already exists
        h5_new_dset = h5_group[dset_name]
        # Make sure it has the correct shape and dtype
        if any((source_dset.shape != h5_new_dset.shape, source_dset.dtype != h5_new_dset.dtype)):
            del h5_new_dset, h5_group[dset_name]
            h5_new_dset = h5_group.create_dataset(dset_name, shape=source_dset.shape, dtype=dtype,
                                                  compression=source_dset.compression, chunks=source_dset.chunks)

    except KeyError:
        h5_new_dset = h5_group.create_dataset(dset_name, shape=source_dset.shape, dtype=dtype,
                                              compression=source_dset.compression, chunks=source_dset.chunks)

    except:
        raise
    # This should link the ancillary datasets correctly
    h5_new_dset = copyAttributes(source_dset, h5_new_dset, skip_refs=skip_refs)
    h5_new_dset.attrs.update(new_attrs)

    return h5_new_dset


def copyAttributes(source, dest, skip_refs=True):
    """
    Copy attributes from one h5object to another
    """
    for attr in source.attrs.keys():
        atval = source.attrs[attr]
        """
        Don't copy references unless asked
        """
        if isinstance(atval, h5py.Reference):
            if isinstance(atval, h5py.RegionReference) or skip_refs:
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
                    for i in range(len(start)):
                        if start[i] == end[i]:
                            ref_slice.append(start[i])
                        else:
                            ref_slice.append(slice(start[i], end[i]))
                except:
                    warn('Could not create new region reference for {} in {}.'.format(attr, source.name))
                    continue

                dest.attrs[attr] = dest.regionref[tuple(ref_slice)]
                continue
            else:
                dest.attrs[attr] = atval
                continue
        dest.attrs[attr] = atval
    if not skip_refs:
        try:
            copyRegionRefs(source, dest)
        except:
            print('Could not create new region reference for {} in {}.'.format(attr, source.name))

    return dest


def checkIfMain(h5_main, verbose=False):
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
        Dataset of interest
    verbose : Boolean (Optional. Default = False)
        Whether or not to print statements

    Returns
    -------
    success : Boolean
        True if all tests pass
    """
    # Check that h5_main is a dataset
    success = isinstance(h5_main, h5py.Dataset)

    if not success:
        if verbose:
            print('{} is not an HDF5 Dataset object.'.format(h5_main))
        return success

    h5_name = h5_main.name.split('/')[-1]

    # Check dimensionality
    success = np.all([success, len(h5_main.shape) == 2])

    if not success:
        if verbose:
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
            if verbose:
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

    for key in h5_source.attrs.keys():
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

    Parameters
    ----------
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

    Returns
    -------
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

    Parameters
    ----------
    h5_source : HDF5 Dataset
            source dataset for region reference copy
    h5_target : HDF5 Dataset
            target dataset for region reference copy
    key : String
            Name of attribute in h5_source that contains
            the Region Reference to copy

    Returns
    -------
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
    if h5_spec_inds.shape[0] > 1:
        '''
        Extract all rows that we want to keep from input indices and values
        '''
        ind_mat = h5_spec_inds[keep_dim, :][:, step_starts]
        val_mat = h5_spec_vals[keep_dim, :][:, step_starts]
        '''
        Create new MicroDatasets to hold the data
        Name them based on basename
        '''
        ds_inds = MicroDataset(basename + '_Indices', ind_mat, dtype=h5_spec_inds.dtype)
        ds_vals = MicroDataset(basename + '_Values', val_mat, dtype=h5_spec_vals.dtype)
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

    else:  # Single spectroscopic dimension:
        ds_inds = MicroDataset('Spectroscopic_Indices', np.array([[0]], dtype=np.uint32))
        ds_vals = MicroDataset('Spectroscopic_Values', np.array([[0]], dtype=np.float32))

        ds_inds.attrs['labels'] = {'Single_Step': (slice(0, None), slice(None))}
        ds_vals.attrs['labels'] = {'Single_Step': (slice(0, None), slice(None))}
        ds_inds.attrs['units'] = ''
        ds_vals.attrs['units'] = ''

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
    while np.prod(unit_chunks) * data_size <= max_chunk_mem:
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
        ichunk = np.argmax(dimensions / unit_chunks)
        unit_chunks[ichunk] += base_chunks[ichunk]

    '''
    Ensure that the size of the chunks is between one and the dimension size.
    '''
    unit_chunks = np.clip(unit_chunks, np.ones_like(unit_chunks), dimensions)

    chunking = tuple(unit_chunks)

    return chunking


def link_as_main(h5_main, h5_pos_inds, h5_pos_vals, h5_spec_inds, h5_spec_vals, anc_dsets=[]):
    """
    Links the object references to the four position and spectrosocpic datasets as
    attributes of `h5_main`

    Parameters
    ----------
    h5_main : h5py.Dataset
        2D Dataset which will have the references added as attributes
    h5_pos_inds : h5py.Dataset
        Dataset that will be linked with the name 'Position_Indices'
    h5_pos_vals : h5py.Dataset
        Dataset that will be linked with the name 'Position_Values'
    h5_spec_inds : h5py.Dataset
        Dataset that will be linked with the name 'Spectroscopic_Indices'
    h5_spec_vals : h5py.Dataset
        Dataset that will be linked with the name 'Spectroscopic_Values'
    anc_dsets : (Optional) list of h5py.Dataset objects
        Datasets that will be linked with their own names
    """
    linkRefAsAlias(h5_main, h5_pos_inds, 'Position_Indices')
    linkRefAsAlias(h5_main, h5_pos_vals, 'Position_Values')
    linkRefAsAlias(h5_main, h5_spec_inds, 'Spectroscopic_Indices')
    linkRefAsAlias(h5_main, h5_spec_vals, 'Spectroscopic_Values')
    for dset in anc_dsets:
        linkRefs(h5_main, dset)


def copy_main_attributes(h5_main, h5_new):
    """
    Copies the units and quantity name from one dataset to another

    Parameters
    ----------
    h5_main : h5py.Dataset
        Dataset containing the target attributes
    h5_new : h5py.Dataset
        Dataset to which the target attributes are to be copied
    """
    for att_name, default_val in zip(['quantity', 'units'], ['Unknown', '']):
        val = default_val
        if att_name in h5_main.attrs:
            val = h5_main.attrs[att_name]
        h5_new.attrs[att_name] = val


def check_for_old(h5_base, tool_name, new_parms=dict(), verbose=False):
    """
    Check to see if the results of a tool already exist and if they 
    were performed with the same parameters.
    
    Parameters
    ----------
    h5_base : h5py.Dataset object
           Dataset on which the tool is being applied to
    tool_name : str
           process or analysis name
    new_parms : dict, optional
           Parameters with which this tool will be performed.
    verbose : bool, optional, default = False
           Whether or not to print debugging statements 
           
    Returns
    -------
    group : h5py.Group or None
           Group with parameters matching those in `new_parms`
    """

    groups = findH5group(h5_base, tool_name)

    for group in groups:
        if verbose:
            print('Looking at group - {}'.format(group.name.split('/')[-1]))

        tests = []
        for key in new_parms.keys():
            
            # HDF5 cannot store None as an attribute anyway. ignore
            if new_parms[key] is None:
                continue
                
            try:
                old_value = get_attr(group, key)
            except KeyError:
                # if parameter was not found assume that something has changed
                if verbose:
                    print('New parm: {} \t- new parm not in group *****'.format(key))
                tests.append(False)
                break
                
            if isinstance(old_value, np.ndarray):
                if not isinstance(new_parms[key], collections.Iterable):
                    if verbose:
                        print('New parm: {} \t- new parm not iterable unlike old parm *****'.format(key))
                    tests.append(False)
                    break
                new_array = np.array(new_parms[key])
                if old_value.size != new_array.size:
                    if verbose:
                        print('New parm: {} \t- are of different sizes ****'.format(key))
                    tests.append(False)
                else:
                    answer = np.all(np.isclose(old_value, new_array))
                    if verbose:
                        print('New parm: {} \t- match: {}'.format(key, answer))
                    tests.append(answer)
            else:
                """if isinstance(new_parms[key], collections.Iterable):
                    if verbose:
                        print('New parm: {} \t- new parm is iterable unlike old parm *****'.format(key))
                    tests.append(False)
                    break"""
                answer = np.all(new_parms[key] == old_value)
                if verbose:
                        print('New parm: {} \t- match: {}'.format(key, answer))
                tests.append(answer)
        if verbose:
              print('')

        if all(tests):
            return group

    return None


def create_spec_inds_from_vals(ds_spec_val_mat):
    """
    Create new Spectroscopic Indices table from the changes in the
    Spectroscopic Values

    Parameters
    ----------
    ds_spec_val_mat : array-like,
        Holds the spectroscopic values to be indexed

    Returns
    -------
    ds_spec_inds_mat : numpy array of uints the same shape as ds_spec_val_mat
        Indices corresponding to the values in ds_spec_val_mat

    """
    ds_spec_inds_mat = np.zeros_like(ds_spec_val_mat, dtype=np.int32)

    """
    Find how quickly the spectroscopic values are changing in each row 
    and the order of row from fastest changing to slowest.
    """
    change_count = [len(np.where([row[i] != row[i - 1] for i in range(len(row))])[0]) for row in ds_spec_val_mat]
    change_sort = np.argsort(change_count)[::-1]

    """
    Determine everywhere the spectroscopic values change and build 
    index table based on those changed
    """
    indices = np.zeros(ds_spec_val_mat.shape[0])
    for jcol in range(1, ds_spec_val_mat.shape[1]):
        this_col = ds_spec_val_mat[change_sort, jcol]
        last_col = ds_spec_val_mat[change_sort, jcol - 1]

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
            indices[changed] += 1
        elif len(changed > 1):
            for change in changed[:-1]:
                indices[change] = 0
            indices[changed[-1]] += 1

        """
        Store the indices for the current column in the dataset
        """
        ds_spec_inds_mat[change_sort, jcol] = indices

    return ds_spec_inds_mat


def get_unit_values(h5_inds, h5_vals, is_spec=True, dim_names=None):
    """
    Gets the unit arrays of values that describe the spectroscopic dimensions

    Parameters
    ----------
    h5_inds : h5py.Dataset
        Spectroscopic or Position Indices dataset
    h5_vals : h5py.Dataset
        Spectroscopic or Position Values dataset
    is_spec : bool, recommended
        Are the provided datasets spectral. Default = True
    dim_names : str, or list of str, Optional
        Names of the dimensions of interest. Default = all

    Note - this function can be extended / modified for ancillary position dimensions as well

    Returns
    -------
    unit_values : dict
        Dictionary containing the unit array for each dimension. The name of the dimensions are the keys.

    """
    # First load to memory
    inds_mat = h5_inds[()]
    vals_mat = h5_vals[()]
    if not is_spec:
        # Convert to spectral shape
        inds_mat = np.transpose(inds_mat)
        vals_mat = np.transpose(vals_mat)

    # For all dimensions, find where the index = 0
    # basically, we are indexing all dimensions to 0
    first_indices = []
    for dim_ind in range(inds_mat.shape[0]):
        first_indices.append(inds_mat[dim_ind] == 0)
    first_indices = np.vstack(first_indices)

    full_dim_names = get_attr(h5_inds, 'labels')
    if dim_names is None:
        dim_names = full_dim_names
    elif not isinstance(dim_names, list):
        dim_names = [dim_names]

    unit_values = dict()
    for dim_name in dim_names:
        # Find the row in the spectroscopic indices that corresponds to the dimensions we want to slice:
        desired_row_ind = np.where(full_dim_names == dim_name)[0][0]

        # Find indices of all other dimensions
        remaining_dims = list(range(inds_mat.shape[0]))
        remaining_dims.remove(desired_row_ind)

        # The intersection of all these indices should give the desired index for the desired row
        intersections = np.all(first_indices[remaining_dims, :], axis=0)

        # apply this slicing to the values dataset:
        unit_values[dim_name] = vals_mat[desired_row_ind, intersections]

    return unit_values


def get_source_dataset(h5_group):
    """
    Find the name of the source dataset used to create the input `h5_group`

    Parameters
    ----------
    h5_group : h5py.Datagroup
        Child group whose source dataset will be returned

    Returns
    -------
    h5_source : h5py.Dataset

    """
    from .pycro_data import PycroDataset

    h5_parent_group = h5_group.parent
    h5_source = h5_parent_group[h5_group.name.split('/')[-1].split('-')[0]]

    if isinstance(h5_source, h5py.Group):
        warn('No source dataset was found.')
        return None
    else:
        return PycroDataset(h5_source)
