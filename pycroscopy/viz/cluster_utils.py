from __future__ import division, print_function, absolute_import, unicode_literals
import sys
from warnings import warn
import h5py
import matplotlib as mpl
import numpy as np
import scipy
from matplotlib import pyplot as plt

if sys.version_info.major == 3:
    unicode = str

from pyUSID import USIDataset
from pyUSID.io.hdf_utils import get_attr
from pyUSID.viz.plot_utils import plot_complex_spectra, plot_map_stack, default_cmap, plot_map, \
    discrete_cmap, plot_line_family, make_scalar_mappable, plot_curves


def plot_cluster_h5_group(h5_group, labels_kwargs=None, centroids_kwargs=None):
    """
    Plots the cluster labels and mean response for each cluster

    Parameters
    ----------
    h5_group : h5py.Datagroup object
        H5 group containing the labels and mean response
    labels_kwargs : dict, optional
        keyword arguments for the labels plot. NOT enabled yet.
    centroids_kwargs : dict, optional
        keyword arguments for the centroids plot. NOT enabled yet.

    Returns
    -------
    fig_labels : figure handle
        Figure containing the labels
    fig_centroids : figure handle
        Figure containing the centroids
    """
    if not isinstance(h5_group, h5py.Group):
        raise TypeError('h5_group should be a h5py.Group')
    h5_labels = USIDataset(h5_group['Labels'])
    h5_centroids = USIDataset(h5_group['Mean_Response'])

    labels_mat = np.squeeze(h5_labels.get_n_dim_form())
    if labels_mat.ndim > 3:
        print('Unable to visualize 4 or more dimensional labels!')
    if labels_mat.ndim == 1:
        fig_labs, axis_labs = plt.subplots(figsize=(5.5, 5))
        axis_labs.plot(h5_labels.get_pos_values(h5_labels.pos_dim_labels[0]), labels_mat)
        axis_labs.set_xlabel(h5_labels.pos_dim_descriptors[0])
        axis_labs.set_ylabel('Cluster index')
        axis_labs.set_title(get_attr(h5_group, 'cluster_algorithm') + ' Labels')
    elif labels_mat.ndim == 2:
        fig_labs, axis_labs = plot_cluster_labels(labels_mat, num_clusters=h5_centroids.shape[0],
                                                  x_label=h5_labels.pos_dim_descriptors[0],
                                                  y_label=h5_labels.pos_dim_descriptors[1],
                                                  x_vec=h5_labels.get_pos_values(h5_labels.pos_dim_labels[0]),
                                                  y_vec=h5_labels.get_pos_values(h5_labels.pos_dim_labels[1]),
                                                  title=get_attr(h5_group, 'cluster_algorithm') + ' Labels')

    # TODO: probably not a great idea to load the entire dataset to memory
    centroids_mat = h5_centroids.get_n_dim_form()
    if len(h5_centroids.spec_dim_labels) == 1:
        legend_mode = 2
        if h5_centroids.shape[0] < 6:
            legend_mode = 1
        fig_cent, axis_cent = plot_cluster_centroids(centroids_mat,
                                                     h5_centroids.get_spec_values(h5_centroids.spec_dim_labels[0]),
                                                     legend_mode=legend_mode,
                                                     x_label=h5_centroids.spec_dim_descriptors[0],
                                                     y_label=h5_centroids.data_descriptor,
                                                     overlayed=h5_centroids.shape[0] < 6,
                                                     title=get_attr(h5_group,
                                                                    'cluster_algorithm') + ' Centroid',
                                                     amp_units=get_attr(h5_centroids, 'units'))
    elif len(h5_centroids.spec_dim_labels) == 2:
        # stack of spectrograms
        if h5_centroids.dtype in [np.complex64, np.complex128, np.complex]:
            fig_cent, axis_cent = plot_complex_spectra(centroids_mat, subtitle_prefix='Cluster',
                                                       title=get_attr(h5_group, 'cluster_algorithm') + ' Centroid',
                                                       x_label=h5_centroids.spec_dim_descriptors[0],
                                                       y_label=h5_centroids.spec_dim_descriptors[1],
                                                       amp_units=get_attr(h5_centroids, 'units'))
        else:
            fig_cent, axis_cent = plot_map_stack(centroids_mat, color_bar_mode='each', evenly_spaced=True,
                                                 title='Cluster',
                                                 heading=get_attr(h5_group,
                                                                  'cluster_algorithm') + ' Centroid')
    return fig_labs, fig_cent


def plot_cluster_labels(labels_mat, num_clusters=None, x_label=None, y_label=None, title=None, axis=None, **kwargs):
    """
    Plots the cluster labels

    Parameters
    ----------
    labels_mat : numpy.ndarray
        1D or 2D unsigned integer array containing the labels of the clusters
    num_clusters : int, optional
        Number of clusters
    x_label : str, optional
        Label for x axis
    y_label : str, optional
        Label for y axis
    title : str, optional
        Title for the plot
    axis : matplotlib.axes.Axes object, optional.
        Axis to plot this image onto. Will create a new figure by default or will use this axis object to plot into
    kwargs : dict
        will be passed on to plot() or plot_map()

    Returns
    -------
    fig : matplotlib.pyplot.Figure object
        figure object
    axis : matplotlib.Axes object
        axis object
    """
    if not isinstance(labels_mat, np.ndarray):
        raise TypeError('labels_mat should be numpy array')
    if labels_mat.ndim > 2:
        raise ValueError('labels_mat should be a 1D or 2D array')
    if not isinstance(num_clusters, int):
        raise TypeError('num_clusters should be an integer')
    if axis is not None:
        if not isinstance(axis, mpl.axes.Axes):
            raise TypeError('axis must be a matplotlib.axes.Axes object')
    if num_clusters is not None:
        if not isinstance(num_clusters, int) or num_clusters < 1:
            raise TypeError('num_clusters should be a positive integer')
    else:
        num_clusters = np.max(labels_mat)

    if axis is None:
        fig, axis = plt.subplots(figsize=kwargs.pop('figsize', (5.5, 5)))
    else:
        fig = None

    if labels_mat.ndim > 1:
        _, _ = plot_map(axis, labels_mat, clim=[0, num_clusters - 1], aspect='auto', show_xy_ticks=True,
                        cmap=discrete_cmap(num_clusters, kwargs.pop('cmap', default_cmap)), **kwargs)
    else:
        x_vec = kwargs.pop('x_vec', np.arange(labels_mat.size))
        axis.plot(x_vec, labels_mat, **kwargs)

    for var, var_name, func in zip([title, x_label, y_label], ['title', 'x_label', 'y_label'],
                                   [axis.set_title, axis.set_xlabel, axis.set_ylabel]):
        if var is not None:
            if not isinstance(var, (str, unicode)):
                raise TypeError(var_name + ' should be a string')
            func(var)

    if fig is not None:
        fig.tight_layout()

    return fig, axis


def plot_cluster_centroids(centroids, x_vec, legend_mode=1, x_label=None, y_label=None, title=None, axis=None,
                           overlayed=True, amp_units=None, **kwargs):
    """

    Parameters
    ----------
    centroids : numpy.ndarray
        2D array. Centroids of clusters
    x_vec : numpy.ndarray
        1D array. Vector against which the curves are plotted
    legend_mode : int, optional. default = 1
        Appearance of legend:
            0 - inside the plot
            1 - outside the plot on the right
            else - colorbar instead of legend
    x_label : str, optional, default = None
        Label for x axis
    y_label : str, optional, default = None
        Label for y axis
    title : str, optional, default = None
        Title for the plot
    axis : matplotlib.axes.Axes object, optional.
        Axis to plot this image onto. Will create a new figure by default or will use this axis object to plot into
    overlayed : bool, optional
        If True - all curves will be plotted overlayed on a single plot. Else, curves will be plotted separately
    amp_units : str, optional
        Units for amplitude
    kwargs : dict
        will be passed on to plot_line_family(), plot_complex_spectra, plot_curves

    Returns
    -------
    fig, axes
    """
    if isinstance(centroids, (list, tuple)):
        centroids = np.array(centroids)
    if not isinstance(centroids, np.ndarray):
        raise TypeError('centroids should be a numpy array')
    if centroids.ndim != 2:
        raise ValueError('centroids should be a 2D numpy array - i.e. - 1D spectra')
    if not isinstance(x_vec, (list, tuple)):
        x_vec = np.array(x_vec)
    if not isinstance(x_vec, np.ndarray):
        raise TypeError('x_vec should be a array-like')
    if x_vec.ndim != 1:
        raise ValueError('x_vec should be a 1D array')
    if not isinstance(legend_mode, int):
        raise TypeError('legend_mode should be an integer')
    if axis is not None:
        if not isinstance(axis, mpl.axes.Axes):
            raise TypeError('axis must be a matplotlib.axes.Axes object')
    if not isinstance(overlayed, bool):
        raise TypeError('overlayed should be a boolean value')
    if amp_units is not None:
        if not (isinstance(amp_units, (str, unicode)) or
                (isinstance(amp_units, np.ndarray) and amp_units.dtype.type == np.str_)):
            raise TypeError('amp_units should be a str')
    else:
        amp_units = 'a.u.'

    cmap = kwargs.get('cmap', default_cmap)
    num_clusters = centroids.shape[0]

    def __overlay_curves(axis, curve_stack):

        plot_line_family(axis, x_vec, curve_stack, label_prefix='Cluster', cmap=cmap)

        if legend_mode == 0:
            axis.legend(loc='best', fontsize=14)
        elif legend_mode == 1:
            axis.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=14)
        else:
            sm = make_scalar_mappable(0, num_clusters - 1, cmap=discrete_cmap(num_clusters, cmap))
            plt.colorbar(sm)

    if overlayed:
        if centroids.dtype in [np.complex64, np.complex128, np.complex]:
            fig, axes = plt.subplots(nrows=2, figsize=kwargs.pop('figsize', (5.5, 2 * 5)))
            for axis, func in zip(axes.flat, [np.abs, np.angle]):
                __overlay_curves(axis, func(centroids))

            for var, var_name, func in zip([y_label, y_label, x_label], ['y_label', 'y_label', 'x_label'],
                                           [axes[1].set_ylabel, axes[1].set_xlabel]):
                if var is not None:
                    if not isinstance(var, (str, unicode)):
                        raise TypeError(var_name + ' should be a string')
                    func(var)

            if title is not None:
                if not isinstance(title, (str, unicode)):
                    raise TypeError('title should be a string')
                for axis, comp_name, units in zip(axes.flat, ['Amplitude', 'Phase'], [amp_units, 'rad']):
                    axis.set_title('{} - {} ({})'.format(title, comp_name, units))

        else:

            if axis is None:
                fig, axis = plt.subplots(figsize=kwargs.pop('figsize', (5.5, 5)))
            else:
                fig = None

            __overlay_curves(axis, centroids)

            for var, var_name, func in zip([title, x_label, y_label], ['title', 'x_label', 'y_label'],
                                           [axis.set_title, axis.set_xlabel, axis.set_ylabel]):
                if var is not None:
                    if not isinstance(var, (str, unicode)):
                        raise TypeError(var_name + ' should be a string')
                    func(var)

        if fig is not None:
            fig.tight_layout()
        return fig, axis

    else:
        if centroids.dtype in [np.complex64, np.complex128, np.complex]:
            return plot_complex_spectra(centroids, x_vec=x_vec, title=title, x_label=x_label, y_label=y_label,
                                        subtitle_prefix='Cluster', amp_units=amp_units, **kwargs)
        else:
            return plot_curves(x_vec, centroids, x_label=x_label, y_label=y_label, title=title,
                               subtitle_prefix='Cluster ', **kwargs)


def plot_cluster_dendrogram(label_mat, e_vals, num_comp, num_cluster, mode='Full', last=None,
                            sort_type='distance', sort_mode=True):
    """
    Creates and plots the dendrograms for the given label_mat and
    eigenvalues

    Parameters
    -------------
    label_mat : 2D real numpy array
        structured as [rows, cols], from KMeans clustering
    e_vals: 3D real numpy array of eigenvalues
        structured as [component, rows, cols]
    num_comp : int
        Number of components used to make eigenvalues
    num_cluster : int
        Number of cluster used to make the label_mat
    mode: str, optional
        How should the dendrograms be created.
        "Full" -- use all clusters when creating the dendrograms
        "Truncated" -- stop showing clusters after 'last'
    last: int, optional - should be provided when using "Truncated"
        How many merged clusters should be shown when using
        "Truncated" mode
    sort_type: {'count', 'distance'}, optional
        What type of sorting should be used when plotting the
        dendrograms.  Options are:
        count - Uses the count_sort from scipy.cluster.hierachy.dendrogram
        distance - Uses the distance_sort from scipy.cluster.hierachy.dendrogram
    sort_mode: {False, True, 'ascending', 'descending'}, optional
        For the chosen sort_type, which mode should be used.
        False - Does no sorting
        'ascending' or True - The child with the minimum of the chosen sort
        parameter is plotted first
        'descending' - The child with the maximum of the chosen sort parameter is
        plotted first

    Returns
    ---------
    fig : matplotlib.pyplot Figure object
        Figure containing the dendrogram
    """
    if mode == 'Truncated' and not last:
        warn('Warning: Truncated dendrograms requested, but no last cluster given.  Reverting to full dendrograms.')
        mode = 'Full'

    if mode == 'Full':
        print('Creating full dendrogram from clusters')
        mode = None
    elif mode == 'Truncated':
        print('Creating truncated dendrogram from clusters.  Will stop at {}.'.format(last))
        mode = 'lastp'
    else:
        raise ValueError('Error: Unknown mode requested for plotting dendrograms. mode={}'.format(mode))

    c_sort = False
    d_sort = False
    if sort_type == 'count':
        c_sort = sort_mode
        if c_sort == 'descending':
            c_sort = 'descendent'
    elif sort_type == 'distance':
        d_sort = sort_mode

    centroid_mat = np.zeros([num_cluster, num_comp])
    for k1 in range(num_cluster):
        [i_x, i_y] = np.where(label_mat == k1)
        u_stack = np.zeros([len(i_x), num_comp])
        for k2 in range(len(i_x)):
            u_stack[k2, :] = np.abs(e_vals[i_x[k2], i_y[k2], :num_comp])

        centroid_mat[k1, :] = np.mean(u_stack, 0)

    # Get the distance between cluster means
    distance_mat = scipy.spatial.distance.pdist(centroid_mat)

    # get hierarchical pairings of clusters
    linkage_pairing = scipy.cluster.hierarchy.linkage(distance_mat, 'weighted')
    linkage_pairing[:, 3] = linkage_pairing[:, 3] / max(linkage_pairing[:, 3])

    fig = plt.figure()
    scipy.cluster.hierarchy.dendrogram(linkage_pairing, p=last, truncate_mode=mode,
                                       count_sort=c_sort, distance_sort=d_sort,
                                       leaf_rotation=90)

    fig.axes[0].set_title('Dendrogram')
    fig.axes[0].set_xlabel('Index or (cluster size)')
    fig.axes[0].set_ylabel('Distance')

    return fig
