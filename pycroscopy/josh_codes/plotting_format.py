from sklearn import (decomposition)
import numpy as np
from scipy import (interpolate)
import matplotlib.pyplot as plt
from matplotlib import (pyplot as plt, animation, colors,
                        ticker, path, patches, patheffects)
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
import string
from sklearn.manifold import TSNE
import sklearn.preprocessing as pre
from os.path import join as pjoin
from igor import binarywave

Path = path.Path
PathPatch = patches.PathPatch


def conduct_PCA(loops, n_components=15, verbose=True):
    """
    Computes the PCA and forms a low-rank representation of a series of response curves
    This code can be applied to all forms of response curves.
    loops = [number of samples, response spectra for each sample]

    Parameters
    ----------
    loops : numpy array
        1 or 2d numpy array - [number of samples, response spectra for each sample]
    n_components : int, optional
        int - sets the number of componets to save
    verbose : bool, optional
        output operational comments

    Returns
    -------
    PCA : object
        results from the PCA
    PCA_reconstructed : numpy array
        low-rank representation of the raw data reconstructed based on PCA denoising
    """

    # resizes the array for hyperspectral data
    if loops.ndim == 3:
        original_size = loops.shape[0]
        loops = loops.reshape(-1, loops.shape[2])
        verbose_print(verbose, 'shape of data resized to [{0}x {1}]'.format(
            loops.shape[0], loops.shape[1]))
    elif loops.ndim == 2:
        pass
    else:
        raise ValueError("data is of an incorrect size")

    if np.isnan(loops).any():
        raise ValueError(
            'data has infinite values consider using a imputer \n see interpolate_missing_points function')

    # Sets the number of components to save
    pca = decomposition.PCA(n_components=n_components)

    # Computes the PCA of the piezoelectric hysteresis loops
    PCA = pca.fit(loops)

    # does the inverse transform - creates a low rank representation of the data
    # this process denoises the data
    PCA_reconstructed = pca.inverse_transform(pca.transform(loops))

    # resized the array for hyperspectral data
    try:
        PCA_reconstructed = PCA_reconstructed.reshape(original_size, original_size, -1)
    except:
        pass

    return(PCA, PCA_reconstructed)


def verbose_print(verbose, *args):
    if verbose:
        print(*args)


def interpolate_missing_points(loop_data):
    """
    Interpolates bad pixels in piezoelectric hystereis loops.\n
    The interpolation of missing points alows for machine learning operations

    Parameters
    ----------
    loop_data : numpy array
        arary of loops

    Returns
    -------
    loop_data_cleaned : numpy array
        arary of loops
    """

    # reshapes the data such that it can run with different data sizes
    if loop_data.ndim == 2:
        loop_data = loop_data.reshape(np.sqrt(loop_data.shape[0]),
                                      np.sqrt(loop_data.shape[0]), -1)
        loop_data = np.expand_dims(loop_data, axis=0)
    elif loop_data.ndim == 3:
        loop_data = np.expand_dims(loop_data, axis=0)

    # Loops around the x index
    for i in range(loop_data.shape[0]):

        # Loops around the y index
        for j in range(loop_data.shape[1]):

            # Loops around the number of cycles
            for k in range(loop_data.shape[3]):

                if any(~np.isfinite(loop_data[i, j, :, k])):

                    true_ind = np.where(~np.isnan(loop_data[i, j, :, k]))
                    point_values = np.linspace(0, 1, loop_data.shape[2])
                    spline = interpolate.InterpolatedUnivariateSpline(point_values[true_ind],
                                                                      loop_data[i, j, true_ind, k].squeeze())
                    ind = np.where(np.isnan(loop_data[i, j, :, k]))
                    val = spline(point_values[ind])
                    loop_data[i, j, ind, k] = val

    return loop_data.squeeze()


def layout_graphs_of_arb_number(graph):
    """
    Sets the layout of graphs in matplotlib in a pretty way based on the number of plots

    Parameters
    ----------
    graphs : int
        number of axes to make

    Returns
    -------
    fig : matplotlib figure
        handel to figure being created.
    axes : numpy array (axes)
        numpy array of axes that are created.
    """

    # Selects the number of columns to have in the graph
    if graph < 3:
        mod = 2
    elif graph < 5:
        mod = 3
    elif graph < 10:
        mod = 4
    elif graph < 17:
        mod = 5
    elif graph < 26:
        mod = 6
    elif graph < 37:
        mod = 7

    # builds the figure based on the number of graphs and selected number of columns
    fig, axes = plt.subplots(graph // mod + (graph % mod > 0), mod,
                             figsize=(3 * mod, 3 * (graph // mod + (graph % mod > 0))))

    # deletes extra unneeded axes
    axes = axes.reshape(-1)
    for i in range(axes.shape[0]):
        if i + 1 > graph:
            fig.delaxes(axes[i])

    return (fig, axes)


def plot_pca_maps(pca, loops, add_colorbars=True, verbose=False, letter_labels=False,
                  add_scalebar=False, filename='./PCA_maps', print_EPS=False,
                  print_PNG=False, dpi=300, num_of_plots=True):
    """
    Adds a colorbar to a imageplot

    Parameters
    ----------
    pca : model
        previously computed pca
    add_colorbars : bool, optional
        adds colorbars to images
    verbose : bool, optional
        sets the verbosity level
    letter_labels : bool, optional
        adds letter labels for use in publications
    add_scalebar : bool, optional
        sets whether a scalebar is added to the maps
    filename : str, optional
        sets the path and filename for the exported images
    print_EPS : bool, optional
        to export as EPS
    print_PNG : bool, optional
        to export as PNG
    dpi : int, optional
        resolution of exported image
    num_of_plots : int, optional
            number of principle componets to show
    """
    if num_of_plots == -1:
        num_of_plots = pca.n_components_

    # creates the figures and axes in a pretty way
    fig, ax = layout_graphs_of_arb_number(num_of_plots)
    # resizes the array for hyperspectral data

    if loops.ndim == 3:
        original_size = loops.shape[0]
        loops = loops.reshape(-1, loops.shape[2])
        verbose_print(verbose, 'shape of data resized to [{0} x {1}]'.format(
            loops.shape[0], loops.shape[1]))
    elif loops.ndim == 2:
        original_size = np.sqrt(loops.shape[0]).astype(int)
    else:
        raise ValueError("data is of an incorrect size")

    PCA_maps = pca_weights_as_embeddings(pca, loops, num_of_components=num_of_plots)

    for i in range(num_of_plots):
        im = ax[i].imshow(PCA_maps[:, i].reshape(original_size, original_size))
        ax[i].set_yticklabels('')
        ax[i].set_xticklabels('')
        #

        if add_colorbars:
            add_colorbar(ax[i], im)

        # labels figures
        if letter_labels:
            labelfigs(ax[i], i)
        labelfigs(ax[i], i, string_add='PC {0}'.format(i + 1), loc='bm')

        if add_scalebar is not False:
            add_scalebar_to_figure(ax[i], add_scalebar[0], add_scalebar[1])

    plt.tight_layout(pad=0, h_pad=0)

    savefig(filename, dpi=300, print_EPS=print_EPS, print_PNG=print_PNG)


def plot_embedding_maps(data, add_colorbars=True, verbose=False, letter_labels=False,
                        add_scalebar=False, filename='./embedding_maps', print_EPS=False,
                        print_PNG=False, dpi=300, num_of_plots=True):
    """
    Adds a colorbar to a imageplot

    Parameters
    ----------
    data : raw data to plot of embeddings
        data of embeddings
    add_colorbars : bool, optional
        adds colorbars to images
    verbose : bool, optional
        sets the verbosity level
    letter_labels : bool, optional
        adds letter labels for use in publications
    add_scalebar : bool, optional
        sets whether a scalebar is added to the maps
    filename : str, optional
        sets the path and filename for the exported images
    print_EPS : bool, optional
        to export as EPS
    print_PNG : bool, optional
        to export as PNG
    dpi : int, optional
        resolution of exported image
    num_of_plots : int, optional
            number of principle componets to show
    """
    if num_of_plots:
        num_of_plots = data.shape[data.ndim - 1]

    # creates the figures and axes in a pretty way
    fig, ax = layout_graphs_of_arb_number(num_of_plots)
    # resizes the array for hyperspectral data

    if data.ndim == 3:
        original_size = data.shape[0].astype(int)
        data = data.reshape(-1, data.shape[2])
        verbose_print(verbose, 'shape of data resized to [{0} x {1}]'.format(
            data.shape[0], data.shape[1]))
    elif data.ndim == 2:
        original_size = np.sqrt(data.shape[0]).astype(int)
    else:
        raise ValueError("data is of an incorrect size")

    for i in range(num_of_plots):
        im = ax[i].imshow(data[:, i].reshape(original_size, original_size))
        ax[i].set_yticklabels('')
        ax[i].set_xticklabels('')
        #

        if add_colorbars:
            add_colorbar(ax[i], im)

        # labels figures
        if letter_labels:
            labelfigs(ax[i], i)
        labelfigs(ax[i], i, string_add='emb. {0}'.format(i + 1), loc='bm')

        if add_scalebar is not False:
            add_scalebar_to_figure(ax[i], add_scalebar[0], add_scalebar[1])

    plt.tight_layout(pad=0, h_pad=0)

    savefig(filename, dpi=300, print_EPS=print_EPS, print_PNG=print_PNG)


def plot_pca_values(voltage, pca, num_of_plots=True, set_ylim=True, letter_labels=False,
                    filename='./PCA_vectors', print_EPS=False,
                    print_PNG=False, dpi=300):
    """
    Adds a colorbar to a imageplot

    Parameters
    ----------
    voltage : numpy array
        voltage vector for the hysteresis loop
    pca : model
        previously computed pca
    num_of_plots : int, optional
        number of principle componets to show
    set_ylim : int, optional
        optional manual overide of y scaler
    letter_labels : bool, optional
        adds letter labels for use in publications
    filename : str, optional
        sets the path and filename for the exported images
    print_EPS : bool, optional
        to export as EPS
    print_PNG : bool, optional
        to export as PNG
    dpi : int, optional
        resolution of exported image
    """
    if num_of_plots:
        num_of_plots = pca.n_components_

    # creates the figures and axes in a pretty way
    fig, ax = layout_graphs_of_arb_number(num_of_plots)

    for i in range(num_of_plots):
        ax[i].plot(voltage, pca.components_[i], 'k')
        ax[i].set_xlabel('Voltage')
        ax[i].set_ylabel('Amplitude (Arb. U.)')
        ax[i].set_yticklabels('')
        #ax[i].set_title('PC {0}'.format(i+1))
        if not set_ylim:
            ax[i].set_ylim(set_ylim[0], set_ylim[1])

        # labels figures
        if letter_labels:
            labelfigs(ax[i], i)
        labelfigs(ax[i], i, string_add='PC {0}'.format(i + 1), loc='bm')

    plt.tight_layout(pad=0, h_pad=0)

    savefig(filename, dpi=300, print_EPS=print_EPS, print_PNG=print_PNG)


def add_colorbar(axes, plot, location='right', size=10, pad=0.05, format='%.1e'):
    """
    Adds a colorbar to a imageplot

    Parameters
    ----------
    axes : matplotlib axes
        axes which to add the plot to
    axes : matplotlib plot
        Plot being references for the scalebar
    location : str, optional
        position to place the colorbar
    size : int, optional
        percent size of colorbar realitive to the plot
    pad : float, optional
        gap between colorbar and plot
    format : str, optional
        string format for the labels on colorbar
    """

    # Adds the scalebar
    divider = make_axes_locatable(axes)
    cax = divider.append_axes(location, size='{0}%'.format(size), pad=pad)
    cbar = plt.colorbar(plot, cax=cax, format=format)

# Function to add text labels to figure


def labelfigs(axes, number, style='wb', loc='br', string_add='', size=14, text_pos='center'):
    """
    Adds labels to figures

    Parameters
    ----------
    axes : matplotlib axes
        axes which to add the plot to
    number : int
        letter number
    style : str, optional
        sets the color of the letters
    loc : str, optional
        sets the location of the label
    string_add : str, optional
        custom string as the label
    size : int, optional
        sets the fontsize for the label
    text_pos : str, optional
        set the justification of the label
    """

    # Sets up various color options
    formating_key = {'wb': dict(color='w',
                                linewidth=1.5),
                     'b': dict(color='k',
                               linewidth=0),
                     'w': dict(color='w',
                               linewidth=0)}

    # Stores the selected option
    formatting = formating_key[style]

    # finds the position for the label
    x_min, x_max = axes.get_xlim()
    y_min, y_max = axes.get_ylim()
    x_value = .08 * (x_max - x_min) + x_min

    if loc == 'br':
        y_value = y_max - .1 * (y_max - y_min)
        x_value = .08 * (x_max - x_min) + x_min
    elif loc == 'tr':
        y_value = y_max - .9 * (y_max - y_min)
        x_value = .08 * (x_max - x_min) + x_min
    elif loc == 'bl':
        y_value = y_max - .1 * (y_max - y_min)
        x_value = x_max - .08 * (x_max - x_min)
    elif loc == 'tl':
        y_value = y_max - .9 * (y_max - y_min)
        x_value = x_max - .08 * (x_max - x_min)
    elif loc == 'tm':
        y_value = y_max - .9 * (y_max - y_min)
        x_value = x_min + (x_max - x_min) / 2
    elif loc == 'bm':
        y_value = y_max - .1 * (y_max - y_min)
        x_value = x_min + (x_max - x_min) / 2
    else:
        raise ValueError(
            'Unknown string format imported please look at code for acceptable positions')

    if string_add == '':

        # Turns to image number into a label
        if number < 26:
            axes.text(x_value, y_value, string.ascii_lowercase[number],
                      size=14, weight='bold', ha=text_pos,
                      va='center', color=formatting['color'],
                      path_effects=[patheffects.withStroke(linewidth=formatting['linewidth'],
                                                           foreground="k")])

        # allows for double letter index
        else:
            axes.text(x_value, y_value, string.ascii_lowercase[0] + string.ascii_lowercase[number - 26],
                      size=14, weight='bold', ha=text_pos,
                      va='center', color=formatting['color'],
                      path_effects=[patheffects.withStroke(linewidth=formatting['linewidth'],
                                                           foreground="k")])
    else:

        axes.text(x_value, y_value, string_add,
                  size=14, weight='bold', ha=text_pos,
                  va='center', color=formatting['color'],
                  path_effects=[patheffects.withStroke(linewidth=formatting['linewidth'],
                                                       foreground="k")])


def add_scalebar_to_figure(axes, image_size, scale_size, units='nm', loc='br'):
    """
    Adds scalebar to figures

    Parameters
    ----------
    axes : matplotlib axes
        axes which to add the plot to
    image_size : int
        size of the image in nm
    scale_size : str, optional
        size of the scalebar in units of nm
    units : str, optional
        sets the units for the label
    loc : str, optional
        sets the location of the label
    """

    x_lim, y_lim = axes.get_xlim(), axes.get_ylim()
    x_size, y_size = np.abs(np.floor(x_lim[1] - x_lim[0])), np.abs(np.floor(y_lim[1] - y_lim[0]))

    fract = scale_size / image_size

    x_point = np.linspace(x_lim[0], x_lim[1], np.floor(image_size))
    y_point = np.linspace(y_lim[0], y_lim[1], np.floor(image_size))

    if loc == 'br':
        x_start = x_point[np.int(.9 * image_size // 1)]
        x_end = x_point[np.int((.9 - fract) * image_size // 1)]
        y_start = y_point[np.int(.1 * image_size // 1)]
        y_end = y_point[np.int((.1 + .025) * image_size // 1)]
        y_label_height = y_point[np.int((.1 + .075) * image_size // 1)]
    elif loc == 'tr':
        x_start = x_point[np.int(.9 * image_size // 1)]
        x_end = x_point[np.int((.9 - fract) * image_size // 1)]
        y_start = y_point[np.int(.9 * image_size // 1)]
        y_end = y_point[np.int((.9 - .025) * image_size // 1)]
        y_label_height = y_point[np.int((.9 - .075) * image_size // 1)]

    path_maker(axes, [x_start, x_end, y_start, y_end], 'w', 'k', '-', 1)

    axes.text((x_start + x_end) / 2,
              y_label_height,
              '{0} {1}'.format(scale_size, units),
              size=14, weight='bold', ha='center',
              va='center', color='w',
              path_effects=[patheffects.withStroke(linewidth=1.5,
                                                   foreground="k")])


def path_maker(axes, locations, facecolor, edgecolor, linestyle, lineweight):
    """
    Adds path to figure

    Parameters
    ----------
    axes : matplotlib axes
        axes which to add the plot to
    locations : numpy array
        location to position the path
    facecolor : str, optional
        facecolor of the path
    edgecolor : str, optional
        edgecolor of the path
    linestyle : str, optional
        sets the style of the line, using conventional matplotlib styles
    lineweight : float, optional
        thickness of the line
    """
    vertices = []
    codes = []
    codes = [Path.MOVETO] + [Path.LINETO] * 3 + [Path.CLOSEPOLY]
    vertices = [(locations[0], locations[2]),
                (locations[1], locations[2]),
                (locations[1], locations[3]),
                (locations[0], locations[3]),
                (0, 0)]
    vertices = np.array(vertices, float)
    path = Path(vertices, codes)
    pathpatch = PathPatch(path, facecolor=facecolor, edgecolor=edgecolor,
                          ls=linestyle, lw=lineweight)
    axes.add_patch(pathpatch)


def savefig(filename, dpi=300, print_EPS=False, print_PNG=False):
    """
    Adds path to figure

    Parameters
    ----------
    filename : str
        path to save file
    dpi : int, optional
        resolution to save image
    print_EPS : bool, optional
        selects if export the EPS
    print_PNG : bool, optional
        selects if print the PNG
    """
    # Saves figures at EPS
    if print_EPS:
        plt.savefig(filename + '.eps', format='eps',
                    dpi=dpi, bbox_inches='tight')

    # Saves figures as PNG
    if print_PNG:
        plt.savefig(filename + '.png', format='png',
                    dpi=dpi, bbox_inches='tight')


def pca_weights_as_embeddings(pca, loops, num_of_components=0, verbose=True):
    """
    Computes the eigenvalue maps computed from PCA

    Parameters
    ----------
    pca : object
        computed PCA
    loops: numpy array
        raw piezoresponse data
    num_of _components: int
        number of PCA components to compute

    Returns
    -------
    fig : matplotlib figure
        handel to figure being created.
    axes : numpy array (axes)
        numpy array of axes that are created.
    """
    if loops.ndim == 3:
        loops = loops.reshape(-1, loops.shape[2])
        verbose_print(verbose, 'shape of data resized to [{0} x {1}]'.format(
            loops.shape[0], loops.shape[1]))

    if num_of_components == 0:
        num_of_components = pca.n_components_

    PCA_embedding = pca.transform(loops)[:, 0:num_of_components]

    return (PCA_embedding)


def T_SNE(encodings, n_components=2, perplexity=30.0, early_exaggeration=12.0,
          learning_rate=200.0, n_iter=1000, n_iter_without_progress=300,
          min_grad_norm=1e-07, metric='euclidean', init='random', verbose=0,
          random_state=None, method='barnes_hut', angle=0.5,
          save_results=False, file_name=''):
    """t-distributed Stochastic Neighbor Embedding.

    t-SNE [1] is a tool to visualize high-dimensional data. It converts
    similarities between data points to joint probabilities and tries
    to minimize the Kullback-Leibler divergence between the joint
    probabilities of the low-dimensional embedding and the
    high-dimensional data. t-SNE has a cost function that is not convex,
    i.e. with different initializations we can get different results.

    It is highly recommended to use another dimensionality reduction
    method (e.g. PCA for dense data or TruncatedSVD for sparse data)
    to reduce the number of dimensions to a reasonable amount (e.g. 50)
    if the number of features is very high. This will suppress some
    noise and speed up the computation of pairwise distances between
    samples. For more tips see Laurens van der Maaten's FAQ [2].

    Read more in the :ref:`User Guide <t_sne>`.

    Parameters
    ----------
    embeddings : numpy array
        "Low" dimensional embedding space for further dimensionality reduction

    n_components : int, optional (default: 2)
        Dimension of the embedded space.

    perplexity : float, optional (default: 30)
        The perplexity is related to the number of nearest neighbors that
        is used in other manifold learning algorithms. Larger datasets
        usually require a larger perplexity. Consider selecting a value
        between 5 and 50. The choice is not extremely critical since t-SNE
        is quite insensitive to this parameter.

    early_exaggeration : float, optional (default: 12.0)
        Controls how tight natural clusters in the original space are in
        the embedded space and how much space will be between them. For
        larger values, the space between natural clusters will be larger
        in the embedded space. Again, the choice of this parameter is not
        very critical. If the cost function increases during initial
        optimization, the early exaggeration factor or the learning rate
        might be too high.

    learning_rate : float, optional (default: 200.0)
        The learning rate for t-SNE is usually in the range [10.0, 1000.0]. If
        the learning rate is too high, the data may look like a 'ball' with any
        point approximately equidistant from its nearest neighbours. If the
        learning rate is too low, most points may look compressed in a dense
        cloud with few outliers. If the cost function gets stuck in a bad local
        minimum increasing the learning rate may help.

    n_iter : int, optional (default: 1000)
        Maximum number of iterations for the optimization. Should be at
        least 250.

    n_iter_without_progress : int, optional (default: 300)
        Maximum number of iterations without progress before we abort the
        optimization, used after 250 initial iterations with early
        exaggeration. Note that progress is only checked every 50 iterations so
        this value is rounded to the next multiple of 50.

        .. versionadded:: 0.17
           parameter *n_iter_without_progress* to control stopping criteria.

    min_grad_norm : float, optional (default: 1e-7)
        If the gradient norm is below this threshold, the optimization will
        be stopped.

    metric : string or callable, optional
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string, it must be one of the options
        allowed by scipy.spatial.distance.pdist for its metric parameter, or
        a metric listed in pairwise.PAIRWISE_DISTANCE_FUNCTIONS.
        If metric is "precomputed", X is assumed to be a distance matrix.
        Alternatively, if metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two arrays from X as input and return a value indicating
        the distance between them. The default is "euclidean" which is
        interpreted as squared euclidean distance.

    init : string or numpy array, optional (default: "random")
        Initialization of embedding. Possible options are 'random', 'pca',
        and a numpy array of shape (n_samples, n_components).
        PCA initialization cannot be used with precomputed distances and is
        usually more globally stable than random initialization.

    verbose : int, optional (default: 0)
        Verbosity level.

    random_state : int, RandomState instance or None, optional (default: None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.  Note that different initializations might result in
        different local minima of the cost function.

    method : string (default: 'barnes_hut')
        By default the gradient calculation algorithm uses Barnes-Hut
        approximation running in O(NlogN) time. method='exact'
        will run on the slower, but exact, algorithm in O(N^2) time. The
        exact algorithm should be used when nearest-neighbor errors need
        to be better than 3%. However, the exact method cannot scale to
        millions of examples.

        .. versionadded:: 0.17
           Approximate optimization *method* via the Barnes-Hut.

    angle : float (default: 0.5)
        Only used if method='barnes_hut'
        This is the trade-off between speed and accuracy for Barnes-Hut T-SNE.
        'angle' is the angular size (referred to as theta in [3]) of a distant
        node as measured from a point. If this size is below 'angle' then it is
        used as a summary node of all points contained within it.
        This method is not very sensitive to changes in this parameter
        in the range of 0.2 - 0.8. Angle less than 0.2 has quickly increasing
        computation time and angle greater 0.8 has quickly increasing error.

    save_results : bool (default: False)
        selects if the user wants to save the file

    file_name : str, optional
        adds a prefix to the filename

    Attributes
    ----------
    embedding_ : array-like, shape (n_samples, n_components)
        Stores the embedding vectors.

    kl_divergence_ : float
        Kullback-Leibler divergence after optimization.

    n_iter_ : int
        Number of iterations run.

    Examples
    --------

    >>> import numpy as np
    >>> from sklearn.manifold import TSNE
    >>> X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    >>> X_embedded = TSNE(n_components=2).fit_transform(X)
    >>> X_embedded.shape
    (4, 2)

    References
    ----------

    [1] van der Maaten, L.J.P.; Hinton, G.E. Visualizing High-Dimensional Data
        Using t-SNE. Journal of Machine Learning Research 9:2579-2605, 2008.

    [2] van der Maaten, L.J.P. t-Distributed Stochastic Neighbor Embedding
        http://homepage.tudelft.nl/19j49/t-SNE.html

    [3] L.J.P. van der Maaten. Accelerating t-SNE using Tree-Based Algorithms.
        Journal of Machine Learning Research 15(Oct):3221-3245, 2014.
        http://lvdmaaten.github.io/publications/papers/JMLR_2014.pdf"""

    TSNE_model = TSNE(n_components=n_components,
                      perplexity=perplexity,
                      early_exaggeration=early_exaggeration,
                      learning_rate=learning_rate,
                      n_iter=n_iter, n_iter_without_progress=n_iter_without_progress,
                      min_grad_norm=min_grad_norm,
                      metric=metric, init=init, verbose=verbose,
                      random_state=random_state, method=method,
                      angle=angle)

    print('Working. TSNE can take some time')

    TSNE_Results = TSNE_model.fit_transform(encodings)

    if save_results:
        folder_path = make_folder('TSNE')
        file_name_ = file_name + '_comp_{0}_per_{1}_lr_{2}_iter_{3}'.format(n_components,
                                                                            perplexity,
                                                                            learning_rate,
                                                                            n_iter)
        file_name_ = prevent_file_overwrite(pjoin(folder_path, file_name_), 'npy')
        np.save(pjoin(file_name_), TSNE_Results)

    return(TSNE_Results)


def make_folder(folder_name, root='./'):
    """
    creates a new folder

    Parameters
    ----------
    folder_name : str
        name of the folder
    root : str, optional
        pass the root path
    """

    folder = pjoin(root, '{}'.format(folder_name))
    os.makedirs(folder, exist_ok=True)

    return (folder)


def prevent_file_overwrite(file_name, ext):
    """
    makes sure file being saved does not overwrite existing

    Parameters
    ----------
    file_name : str
        name of the file
    ext : str
        the file extension
    """

    count = 0
    file_name = file_name + '.' + ext
    while os.path.isfile(file_name):
        if file_name.find('_run') == -1:
            file_name = file_name[:-4]
            file_name = file_name + '_run_{0:02d}.'.format(count) + ext
        else:
            count = + 1
            file_name = file_name[:file_name.find('run') + 4] + '{0:02d}.npy'.format(count) + ext

    return(file_name[:-len(ext) - 1])


def rgb_color_map(data, add_scalebar=False, print_EPS=False,
                  print_PNG=False, filename='RGB_maps', dpi=300):
    """
    Plots 3d hyperspectral data as an RGB image for visualization
    TODO: need to improve colormaps used

    Parameters
    ----------
    data : numpy array
        voltage vector for the hysteresis loop
    add_scalebar : int, optional
        vector with 2 values first is the scan size second is the marker size
    filename : str, optional
        sets the path and filename for the exported images
    print_EPS : bool, optional
        to export as EPS
    print_PNG : bool, optional
        to export as PNG
    dpi : int, optional
        resolution of exported image
    """
    fig = plt.figure(figsize=(3, 3), dpi=300)
    ax = fig.add_subplot(111)  # 1 Row, 1 Column and the first axes in this grid

    # Pre-allocates the matrix
    RGB = np.ones((data.shape[0], 3))

    for i in range(3):
        data[:, i] -= np.nanmin(data[:, i])
        RGB[:, i] = (data[:, i] / (np.nanmax(data[:, i])))

    size = np.sqrt(data.shape[0]).astype('int')
    im = ax.imshow(RGB.reshape(size, size, 3))
    ax.set_yticklabels('')
    ax.set_xticklabels('')

    if add_scalebar is not False:
        add_scalebar_to_figure(ax, add_scalebar[0], add_scalebar[1])

    savefig(filename, dpi=dpi, print_EPS=print_EPS, print_PNG=print_PNG)

def load_AFM_data(file_path):
    ## set the folder path
    d={}
    dat = []

    ## Importing image data from .ibw format and assigning to list dat
    d = binarywave.load(file_path)
    dat.append((d['wave']['wData']))

    return d, dat[0]

def normalize_AFM_slowscan(dat):

    norm_dat = np.zeros(dat.shape)

    for i in range(dat.shape[2]):

        for j in range(dat.shape[1]):

            norm_dat[:,j,i] = pre.StandardScaler().fit_transform(dat[:,j,i].reshape(-1,1)).squeeze()

    return norm_dat

def plot_AFM_images(d, dat, normal_data):

    # Code which sets the dimensions of the figure based on aspect ratio
    x_aspect = dat.shape[1]/np.max(dat.shape)
    y_aspect = dat.shape[0]/np.max(dat.shape)

    if normal_data.shape[2] == 11:

        # Defines the figures and the axes
        fig, axes = plt.subplots(2, 3, figsize=(9*y_aspect+.5, 6*x_aspect+.5))
        axes = axes.reshape(6)

        labels = ['Height', 'Amp 1', 'Amp 2', 'Phase 1', 'Phase 2', 'Resonance Frequency']
        ind = [0,1,3,5,7,9]

        for i in range(6):
            axes[i].imshow(np.rollaxis(normal_data[:,:,ind[i]],1).T)
            axes[i].set_title(labels[i])
            axes[i].set_xticks([0,dat.shape[0]])
            axes[i].set_yticks([0,dat.shape[1]])

            notes = d['wave']['note'].decode('Windows-1252').split('\r')

            fast_scan_size = float(notes[1].split(': ')[1])
            slow_scan_size = float(notes[2].split(': ')[1])

            if fast_scan_size > 1e-6:
                axes[i].set_xticklabels([0,str(fast_scan_size*1e6) +' ${\mu}$m'])
            else:
                axes[i].set_xticklabels([0,str(fast_scan_size*1e9) +' nm'])

            if slow_scan_size > 1e-6:
                axes[i].set_yticklabels([0,str(slow_scan_size*1e6) +' ${\mu}$m'])
            else:
                axes[i].set_yticklabels([0,str(slow_scan_size*1e9) +' nm'])

    if normal_data.shape[2] == 6:

        # Defines the figures and the axes
        fig, axes = plt.subplots(2, 3, figsize=(9*y_aspect+.5, 6*x_aspect+.5))
        axes = axes.reshape(6)

        labels = ['Height', 'Amp 1', 'Amp 2', 'Phase 1', 'Phase 2', 'Resonance Frequency']

        for i in range(6):
            axes[i].imshow(np.rollaxis(normal_data[:,:,i],1).T)
            axes[i].set_title(labels[i])
            axes[i].set_xticks([0,dat.shape[0]])
            axes[i].set_yticks([0,dat.shape[1]])

            notes = d['wave']['note'].decode('Windows-1252').split('\r')

            fast_scan_size = float(notes[1].split(': ')[1])
            slow_scan_size = float(notes[2].split(': ')[1])

            if fast_scan_size > 1e-6:
                axes[i].set_xticklabels([0,str(fast_scan_size*1e6) +' ${\mu}$m'])
            else:
                axes[i].set_xticklabels([0,str(fast_scan_size*1e9) +' nm'])

            if slow_scan_size > 1e-6:
                axes[i].set_yticklabels([0,str(slow_scan_size*1e6) +' ${\mu}$m'])
            else:
                axes[i].set_yticklabels([0,str(slow_scan_size*1e9) +' nm'])

    if normal_data.shape[2] == 4:

        # Defines the figures and the axes
        fig, axes = plt.subplots(2, 2, figsize=(6*y_aspect+.5, 6*x_aspect+.5))
        axes = axes.reshape(4)

        labels = ['Height', 'Amplitude', 'Phase', 'z sensor']

        for i in range(4):
            axes[i].imshow(np.rollaxis(normal_data[:,:,i],1).T)
            axes[i].set_title(labels[i])
            axes[i].set_xticks([0,dat.shape[0]])
            axes[i].set_yticks([0,dat.shape[1]])

            notes = d['wave']['note'].decode('Windows-1252').split('\r')

            fast_scan_size = float(notes[1].split(': ')[1])
            slow_scan_size = float(notes[2].split(': ')[1])

            if fast_scan_size > 1e-6:
                axes[i].set_xticklabels([0,str(fast_scan_size*1e6) +' ${\mu}$m'])
            else:
                axes[i].set_xticklabels([0,str(fast_scan_size*1e9) +' nm'])

            if slow_scan_size > 1e-6:
                axes[i].set_yticklabels([0,str(slow_scan_size*1e6) +' ${\mu}$m'])
            else:
                axes[i].set_yticklabels([0,str(slow_scan_size*1e9) +' nm'])

            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
