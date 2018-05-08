import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np


def show_grid(arr, sweep_signal):
    """
    Plot 2 out of 3 dimensions of a Grid as an image with a slider to move along third dimension
    
    Parameters
    ----------
    arr : array_like
        A 3d array consisting of (Ix, Iy, E) or (qx, qy, E) data.
    sweep_signal : array_like
        A 1d array consiting of the bias values or other relevant spectroscopic sweep parameter.
        For example can be passed from Grid.signals['sweep_signal'].


    Returns
    -------
    ?
    """
    # starting + min/max values for slider
    default_energy_index = arr.shape[-1] // 2
    energy_min = 0
    energy_max = arr.shape[-1] - 1

    fig, ax = plt.subplots()
    ax.set_position([0.125, 0.175, 0.80, 0.80])
    s_ax = fig.add_axes([0.2, 0.10, 0.65, 0.03])  # axis for slider
    s_energy_ind = Slider(s_ax, 'Energy', energy_min, energy_max, valinit=default_energy_index)

    im = ax.imshow(arr[:, :, int(s_energy_ind.val)])
    s_energy_ind.valtext.set_text('{:.2f} mV'.format(sweep_signal[int(s_energy_ind.val)] * 1000))

    def update(val):
        im.set_data(arr[:, :, int(s_energy_ind.val)])
        im.autoscale()

        s_energy_ind.valtext.set_text('{:.2f} mV'.format(sweep_signal[int(s_energy_ind.val)] * 1000))

    s_energy_ind.on_changed(update)
    plt.show()

    return fig, ax, s_ax, im, s_energy_ind
