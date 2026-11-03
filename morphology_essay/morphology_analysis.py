import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skimage.io as skio
from skimage import measure
import os
import seaborn as sns
from collections import defaultdict


# region FUNCTIONS
def read_file(filename, scale):
    """
    Reads mask analysis file (.csv) as produced by the YeaZ segmentation GUI.
    :param filename: path to data file (.csv format)
    :param scale: scaling factor (#pixels / um)
    :return: lists of cell areas, major axes and minor axes per segmented cell
    """

    image_dimension = 512

    data = pd.read_csv(filename)

    data = data[data['Length Major Axis'] > 0]      # PREVENT DIVISION BY ZERO

    com_x = data['Center of Mass X']
    com_y = data['Center of Mass Y']
    area = data['Area'] * scale**2
    maj_ax = data['Length Major Axis'] * scale
    min_ax = data['Length Minor Axis'] * scale

    ii = (area < 200) * (area > 5)     # FILTER OBJECTS OUTSIDE REALISTIC CELL SIZE RANGE
    jj = (com_x > 0.1*image_dimension) * (com_x < 0.9*image_dimension) * (com_y > 0.1*image_dimension) * (com_y < 0.9*image_dimension)  # EXCLUDE CELLS CLOSE TO IMAGE BORDERS
    ii *= jj

    area = area[ii]
    maj_ax = maj_ax[ii]
    min_ax = min_ax[ii]

    return area, maj_ax, min_ax


def elongation(maj_ax, min_ax):
    """
    Calculates cell elongation from major and minor axis lengths.

    :param maj_ax: major axis lengths
    :param min_ax: minor axis lengths
    :return: elongation parameter
    """

    return (maj_ax - min_ax) / maj_ax


def pool_data(vars, pool):
    """
    Pools arrays according to a specified grouping.

    :param vars: list of arrays
    :param pool: grouping identifiers for each array
    :return: pooled arrays
    """

    groups = defaultdict(list)

    for arr, label in zip(vars, pool):
        groups[label].append(arr)

    return [np.concatenate(groups[k]) for k in sorted(groups)]


def mirror_histogram(var, bin_boundaries):
    """
    Calculates histogram values mirrored with respect to the x-axis.

    :param var: variable to histogram
    :param bin_boundaries: histogram bin boundaries
    :return: histogram heights, bin positions and bin width
    """

    heights, bins = np.histogram(var, density=True, bins=bin_boundaries)

    heights *= -1

    bin_width = np.diff(bins)[0]
    bin_pos = (bins[:-1] + bin_width / 2)

    return heights, bin_pos, bin_width


def plot_histograms(vars, labels, mirrored, reference, ax, cmap=None, dashes=None):
    """
    Takes a set of variables and draws a histogram of each variable on one set of axes. Indicated variables can be drawn
    mirrored with respect to the x-axis.

    :param vars: list of variable arrays to plot
    :param labels: list of labels for each variable
    :param mirrored: list of booleans; variable will be plotted mirrored if the corresponding boolean is True
    :param ax: axis to plot on
    :param cmap: list of colors for plotting (NB: mirrored plots are all shown in grey)
    """

    # CALCULATE COMMON HISTOGRAM BINNING

    if cmap is None:
        cmap = ['tab:blue', 'tab:orange', 'darkgreen', 'grey']

    if dashes is None:
        dashes = [(1, 0), (3, 1), (1, 1, 3, 1), (1, 1), (0.5, 0.5), (1, 1), (1,1), (1,1)]

    flat_vars = [item for sublist in vars for item in sublist]
    bin_boundaries = np.linspace(0, max(flat_vars), 100)

    # INITIALIZE PLOT VARIABLES

    cell_counts = np.zeros(len(vars))
    j = 0
    limits = np.zeros(len(vars)+1)

    for i in range(len(vars)):

        # DEFINE PLOT PARAMETERS

        cell_counts[i] = len(vars[i])
        # label = labels[i] + ' (%i cells)' %cell_counts[i]
        label = labels[i]
        mean = np.median(vars[i])

        # CREATE MIRRORED HISTOGRAM

        if mirrored[i]:

            heights, bin_pos, bin_width = mirror_histogram(vars[i], bin_boundaries)
            limits[i] = min(heights)

            # ax.bar(bin_pos, heights, width=bin_width, alpha=0.6, label=label, color='grey', linewidth=0.5, edgecolor='grey')

            ax.plot(bin_pos, heights, label=label, color='grey', linewidth=1)
            ax.plot((mean, mean), (0, -1000), c='grey', ls='dashed', lw=1.5)

        # CREATE STANDARD HISTOGRAM

        else:

            # n, bins, patches = ax.hist(vars[i], bins=bin_boundaries, density=True, color=cmap[j], alpha=0.6, label=label, linewidth=0.5, edgecolor='grey')

            heights, bins = np.histogram(vars[i], bins=bin_boundaries, density=True)

            bin_width = np.diff(bins)[0]
            bin_pos = (bins[:-1] + bin_width / 2)

            if reference[i]:

                ax.plot(bin_pos, heights, color='grey', label=label, lw=1.5)
                ax.fill_between(bin_pos, heights, color='grey', alpha=0.5)

                # ax.plot((mean, mean), (0, 1000), c='grey', ls='dashed', lw=1.5)

            else:

                ax.plot(bin_pos, heights, color=cmap[j], dashes=dashes[j], label=label, lw=1.5)

                # ax.plot((mean, mean), (0, 1000), c=cmap[j], ls='dashed', lw=1.5)

                j += 1

            limits[i] = max(heights)

    limits[-1] = 0
    limits = np.nan_to_num(limits)

    # FORMAT AXES

    ax.legend(frameon=False)
    ax.set_ylim(1.05*min(limits), 1.05*max(limits))
    ax.tick_params(labelleft=False)
    ax.set_ylabel('Density')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def create_figure(files, mask_files, labels, pool, mirrored, reference, scales, title='', cmap=None):
    """
    Creates a 2x2 figure, containing histograms for the distribution of cell area, minor axis length and major axis
    length obtained from YeaZ extracted .csv files, as well as a calculated elongation parameter. Each subplot contains
    the histograms from all files provided.
    :param files: list of file directories to be used
    :param labels: list of labels to attach to each respective file
    :param mirrored: list of tuples indicating if the respective histogram should be mirrored wrt the x-axis (True) or
    not (False)
    :param title: title to give to the full figure
    :param scales: scale factors (#pixels / um)
    :param cmap: list of colors to plot with (use default list if None)
    """

    # INITIALIZE DATA LISTS

    areas = []
    maj_axs = []
    min_axs = []
    elongs = []
    circs = []

    # READ DATASETS

    i = 0

    for cvs_file, mask_file in zip(files, mask_files):

        print(f'Working on dataset {i+1}')

        area, maj_ax, min_ax = read_file(cvs_file, scales[i])
        elong = elongation(maj_ax, min_ax)

        areas.append(area)
        maj_axs.append(maj_ax)
        min_axs.append(min_ax)
        elongs.append(elong)

        if mask_file is not None:
            circularities = read_mask(mask_file, scales[i])
            circs.append(1/circularities)
        else:
            circs.append(np.nan*np.zeros(5))

        i += 1

    # POOL DATASETS

    areas = pool_data(areas, pool)
    elongs = pool_data(elongs, pool)

    # CREATE FIGURE

    fig, ax = plt.subplots(1, 2, figsize=(6.5, 2.5))
    fig.suptitle(title)

    plot_histograms(areas, labels=labels, mirrored=mirrored,
                    reference=reference, ax=ax[0], cmap=cmap)

    ax[0].set_title('Cell size')
    ax[0].set_xlabel(r'Area (µm$^2$)')
    ax[0].set_xlim([0, 130])

    plot_histograms(elongs, labels=labels, mirrored=mirrored,
                    reference=reference, ax=ax[1], cmap=cmap)

    ax[1].set_title('Elongation parameter')
    ax[1].set_xlabel(r'Elongation (0-1 $\longleftrightarrow$ round-narrow)')
    ax[1].get_legend().remove()

    plt.tight_layout()


def draw_from_dict(data_collection, labels='default', pool=None, do_mirror=None,
                   reference=None, title='', cmap=None):
    """
    Draw histogram figure from datasets stored as dictionaries, where each dictionary contains a 'path', 'label' and 'scale'.
    :param data_collection: tuple of datasets dictionaries to plot
    :param labels: list of legend labels for each dataset; default is to use labels from dictionary
    :param do_mirror: list of booleans corresponding to each dataset; dataset is plotted mirrored with respect to the
    x-axis if True (default is all False)
    :param title: title of the figure
    :param cmap: list of colors to plot with (use default list if None)
    """

    files = []
    mask_files = []
    plot_labels = []
    scales = []
    mirrored = []

    for dataset in data_collection:

        files.append(dataset["path"])

        try:
            mask_files.append(dataset["mask_path"])
        except KeyError:
            mask_files.append(None)

        if labels == 'default':
            plot_labels.append(dataset["label"])

        scales.append(dataset["scale"])

        if do_mirror is None:
            mirrored.append(False)

    if do_mirror is not None:
        mirrored = do_mirror

    if labels != 'default':
        plot_labels = labels

    create_figure(files, mask_files, plot_labels, pool, mirrored,
                  reference, scales, title=title, cmap=cmap)
# endregion


# Dataset dictionaries contain:
#   path      : YeaZ measurement file (.csv)
#   mask_path : optional segmentation mask (.tif)
#   label     : dataset label used in plots
#   scale     : spatial calibration (µm/pixel)

experiment_dict_1 = {
    "path": "path_to_csv1.csv",
    "label": "Experiment label 1",
    "scale": 0.123
}

experiment_dict_2 = {
    "path": "path_to_csv2.csv",
    "label": "Experiment label 2",
    "scale": 0.123
}

experiment_dict_3 = {
    "path": "path_to_csv3.csv",
    "label": "Experiment label 3",
    "scale": 0.123
}



data_collection = (experiment_dict_1, experiment_dict_2, experiment_dict_3)
pool = [1, 1, 2]
mirrored = [False, False, False]    # DATA IS PLOTTED MIRRORED WITH RESPECT TO THE X-AXIS IF TRUE
reference = [True, False, False]    # DATA IS PLOTTED AS A GREY FILLED IN REFERENCE IF TRUE
labels = ['Alternative label 1', 'Alternative label 2', 'Alternative label 3']
cmap = [sns.color_palette('Set2')[1], sns.color_palette('Set2')[2], sns.color_palette('Set2')[0]]
draw_from_dict(data_collection, labels=labels, pool=pool, do_mirror=mirrored, reference=reference, title='', cmap=cmap)


plt.show()
