import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import Line2D
import pandas as pd
import seaborn as sns
import scienceplots
from scipy.stats import mannwhitneyu
from statannotations.Annotator import Annotator

# Ensure text remains editable in SVG export
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams["mathtext.fontset"] = 'dejavuserif'

from lib.utilities import get_signal_start_end
from lib.prepare_properties import create_property_dataframe

# =========================
# SELECT PROPERTY TO PLOT
# =========================
# Choose ONE property by uncommenting it

# property_to_plot, axis_label = 'window_length', r'Establishment time $\tau_e$ (min)'
# property_to_plot, axis_label = 'step_size_during', 'Mobility during\n' + r'establishment $\Delta s_e$ (µm/20 s)'
# property_to_plot, axis_label = 'increase_rate', r'Signal increase rate $r$ (/min)'

# property_to_plot, axis_label = 'peak_volume', r'Peak spot volume $V$ (µm$^3$)'
# property_to_plot, axis_label = 'circularity', r'Peak spot circularity $c$'
# property_to_plot, axis_label = 'spot_density', r'Peak spot density $\rho$ (µm$^{-3}$)'
# property_to_plot, axis_label = 'peak_intensity', r'Peak spot intensity'

# property_to_plot, axis_label = 'cytosol_intensity', r'Absolute cytosolic intensity'
# property_to_plot, axis_label = 'cell_size', r'Cell size (µm$^2$)'

property_to_plot, axis_label = 'time_to_budding', r'Time before bud emergence $\tau_m$ (min)'
# property_to_plot, axis_label = 'step_size_after', 'Mobility during\n' + r'maintenance $\Delta s_m$ (µm/20 s)'

# =========================
# USER INPUT
# =========================

directories = [
    "path_to_dataset_1/",
    "path_to_dataset_2/",
    "path_to_dataset_3/"
]

dataset_labels = ['Dataset 1', 'Dataset 2', 'Dataset 3']
palette = sns.color_palette('Set1')

subcat = False

# Optional subcategory settings
overall_label = "Combined"
category_labels = ['Group A', 'Group B']
category_colors = ['k', 'r']
category_markers = ['.', '*']

# Scaling parameters
scale = 63.69 / 512   # µm per pixel
frame_time = 20       # seconds per frame

# =========================
# PLOTTING SETUP
# =========================
sns.set_theme('paper')
sns.set_style('whitegrid')
plt.figure(figsize=(2.2, 2.4))

# =========================
# LOAD DATA
# =========================
df = create_property_dataframe(
    directories,
    dataset_labels,
    frame_time=frame_time,
    scale=scale
)

# =========================
# PLOTTING
# =========================
if not subcat:

    ax = sns.stripplot(
        data=df, y=property_to_plot, x='dataset',
        color='k', marker='.', size=3.5, alpha=0.8
    )

    sns.violinplot(
        data=df, y=property_to_plot, x='dataset',
        hue='dataset', legend=False,
        palette=palette, inner='quart',
        cut=1, linewidth=1, alpha=0.9
    )

else:
    # Subcategory handling (unchanged)
    df['subgroup'] = df['dataset'].apply(
        lambda x: x if x in [dataset_labels[1], dataset_labels[2]] else dataset_labels[2]
    )

    df['dataset'] = df['dataset'].replace({
        dataset_labels[1]: overall_label,
        dataset_labels[2]: overall_label
    })

    ax = sns.scatterplot(
        data=df, y=property_to_plot, x='dataset',
        hue='subgroup', style='subgroup',
        palette=category_colors,
        markers=category_markers,
        alpha=0.8, zorder=10, linewidth=0, size=3.5
    )

    # Jitter
    for points in ax.collections:
        vertices = points.get_offsets().data
        if len(vertices) > 0:
            vertices[:, 0] += np.random.uniform(-0.15, 0.15, vertices.shape[0])
            points.set_offsets(vertices)

    xticks = ax.get_xticks()
    ax.set_xlim(xticks[0] - 0.5, xticks[-1] + 0.5)

    sns.violinplot(
        data=df, y=property_to_plot, x='dataset',
        hue='dataset', legend=False,
        palette=palette, inner='quart',
        cut=1, linewidth=1, alpha=0.9, zorder=1
    )

# =========================
# FORMATTING
# =========================
ax.set_xlabel("")
ax.set_ylabel(axis_label)
ax.tick_params(top=False, right=False)
ax.ticklabel_format(style='sci', axis='y', scilimits=(-2, 2))

# =========================
# STATISTICS
# =========================
if len(dataset_labels) > 1:
    pairs = [
        [dataset_labels[i], dataset_labels[j]]
        for i in range(len(dataset_labels))
        for j in range(i + 1, len(dataset_labels))
    ]

    annotator = Annotator(ax, pairs, data=df, x='dataset', y=property_to_plot)
    annotator.configure(test='Mann-Whitney', text_format='star')
    annotator.apply_and_annotate()

# =========================
# SHOW
# =========================
plt.tight_layout()
plt.show()