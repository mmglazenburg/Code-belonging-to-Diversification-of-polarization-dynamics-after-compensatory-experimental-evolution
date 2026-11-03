import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scienceplots
from scipy.stats import mannwhitneyu

# Ensure text remains editable in SVG export
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams["mathtext.fontset"] = 'dejavuserif'

from lib.prepare_properties import create_property_dataframe

# =========================
# SELECT DATASETS
# =========================
# Choose ONE configuration by uncommenting

# Example 1: Multiple conditions
directories = [
    "path_to_dataset_1/",
    "path_to_dataset_2/",
    "path_to_dataset_3/",
    "path_to_dataset_4/",
    "path_to_dataset_5/"
]

dataset_labels = [
    'Condition 1',
    'Condition 2',
    'Condition 3',
    'Condition 4',
    'Condition 5'
]

# Example 2: Simple comparison
# directories = ["path_to_dataset_1/", "path_to_dataset_2/"]
# dataset_labels = ['WT', 'Mutant']

# =========================
# PARAMETERS
# =========================
scale = 63.69 / 512   # µm per pixel
frame_time = 20       # seconds per frame

# Labels for plotted properties (order must match dataframe columns)
property_labels = [
    r'$\tau_e$', r'$r$', r'$\Delta s_e$',
    r'$V$', r'$\rho$', r'$c$',
    r'$\tau_m$', r'$\Delta s_m$'
]

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
# COMPUTE NORMALIZED MEANS + STATISTICS
# =========================
statistics_labels = []

for i, label in enumerate(dataset_labels):

    # Select dataset
    data = df[df['dataset'] == label]

    # Remove non-numeric or unused columns
    data = data.drop(columns=[
        'dataset',
        'replicate',
        'peak_intensity',
        'cytosol_intensity',
        'cell_size'
    ])

    if i == 0:
        # Use first dataset as baseline
        df_normalization = data.mean().to_frame().T
        mean_matrix = df_normalization / df_normalization
        baseline_data = data
        statistics_labels = ['' for _ in range(len(data.columns))]
        continue

    # Normalize means relative to baseline
    data_mean = data.mean().to_frame().T / df_normalization
    mean_matrix = pd.concat([mean_matrix, data_mean], ignore_index=True)

    # Perform statistical tests (Mann–Whitney U)
    for column in data:

        column_data = data[column].dropna()
        baseline_column = baseline_data[column].dropna()

        res = mannwhitneyu(baseline_column, column_data)

        # Convert p-value to significance stars
        if res.pvalue <= 0.0001:
            statistic_string = '****'
        elif res.pvalue <= 0.001:
            statistic_string = '***'
        elif res.pvalue <= 0.01:
            statistic_string = '**'
        elif res.pvalue <= 0.05:
            statistic_string = '*'
        else:
            statistic_string = ''

        statistics_labels.append(statistic_string)

# Reshape statistics labels into matrix
statistics_labels = np.array(statistics_labels)
statistics_labels = statistics_labels.reshape(
    (len(dataset_labels), len(property_labels))
)

# Label rows
mean_matrix.index = dataset_labels

# =========================
# PLOTTING
# =========================
sns.set_theme('paper')
sns.set_style('white')

plt.figure(figsize=(6.2, 2))

ax = sns.heatmap(
    mean_matrix,
    center=1,
    cmap='coolwarm',
    linewidth=5,
    annot=statistics_labels,
    fmt='',
    cbar_kws={'label': 'Fold change with respect to WT'}
)

# Place x-axis labels on top
ax.xaxis.tick_top()
ax.set_xticklabels(property_labels)

plt.tight_layout()
plt.show()
