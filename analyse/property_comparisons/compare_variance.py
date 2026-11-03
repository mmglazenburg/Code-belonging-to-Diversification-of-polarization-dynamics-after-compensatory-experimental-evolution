import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Ensure text remains editable in SVG export
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams["mathtext.fontset"] = 'dejavuserif'

from lib.prepare_properties import create_property_dataframe

# =========================
# SELECT DATASETS
# =========================
# Choose ONE configuration by uncommenting

# Example 1: Bem1-AID comparison
directories = [
    "path_to_dataset_1/",
    "path_to_dataset_2/",
    "path_to_dataset_3/"
]
dataset_labels = ['WT', 'Condition I', 'Condition IV']

# Example 2: Spa2 variants
# directories = [
#     "path_to_dataset_1/",
#     "path_to_dataset_2/",
#     "path_to_dataset_3/",
#     "path_to_dataset_4/",
#     "path_to_dataset_5/"
# ]
# dataset_labels = ['Spa2 (WT)', 'Mutant', 'T3', 'A1', 'A10']

# =========================
# OPTIONAL: COMBINE CATEGORIES
# =========================
# Combine last two categories into one (e.g., grouping similar conditions)
overall_label = 'Combined'
# overall_label = None  # Uncomment to disable grouping

# =========================
# PARAMETERS
# =========================
scale = 63.69 / 512   # µm per pixel
frame_time = 20       # seconds per frame

# =========================
# PLOTTING SETUP
# =========================
sns.set_theme('paper')
sns.set_style('whitegrid')

# =========================
# LOAD DATA
# =========================
df = create_property_dataframe(
    directories,
    dataset_labels,
    frame_time=frame_time,
    scale=scale
)

# Combine last two categories if enabled
if overall_label is not None:
    df['dataset'] = df['dataset'].replace({
        dataset_labels[1]: overall_label,
        dataset_labels[2]: overall_label
    })
    dataset_labels = [dataset_labels[0], overall_label]

# =========================
# SELECT PROPERTIES
# =========================
properties = [
    'window_length',
    'increase_rate',
    'step_size_during',
    'peak_volume',
    'spot_density',
    'circularity'
]

property_labels = [
    r'Establishment time $\tau_e$',
    r'Signal increase rate $r$',
    'Mobility during\n' + r'establishment $\Delta s_e$',
    r'Peak spot volume $V$',
    r'Peak spot density $\rho$',
    r'Peak spot circularity $c$'
]

# =========================
# COMPUTE STANDARD DEVIATIONS
# =========================
stds = np.zeros((len(properties), len(dataset_labels)))

for i, prop in enumerate(properties):
    for j, label in enumerate(dataset_labels):

        # Extract data for current dataset and property
        df_temp = df.loc[df['dataset'] == label]
        data = np.array(df_temp[prop])

        stds[i, j] = np.nanstd(data)

# Normalize each property relative to first dataset
stds_normalized = stds / stds[:, [0]]

# Convert to DataFrame (exclude reference column)
stds_df = pd.DataFrame(stds_normalized[:, 1:])

# =========================
# PLOTTING
# =========================
fig, ax = plt.subplots(figsize=(2, 2.3))

sns.heatmap(
    stds_df,
    cmap='binary',
    linewidth=0.5,
    annot=True,
    fmt='.1f',
    cbar=False
)

# Toggle x-axis labels visibility if desired
# ax.set_xticklabels(dataset_labels[1:])
ax.set_xticklabels([])

ax.set_yticklabels(property_labels, rotation=0)
ax.set_title(r'$\sigma_{\mathrm{other}}$ / $\sigma_{\mathrm{WT}}$')

# =========================
# DISPLAY
# =========================
plt.tight_layout()
plt.show()